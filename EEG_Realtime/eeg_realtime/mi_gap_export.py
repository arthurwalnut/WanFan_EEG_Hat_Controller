"""Derived MI gap exports; canonical raw recordings remain unchanged.

Only declared packet-loss gaps with two measured CSV endpoints are filled.
The resulting CSV is for inspection/export. Opt-in training reconstructs the
same endpoint means from raw samples; it never trusts an edited derived CSV.
Planned cue labels do not certify display timing or usable EEG.
"""
from __future__ import annotations

import csv
from bisect import bisect_right
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from .mi_recording import COLUMNS, FS, _validate_manifest


METHOD = "channelwise_endpoint_mean_v1"
OUTPUT_FILE = "samples_filled.csv"
REPORT_FILE = "gap_fill.json"
EXPORT_COLUMNS = (*COLUMNS, "is_interpolated", "gap_id", "interpolation_method", "label", "trial")
MAX_MANIFEST_BYTES = 2_000_000
MAX_EVENT_LINE_BYTES = 131_072
MAX_INDEX = (1 << 63) - 1


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= MAX_INDEX:
        raise ValueError(f"{name} must be an integer between {minimum} and {MAX_INDEX}")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _json(text):
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(text, object_pairs_hook=_object, parse_constant=nonfinite)


def _file(root, name, *, required=True):
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError(f"Export refuses linked or external files: {name}")
    if path.exists() and not path.is_file():
        raise ValueError(f"Expected a regular file: {name}")
    if required and not path.is_file():
        raise ValueError(f"Recording file is missing: {name}")
    return path


def _fingerprint(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _atomic_report(path, report):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _loss_events(path, limit):
    events, seen_ids = [], set()
    declared_count = 0
    # Heartbeat evidence can dominate this file. Retain packet-loss events only.
    with path.open("rb") as handle:
        line_number = 0
        while True:
            line = handle.readline(MAX_EVENT_LINE_BYTES + 1)
            if not line:
                break
            line_number += 1
            if len(line) > MAX_EVENT_LINE_BYTES:
                raise ValueError(f"Event line {line_number} exceeds the bounded size")
            if not line.strip():
                continue
            row = _json(line.decode("utf-8-sig"))
            if not isinstance(row, dict):
                raise ValueError(f"Event line {line_number} must be an object")
            if row.get("kind") != "packet_loss":
                continue
            event = row.get("details")
            if not isinstance(event, dict):
                raise ValueError(f"packet_loss line {line_number} requires details")
            event = dict(event)
            identifier = _integer(event.get("gap_id"), "gap_id", 1)
            if identifier in seen_ids:
                raise ValueError(f"Duplicate or contradictory packet_loss gap_id: {identifier}")
            seen_ids.add(identifier)
            count = _integer(event.get("missing_samples"), "missing_samples", 1)
            packets = _integer(event.get("missing_packets"), "missing_packets", 1)
            start = _integer(event.get("missing_start_index"), "missing_start_index")
            following = _integer(event.get("next_sample_index"), "next_sample_index", 1)
            if count != packets * 50 or following - start != count:
                raise ValueError(f"Packet/sample/index counts disagree for gap {identifier}")
            if count > limit:
                raise ValueError(f"Gap {identifier} exceeds duration_s * {FS} fill limit")
            _number(event.get("detected_s"), "detected_s")
            _number(event.get("detected_monotonic"), "detected_monotonic")
            if type(event.get("allowed")) is not bool:
                raise ValueError(f"Gap {identifier} requires an explicit allowed boolean")
            previous_interval = event.get("interval_since_previous_s")
            if previous_interval is not None:
                _number(previous_interval, "interval_since_previous_s")
            if _number(event.get("minimum_interval_s"), "minimum_interval_s") != 15:
                raise ValueError(f"Gap {identifier} has an unsupported minimum interval")
            events.append(event)
            declared_count += count
            if declared_count > limit:
                raise ValueError(f"Declared gaps exceed duration_s * {FS} total fill limit")
    events.sort(key=lambda item: item["missing_start_index"])
    for previous, current in zip(events, events[1:]):
        if current["missing_start_index"] < previous["next_sample_index"]:
            raise ValueError("Packet-loss declarations overlap or contradict each other")
    return events


def _sample(row, line):
    if len(row) != len(COLUMNS):
        raise ValueError(f"Malformed raw CSV column count at row {line}")
    try:
        index, segment = int(row[0]), int(row[-2])
        timestamp, received = float(row[1]), float(row[-1])
        values = tuple(float(value) for value in row[2:10])
    except ValueError as exc:
        raise ValueError(f"Invalid raw CSV value at row {line}") from exc
    if (not 0 <= index <= MAX_INDEX or segment < 0 or timestamp < 0 or received < 0
            or not all(math.isfinite(value) for value in (timestamp, received, *values))):
        raise ValueError(f"Nonfinite/negative raw CSV value at row {line}")
    return {"index": index, "time": timestamp, "values": values,
            "segment": segment, "received": received}


def _cue(plan, starts, timestamp):
    index = bisect_right(starts, timestamp) - 1
    if index >= 0:
        phase = plan[index]
        if timestamp < phase["end_s"]:
            return phase["label"], phase["trial"]
    return "interval", 0


def _gap_summary(event, reason=None):
    summary = {key: event[key] for key in ("gap_id", "missing_samples", "missing_packets",
                                         "missing_start_index", "next_sample_index", "allowed")}
    if reason is not None:
        summary["reason"] = reason
    return summary


def _write_csv(source, temporary, events, metadata, report):
    by_start = {event["missing_start_index"]: event for event in events}
    seen = set()
    raw_count = 0
    previous = None
    plan = metadata["plan"]
    phase_starts = [phase["start_s"] for phase in plan]
    with source.open(encoding="utf-8-sig", newline="") as raw, temporary.open("x", encoding="utf-8", newline="") as output:
        reader, writer = csv.reader(raw), csv.writer(output)
        if tuple(next(reader, ())) != COLUMNS:
            raise ValueError("Raw samples.csv header/channel order does not match MI schema")
        writer.writerow(EXPORT_COLUMNS)
        for line_number, row in enumerate(reader, 2):
            current = _sample(row, line_number)
            if previous is not None:
                if (current["index"] <= previous["index"] or current["time"] <= previous["time"]
                        or current["segment"] < previous["segment"] or current["received"] < previous["received"]):
                    raise ValueError(f"Nonmonotonic raw CSV metadata at row {line_number}")
                missing_start = previous["index"] + 1
                missing = current["index"] - missing_start
                if missing:
                    event = by_start.get(missing_start)
                    if event is None:
                        report["unfilled_gaps"].append({"gap_id": None, "missing_start_index": missing_start,
                            "next_sample_index": current["index"], "missing_samples": missing,
                            "reason": "undeclared_gap"})
                    else:
                        seen.add(event["gap_id"])
                        span = current["time"] - previous["time"]
                        expected = (missing + 1) / FS
                        reason = None
                        if missing != event["missing_samples"] or current["index"] != event["next_sample_index"]:
                            reason = "csv_indices_disagree_with_event"
                        elif not .75 * expected - 1e-9 <= span <= 1.25 * expected + 1e-9:
                            reason = "timestamp_span_disagrees_with_missing_samples"
                        elif (previous["time"] >= metadata["duration_s"]
                              or current["time"] > metadata["duration_s"] + 1.25 / FS):
                            reason = "gap_outside_recording_duration"
                        if reason:
                            report["unfilled_gaps"].append(_gap_summary(event, reason))
                        else:
                            means = tuple(left / 2 + right / 2 for left, right in zip(previous["values"], current["values"]))
                            for offset in range(1, missing + 1):
                                timestamp = previous["time"] + span * (offset / (missing + 1))
                                if not previous["time"] < timestamp < current["time"]:
                                    raise ValueError("Interpolated timestamp is not strictly bracketed")
                                label, trial = _cue(plan, phase_starts, timestamp)
                                writer.writerow((missing_start + offset - 1, timestamp, *means, -1, "",
                                                 1, event["gap_id"], METHOD, label, trial))
                            report["filled_sample_count"] += missing
                            report["filled_gaps"].append(dict(_gap_summary(event),
                                left_index=previous["index"], right_index=current["index"],
                                left_time_s=previous["time"], right_time_s=current["time"]))
            label, trial = _cue(plan, phase_starts, current["time"])
            # Retain every original field string. Only derived columns are added.
            writer.writerow((*row, 0, "", "", label, trial))
            raw_count += 1
            if raw_count > metadata["sample_count"]:
                raise ValueError("Raw CSV contains more rows than the closed manifest")
            previous = current
        if raw_count != metadata["sample_count"]:
            raise ValueError("Raw CSV row count differs from the closed manifest")
        for event in events:
            if event["gap_id"] not in seen:
                report["unfilled_gaps"].append(_gap_summary(event, "no_matching_measured_endpoints"))
        output.flush()
        os.fsync(output.fileno())
    report["raw_sample_count"] = raw_count
    report["raw_sample_count_verified"] = True
    report["total_sample_count"] = raw_count + report["filled_sample_count"]


def export_gap_filled(path):
    """Return/persist an audit and, when needed, an atomically published export.

    Call after the recorder has closed, outside the acquisition lock. Invalid
    declarations/raw CSV return status ``error`` with no newly published CSV.
    A failure to persist the report propagates to the caller. Original files
    are opened read-only, and recording time/labels are never shifted.
    """
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Gap export requires a regular recording directory")
    root = root.resolve()
    report_path = _file(root, REPORT_FILE, required=False)
    output_path = _file(root, OUTPUT_FILE, required=False)
    report = {"schema_version": 1, "status": "error", "output_file": None, "method": METHOD,
              "raw_sample_count": 0, "filled_sample_count": 0, "total_sample_count": 0,
              "raw_sample_count_verified": False, "filled_gaps": [], "unfilled_gaps": [], "errors": [],
              "label_source": "original_plan_at_unchanged_time_s",
              "training_source": "samples.csv",
              "note": "Inserted values are synthetic endpoint means, not measured EEG. Replay uses raw samples; optional filled-data training rebuilds verified gaps from the raw files."}
    temporary = output_path.with_name(f".{OUTPUT_FILE}.{uuid4().hex}.tmp")
    try:
        manifest_path = _file(root, "manifest.json")
        events_path = _file(root, "events.jsonl")
        raw_path = _file(root, "samples.csv")
        original_state = {item: _fingerprint(item) for item in (manifest_path, events_path, raw_path)}
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest exceeds the bounded size")
        metadata = _json(manifest_path.read_text(encoding="utf-8-sig"))
        _validate_manifest(metadata, root)
        if metadata["status"] == "recording":
            raise ValueError("Gap export requires a closed recording")
        report["raw_sample_count"] = report["total_sample_count"] = metadata["sample_count"]
        limit = math.floor(metadata["duration_s"] * FS)
        events = _loss_events(events_path, limit)
        if events:
            _write_csv(raw_path, temporary, events, metadata, report)
        for original, fingerprint in original_state.items():
            if _fingerprint(original) != fingerprint:
                raise ValueError(f"Recording changed during gap export: {original.name}")
        if events:
            os.replace(temporary, output_path)
            report["status"] = "partial" if report["unfilled_gaps"] else "complete"
            report["output_file"] = OUTPUT_FILE
        else:
            report["status"] = "not_needed"
    except Exception as exc:
        report.update(status="error", output_file=None, filled_sample_count=0,
                      total_sample_count=report["raw_sample_count"], filled_gaps=[])
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if temporary.exists():
            temporary.unlink()
    _atomic_report(report_path, report)
    return report


def fill_training_samples(path, metadata, samples, times, indices, segments, received):
    """Build an in-memory training view, preserving raw files and clock evidence.

    Modern records require matching packet-loss declarations. Older records can
    identify missing packets through their sample indices only when packet size,
    timestamp span and segment transition agree. Every other boundary remains.
    """
    import numpy as np

    limit = math.floor(metadata["duration_s"] * FS)
    events = _loss_events(Path(path) / "events.jsonl", limit)
    declared = {event["missing_start_index"]: event for event in events}
    modern = "packet_loss_policy" in metadata or bool(events)
    report = {"use_filled": True, "method": METHOD, "status": "not_needed",
              "source_file": "samples.csv", "derived_csv_trusted": False,
              "raw_sample_count": len(samples), "filled_sample_count": 0,
              "filled_gaps": [], "unfilled_gaps": [], "cross_trial_rejected_ids": [],
              "note": "Synthetic endpoint means; raw timestamps and display/clock checks retained."}
    matches, seen = {}, set()
    corrections = np.zeros(len(indices), dtype=np.int64)
    for right in np.flatnonzero(np.diff(indices) > 1) + 1:
        left = int(right) - 1
        start, following = int(indices[left]) + 1, int(indices[right])
        missing = following - start
        event = declared.get(start)
        item = {"missing_start_index": start, "next_sample_index": following,
                "missing_samples": missing, "source": "packet_loss_event" if event else "legacy_index_and_time"}
        reason = None
        if event:
            seen.add(event["gap_id"])
            item["gap_id"] = event["gap_id"]
        if metadata.get("source") != "device":
            reason = "not_device_recording"
        elif modern and event is None:
            reason = "undeclared_gap"
        elif event and (event["missing_samples"] != missing or event["next_sample_index"] != following):
            reason = "csv_indices_disagree_with_event"
        elif missing % 50:
            reason = "not_whole_packets"
        elif missing + report["filled_sample_count"] > limit:
            reason = "gap_exceeds_recording_duration"
        elif int(segments[right]) - int(segments[left]) not in (0, 1):
            reason = "unexplained_segment_transition"
        elif received[right] <= received[left]:
            reason = "missing_distinct_packet_receipts"
        elif not .75 * (missing + 1) / FS - 1e-9 <= times[right] - times[left] <= 1.25 * (missing + 1) / FS + 1e-9:
            reason = "timestamp_span_disagrees_with_missing_samples"
        elif times[right] > metadata["duration_s"] + 1.25 / FS:
            reason = "gap_outside_recording_duration"
        if reason:
            report["unfilled_gaps"].append(dict(item, reason=reason))
            continue
        item.update(left_index=int(indices[left]), right_index=following,
                    left_time_s=float(times[left]), right_time_s=float(times[right]))
        task_at = lambda value: next((phase["trial"] for phase in metadata["plan"]
                                    if phase["phase"] == "task" and phase["start_s"] <= value < phase["end_s"]), 0)
        first_trial, last_trial = task_at(times[left]), task_at(times[right])
        if first_trial and last_trial and first_trial != last_trial:
            affected = [phase["trial"] for phase in metadata["plan"] if phase["phase"] == "task"
                        and phase["start_s"] <= times[right] and phase["end_s"] > times[left]]
            report["cross_trial_rejected_ids"] = sorted(set(report["cross_trial_rejected_ids"]) | set(affected))
            item["cross_trial_rejected_ids"] = affected
        report["filled_gaps"].append(item)
        report["filled_sample_count"] += missing
        matches[int(right)] = missing
        corrections[right] = int(segments[right]) - int(segments[left])
    for event in events:
        if event["gap_id"] not in seen:
            report["unfilled_gaps"].append(dict(_gap_summary(event), reason="no_matching_measured_endpoints"))
    # A segment change without a missing index remains an unbridged boundary.
    for right in np.flatnonzero((np.diff(segments) != 0) & (np.diff(indices) == 1)) + 1:
        report["unfilled_gaps"].append({"missing_start_index": int(indices[right]),
                                       "missing_samples": 0, "reason": "segment_boundary_without_missing_indices"})
    report["status"] = "partial" if report["unfilled_gaps"] else "complete" if matches else "not_needed"
    if not matches:
        return samples, times, indices, segments, received, np.zeros(len(samples), dtype=bool), report
    joined_segments = segments - np.cumsum(corrections)
    parts = [[] for _ in range(6)]
    start = 0
    for right, missing in matches.items():
        left = right - 1
        raw_parts = (samples[start:right], times[start:right], indices[start:right],
                     joined_segments[start:right], received[start:right], np.zeros(right - start, dtype=bool))
        for bucket, value in zip(parts, raw_parts):
            bucket.append(value)
        offsets = np.arange(1, missing + 1, dtype=np.int64)
        fill_times = times[left] + (times[right] - times[left]) * offsets / (missing + 1)
        mean = samples[left] / 2 + samples[right] / 2
        fill_parts = (np.broadcast_to(mean, (missing, len(mean))), fill_times, indices[left] + offsets,
                      np.full(missing, joined_segments[left], dtype=np.int64),
                      np.full(missing, np.nan), np.ones(missing, dtype=bool))
        for bucket, value in zip(parts, fill_parts):
            bucket.append(value)
        start = right
    tail = (samples[start:], times[start:], indices[start:], joined_segments[start:], received[start:],
            np.zeros(len(samples) - start, dtype=bool))
    for bucket, value in zip(parts, tail):
        bucket.append(value)
    return *(np.concatenate(bucket) for bucket in parts), report
