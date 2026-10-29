"""Raw eight-channel MI sessions, reproducible plans and conservative cue alignment.

This module neither connects to the EEG board nor modifies vendor recordings.
All timestamps are software estimates; display acknowledgements are not hardware
triggers. A recording remains replayable when its cues are unsuitable for training.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

FS = 250
CHANNELS = ("O2", "O1", "P8", "P7", "C4", "C3", "FP1", "FP2")
LABELS = ("rest", "left_mi", "right_mi")
ACK_TOLERANCE_S = 0.25
SCHEMA_VERSION = 1
RECEIPT_CLOCK_METHOD = "receipt-envelope-5s-bounded-interpolation-v1"
DISPLAY_TIMING_METHOD = "raf-visible-heartbeat-v1"
HEARTBEAT_MAX_GAP_S = 0.35
DISPLAY_CLOCK_UNCERTAINTY_S = 0.05
COLUMNS = ("sample_index", "time_s", *CHANNELS, "segment", "receive_monotonic")
TIMING_NOTE = (
    "Raw samples are unfiltered microvolts. time_s is an estimated sampling time "
    "relative to recording start, not a device timestamp. Cue acknowledgements "
    "describe software display timing, not hardware synchronization. Only complete "
    "continuous tasks with a first display acknowledgement within 0.25 seconds of "
    "planned onset are eligible for training. New display-timing recordings also "
    "require continuing visible-frame evidence through the task end. Receipt-clock "
    "recordings require a valid clock anchor; sample labels retain the original plan."
)


def _number(value, name, *, minimum=0.0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _duration(value):
    duration = _number(value, "duration_s")
    if duration <= 0 or duration > 86400:
        raise ValueError("duration_s must be greater than zero and at most 86400")
    return duration


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def make_plan(duration_s, seed=None):
    """Start with a 2 s cross, then balanced random 5 s tasks separated by 2 s.

    A tail too short for another complete task remains one interval. Task trials
    are one-based; interval trials are zero. Phase indices are zero-based.
    """
    duration = _duration(duration_s)
    if seed is not None:
        seed = _integer(seed, "seed")
    rng = random.Random(seed)
    phases, bag = [], []
    position, trial = 0.0, 0
    while position < duration:
        interval_end = min(duration, position + 2.0)
        if duration - interval_end < 5.0:
            interval_end = duration
        phases.append({"index": len(phases), "phase": "interval", "label": "interval", "trial": 0,
                       "start_s": position, "end_s": interval_end})
        position = interval_end
        if position >= duration:
            break
        if not bag:
            bag = list(LABELS)
            rng.shuffle(bag)
        trial += 1
        phases.append({"index": len(phases), "phase": "task", "label": bag.pop(), "trial": trial,
                       "start_s": position, "end_s": position + 5.0})
        position += 5.0
    return phases


def _atomic_manifest(path, manifest):
    payload = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _json_load(text, where):
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid JSON in {where}: {exc}") from exc


def _validate_manifest(metadata, path):
    if not isinstance(metadata, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported MI recording schema: {path}")
    if metadata.get("sample_rate") != FS or isinstance(metadata.get("sample_rate"), bool):
        raise ValueError("MI recording sample_rate must be 250")
    if metadata.get("channel_names") != list(CHANNELS):
        raise ValueError("MI recording channel_names/order do not match the fixed eight electrodes")
    if metadata.get("units") != "uV":
        raise ValueError("MI recording units must be uV; implicit rescaling is not allowed")
    duration = _duration(metadata.get("duration_s"))
    seed = _integer(metadata.get("seed"), "manifest seed")
    if metadata.get("plan") != make_plan(duration, seed):
        raise ValueError("Manifest plan does not match duration, seed or the fixed paradigm")
    if metadata.get("status") not in ("recording", "complete", "aborted", "error", "stopped"):
        raise ValueError("Invalid MI recording status")
    for key in ("id", "source", "created_utc", "timing_note"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"Manifest {key} must be a nonempty string")
    if not isinstance(metadata.get("subject", ""), str):
        raise ValueError("Manifest subject must be a string")
    if metadata.get("display_timing_method") not in (None, DISPLAY_TIMING_METHOD):
        raise ValueError("Unsupported recording display_timing_method")
    _integer(metadata.get("sample_count"), "manifest sample_count")
    return metadata


def _ack_details(details, plan):
    if not isinstance(details, dict):
        raise ValueError("Acknowledgement details must be an object")
    index = _integer(details.get("phase_index"), "phase_index")
    if index >= len(plan):
        raise ValueError("Acknowledgement phase_index is outside the plan")
    onset = _number(details.get("onset_s"), "onset_s")
    received = _number(details.get("received_s"), "received_s")
    if received < onset:
        raise ValueError("Acknowledgement receipt cannot precede its displayed onset")
    valid = abs(onset - plan[index]["start_s"]) <= ACK_TOLERANCE_S + 1e-9
    return {"phase_index": index, "onset_s": onset, "received_s": received, "timing_valid": valid}


class SessionRecorder:
    """One writer-owned session. The caller serializes write/event/close calls."""

    def __init__(self, folder_base, source, duration_s, seed=None, subject="", clock_method=None,
                 display_timing_method=None):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a nonempty string")
        if not isinstance(subject, str):
            raise ValueError("subject must be a string")
        if clock_method is not None and (not isinstance(clock_method, str) or not clock_method.strip()):
            raise ValueError("clock_method must be a nonempty string when provided")
        if display_timing_method is not None and display_timing_method != DISPLAY_TIMING_METHOD:
            raise ValueError("Unsupported display_timing_method")
        duration = _duration(duration_s)
        actual_seed = random.SystemRandom().randrange(2**63) if seed is None else _integer(seed, "seed")
        self.id = f"mi_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"
        self.path = Path(folder_base) / self.id
        self.plan = make_plan(duration, actual_seed)
        self.manifest = {"schema_version": SCHEMA_VERSION, "id": self.id, "source": source.strip(),
                         "created_utc": _utc_now(), "sample_rate": FS, "channel_names": list(CHANNELS),
                         "units": "uV", "duration_s": duration, "seed": actual_seed, "subject": subject,
                         "plan": self.plan, "status": "recording", "sample_count": 0,
                         "timing_note": TIMING_NOTE, "acknowledgements": []}
        if clock_method is not None:
            self.manifest["clock_method"] = clock_method
        if display_timing_method is not None:
            self.manifest["display_timing_method"] = display_timing_method
        self._last_index = self._last_time = self._last_segment = self._last_received = None
        self._acks = {}
        self._closed = False
        self._write_recovery_error = None
        self.path.mkdir(parents=True, exist_ok=False)
        self._samples_handle = self._events_handle = None
        try:
            self._samples_handle = (self.path / "samples.csv").open("x", encoding="utf-8", newline="")
            self._events_handle = (self.path / "events.jsonl").open("x", encoding="utf-8", newline="\n")
            self._writer = csv.writer(self._samples_handle)
            self._writer.writerow(COLUMNS)
            self._samples_handle.flush()
            _atomic_manifest(self.path / "manifest.json", self.manifest)
            self.event("session_started", {"source": source.strip(), "duration_s": duration, "seed": actual_seed})
        except Exception:
            for handle in (self._samples_handle, self._events_handle):
                if handle is not None:
                    handle.close()
            self._closed = True
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("MI recording is already closed")

    def write(self, samples, sample_indices, times, segment, received_monotonic):
        self._ensure_open()
        values = np.asarray(samples, dtype=np.float64)
        indices = np.asarray(sample_indices)
        timestamps = np.asarray(times, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(CHANNELS):
            raise ValueError("samples must have shape (N, 8)")
        if indices.shape != (len(values),) or timestamps.shape != (len(values),):
            raise ValueError("sample_indices and times must each have shape (N,)")
        if indices.dtype.kind not in "iu" or indices.dtype.kind == "b":
            raise ValueError("sample_indices must contain integers")
        if not np.isfinite(values).all() or not np.isfinite(timestamps).all():
            raise ValueError("Samples and timestamps must be finite")
        if np.any(indices < 0) or np.any(indices > np.iinfo(np.int64).max) or np.any(timestamps < 0):
            raise ValueError("Sample indices/timestamps must be nonnegative and indices must fit int64")
        indices = indices.astype(np.int64, copy=False)
        segment = _integer(segment, "segment")
        received = _number(received_monotonic, "received_monotonic")
        if not len(values):
            return
        if (np.any(np.diff(indices) <= 0) or np.any(np.diff(timestamps) <= 0)
                or (self._last_index is not None and indices[0] <= self._last_index)
                or (self._last_time is not None and timestamps[0] <= self._last_time)):
            raise ValueError("Sample indices and timestamps must increase strictly across the session")
        if self._last_segment is not None and segment < self._last_segment:
            raise ValueError("Segment identifiers cannot decrease")
        if self._last_received is not None and received < self._last_received:
            raise ValueError("Receive monotonic time cannot decrease")
        rows = [[int(index), float(timestamp), *map(float, sample), segment, received]
                for index, timestamp, sample in zip(indices, timestamps, values)]
        position = self._samples_handle.tell()
        try:
            self._writer.writerows(rows)
            self._samples_handle.flush()
        except Exception as exc:
            try:
                self._samples_handle.seek(position)
                self._samples_handle.truncate()
                self._samples_handle.flush()
                os.fsync(self._samples_handle.fileno())
            except OSError as recovery_error:
                self._write_recovery_error = str(recovery_error)
                self.manifest.update(recovery_required=True, write_error=str(exc),
                                     rollback_error=str(recovery_error))
                try:
                    _atomic_manifest(self.path / "manifest.json", self.manifest)
                except OSError:
                    pass  # Both failures still propagate to the caller.
                raise OSError(f"Sample write failed ({exc}); packet rollback also failed ({recovery_error}); "
                              "recording requires explicit recovery") from exc
            raise
        self.manifest["sample_count"] += len(values)
        self._last_index, self._last_time = int(indices[-1]), float(timestamps[-1])
        self._last_segment, self._last_received = segment, received

    def event(self, kind, details):
        self._ensure_open()
        if not isinstance(kind, str) or not kind or not isinstance(details, dict):
            raise ValueError("Events require a nonempty kind and dictionary details")
        payload = json.dumps({"kind": kind, "recorded_utc": _utc_now(), "details": details},
                             ensure_ascii=False, allow_nan=False)
        self._events_handle.write(payload + "\n")
        self._events_handle.flush()

    def acknowledge(self, phase_index, onset_s, received_s):
        self._ensure_open()
        ack = _ack_details({"phase_index": phase_index, "onset_s": onset_s, "received_s": received_s}, self.plan)
        duplicate = phase_index in self._acks
        self.event("display_ack", {**ack, "duplicate": duplicate})
        if duplicate:
            return False
        self._acks[phase_index] = ack
        self.manifest["acknowledgements"].append(dict(ack))
        return ack["timing_valid"]

    def close(self, status="complete", reason=""):
        if self._closed:
            return
        if status not in ("complete", "aborted", "error", "stopped") or not isinstance(reason, str):
            raise ValueError("Invalid close status/reason")
        failure = None
        try:
            if self._write_recovery_error:
                raise OSError(f"Cannot finalize recording after failed packet rollback: {self._write_recovery_error}")
            self.event("session_closed", {"status": status, "reason": reason,
                                           "sample_count": self.manifest["sample_count"]})
            for handle in (self._samples_handle, self._events_handle):
                handle.flush()
                os.fsync(handle.fileno())
            final_manifest = dict(self.manifest, status=status, reason=reason, closed_utc=_utc_now())
            _atomic_manifest(self.path / "manifest.json", final_manifest)
            self.manifest.update(final_manifest)
        except BaseException as exc:
            failure = exc
        finally:
            self._closed = True
            for handle in (self._samples_handle, self._events_handle):
                try:
                    handle.close()
                except OSError as exc:
                    if failure is None:
                        failure = exc
        if failure is not None:
            raise failure


def _read_manifest(path):
    metadata = _json_load((path / "manifest.json").read_text(encoding="utf-8-sig"), path / "manifest.json")
    return _validate_manifest(metadata, path)


def _timestamp_audit(metadata, events, times, received, trials):
    """Check software sample estimates against same-process packet receipt time.

    This establishes a necessary clock-consistency condition, not hardware cue
    synchronization. Missing anchors remain visible as unverified legacy data.
    Never repair or relabel the original recording while loading it.
    """
    required = metadata.get("clock_method") == RECEIPT_CLOCK_METHOD
    audit = {"schema_version": 1, "available": False, "status": "unverified", "anchor_required": required,
             "threshold_s": ACK_TOLERANCE_S, "packet_count": 0, "bad_packet_count": 0,
             "bad_sample_count": 0, "max_future_lead_s": None, "min_packet_lead_s": None,
             "max_packet_lead_s": None, "rejected_trial_ids": [], "trial_details": [],
             "warning": "缺少可校验的录制时钟锚点；未证明采样时间与显示提示同步。",
             "note": "按相同接收时间的连续样本组检查包末时间；未检出超前不代表硬件同步，网络延迟仍可能存在。"}
    bad_samples = np.zeros(len(times), dtype=bool)
    def unverified():
        if required:
            audit["rejected_trial_ids"] = [p["trial"] for p in metadata["plan"] if p["phase"] == "task"]
            audit["warning"] = "新录制缺少唯一有效的时钟锚点，所有任务均排除训练；原始数据仍可回放。"
        return audit, bad_samples
    anchors = [event["details"].get("start_monotonic") for event in events if event["kind"] == "clock_anchor"]
    if len(anchors) != 1:
        return unverified()
    try:
        anchor = _number(anchors[0], "clock anchor")
    except ValueError:
        return unverified()
    audit.update(available=True, status="consistent", warning="", max_future_lead_s=0.0)
    if not len(times):
        return audit, bad_samples
    boundaries = [0, *map(int, np.flatnonzero(np.diff(received) != 0) + 1), len(times)]
    leads, details = [], {}
    phases = {phase["trial"]: phase for phase in metadata["plan"] if phase["phase"] == "task"}
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        lead = float(anchor + times[end - 1] - received[end - 1])
        leads.append(lead)
        if lead <= ACK_TOLERANCE_S + 1e-9:
            continue
        audit["bad_packet_count"] += 1
        bad_samples[start:end] = True
        for trial in np.unique(trials[start:end]):
            trial = int(trial)
            if trial <= 0:
                continue
            phase = phases[trial]
            detail = details.setdefault(trial, {"trial_id": trial, "label": phase["label"],
                "planned_start_s": phase["start_s"], "planned_end_s": phase["end_s"],
                "bad_packet_count": 0, "bad_sample_count": 0, "max_future_lead_s": 0.0})
            detail["bad_packet_count"] += 1
            detail["bad_sample_count"] += int(np.sum(trials[start:end] == trial))
            detail["max_future_lead_s"] = max(detail["max_future_lead_s"], lead)
    audit.update(packet_count=len(leads), bad_sample_count=int(bad_samples.sum()),
                 min_packet_lead_s=min(leads), max_packet_lead_s=max(leads),
                 max_future_lead_s=max(0.0, max(leads)), rejected_trial_ids=sorted(details),
                 trial_details=[details[trial] for trial in sorted(details)])
    if audit["bad_packet_count"]:
        audit.update(status="suspect", warning=(f"估计采样时间比实际接收最多超前 {audit['max_future_lead_s']:.3f} 秒；"
                     f"{len(details)} 个涉及任务因时钟不可信排除训练，原数据与计划标签保留。"))
    return audit, bad_samples


def _display_audit(metadata, events, acks, reasons):
    """Audit complete task eligibility without relabeling any original sample.

    New recordings need continuing visible-frame evidence, including the task
    tail. An onset acknowledgement alone cannot prove the cue stayed visible.
    Legacy recordings retain their old eligibility, explicitly unverified here.
    """
    method = metadata.get("display_timing_method")
    required = method == DISPLAY_TIMING_METHOD
    tasks = [phase for phase in metadata["plan"] if phase["phase"] == "task"]
    heartbeats, invalid_heartbeats, interruptions = [], [], []
    last_observed = None
    for event in events:
        details = event["details"]
        if event["kind"] == "display_interruption":
            since = _number(details.get("invalid_since_s"), "display interruption time")
            received = _number(details.get("received_s"), "display interruption receipt")
            if since > received + DISPLAY_CLOCK_UNCERTAINTY_S + 1e-9:
                raise ValueError("Display interruption cannot follow its receipt")
            interruptions.append(since)
        elif event["kind"] == "display_heartbeat":
            observed = _number(details.get("observed_s"), "display heartbeat time")
            received = _number(details.get("received_s"), "display heartbeat receipt")
            uncertainty = _number(details.get("clock_uncertainty_s"), "display clock uncertainty")
            index = details.get("phase_index")
            valid = (uncertainty <= DISPLAY_CLOCK_UNCERTAINTY_S + 1e-9
                     and -uncertainty - 1e-9 <= received - observed <= .6 + 1e-9
                     and observed <= metadata["duration_s"] + .6 + 1e-9
                     and details.get("visible", True) is True)
            if index is None:
                valid = valid and observed >= metadata["duration_s"] - 1e-9
            else:
                index = _integer(index, "display heartbeat phase_index")
                if index >= len(metadata["plan"]):
                    raise ValueError("Display heartbeat phase_index is outside the plan")
                phase = metadata["plan"][index]
                valid = valid and (phase["start_s"] - ACK_TOLERANCE_S - 1e-9 <= observed
                                   <= phase["end_s"] + HEARTBEAT_MAX_GAP_S + 1e-9)
            if last_observed is not None and observed < last_observed - 1e-9:
                valid = False
            last_observed = max(observed, last_observed) if last_observed is not None else observed
            (heartbeats if valid else invalid_heartbeats).append((observed, index))
    since = min(interruptions) if interruptions else None
    for phase in tasks:
        failures = reasons[phase["trial"]]
        if since is not None and phase["end_s"] > since + 1e-9:
            failures.append("display_interrupted")
        if not required:
            continue
        ack = acks.get(phase["index"])
        if not ack or not ack["timing_valid"]:
            continue  # The onset failure is already recorded by the loader.
        start = max(phase["start_s"], ack["onset_s"])
        next_acks = [value["onset_s"] for index, value in acks.items() if index > phase["index"]]
        end = min([phase["end_s"], *next_acks])
        if any(start <= observed < end for observed, _ in invalid_heartbeats):
            failures.append("heartbeat_invalid")
        if any(start <= observed < end and index != phase["index"] for observed, index in heartbeats):
            failures.append("heartbeat_phase_mismatch")
        inside = [observed for observed, index in heartbeats
                  if index == phase["index"] and start <= observed < end]
        if not inside:
            failures.append("heartbeat_missing")
        # A later visible phase (including the terminal screen) or an actual
        # next-phase ACK can close the coverage; a near-end heartbeat cannot.
        tails = [observed for observed, index in heartbeats
                 if observed >= end - 1e-9 and (index is None or index >= phase["index"])]
        tails.extend(onset for onset in next_acks if onset >= end - 1e-9)
        if not tails:
            failures.append("heartbeat_tail")
        else:
            proof = sorted([start, *inside, min(tails)])
            if any(b - a > HEARTBEAT_MAX_GAP_S + 1e-9 for a, b in zip(proof, proof[1:])):
                failures.append("heartbeat_gap")
    per_class = {label: {"planned_trials": 0, "eligible_trials": 0, "rejected_trials": 0} for label in LABELS}
    details, eligible, rejected, counts = [], [], [], {}
    for phase in tasks:
        trial = phase["trial"]
        failures = sorted(set(reasons[trial]))
        ok = not failures
        (eligible if ok else rejected).append(trial)
        per_class[phase["label"]]["planned_trials"] += 1
        per_class[phase["label"]]["eligible_trials" if ok else "rejected_trials"] += 1
        for reason in failures:
            counts[reason] = counts.get(reason, 0) + 1
        details.append({"trial_id": trial, "label": phase["label"], "eligible": ok, "reasons": failures})
    status = ("suspect" if rejected else "verified") if required else "unverified"
    return {"schema_version": 1, "method": method, "status": status,
            "planned_trials": len(tasks), "eligible_trials": len(eligible), "rejected_trials": len(rejected),
            "eligible_trial_ids": eligible, "rejected_trial_ids": rejected, "per_class": per_class,
            "rejection_reasons": counts, "trial_details": details,
            "heartbeat_count": len(heartbeats), "invalid_heartbeat_count": len(invalid_heartbeats),
            "max_heartbeat_gap_s": HEARTBEAT_MAX_GAP_S,
            "warning": ("旧录制只有起始显示确认，未验证提示持续可见；保留原有时序筛选规则。" if not required
                        else ("部分任务未通过采样时序或持续显示校验，已排除训练。" if rejected else "")),
            "note": "eligible 仅表示时序筛选通过，信号硬质量及训练样本量另行校验；软件显示确认不是硬件同步。"}


def load_session(path, *, use_filled=False):
    """Read raw samples, optionally reconstructing verified packet-loss gaps.

    A valid manifest left in ``recording`` status by interruption is recoverable
    from its flushed CSV and event log. Corrupted/truncated rows are reported,
    never silently discarded. Completed earlier tasks can survive an aborted run.
    """
    if type(use_filled) is not bool:
        raise ValueError("use_filled must be an explicit boolean")
    path = Path(path)
    metadata = _read_manifest(path)
    events, acks = [], {}
    with (path / "events.jsonl").open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            event = _json_load(line, f"{path / 'events.jsonl'}:{line_number}")
            if not isinstance(event, dict) or not isinstance(event.get("kind"), str) or not isinstance(event.get("details"), dict):
                raise ValueError(f"Malformed event at line {line_number}")
            events.append(event)
            if event["kind"] == "display_ack":
                ack = _ack_details(event["details"], metadata["plan"])
                acks.setdefault(ack["phase_index"], ack)
    indices, timestamps, values, segments, received = [], [], [], [], []
    with (path / "samples.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader, ())) != COLUMNS:
            raise ValueError("MI samples.csv header/channel order does not match the schema")
        for row_number, row in enumerate(reader, 2):
            if len(row) != len(COLUMNS):
                raise ValueError(f"Malformed sample column count at CSV row {row_number}")
            try:
                index, segment = int(row[0]), int(row[-2])
                timestamp, arrival = float(row[1]), float(row[-1])
                sample = [float(value) for value in row[2:10]]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sample value at CSV row {row_number}") from exc
            if (index < 0 or index > np.iinfo(np.int64).max or segment < 0 or timestamp < 0 or arrival < 0
                    or not all(math.isfinite(value) for value in [timestamp, arrival, *sample])):
                raise ValueError(f"Nonfinite/negative sample metadata at CSV row {row_number}")
            if indices and (index <= indices[-1] or timestamp <= timestamps[-1] or segment < segments[-1] or arrival < received[-1]):
                raise ValueError(f"Nonmonotonic sample metadata at CSV row {row_number}")
            indices.append(index); timestamps.append(timestamp); values.append(sample); segments.append(segment); received.append(arrival)
    if metadata["status"] != "recording" and metadata["sample_count"] != len(values):
        raise ValueError("Manifest sample_count differs from the closed CSV")
    if metadata["status"] == "recording":
        metadata = dict(metadata, recovered=True, sample_count=len(values))
    samples = np.asarray(values, dtype=np.float64).reshape((-1, len(CHANNELS)))
    times = np.asarray(timestamps, dtype=np.float64)
    indices_array, segment_array = np.asarray(indices, dtype=np.int64), np.asarray(segments, dtype=np.int64)
    receipt_array = np.asarray(received, dtype=np.float64)
    interpolated = np.zeros(len(samples), dtype=bool)
    training_data = {"use_filled": False, "status": "disabled", "source_file": "samples.csv",
                     "raw_sample_count": len(samples), "filled_sample_count": 0,
                     "filled_gaps": [], "unfilled_gaps": []}
    if use_filled:
        from .mi_gap_export import fill_training_samples
        samples, times, indices_array, segment_array, receipt_array, interpolated, training_data = fill_training_samples(
            path, metadata, samples, times, indices_array, segment_array, receipt_array)
    labels = np.full(len(times), "interval", dtype="U16")
    trials = np.zeros(len(times), dtype=np.int64)
    timing_valid = np.zeros(len(times), dtype=bool)
    reasons = {phase["trial"]: [] for phase in metadata["plan"] if phase["phase"] == "task"}
    receipt_clock = metadata.get("clock_method") == RECEIPT_CLOCK_METHOD
    for phase in metadata["plan"]:
        mask = (times >= phase["start_s"]) & (times < phase["end_s"])
        labels[mask] = phase["label"]
        trials[mask] = phase["trial"]
        if phase["phase"] != "task":
            continue
        positions = np.flatnonzero(mask)
        # An aborted/short trial, an internal gap or a segment boundary never
        # becomes eligible just because the remaining timestamps have a label.
        # Receipt-aware clocks reflect a slightly fast/slow device crystal. A
        # complete five-second wall-clock task need not contain exactly 1250
        # samples. Coverage plus index/segment/dt checks still reject missing data.
        edge_tolerance = (1.25 if receipt_clock else 1.0) / FS
        enough_samples = len(positions) >= (2 if receipt_clock else round(5 * FS))
        complete = (enough_samples and times[positions[0]] <= phase["start_s"] + edge_tolerance + 1e-9
                    and times[positions[-1]] >= phase["end_s"] - edge_tolerance - 1e-9)
        if not complete:
            reasons[phase["trial"]].append("incomplete_task")
            continue
        task_indices = indices_array[positions]
        task_times = times[positions]
        if (len(np.unique(segment_array[positions])) != 1 or np.any(np.diff(task_indices) != 1)
                or np.any(np.abs(np.diff(task_times) - 1 / FS) > (0.25 if receipt_clock else 0.5) / FS + 1e-9)):
            reasons[phase["trial"]].append("sample_discontinuity")
            continue
        ack = acks.get(phase["index"])
        if not ack or not ack["timing_valid"]:
            reasons[phase["trial"]].append("missing_or_late_ack")
            continue
        valid_start = max(phase["start_s"], ack["onset_s"])
        later_onsets = [value["onset_s"] for index, value in acks.items() if index > phase["index"]]
        valid_end = min([phase["end_s"], *later_onsets])
        if valid_end <= valid_start:
            reasons[phase["trial"]].append("invalid_display_order")
        timing_valid[mask & (times >= valid_start) & (times < valid_end)] = True
    # Synthetic rows have no receipt evidence. Audit the original packets only;
    # interpolation cannot dilute or conceal an invalid real packet clock.
    measured = ~interpolated
    audit, _ = _timestamp_audit(metadata, events, times[measured], receipt_array[measured], trials[measured])
    excluded = np.isin(trials, audit["rejected_trial_ids"])
    audit["excluded_timing_valid_samples"] = int(np.sum(timing_valid & excluded))
    timing_valid[excluded] = False
    for trial in audit["rejected_trial_ids"]:
        reasons[trial].append("clock_future" if audit["available"] else "clock_unverified")
    for trial in training_data.get("cross_trial_rejected_ids", []):
        reasons[trial].append("interpolation_cross_trial")
    if use_filled:
        for phase in metadata["plan"]:
            if phase["phase"] == "task" and not np.any(measured & (trials == phase["trial"])):
                reasons[phase["trial"]].append("task_without_measured_samples")
    display_audit = _display_audit(metadata, events, acks, reasons)
    timing_valid[np.isin(trials, display_audit["rejected_trial_ids"])] = False
    metadata = dict(metadata, timestamp_audit=audit, display_audit=display_audit, training_data=training_data)
    return {"metadata": metadata, "samples": samples, "times": times, "indices": indices_array,
            "segments": segment_array, "labels": labels, "trials": trials, "timing_valid": timing_valid,
            "events": events, "acknowledgements": list(acks.values()),
            "receive_monotonic": receipt_array, "is_interpolated": interpolated}


def list_sessions(base):
    """List manifest summaries only, without opening raw sample/event files."""
    base = Path(base)
    if not base.exists():
        return []
    fields = ("id", "source", "created_utc", "sample_rate", "channel_names", "units",
              "duration_s", "subject", "status", "sample_count")
    sessions = []
    for path in sorted(base.iterdir(), reverse=True):
        if path.is_symlink() or not path.is_dir() or not (path / "manifest.json").is_file():
            continue
        metadata = _read_manifest(path)
        sessions.append({**{key: metadata.get(key) for key in fields}, "path": str(path.resolve())})
    return sessions
