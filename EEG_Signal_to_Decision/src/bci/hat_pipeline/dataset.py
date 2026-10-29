"""Build provenance-preserving, task-bounded epochs from new Hat recordings."""
from __future__ import annotations

import numpy as np

from .contract import HOP, LABELS, WINDOW, check_recording_metadata, input_contract
from .processing import StreamingPipeline


class DatasetBuildError(ValueError):
    """A valid recording set yielded no training epochs, with JSON-safe evidence."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _new_summary():
    return {"windows": 0, "candidate_windows": 0, "accepted_windows": 0,
            "outside_complete_task": 0, "warmup_rejected": 0, "quality_rejected": 0,
            "warning_windows": 0, "accepted_warning_windows": 0,
            "clock_rejected_trials": 0, "clock_unverified_recordings": 0,
            "interpolated_samples": 0, "interpolated_windows": 0,
            "recordings_with_interpolation": 0,
            "per_class": {label: {"candidate_windows": 0, "accepted_windows": 0,
                                  "candidate_trials": 0, "accepted_trials": 0} for label in LABELS},
            "rejection_reasons": {}, "channel_rejections": {},
            "warning_reasons": {}, "channel_warnings": {},
            "_candidate_groups": {label: set() for label in LABELS},
            "_accepted_groups": {label: set() for label in LABELS}}


def _increment(counts, key):
    counts[key] = counts.get(key, 0) + 1


def _quality_counts(summary, quality, *, warnings=False, fallback="hard_quality"):
    """Count each reason and channel/reason at most once per candidate window.

    A window can have multiple reasons; those counts are deliberately not an
    additive partition. Quality rejection totals remain an additive partition.
    """
    pairs = set()
    if warnings:
        for item in quality.get("warnings", []):
            if isinstance(item, dict):
                pairs.add((str(item.get("channel", "unknown")), str(item.get("reason", "warning"))))
        reason_key, channel_key = "warning_reasons", "channel_warnings"
    else:
        for item in quality.get("channels", []):
            if isinstance(item, dict) and item.get("state") not in (None, "ok"):
                pairs.add((str(item.get("name", "unknown")), str(item["state"])))
        reason_key, channel_key = "rejection_reasons", "channel_rejections"
    reasons = {reason for _, reason in pairs}
    if warnings and reasons:
        summary["warning_windows"] += 1
    if not warnings and not reasons:
        reasons.add(fallback)
    for reason in reasons:
        _increment(summary[reason_key], reason)
    for channel, reason in pairs:
        _increment(summary[channel_key].setdefault(channel, {}), reason)


def _finish_summary(summary):
    for stage in ("candidate", "accepted"):
        groups = summary.pop(f"_{stage}_groups")
        summary[f"{stage}_trials"] = sum(len(values) for values in groups.values())
        for label in LABELS:
            summary["per_class"][label][f"{stage}_trials"] = len(groups[label])
    return summary


def _empty_dataset_message(diagnostics):
    names = {"flat": "平线", "saturated": "饱和", "nonfinite": "非有限值",
             "filter_warmup": "滤波预热", "hard_quality": "硬质量检查"}
    reasons = "、".join(f"{names.get(reason, reason)} {count} 窗"
                        for reason, count in sorted(diagnostics["rejection_reasons"].items())) or "无"
    return ("No valid full task windows：没有可训练的完整任务窗口。"
            f"共 {diagnostics['windows']} 窗，完整任务候选 {diagnostics['candidate_windows']} 窗，"
            f"保留 {diagnostics['accepted_windows']} 窗；"
            f"标注/完整任务范围外 {diagnostics['outside_complete_task']} 窗，"
            f"预热拒绝 {diagnostics['warmup_rejected']} 窗，"
            f"硬质量拒绝 {diagnostics['quality_rejected']} 窗。原因：{reasons}。"
            f"采样时钟不可信排除 {diagnostics['clock_rejected_trials']} 个任务试次。"
            "原因可在同一窗口重叠；完整试次数与各通道原因见诊断统计。")


def build_dataset(sessions):
    sessions = list(sessions)
    if not sessions:
        raise ValueError("Provide at least one newly recorded session")
    for session in sessions:
        check_recording_metadata(session["metadata"])
    # Recorder-generated IDs begin with their capture timestamp. Canonical
    # ordering makes the same selected recordings independent of UI click order.
    sessions.sort(key=lambda session: session["metadata"]["id"])
    xs, ys, groups, ids, starts, ends, filled_counts = [], [], [], [], [], [], []
    sources, seen = set(), set()
    rejected = {"quality_or_warmup": 0, "outside_complete_task": 0}
    diagnostics = _new_summary()
    diagnostics.update(schema_version=1, window_samples=WINDOW, hop_samples=HOP,
                       label_order=list(LABELS), per_recording=[], use_filled=False)
    for session in sessions:
        metadata = session["metadata"]
        identifier = metadata["id"]
        if identifier in seen:
            raise ValueError("Duplicate session id would leak repeated data between groups")
        seen.add(identifier); sources.add(metadata["source"])
        if len(sources) != 1:
            raise ValueError("Demo and device recordings must never be mixed in a model")
        data = np.asarray(session["samples"], dtype=float)
        arrays = {key: np.asarray(session[key]) for key in ("times", "indices", "segments", "labels", "trials", "timing_valid")}
        if data.ndim != 2 or data.shape[1] != 8 or any(a.ndim != 1 or len(a) != len(data) for a in arrays.values()):
            raise ValueError("Session arrays must have matching sample counts and eight channels")
        if not np.issubdtype(arrays["trials"].dtype, np.integer) or not np.issubdtype(arrays["segments"].dtype, np.integer):
            raise ValueError("Trial and segment ids must be integers")
        if arrays["timing_valid"].dtype != np.bool_:
            raise ValueError("timing_valid must be an explicit boolean per sample")
        interpolated = np.asarray(session.get("is_interpolated", np.zeros(len(data), dtype=bool)))
        if interpolated.ndim != 1 or len(interpolated) != len(data) or interpolated.dtype != np.bool_:
            raise ValueError("is_interpolated must be an explicit boolean per sample")
        training_data = metadata.get("training_data", {"use_filled": False, "status": "disabled"})
        if interpolated.any() and training_data.get("use_filled") is not True:
            raise ValueError("Interpolated samples require explicit filled-data training metadata")
        recording = _new_summary()
        recording.update(id=identifier, subject=metadata.get("subject", ""), source=metadata["source"])
        recording["training_data"] = training_data
        recording["use_filled"] = training_data.get("use_filled") is True
        recording["interpolated_samples"] = int(interpolated.sum())
        recording["recordings_with_interpolation"] = int(interpolated.any())
        diagnostics["use_filled"] |= recording["use_filled"]
        diagnostics["interpolated_samples"] += recording["interpolated_samples"]
        diagnostics["recordings_with_interpolation"] += recording["recordings_with_interpolation"]
        audit = metadata.get("timestamp_audit", {})
        recording["timestamp_audit"] = audit
        recording["clock_rejected_trials"] = len(audit.get("rejected_trial_ids", []))
        recording["clock_unverified_recordings"] = int(not audit.get("available", False))
        diagnostics["clock_rejected_trials"] += recording["clock_rejected_trials"]
        diagnostics["clock_unverified_recordings"] += recording["clock_unverified_recordings"]
        diagnostics["per_recording"].append(recording)
        pipe = StreamingPipeline(source_kind=metadata["source"])
        boundaries = np.flatnonzero(np.diff(arrays["segments"]) != 0) + 1
        start = 0
        for end in list(boundaries) + [len(data)]:
            block_indices = arrays["indices"][start:end]
            if len(block_indices) and np.any(np.diff(block_indices) <= 0):
                raise ValueError("Within-segment sample indices must increase strictly")
            rows = pipe._push_windows(data[start:end], block_indices, arrays["times"][start:end], int(arrays["segments"][start]) if end > start else 0)
            for row, epoch in rows:
                for summary in (diagnostics, recording):
                    summary["windows"] += 1
                local = int(np.searchsorted(block_indices, row["window_start_index"]))
                lo, hi = start + local, start + local + WINDOW
                labels, trials = arrays["labels"][lo:hi], arrays["trials"][lo:hi]
                valid_task = (len(labels) == WINDOW and np.all(arrays["timing_valid"][lo:hi]) and
                              trials[0] > 0 and np.all(trials == trials[0]) and
                              labels[0] in LABELS and np.all(labels == labels[0]))
                if not valid_task:
                    rejected["outside_complete_task"] += 1
                    for summary in (diagnostics, recording):
                        summary["outside_complete_task"] += 1
                    continue
                label = str(labels[0])
                group = f"{identifier}:trial:{int(trials[0])}"
                quality = row.get("quality") or {}
                for summary in (diagnostics, recording):
                    summary["candidate_windows"] += 1
                    summary["per_class"][label]["candidate_windows"] += 1
                    summary["_candidate_groups"][label].add(group)
                    _quality_counts(summary, quality, warnings=True)
                if row["status"] != "ready":
                    rejected["quality_or_warmup"] += 1
                    for summary in (diagnostics, recording):
                        if row["status"] == "warming_up":
                            summary["warmup_rejected"] += 1
                            _increment(summary["rejection_reasons"], "filter_warmup")
                        else:
                            summary["quality_rejected"] += 1
                            _quality_counts(summary, quality)
                    continue
                for summary in (diagnostics, recording):
                    summary["accepted_windows"] += 1
                    summary["interpolated_windows"] += int(interpolated[lo:hi].any())
                    if any(isinstance(item, dict) for item in quality.get("warnings", [])):
                        summary["accepted_warning_windows"] += 1
                    summary["per_class"][label]["accepted_windows"] += 1
                    summary["_accepted_groups"][label].add(group)
                xs.append(epoch); ys.append(LABELS.index(str(labels[0])))
                groups.append(group); ids.append(identifier)
                starts.append(row["window_start_index"]); ends.append(row["window_end_index"])
                filled_counts.append(int(interpolated[lo:hi].sum()))
            start = int(end)
    _finish_summary(diagnostics)
    for recording in diagnostics["per_recording"]:
        _finish_summary(recording)
    if not xs:
        raise DatasetBuildError(_empty_dataset_message(diagnostics), diagnostics)
    return {"X": np.stack(xs), "y": np.asarray(ys, dtype=np.int64),
            "groups": np.asarray(groups), "session_ids": np.asarray(ids),
            "window_start_indices": np.asarray(starts), "window_end_indices": np.asarray(ends),
            "window_interpolated_samples": np.asarray(filled_counts, dtype=np.int64),
            "metadata": {"input": input_contract(), "source": sources.pop(), "session_ids": sorted(seen),
                         "training_data": {key: diagnostics[key] for key in
                                           ("use_filled", "interpolated_samples", "interpolated_windows", "recordings_with_interpolation")},
                         "dataset_diagnostics": diagnostics},
            "rejected": rejected}
