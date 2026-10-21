"""Timestamp-defined SSVEP recordings; EEG samples remain unfiltered microvolts."""
from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np

from .cvep_recording import (CVEPRecorder, FS, CHANNELS, COLUMNS, finite, integer,
    atomic_json, _file_fingerprint, _read_json, _validate_subject)

CLOCK_DOMAIN = "perf_counter"
DISPLAY_METHOD = "ssvep-timestamp-frames-v1"
CLIENT_BUILD = "20260929-ssvep-1"
FRAME_TIMING_ACTION = "warn_continue"
FRAME_PERIOD = 1 / 60
MAX_UNCERTAINTY = .05
TASK_SECONDS = 6.
REST_SECONDS = 2.
FREQUENCIES = (8., 10., 12., 14.)
FRAME_INTERVAL_POLICY = dict(id="ssvep-timestamp-advisory-v1", min_s=.003, max_s=.040, mean_tolerance_s=.003)


def _pipeline():
    from .ssvep_engine import _core
    return _core()


def frame_interval_policy(value=None):
    if value is not None and value != FRAME_INTERVAL_POLICY:
        raise ValueError("SSVEP显示规则不兼容，请刷新页面。")
    return dict(FRAME_INTERVAL_POLICY)


def make_plan(kind, trials_per_target=6, trial_count=12, seed=None, target_count=4):
    if type(target_count) is not int or target_count != 4 or kind not in ("calibration", "online"):
        raise ValueError("SSVEP固定使用四类8/10/12/14Hz。")
    rng = random.Random(seed)
    targets = []
    if kind == "calibration":
        for _ in range(integer(trials_per_target, "每类试次数", 1, 20)):
            bag = list(range(4)); rng.shuffle(bag); targets.extend(bag)
    else:
        targets = [None] * integer(trial_count, "在线试次数", 1, 120)
    plan, cursor = [], 0.
    for trial, target in enumerate(targets, 1):
        for phase, duration in (("rest", REST_SECONDS), ("task", TASK_SECONDS)):
            plan.append(dict(index=len(plan), phase=phase, trial_id=trial, target=target,
                             start_s=cursor, end_s=cursor + duration))
            cursor += duration
    return plan


def _validate_plan(plan, kind):
    if not isinstance(plan, list) or not plan or len(plan) % 2 or len(plan) > 240:
        raise ValueError("SSVEP计划必须包含休息/任务配对。")
    cursor = 0.
    for index, phase in enumerate(plan):
        if not isinstance(phase, dict):
            raise ValueError("SSVEP计划阶段无效。")
        duration = REST_SECONDS if index % 2 == 0 else TASK_SECONDS
        if (phase.get("index") != index or phase.get("phase") != ("rest" if index % 2 == 0 else "task")
                or integer(phase.get("trial_id"), "试次", 1) != index // 2 + 1
                or abs(finite(phase.get("start_s"), "开始") - cursor) > 1e-8
                or abs(finite(phase.get("end_s"), "结束") - cursor - duration) > 1e-8):
            raise ValueError("SSVEP计划需要2秒休息和6秒任务。")
        if kind == "calibration":
            integer(phase.get("target"), "目标", 0, 3)
        elif phase.get("target") is not None:
            raise ValueError("在线数据不能写入伪造的意图标签。")
        if index % 2 and phase["target"] != plan[index - 1]["target"]:
            raise ValueError("试次提示与标签不一致。")
        cursor += duration


class SSVEPRecorder(CVEPRecorder):
    """Reuse only the audited packet writer, fsync/rollback and close primitives."""
    def __init__(self, base, source, kind, plan, contract, start_time, clock_uncertainty_s,
                 subject="", codebook=None, initial_clock_sync_uncertainty_s=None,
                 software_timing_budget_s=MAX_UNCERTAINTY, recording_name=""):
        if source not in ("demo", "device") or kind not in ("calibration", "online"):
            raise ValueError("SSVEP录制来源或用途无效。")
        _validate_plan(plan, kind)
        subject, recording_name = _validate_subject(subject), _validate_subject(recording_name)
        if contract != _pipeline().get_contract() or codebook != _pipeline().get_config():
            raise ValueError("SSVEP频率/信号处理契约不兼容。")
        initial = finite(clock_uncertainty_s if initial_clock_sync_uncertainty_s is None else initial_clock_sync_uncertainty_s,
                         "初始同步预算", 0, MAX_UNCERTAINTY)
        self.id = f"ssvep_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"
        self.path = Path(base) / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.manifest = dict(schema_version=1, workflow="ssvep", id=self.id, source=source, kind=kind,
            subject=subject, recording_name=recording_name, target_count=4,
            sample_rate=FS, channel_names=list(CHANNELS), units="uV", clock_domain=CLOCK_DOMAIN,
            clock_method="receipt-envelope-5s-bounded-interpolation-v1", start_time=finite(start_time, "时钟锚点"),
            display_timing_method=DISPLAY_METHOD, client_build=CLIENT_BUILD,
            frame_timing_action=FRAME_TIMING_ACTION, frame_interval_policy=dict(FRAME_INTERVAL_POLICY),
            initial_clock_sync_uncertainty_s=initial, clock_uncertainty_s=MAX_UNCERTAINTY,
            software_timing_budget_s=MAX_UNCERTAINTY, contract=contract, stimulus=codebook,
            plan=plan, status="recording", sample_count=0, duration_s=plan[-1]["end_s"],
            created_utc=datetime.now(timezone.utc).isoformat(),
            timing_note="Host receipt sample estimates and timestamp-driven software luminance; no hardware/optical synchronization. Advisory frame jitter alone does not invalidate SSVEP trials.")
        self.samples = self.events = None
        self.last, self.closed, self.write_error, self.event_count = None, False, None, 0
        try:
            self.samples = (self.path / "samples.csv").open("x", encoding="utf-8", newline="")
            self.events = (self.path / "events.jsonl").open("x", encoding="utf-8", newline="")
            csv.writer(self.samples).writerow(COLUMNS); self.samples.flush()
            atomic_json(self.path / "manifest.json", self.manifest)
        except Exception:
            for handle in (self.samples, self.events):
                if handle: handle.close()
            raise


def read_metadata(path):
    path = Path(path)
    for name, limit in (("manifest.json", 2 * 1024 ** 2), ("events.jsonl", 200 * 1024 ** 2), ("samples.csv", 1024 ** 3)):
        member = path / name
        if member.is_symlink() or member.resolve().parent != path.resolve() or member.stat().st_size > limit:
            raise ValueError("SSVEP录制子文件路径或大小非法。")
    m = _read_json((path / "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(m, dict) or m.get("schema_version") != 1 or m.get("workflow") != "ssvep"
            or m.get("id") != path.name or m.get("sample_rate") != FS or m.get("channel_names") != list(CHANNELS)
            or m.get("units") != "uV" or m.get("clock_domain") != CLOCK_DOMAIN or m.get("display_timing_method") != DISPLAY_METHOD
            or m.get("clock_method") != "receipt-envelope-5s-bounded-interpolation-v1"
            or m.get("source") not in ("demo", "device") or m.get("kind") not in ("calibration", "online")
            or m.get("status") not in ("recording", "complete", "aborted", "error") or type(m.get("target_count")) is not int or m["target_count"] != 4
            or m.get("contract") != _pipeline().get_contract() or m.get("stimulus") != _pipeline().get_config()):
        raise ValueError("SSVEP录制元数据或频率契约不兼容。")
    _validate_subject(m.get("subject")); _validate_subject(m.get("recording_name", ""))
    finite(m.get("start_time"), "时钟锚点")
    finite(m.get("initial_clock_sync_uncertainty_s"), "初始同步预算", 0, MAX_UNCERTAINTY)
    integer(m.get("sample_count"), "样本数")
    frame_interval_policy(m.get("frame_interval_policy"))
    _validate_plan(m.get("plan"), m["kind"])
    return m


def _load_raw(path, metadata):
    recoverable = metadata["status"] in ("recording", "error")
    recovery = dict(unclean_shutdown=recoverable, discarded_event_tail=False, discarded_sample_tail=False,
                    file_integrity="unverified", warnings=[])
    fingerprints = metadata.get("files")
    if not recoverable and metadata["display_timing_method"] == DISPLAY_METHOD and fingerprints is None:
        raise ValueError("SSVEP已关闭记录缺少最终文件完整性清单。")
    if not recoverable and fingerprints is not None:
        if not isinstance(fingerprints, dict) or set(fingerprints) != {"samples.csv", "events.jsonl"}:
            raise ValueError("录制文件完整性清单无效。")
        for name, expected in fingerprints.items():
            if expected != _file_fingerprint(path / name):
                raise ValueError(f"{name}完整性校验失败；样本或事件文件已截断或改变。")
        recovery["file_integrity"] = "verified"
    else:
        recovery["warnings"].append("未正常关闭或旧记录缺少最终文件校验；仅逐试次恢复有完整证据的数据。")
    events = []
    lines = (path / "events.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    for n, line in enumerate(lines):
        if recoverable and n == len(lines) - 1 and not line.endswith("\n"):
            recovery["discarded_event_tail"] = True
            continue
        try:
            item = _read_json(line)
            if not isinstance(item, dict) or not isinstance(item.get("details"), dict) or not isinstance(item.get("kind"), str):
                raise ValueError("Invalid SSVEP event")
            events.append(item)
        except (ValueError, TypeError):
            raise
    closes = [(n, item["details"]) for n, item in enumerate(events) if item["kind"] == "close"]
    if closes and (len(closes) != 1 or closes[0][0] != len(events) - 1):
        raise ValueError("结束事件重复或结束后仍有新事件。")
    if not recoverable:
        if not closes or closes[0][1].get("status") != metadata["status"]:
            raise ValueError("已关闭清单与最后结束事件不一致。")
        if "event_count" in metadata and integer(metadata["event_count"], "保存事件数") != len(events):
            raise ValueError("录制事件数与完成清单不一致。")
        if metadata["display_timing_method"] == DISPLAY_METHOD and "event_count" not in metadata:
            raise ValueError("SSVEP已关闭记录缺少最终事件数。")
    rows = []
    with (path / "samples.csv").open(encoding="utf-8", newline="") as handle:
        if tuple(next(csv.reader([handle.readline()]), [])) != COLUMNS:
            raise ValueError("SSVEP原始CSV列与固定8通道不符。")
        raw = handle.readlines()
        for n, line in enumerate(raw):
            if recoverable and n == len(raw) - 1 and not line.endswith("\n"):
                recovery["discarded_sample_tail"] = True
                continue
            try:
                row = next(csv.reader([line]))
                if len(row) != len(COLUMNS):
                    raise ValueError("Incomplete raw row")
                i, seg = int(row[0]), int(row[-2])
                values = [float(v) for v in row[1:-2]]
                received = float(row[-1])
                if i < 0 or seg < 0 or not all(math.isfinite(v) for v in [*values, received]) or values[0] < 0 or received < 0:
                    raise ValueError("Invalid raw values")
                if rows and (i <= rows[-1][0] or values[0] <= rows[-1][1] or seg < rows[-1][-2] or received < rows[-1][-1]):
                    raise ValueError("Raw indices/times must increase")
                rows.append([i, *values, seg, received])
            except ValueError:
                raise
    data = np.asarray(rows, float).reshape((-1, 12))
    if not recoverable and len(rows) != integer(metadata.get("sample_count"), "保存样本数"):
        raise ValueError("录制样本数与完成清单不一致，文件可能不完整。")
    indices, times = data[:, 0].astype(np.int64), data[:, 1]
    samples, segments, received = data[:, 2:10], data[:, 10].astype(np.int64), data[:, 11]
    return samples, times, indices, segments, received, events, recovery


def load_session(path):
    path = Path(path)
    metadata = read_metadata(path)
    samples, times, indices, segments, received, events, recovery = _load_raw(path, metadata)
    tasks = [p for p in metadata["plan"] if p["phase"] == "task"]
    ids = {p["trial_id"] for p in tasks}
    frames, by_trial, tails, errors, warnings, interruptions = [], {}, {}, {}, [], []
    audits, final_audit = {}, None
    global_index = 0
    for event in events:
        d = event["details"]
        if event["kind"] == "interruption":
            interruptions.append(finite(d.get("invalid_since_s"), "失效时刻"))
        if event["kind"] == "final_clock_audit":
            trial = integer(d.get("trial_id"), "结束复核试次", 1)
            if trial not in tails or trial != max(ids): raise ValueError("SSVEP结束复核试次无效。")
            checked = finite(d.get("checked_at_s"), "复核时间")
            delta = finite(d.get("offset_delta_s"), "偏移", -float("inf"))
            new = finite(d.get("new_sync_uncertainty_s"), "同步预算")
            effective = finite(d.get("effective_sync_uncertainty_s"), "有效预算")
            receipt = finite(d.get("received_s"), "复核接收")
            if abs(effective - abs(delta) - new) > 1e-9 or checked < tails[trial] or checked > receipt + effective + 1e-6:
                raise ValueError("SSVEP结束复核声明不一致。")
            if final_audit is not None: raise ValueError("SSVEP结束时钟复核重复。")
            final_audit = dict(d)
            if effective > MAX_UNCERTAINTY: errors.setdefault(trial, set()).add("post_clock_uncertainty")
        if event["kind"] == "timing_warning":
            trial = integer(d.get("trial_id"), "告警试次", 1)
            if trial not in ids or not isinstance(d.get("reason"), str) or not isinstance(d.get("code"), str):
                raise ValueError("SSVEP告警事件格式无效。")
            warnings.append(dict(d))
            if d.get("severity") == "invalid": errors.setdefault(trial, set()).add(d["code"])
        if event["kind"] != "frames": continue
        trial = integer(d.get("trial_id"), "帧试次", 1)
        if trial not in ids or trial in tails or trial != len(tails) + 1 or d.get("visible") is not True:
            raise ValueError("SSVEP帧次序或可见性证据无效。")
        proof = by_trial.setdefault(trial, [])
        receipt = finite(d.get("received_s"), "报告时刻")
        sync = finite(d.get("clock_sync_uncertainty_s", metadata["initial_clock_sync_uncertainty_s"]), "同步预算")
        audit = d.get("clock_audit")
        if not isinstance(audit, dict) or integer(audit.get("trial_id"), "复核试次", 1) != trial:
            raise ValueError("SSVEP逐试次时钟复核缺失。")
        checked = finite(audit.get("checked_at_s"), "复核时刻")
        delta = finite(audit.get("offset_delta_s"), "复核偏移", -float("inf"))
        new = finite(audit.get("new_sync_uncertainty_s"), "复核预算")
        effective = finite(audit.get("effective_sync_uncertainty_s"), "复核有效预算")
        if (abs(effective - abs(delta) - new) > 1e-9 or abs(sync - effective) > 1e-9
                or checked < tails.get(trial - 1, 0.) or checked > receipt + sync + 1e-6
                or trial in audits and audits[trial] != audit):
            raise ValueError("SSVEP时钟复核声明不一致。")
        audits[trial] = dict(audit)
        if sync > MAX_UNCERTAINTY: errors.setdefault(trial, set()).add("clock_uncertainty")
        incoming = d.get("frames", [])
        if not isinstance(incoming, list) or len(incoming) > 32: raise ValueError("SSVEP帧批量无效。")
        for f in incoming:
            code = integer(f.get("code_frame"), "试次帧号")
            onset = finite(f.get("observed_s"), "显示时间")
            lag = finite(f.get("callback_end_lag_s", 0), "回调延迟")
            if integer(f.get("frame_index"), "全局帧号") != global_index or code != len(proof):
                raise ValueError("SSVEP帧缺失、重复或倒序。")
            if proof and onset <= proof[-1]["observed_s"]: raise ValueError("SSVEP显示时间必须递增。")
            first = proof[0]["observed_s"] if proof else onset
            if checked > first + 1e-6 or first - checked > 3.: errors.setdefault(trial, set()).add("clock_audit_age")
            levels = np.asarray(f.get("luminances"), float)
            expected = .5 + .5 * np.sin(2 * np.pi * np.asarray(FREQUENCIES) * (onset - first))
            if levels.shape != (4,) or not np.isfinite(levels).all() or np.any(levels < 0) or np.any(levels > 1) or np.max(np.abs(levels - expected)) > .01:
                errors.setdefault(trial, set()).add("stimulus_luminance_mismatch")
            if receipt - onset < lag - sync - 1e-6: errors.setdefault(trial, set()).add("frame_future")
            if lag > .25 or (proof and onset - proof[-1]["observed_s"] > .25):
                errors.setdefault(trial, set()).add("display_gap")
            row = dict(f, trial_id=trial)
            frames.append(row); proof.append(row); global_index += 1
        if d.get("tail_s") is not None:
            tail = finite(d["tail_s"], "清屏时间")
            tail_lag = finite(d.get("tail_callback_end_lag_s", 0.), "清屏回调延迟")
            if not proof or tail <= proof[-1]["observed_s"]: raise ValueError("SSVEP清屏时刻无效。")
            if tail - proof[-1]["observed_s"] > .25 or tail_lag > .25: errors.setdefault(trial, set()).add("display_gap")
            if receipt - tail < tail_lag - sync - 1e-6: errors.setdefault(trial, set()).add("frame_future")
            tails[trial] = tail
    trials, reasons = [], {}
    for task in tasks:
        trial = task["trial_id"]; proof = by_trial.get(trial, [])
        onset = proof[0]["observed_s"] if proof else None
        tail = tails.get(trial); failures = set(errors.get(trial, set()))
        if not proof or tail is None: failures.add("incomplete_frames")
        positions = np.empty(0, dtype=int)
        if onset is not None and tail is not None:
            post = audits.get(trial + 1) if trial != max(ids) else final_audit
            if post is None: failures.add("missing_post_clock_audit")
            elif (post["checked_at_s"] < tail or post["checked_at_s"] - tail > 3.
                  or post["effective_sync_uncertainty_s"] > MAX_UNCERTAINTY): failures.add("post_clock_uncertainty")
            if not TASK_SECONDS - 1e-6 <= tail - onset <= TASK_SECONDS + .25: failures.add("task_duration")
            if onset < tails.get(trial - 1, 0.) + 1.95: failures.add("rest_too_short")
            if interruptions and tail > min(interruptions) + 1e-9: failures.add("display_interrupted")
            positions = np.flatnonzero((times >= onset) & (times < tail))
            if (len(positions) < 1400 or times[positions[0]] > onset + .005
                    or times[positions[-1]] < tail - .005): failures.add("incomplete_samples")
            elif (np.any(np.diff(indices[positions]) != 1) or len(set(segments[positions])) != 1
                  or np.any(np.abs(np.diff(times[positions]) - 1 / FS) > .25 / FS + 1e-9)):
                failures.add("sample_gap")
            if np.any(metadata["start_time"] + times[positions] > received[positions] + 1e-6): failures.add("sample_clock_future")
        if metadata.get("write_error") or metadata.get("recovery_required"): failures.add("write_error")
        timing_valid = not failures
        quality = _pipeline().raw_quality(samples[positions]) if len(positions) else None
        quality_valid = bool(quality and quality["valid"])
        if not quality_valid: failures.add("raw_quality")
        for reason in failures: reasons[reason] = reasons.get(reason, 0) + 1
        trials.append(dict(trial_id=trial, target=task["target"], onset_s=onset, tail_s=tail,
                           valid=not failures, timing_valid=timing_valid, quality_valid=quality_valid,
                           quality=quality, reasons=sorted(failures), frame_count=len(proof)))
    diagnostics = dict(target_count=4, frequencies=list(FREQUENCIES), planned_trials=len(tasks),
        valid_trials=sum(t["valid"] for t in trials), rejected_trials=sum(not t["valid"] for t in trials),
        timing_valid_trials=sum(t["timing_valid"] for t in trials), quality_valid_trials=sum(t["quality_valid"] for t in trials),
        rejection_reasons=reasons, per_target={str(k): sum(t["valid"] and t["target"] == k for t in trials) for k in range(4)},
        raw_samples=len(samples), frame_count=len(frames), timing_warnings=warnings,
        timing_warning_trials=len({w["trial_id"] for w in warnings}), recovery=recovery, timing_note=metadata["timing_note"])
    return dict(metadata=metadata, samples=samples, times=times, indices=indices, segments=segments,
                events=events, frames=frames, trials=trials, diagnostics=diagnostics)


def list_sessions(base):
    results = []
    for path in sorted(Path(base).glob("ssvep_*"), reverse=True):
        if not path.is_dir() or path.is_symlink(): continue
        try:
            m = read_metadata(path)
            results.append({key: m.get(key) for key in ("id", "subject", "recording_name", "kind", "source", "status", "created_utc", "sample_count", "duration_s", "target_count")})
        except (ValueError, OSError) as exc:
            results.append(dict(id=path.name, status="error", error=str(exc)))
    return results
