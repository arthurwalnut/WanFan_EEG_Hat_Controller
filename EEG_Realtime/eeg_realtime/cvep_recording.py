"""Independent c-VEP raw sessions and conservative software-frame evidence."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

FS = 250
CHANNELS = ("O2", "O1", "P8", "P7", "C4", "C3", "FP1", "FP2")
DISPLAY_METHOD = "raf-frame-evidence-v2"
LEGACY_DISPLAY_METHOD = "raf-frame-evidence-v1"
CLOCK_DOMAIN = "perf_counter"
FRAME_PERIOD = 1 / 60
FRAMES_PER_TRIAL = 378
MAX_UNCERTAINTY = .008
FRAME_INTERVAL_POLICY = dict(id="raf-60hz-record-through-v1", min_s=.0125,
                             max_s=FRAME_PERIOD * 1.25, mean_tolerance_s=.001)
FRAME_TIMING_ACTION = "warn_continue"
CLIENT_BUILD = "20260929-recording-identity-1"
PREVIOUS_FRAME_INTERVAL_POLICY = dict(id="raf-60hz-frame-25pct-v1", min_s=FRAME_PERIOD * .75,
                                      max_s=FRAME_PERIOD * 1.25, mean_tolerance_s=.001)
LEGACY_FRAME_INTERVAL_POLICY = dict(id="raf-60hz-frame-20pct-v1", min_s=FRAME_PERIOD * .8,
                                    max_s=FRAME_PERIOD * 1.2, mean_tolerance_s=.001)
COLUMNS = ("sample_index", "time_s", *CHANNELS, "segment", "receive_perf_counter")


def _pipeline():
    # Resolve the sibling application from this file, including portable layouts.
    source = Path(__file__).resolve().parents[2] / "EEG_Signal_to_Decision" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from bci import cvep_pipeline
    return cvep_pipeline


def _validate_subject(value):
    if not isinstance(value, str) or len(value) > 80 or any(ord(c) < 32 for c in value):
        raise ValueError("受试者编号最多80字，不能含控制字符。")
    return value.strip()


def _validate_recording_name(value):
    if not isinstance(value, str) or len(value) > 80 or any(ord(c) < 32 for c in value):
        raise ValueError("单次录制代号最多80字，不能含控制字符。")
    return value.strip()


def _validate_plan(plan, kind, target_count=6):
    target_count = _pipeline().validate_target_count(target_count)
    if not isinstance(plan, list) or not plan or len(plan) % 2 or len(plan) > 240:
        raise ValueError("c-VEP计划必须包含完整的休息/任务配对。")
    cursor = 0.
    for index, phase in enumerate(plan):
        if not isinstance(phase, dict):
            raise ValueError("c-VEP计划阶段格式无效。")
        expected_phase = "rest" if index % 2 == 0 else "task"
        duration = 2. if expected_phase == "rest" else FRAMES_PER_TRIAL * FRAME_PERIOD
        if (phase.get("phase") != expected_phase or integer(phase.get("index"), "阶段编号") != index
                or integer(phase.get("trial_id"), "计划试次", 1) != index // 2 + 1
                or abs(finite(phase.get("start_s"), "计划开始") - cursor) > 1e-8
                or abs(finite(phase.get("end_s"), "计划结束") - cursor - duration) > 1e-8):
            raise ValueError("c-VEP计划顺序或2秒休息/6.3秒任务时长无效。")
        if kind == "calibration":
            integer(phase.get("target"), "校准目标", 0, target_count - 1)
        elif phase.get("target") is not None:
            raise ValueError("在线录制不能存储伪造的注视标签。")
        if index % 2 and phase.get("target") != plan[index - 1].get("target"):
            raise ValueError("同一试次的休息提示与任务标签不一致。")
        cursor += duration


def _file_fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return dict(size_bytes=path.stat().st_size, sha256=digest.hexdigest())


def finite(value, name, minimum=0., maximum=float("inf")):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} 必须是有限数字。")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} 超出允许范围。")
    return value


def frame_interval_policy(value=None):
    """Resolve a recorded policy; an absent field keeps the original limits."""
    if value is None:
        return dict(LEGACY_FRAME_INTERVAL_POLICY)
    if not isinstance(value, dict) or set(value) != set(FRAME_INTERVAL_POLICY):
        raise ValueError("帧间隔规则必须包含已知标识与完整时序范围。")
    policies = (FRAME_INTERVAL_POLICY, PREVIOUS_FRAME_INTERVAL_POLICY, LEGACY_FRAME_INTERVAL_POLICY)
    expected = next((policy for policy in policies if value.get("id") == policy["id"]), None)
    if expected is None:
        raise ValueError("帧间隔规则标识不受支持。")
    for field in ("min_s", "max_s", "mean_tolerance_s"):
        if finite(value[field], "帧间隔规则范围") != expected[field]:
            raise ValueError("帧间隔规则范围与其标识不一致。")
    return dict(expected)


def integer(value, name, minimum=0, maximum=2 ** 63 - 1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须是允许范围内的整数。")
    return int(value)


def atomic_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def make_plan(kind, trials_per_target=6, trial_count=12, seed=None, target_count=6):
    target_count = _pipeline().validate_target_count(target_count)
    if kind not in ("calibration", "online"):
        raise ValueError("请选择 calibration 或 online。")
    rng = random.Random(seed)
    if kind == "calibration":
        count = integer(trials_per_target, "每目标试次数", 1, 20)
        targets = []
        for _ in range(count):
            bag = list(range(target_count))
            rng.shuffle(bag)
            targets.extend(bag)
    else:
        targets = [None] * integer(trial_count, "在线试次数", 1, 120)
    phases, cursor = [], 0.
    for trial, target in enumerate(targets, 1):
        phases.append(dict(index=len(phases), phase="rest", trial_id=trial, target=target,
                           start_s=cursor, end_s=cursor + 2))
        cursor += 2
        phases.append(dict(index=len(phases), phase="task", trial_id=trial, target=target,
                           start_s=cursor, end_s=cursor + FRAMES_PER_TRIAL * FRAME_PERIOD))
        cursor += FRAMES_PER_TRIAL * FRAME_PERIOD
    return phases


class CVEPRecorder:
    def __init__(self, base, source, kind, plan, contract, start_time, clock_uncertainty_s, subject="", codebook=None,
                 initial_clock_sync_uncertainty_s=None, software_timing_budget_s=MAX_UNCERTAINTY, recording_name=""):
        if source not in ("demo", "device") or kind not in ("calibration", "online"):
            raise ValueError("录制来源或用途不合法。")
        subject = _validate_subject(subject)
        recording_name = _validate_recording_name(recording_name)
        target_count = _pipeline().target_count_from_contract(contract)
        _validate_plan(plan, kind, target_count)
        if codebook != _pipeline().get_codebook(target_count):
            raise ValueError("录制输入契约或精确编码表不兼容。")
        budget = finite(software_timing_budget_s, "总软件时序预算", 0, MAX_UNCERTAINTY)
        initial = finite(clock_uncertainty_s if initial_clock_sync_uncertainty_s is None else initial_clock_sync_uncertainty_s,
                         "初始同步误差", 0, budget)
        self.id = f"cvep_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"
        self.path = Path(base) / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.manifest = dict(schema_version=1, id=self.id, kind=kind, source=source, subject=subject,
            recording_name=recording_name, target_count=target_count,
            sample_rate=FS, channel_names=list(CHANNELS), units="uV", clock_domain=CLOCK_DOMAIN,
            clock_method="receipt-envelope-5s-bounded-interpolation-v1", display_timing_method=DISPLAY_METHOD,
            frame_interval_policy=dict(FRAME_INTERVAL_POLICY), frame_timing_action=FRAME_TIMING_ACTION,
            client_build=CLIENT_BUILD,
            clock_uncertainty_s=budget, software_timing_budget_s=budget,
            initial_clock_sync_uncertainty_s=initial,
            start_time=finite(start_time, "时钟锚点"), contract=contract, codebook=codebook, plan=plan,
            status="recording", sample_count=0, duration_s=plan[-1]["end_s"], created_utc=datetime.now(timezone.utc).isoformat(),
            timing_note="EEG times are host receipt estimates; rAF evidence is software presentation evidence, not optical or hardware synchronization.")
        self.samples = self.events = None
        self.last = None
        self.closed = False
        self.write_error = None
        self.event_count = 0
        try:
            self.samples = (self.path / "samples.csv").open("x", encoding="utf-8", newline="")
            self.events = (self.path / "events.jsonl").open("x", encoding="utf-8", newline="")
            csv.writer(self.samples).writerow(COLUMNS)
            self.samples.flush()
            atomic_json(self.path / "manifest.json", self.manifest)
        except Exception:
            for handle in (self.samples, self.events):
                if handle:
                    handle.close()
            raise

    def _write(self, handle, payload):
        if self.closed or self.write_error:
            raise RuntimeError("录制已关闭或存在写入错误。")
        position = handle.tell()
        try:
            handle.write(payload)
            handle.flush()
        except Exception as exc:
            self.write_error = str(exc)
            try:
                handle.seek(position)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                self.manifest["recovery_required"] = True
            self.manifest.update(status="error", write_error=str(exc))
            try:
                atomic_json(self.path / "manifest.json", self.manifest)
            except OSError:
                pass
            raise

    def write(self, samples, indices, times, segment, received):
        values, timestamps, idx = np.asarray(samples, float), np.asarray(times, float), np.asarray(indices)
        if values.ndim != 2 or values.shape[1] != 8 or timestamps.shape != (len(values),) or idx.shape != (len(values),):
            raise ValueError("原始样本应为 N×8，索引和时间应为 N。")
        if not len(values):
            return
        if not np.issubdtype(idx.dtype, np.integer) or not np.isfinite(values).all() or not np.isfinite(timestamps).all():
            raise ValueError("样本/时间必须有限，索引必须是整数。")
        segment = integer(segment, "连续段")
        received = finite(received, "QPC接收时间")
        if np.any(idx < 0) or np.any(timestamps < 0) or np.any(np.diff(idx) <= 0) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("样本索引与时刻必须非负且严格递增。")
        if self.last and (idx[0] <= self.last[0] or timestamps[0] <= self.last[1] or segment < self.last[2] or received < self.last[3]):
            raise ValueError("跨包索引、时刻或连续段倒退。")
        output = io.StringIO(newline="")
        csv.writer(output).writerows((int(i), float(t), *map(float, row), segment, received)
                                     for i, t, row in zip(idx, timestamps, values))
        self._write(self.samples, output.getvalue())
        self.last = (int(idx[-1]), float(timestamps[-1]), segment, received)
        self.manifest["sample_count"] += len(values)

    def event(self, kind, details):
        payload = json.dumps(dict(kind=kind, details=details), ensure_ascii=False, allow_nan=False) + "\n"
        self._write(self.events, payload)
        self.event_count += 1

    def close(self, status="complete", reason=""):
        if self.closed:
            return
        if status not in ("complete", "aborted", "error"):
            raise ValueError("不合法的录制结束状态。")
        try:
            if self.write_error:
                status = "error"
                reason = self.write_error
            else:
                self.event("close", dict(status=status, reason=str(reason)))
            for handle in (self.samples, self.events):
                handle.flush()
                os.fsync(handle.fileno())
            final = dict(self.manifest, status=status, reason=str(reason),
                         duration_s=self.last[1] if self.last else 0., closed_utc=datetime.now(timezone.utc).isoformat(),
                         event_count=self.event_count,
                         files={name: _file_fingerprint(self.path / name) for name in ("samples.csv", "events.jsonl")})
            atomic_json(self.path / "manifest.json", final)
            self.manifest = final
        finally:
            self.closed = True
            for handle in (self.samples, self.events):
                handle.close()


def _read_json(text):
    def invalid(value):
        raise ValueError(f"Invalid JSON constant {value}")
    return json.loads(text, parse_constant=invalid)


def read_metadata(path):
    path = Path(path)
    for name, limit in (("manifest.json", 2 * 1024 * 1024), ("events.jsonl", 200 * 1024 * 1024),
                        ("samples.csv", 1024 * 1024 * 1024)):
        member = path / name
        if member.is_symlink() or member.resolve().parent != path.resolve() or member.stat().st_size > limit:
            raise ValueError("录制子文件路径或大小不合法。")
    metadata = _read_json((path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("c-VEP录制清单必须是对象。")
    if (metadata.get("schema_version") != 1 or metadata.get("id") != path.name or metadata.get("sample_rate") != FS
            or metadata.get("channel_names") != list(CHANNELS) or metadata.get("units") != "uV"
            or metadata.get("clock_domain") != CLOCK_DOMAIN or metadata.get("display_timing_method") not in (DISPLAY_METHOD, LEGACY_DISPLAY_METHOD)
            or metadata.get("clock_method") != "receipt-envelope-5s-bounded-interpolation-v1"
            or metadata.get("source") not in ("device", "demo") or metadata.get("kind") not in ("calibration", "online")
            or metadata.get("status") not in ("recording", "complete", "aborted", "error")):
        raise ValueError("c-VEP录制元数据、时钟域或通道契约不兼容。")
    _validate_subject(metadata.get("subject"))
    # Older recordings used one visible field; preserve the stored identity and
    # provide a display-name fallback in memory without rewriting the manifest.
    metadata["recording_name"] = _validate_recording_name(metadata.get("recording_name", metadata["subject"]))
    if "frame_interval_policy" in metadata and metadata["frame_interval_policy"] is None:
        raise ValueError("帧间隔规则不能为 null；旧格式应省略此字段。")
    # Old recordings retain their original acceptance range; normalization is in memory only.
    metadata["frame_interval_policy"] = frame_interval_policy(metadata.get("frame_interval_policy"))
    if (metadata["frame_interval_policy"]["id"] == FRAME_INTERVAL_POLICY["id"]
            and metadata.get("frame_timing_action") != FRAME_TIMING_ACTION):
        raise ValueError("持续录制的帧时序处理规则不兼容。")
    target_count = _pipeline().target_count_from_contract(metadata.get("contract"))
    if "target_count" in metadata and _pipeline().validate_target_count(metadata["target_count"]) != target_count:
        raise ValueError("录制目标数量与输入契约不一致。")
    # Old six-target manifests are normalized in memory only; never rewrite raw files.
    metadata["target_count"] = target_count
    if not isinstance(metadata.get("codebook"), dict) or metadata["codebook"].get("hash") != metadata["contract"].get("codebook_hash"):
        raise ValueError("录制缺少精确编码表或其hash与输入契约不符。")
    book = metadata["codebook"]
    bits = np.asarray(book.get("bits"))
    frames = np.asarray(book.get("frames"))
    if (book.get("frames_per_bit") != 2 or book.get("refresh_hz") != 60
            or bits.shape != (target_count, 63) or frames.shape != (target_count, 126) or not np.all((bits == 0) | (bits == 1))
            or not np.array_equal(frames, np.repeat(bits, 2, axis=1))):
        raise ValueError("保存的精确编码表无效。")
    # Same canonical payload as the separately versioned codebook generator.
    payload = dict(bits=bits.tolist(), frames_per_bit=2, refresh_hz=60)
    expected_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    if expected_hash != book["hash"]:
        raise ValueError("保存的编码位序列与hash不一致。")
    finite(metadata.get("start_time"), "时钟锚点")
    finite(metadata.get("clock_uncertainty_s"), "时钟预算", 0, MAX_UNCERTAINTY)
    if metadata["display_timing_method"] == DISPLAY_METHOD:
        budget = finite(metadata.get("software_timing_budget_s"), "总软件时序预算", 0, MAX_UNCERTAINTY)
        finite(metadata.get("initial_clock_sync_uncertainty_s"), "初始同步误差", 0, budget)
        if abs(metadata["clock_uncertainty_s"] - budget) > 1e-9:
            raise ValueError("时钟预算别名与总软件时序预算不一致。")
    integer(metadata.get("sample_count"), "保存样本数")
    _validate_plan(metadata.get("plan"), metadata["kind"], target_count)
    return metadata


def load_session(path):
    path = Path(path)
    metadata = read_metadata(path)
    interval_policy = metadata["frame_interval_policy"]
    record_through = interval_policy["id"] == FRAME_INTERVAL_POLICY["id"]
    evidence_limit = float("inf") if record_through else MAX_UNCERTAINTY
    target_count = metadata["target_count"]
    recoverable = metadata["status"] in ("recording", "error")
    recovery = dict(unclean_shutdown=recoverable, discarded_event_tail=False, discarded_sample_tail=False,
                    file_integrity="unverified", warnings=[])
    fingerprints = metadata.get("files")
    if not recoverable and metadata["display_timing_method"] == DISPLAY_METHOD and fingerprints is None:
        raise ValueError("v2已关闭记录缺少最终文件完整性清单。")
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
                raise ValueError("Invalid c-VEP event")
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
            raise ValueError("v2已关闭记录缺少最终事件数。")
    rows = []
    with (path / "samples.csv").open(encoding="utf-8", newline="") as handle:
        if tuple(next(csv.reader([handle.readline()]), [])) != COLUMNS:
            raise ValueError("c-VEP原始CSV列与固定8通道不符。")
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
    frames, tails, interruptions, audits, max_lags, final_audits = [], {}, [], {}, {}, []
    task_ids = {p["trial_id"] for p in metadata["plan"] if p["phase"] == "task"}
    frame_failures = set()
    audit_failures = set()
    timing_warnings, warning_codes = [], {}
    previous_index = -1
    previous_receipt = None
    for event_index, event in enumerate(events):
        details = event["details"]
        if event["kind"] == "interruption":
            interruptions.append(finite(details.get("invalid_since_s"), "中断时刻"))
        if event["kind"] == "final_clock_audit":
            final_audits.append((event_index, details))
        if event["kind"] == "timing_warning":
            trial = integer(details.get("trial_id"), "时序告警试次", 1)
            code, reason = details.get("code"), details.get("reason")
            if (trial not in task_ids or not isinstance(code, str) or not code.strip()
                    or not isinstance(reason, str) or not reason.strip()
                    or not isinstance(details.get("details"), dict)):
                raise ValueError("时序告警必须引用计划试次并保存代码、原因和诊断信息。")
            timing_warnings.append(dict(details))
            warning_codes.setdefault(trial, set()).add(code)
        if event["kind"] != "frames":
            continue
        trial = integer(details.get("trial_id"), "帧试次", 1)
        if trial not in task_ids:
            raise ValueError("帧证据引用了计划之外的试次。")
        receipt = finite(details.get("received_s"), "帧接收时刻")
        if previous_receipt is not None and receipt < previous_receipt:
            frame_failures.add(trial)
        previous_receipt = receipt
        budget = finite(details.get("clock_uncertainty_s"), "帧时钟预算", 0, evidence_limit)
        if budget > metadata["clock_uncertainty_s"] + 1e-9 or details.get("visible") is not True:
            frame_failures.add(trial)
        incoming = details.get("frames", [])
        if not isinstance(incoming, list) or len(incoming) > 16 or (not incoming and details.get("tail_s") is None):
            raise ValueError("帧事件应包含1～16帧或明确末尾证据。")
        if any(not isinstance(frame, dict) for frame in incoming):
            raise ValueError("逐帧证据格式无效。")
        if metadata["display_timing_method"] == DISPLAY_METHOD:
            audit = details.get("clock_audit")
            if not isinstance(audit, dict):
                raise ValueError("v2帧证据缺少每试次时钟审计。")
            audit_trial = integer(audit.get("trial_id"), "时钟审计试次", 1)
            checked = finite(audit.get("checked_at_s"), "时钟检查时刻")
            delta = finite(audit.get("offset_delta_s"), "同步偏移变化", -float("inf"))
            new_uncertainty = finite(audit.get("new_sync_uncertainty_s"), "新同步误差", 0, evidence_limit)
            effective = finite(audit.get("effective_sync_uncertainty_s"), "有效同步误差", 0, evidence_limit)
            if audit_trial != trial or abs(effective - abs(delta) - new_uncertainty) > 1e-9:
                frame_failures.add(trial)
                audit_failures.add(trial)
            normalized_audit = dict(trial_id=audit_trial, checked_at_s=checked, offset_delta_s=delta,
                                    new_sync_uncertainty_s=new_uncertainty, effective_sync_uncertainty_s=effective)
            if trial in audits and audits[trial] != normalized_audit:
                frame_failures.add(trial)
                audit_failures.add(trial)
            audits.setdefault(trial, normalized_audit)
            sync = finite(details.get("clock_sync_uncertainty_s"), "批次同步误差", 0, evidence_limit)
            callback_lags = [finite(f.get("callback_end_lag_s"), "帧回调延迟", 0, evidence_limit) for f in incoming]
            if details.get("tail_s") is not None:
                callback_lags.append(finite(details.get("tail_callback_end_lag_s"), "尾帧回调延迟", 0, evidence_limit))
            callback_max = max(callback_lags, default=0.)
            max_lags[trial] = max(callback_max, max_lags.get(trial, 0.))
            declared_lag = finite(details.get("callback_end_lag_s"), "批次回调延迟", 0, evidence_limit)
            if (abs(sync - effective) > 1e-9 or abs(declared_lag - callback_max) > 1e-9
                    or abs(budget - effective - callback_max) > 1e-9
                    or effective + callback_max > metadata["software_timing_budget_s"] + 1e-9):
                frame_failures.add(trial)
        for frame in incoming:
            if not isinstance(frame, dict):
                raise ValueError("逐帧证据格式无效。")
            index = integer(frame.get("frame_index"), "帧编号")
            code = integer(frame.get("code_frame"), "编码帧", 0, FRAMES_PER_TRIAL - 1)
            observed = finite(frame.get("observed_s"), "帧显示时刻")
            earliest = (frame["callback_end_lag_s"] - effective if metadata["display_timing_method"] == DISPLAY_METHOD else -budget)
            if index != previous_index + 1 or not earliest - 1e-9 <= receipt - observed <= .35:
                frame_failures.add(trial)
            previous_index = index
            frames.append(dict(frame_index=index, trial_id=trial, code_frame=code, observed_s=observed,
                               **({"callback_end_lag_s": frame["callback_end_lag_s"]} if metadata["display_timing_method"] == DISPLAY_METHOD else {})))
        if details.get("tail_s") is not None:
            tail = finite(details["tail_s"], "刺激尾帧")
            earliest = (details["tail_callback_end_lag_s"] - effective if metadata["display_timing_method"] == DISPLAY_METHOD else -budget)
            if trial in tails or not earliest - 1e-9 <= receipt - tail <= .35:
                frame_failures.add(trial)
            tails[trial] = tail
    trials, counts, quality_counts = [], {}, {}
    tasks = [p for p in metadata["plan"] if p["phase"] == "task"]
    final_effective = None
    final_reason = "missing_final_clock_audit"
    if final_audits and metadata["display_timing_method"] == DISPLAY_METHOD:
        final_reason = "invalid_final_clock_audit"
        try:
            if len(final_audits) != 1:
                raise ValueError("duplicate final audit")
            position, details = final_audits[0]
            trial = integer(details.get("trial_id"), "结束复核试次", 1)
            checked = finite(details.get("checked_at_s"), "结束复核时刻")
            delta = finite(details.get("offset_delta_s"), "结束时钟偏移", -evidence_limit, evidence_limit)
            new = finite(details.get("new_sync_uncertainty_s"), "结束同步误差", 0, evidence_limit)
            effective = finite(details.get("effective_sync_uncertainty_s"), "结束有效误差", 0, evidence_limit)
            receipt = finite(details.get("received_s"), "结束审计接收时间")
            if (trial != tasks[-1]["trial_id"] or trial not in tails
                    or not tails[trial] <= checked <= tails[trial] + 3.
                    or abs(effective - abs(delta) - new) > 1e-9 or checked > receipt + effective + 1e-9
                    or (previous_receipt is not None and receipt < previous_receipt)
                    or any(item["kind"] == "frames" for item in events[position + 1:])):
                raise ValueError("invalid final audit interval")
            final_effective = effective
        except (ValueError, TypeError, KeyError):
            pass
    previous_tail = None
    for task in tasks:
        trial = task["trial_id"]
        proof = [f for f in frames if f["trial_id"] == trial]
        reasons = []
        onset = proof[0]["observed_s"] if proof else None
        tail = tails.get(trial)
        if metadata["display_timing_method"] != DISPLAY_METHOD:
            reasons.append("unverified_display_callback")
        if len(proof) != FRAMES_PER_TRIAL or [f["code_frame"] for f in proof] != list(range(FRAMES_PER_TRIAL)):
            reasons.append("incomplete_frames")
        if trial in frame_failures:
            reasons.append("invalid_frame_evidence")
        if trial in warning_codes:
            reasons.append("timing_warning")
        if tail is None:
            reasons.append("missing_tail")
        if proof and tail is not None:
            spacing = np.diff([f["observed_s"] for f in proof] + [tail])
            if (np.any((spacing < interval_policy["min_s"] - 1e-9) | (spacing > interval_policy["max_s"] + 1e-9))
                    or abs(float(np.mean(spacing)) - FRAME_PERIOD) > interval_policy["mean_tolerance_s"] + 1e-9):
                reasons.append("dropped_or_irregular_frame")
            if onset < (previous_tail + 1.95 if previous_tail is not None else 1.95):
                reasons.append("rest_too_short")
            if metadata["display_timing_method"] == DISPLAY_METHOD:
                audit = audits.get(trial)
                if (audit is None or audit["checked_at_s"] < (previous_tail or 0.) - 1e-9
                        or audit["checked_at_s"] > onset + 1e-9 or onset - audit["checked_at_s"] > 3.):
                    reasons.append("invalid_clock_audit")
                # Each trial needs a later clock check. A preserved tail proves
                # frame completion, not that the frozen mapping remained valid.
                if trial == tasks[-1]["trial_id"]:
                    if final_effective is None:
                        reasons.append(final_reason)
                    elif final_effective + max_lags.get(trial, 0.) > metadata["software_timing_budget_s"] + 1e-9:
                        reasons.append("post_clock_budget_exceeded")
                else:
                    post = audits.get(trial + 1)
                    next_proof = [frame for frame in frames if frame["trial_id"] == trial + 1]
                    if post is None:
                        reasons.append("missing_post_trial_clock_audit")
                    elif (trial + 1 in audit_failures or not tail - 1e-9 <= post["checked_at_s"] <= tail + 3.
                          or not next_proof
                          # In record-through sessions a later static wait does
                          # not undo an already timely post-trial clock check.
                          # The next trial checks its own audit age above.
                          or (not record_through and not 0 <= next_proof[0]["observed_s"] - post["checked_at_s"] <= 3.)
                          or post["effective_sync_uncertainty_s"] + max_lags.get(trial, 0.) > metadata["software_timing_budget_s"] + 1e-9):
                        reasons.append("invalid_post_trial_clock_audit")
            mask = (times >= onset) & (times < tail)
            positions = np.flatnonzero(mask)
            if (not len(positions) or times[positions[0]] > onset + 1.25 / FS
                    or times[positions[-1]] < tail - 1.25 / FS
                    or len(positions) < max(1400, int(.95 * FS * (tail - onset)))):
                reasons.append("incomplete_samples")
            elif (len(np.unique(segments[mask])) != 1 or np.any(np.diff(indices[mask]) != 1)
                    or np.any(np.abs(np.diff(times[mask]) - 1 / FS) > .25 / FS + 1e-9)):
                reasons.append("sample_gap")
            if np.any(metadata["start_time"] + times[mask] > received[mask] + 1e-6):
                reasons.append("sample_clock_future")
            if interruptions and tail > min(interruptions) + 1e-9:
                reasons.append("display_interrupted")
            previous_tail = tail
        if metadata.get("write_error") or metadata.get("recovery_required"):
            reasons.append("write_error")
        timing_valid = not reasons
        quality = None
        if onset is not None and tail is not None and tail > onset:
            raw_trial = samples[(times >= onset) & (times < tail)]
            if len(raw_trial):
                quality = _pipeline().raw_quality(raw_trial)
        quality_valid = bool(quality and quality["valid"])
        if not quality_valid:
            reasons.append("raw_quality" if quality else "missing_quality_samples")
            if quality:
                quality_counts[quality["reason"]] = quality_counts.get(quality["reason"], 0) + 1
        for reason in sorted(set(reasons)):
            counts[reason] = counts.get(reason, 0) + 1
        trials.append(dict(trial_id=trial, target=task["target"], onset_s=onset, tail_s=tail,
                           valid=not reasons, timing_valid=timing_valid, quality_valid=quality_valid,
                           quality=quality, reasons=sorted(set(reasons)), frame_count=len(proof),
                           timing_warning_codes=sorted(warning_codes.get(trial, set()))))
    diagnostics = dict(target_count=target_count, planned_trials=len(tasks), valid_trials=sum(t["valid"] for t in trials),
        rejected_trials=sum(not t["valid"] for t in trials), rejection_reasons=counts,
        raw_samples=len(samples), frame_count=len(frames), timing_note=metadata["timing_note"],
        per_target={str(k): sum(t["valid"] and t["target"] == k for t in trials) for k in range(target_count)},
        timing_valid_trials=sum(t["timing_valid"] for t in trials), quality_valid_trials=sum(t["quality_valid"] for t in trials),
        per_target_timing={str(k): sum(t["timing_valid"] and t["target"] == k for t in trials) for k in range(target_count)},
        per_target_quality={str(k): sum(t["quality_valid"] and t["target"] == k for t in trials) for k in range(target_count)},
        quality_rejection_reasons=quality_counts, quality_channels=[CHANNELS[i] for i in _pipeline().SELECTED],
        quality_warning_trials=sum(bool(t["quality"] and t["quality"]["warnings"]) for t in trials),
        recovery=recovery, display_method_verified=metadata["display_timing_method"] == DISPLAY_METHOD,
        frame_interval_policy=dict(interval_policy), timing_warnings=timing_warnings,
        timing_warning_trials=len(warning_codes),
        timing_warning_reasons={code: sum(code in codes for codes in warning_codes.values())
                                for code in sorted({c for codes in warning_codes.values() for c in codes})})
    return dict(metadata=metadata, samples=samples, times=times, indices=indices, segments=segments,
                events=events, frames=frames, trials=trials, diagnostics=diagnostics)


def list_sessions(base):
    result = []
    for path in sorted(Path(base).glob("cvep_*"), reverse=True):
        if not path.is_dir() or path.is_symlink() or not (path / "manifest.json").is_file():
            continue
        try:
            m = read_metadata(path)
            result.append({key: m.get(key) for key in ("id", "kind", "source", "subject", "recording_name", "created_utc", "status", "sample_count", "duration_s", "target_count")})
        except (ValueError, OSError) as exc:
            result.append(dict(id=path.name, status="error", error=str(exc)))
    return result
