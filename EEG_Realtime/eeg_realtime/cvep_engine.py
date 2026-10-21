"""Independent c-VEP capture, frame-evidence recording and template inference."""
from __future__ import annotations

import copy
import json
import logging
import math
import secrets
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .cvep_recording import (CHANNELS, CLOCK_DOMAIN, DISPLAY_METHOD, FRAME_PERIOD,
    FRAMES_PER_TRIAL, FS, MAX_UNCERTAINTY, CVEPRecorder, atomic_json, finite, integer,
    list_sessions, load_session, make_plan, FRAME_INTERVAL_POLICY, frame_interval_policy,
    CLIENT_BUILD, FRAME_TIMING_ACTION)
from .sample_clock import ReceiptSampleClock
from .sources import DemoSource, DeviceSource
from .storage import unique_name, utc_now

LOGGER = logging.getLogger(__name__)
# Hat packets contain 50 samples at 250 Hz (200 ms nominal receipt cadence).
# Arrival jitter and local processing backlog are distinct conditions.
MAX_RECEIPT_GAP_S = 1.25
MAX_PROCESSING_LAG_S = .25
START_FRESHNESS_S = .25
PREDICTION_FRESHNESS_S = .25
SOURCE_STAT_FIELDS = ("packets", "lost_packets", "crc_errors", "bad_packets",
    "reordered_packets", "reconnects", "receiver_queue_depth", "receiver_queue_overflows",
    "receiver_dropped_packets", "receiver_max_queue_depth")

PIPELINE_SRC = Path(__file__).resolve().parents[2] / "EEG_Signal_to_Decision" / "src"
if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))


def _core():
    from bci import cvep_pipeline
    return cvep_pipeline


def unavailable(reason, status="unavailable"):
    return dict(status=status, label=None, candidate_target=None, correlations=[], margin=None,
                accepted=False, reason=reason)


class _RetroactiveTimingError(ValueError):
    def __init__(self, message, invalid_since):
        super().__init__(message)
        self.invalid_since = invalid_since


class CVEPEngine:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.recordings = self.root / "data/cvep_recordings"
        self.models = self.root / "data/cvep_models"
        self.output = self.root / "data/cvep_predictions"
        for path in (self.recordings, self.models, self.output):
            path.mkdir(parents=True, exist_ok=True)
        self.lock, self.lifecycle = threading.RLock(), threading.RLock()
        self.closed, self.stop_event = threading.Event(), threading.Event()
        self.worker = self.training_thread = None
        self.source = None
        self.source_kind = "demo"
        self.status, self.message = "idle", "c-VEP 独立工作台；尚未开始采集。"
        self.events = deque(maxlen=60)
        self.sample_clock = ReceiptSampleClock(FS)
        self.preprocessor = _core().ContinuousPreprocessor()
        self.raw = self.filtered = np.empty((0, 8))
        self.times = np.empty(0)
        self.indices = np.empty(0, dtype=np.int64)
        self.cursor = self.segment = 0
        self.origin = self.last_received = self.last_sample = self.stable_since = None
        self.last_batch_info = None
        self.last_start_failure = None
        self.filter_start = None
        self.quality = dict(valid=False, reason="等待至少2秒新鲜的枕顶通道数据。", warnings=[])
        self.recorder = None
        self.owner_token = None
        self.session_start = None
        self.frames_by_trial, self.tails, self.decoded_ends = {}, {}, {}
        self.clock_audits = {}
        self.callback_max_by_trial = {}
        self._timing_invalid_trials = set()
        self.timing_warnings = []
        self.decode_starts, self.decode_done = {}, set()
        self.next_frame_index = 0
        self.last_frame_received = None
        self.finish_requested = False
        self.waiting_final_audit = False
        self.last_session = dict(active=False, id=None, plan=[])
        self.model = self.model_id = None
        self.prediction = unavailable("尚未加载 c-VEP 个人模型。", "no_model")
        self.predictions = deque(maxlen=5000)
        self.prediction_invalidations = deque(maxlen=5000)
        self.prediction_id = 0
        self.prediction_handle = None
        self.training = dict(status="idle", message="请选择校准录制；模拟与实机不能混合训练。")
        self.demo_target = 0
        self.rng = np.random.default_rng(7419)
        self.replay = self.replay_id = None
        self.replay_position, self.replay_speed = 0., 1.
        self.replay_playing = False
        self.replay_clock = time.perf_counter()
        self.monitor = threading.Thread(target=self._monitor, daemon=True, name="CVEP-monitor")
        self.monitor.start()

    def _event(self, message, level="info"):
        self.events.append(dict(time=utc_now(), level=level, message=str(message)))

    def _timing_warning(self, trial, code, reason, **details):
        """Persist the first occurrence per trial/reason while raw capture continues."""
        first_warning = trial not in self._timing_invalid_trials
        self._timing_invalid_trials.add(trial)
        if not any(item["trial_id"] == trial and item["code"] == code for item in self.timing_warnings):
            item = dict(trial_id=trial, code=code, reason=reason, details=details)
            if self.recorder:
                self.recorder.event("timing_warning", item)
            self.timing_warnings.append(item)
            self._event(f"试次{trial}时序告警：{reason}；EEG继续保存，该试次不用于训练/分类。", "warning")
        if first_warning:
            for record in self.predictions:
                if record["trial_id"] == trial:
                    record.update(timing_valid=False, accepted=False, label=None, status="invalidated",
                                  invalidation_reason=reason)
            self.prediction_id += 1
            invalidation = dict(type="trial_timing_invalidated", id=self.prediction_id, trial_id=trial,
                                code=code, reason=reason, timing_valid=False, accepted=False,
                                server_time=time.perf_counter())
            self.prediction_invalidations.append(invalidation)
            # Journal entries retain the original decision as it was emitted.
            # Replay consumers apply this later event to the entire parent trial.
            self._export_prediction(invalidation)
        if trial == len(self.tails) + 1:
            self.prediction = unavailable("当前试次时序不合格；EEG继续保存，等待后续正常试次。", "timing_warning")

    def _export_prediction(self, item):
        if self.prediction_handle:
            try:
                self.prediction_handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                self.prediction_handle.flush()
            except (OSError, ValueError, TypeError) as exc:
                self._event(f"预测导出失败：{exc}", "error")
                self.prediction_handle.close()
                self.prediction_handle = None

    @staticmethod
    def _batch_info(batch):
        def number(value):
            return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
        return dict(sequence=number(batch.sequence), gap=bool(batch.gap),
                    missing_samples=number(batch.missing_samples), sample_count=len(batch.samples),
                    received_perf_counter=number(batch.received_perf_counter))

    def _acquisition_diagnostics(self, trigger, *, now=None, batch=None):
        """Only numeric transport facts; never serialize source config or request tokens."""
        now = time.perf_counter() if now is None else now
        stats = getattr(self.source, "stats", {})
        stats = stats if isinstance(stats, dict) else {}
        selected = {key: value for key in SOURCE_STAT_FIELDS
                    if isinstance((value := stats.get(key)), (int, float))
                    and not isinstance(value, bool) and math.isfinite(value)}
        result = dict(trigger=trigger, clock_domain=CLOCK_DOMAIN, observed_perf_counter=now,
            source=self.source_kind, status=self.status, segment=self.segment, next_sample_index=self.cursor,
            raw_samples=len(self.raw),
            threshold_s=(MAX_RECEIPT_GAP_S if trigger in ("receipt_gap", "receipt_timeout") else
                         MAX_PROCESSING_LAG_S if trigger == "processing_lag" else
                         START_FRESHNESS_S if trigger == "session_start_rejected" else None),
            receipt_timeout_s=MAX_RECEIPT_GAP_S, processing_lag_limit_s=MAX_PROCESSING_LAG_S,
            start_freshness_s=START_FRESHNESS_S, prediction_freshness_s=PREDICTION_FRESHNESS_S,
            required_stable_s=2., source_stats=selected,
            last_received_perf_counter=self.last_received,
            receipt_age_s=None if self.last_received is None else now - self.last_received,
            stable_for_s=None if self.stable_since is None else max(0., now - self.stable_since),
            session_elapsed_s=None if self.session_start is None else now - self.session_start,
            latest_batch=copy.deepcopy(self.last_batch_info))
        if batch is not None:
            info = self._batch_info(batch)
            received = info["received_perf_counter"]
            result.update(trigger_batch=info,
                receipt_interval_s=None if received is None or self.last_received is None else received - self.last_received,
                processing_lag_s=None if received is None else now - received)
        return result

    def bootstrap(self):
        books = {}
        for count in (2, 4, 6):
            book = dict(_core().get_codebook(count))
            book.update(frames_per_bit=2, cycles=3, nominal_refresh=60)
            book.setdefault("id", book.get("hash"))
            book.setdefault("version", 1)
            books[str(count)] = book
        return dict(server_time=time.perf_counter(), clock_domain=CLOCK_DOMAIN, sample_rate=FS,
            channels=list(CHANNELS), contract=_core().get_contract(), codebook=books["6"],
            target_counts=[2, 4, 6], codebooks=books,
            frame_interval_policy=dict(FRAME_INTERVAL_POLICY),
            client_build=CLIENT_BUILD, frame_timing_action=FRAME_TIMING_ACTION,
            sessions=self.sessions(), models=self.model_files(), version="1.0.0")

    def sessions(self):
        return list_sessions(self.recordings)

    def model_files(self):
        result = []
        for path in sorted(self.models.glob("*.json"), reverse=True):
            if path.is_symlink():
                continue
            try:
                model = _core().load_model(path)
                metadata = model.get("metadata") if isinstance(model.get("metadata"), dict) else {}
                result.append(dict(id=path.stem, source=model.get("source"),
                    subject=metadata.get("subject", ""), recording_names=metadata.get("recording_names", []),
                    target_count=_core().target_count_from_contract(model["contract"]), metrics=model.get("metrics", {})))
            except (ValueError, OSError) as exc:
                result.append(dict(id=path.stem, status="incompatible", reason=str(exc)))
        return result

    @staticmethod
    def _path(base, identifier, suffix=""):
        if not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier or any(c in identifier for c in "/\\:"):
            raise ValueError("请选择本项目内的有效文件编号。")
        path = base / (identifier + suffix)
        if path.is_symlink() or path.resolve().parent != base.resolve():
            raise ValueError("文件路径不合法。")
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _reset_buffer(self):
        self.raw = self.filtered = np.empty((0, 8))
        self.times = np.empty(0)
        self.indices = np.empty(0, dtype=np.int64)
        self.preprocessor.reset()
        self.decoded_ends.clear()
        self.decode_starts.clear()
        self.decode_done.clear()

    def start(self, options):
        kind = options.get("source", "demo")
        if kind not in ("device", "demo"):
            raise ValueError("仅支持真实设备或演示采集；文件请使用回放。")
        if options.get("sample_rate", FS) != FS or options.get("channels", 8) != 8:
            raise ValueError("c-VEP 固定为250Hz、8通道。")
        source = (DeviceSource if kind == "device" else DemoSource)(dict(sample_rate=FS, channels=8,
            mode="eeg", device_ip=options.get("device_ip", "")))
        with self.lifecycle:
            with self.lock:
                if self.recorder:
                    raise RuntimeError("刺激录制正在进行，请先由拥有页结束或明确紧急停止。")
                if self.training.get("status") == "running":
                    raise RuntimeError("正在训练，请等待完成后采集。")
            self.stop()
            with self.lock:
                self.source, self.source_kind = source, kind
                self.stop_event = threading.Event()
                self.status, self.message = "connecting", "等待新鲜的八通道数据。"
                self.cursor = self.segment = 0
                self.sample_clock.reset()
                self.origin = self.last_received = self.last_sample = self.stable_since = None
                self.last_batch_info = None
                self.last_start_failure = None
                self.replay, self.replay_id, self.replay_playing = None, None, False
                self._reset_buffer()
                self.prediction = unavailable("等待刺激及连续数据。")
                self.predictions.clear()
                self.worker = threading.Thread(target=self._run, args=(source, self.stop_event), daemon=True, name="CVEP-acquisition")
                self.worker.start()
                self._event(f"开始{kind}采集；使用独立c-VEP时钟和录制目录。")
        return dict(ok=True)

    def stop(self):
        with self.lifecycle:
            with self.lock:
                self.stop_event.set()
                source, worker = self.source, self.worker
                if source:
                    source.close()
                self._finish("aborted", "采集已停止。")
                self.replay_playing = False
            if worker and worker is not threading.current_thread():
                worker.join(timeout=5)
                if worker.is_alive():
                    raise RuntimeError("c-VEP采集线程尚未退出，请稍后重试。")
            with self.lock:
                self.source = self.worker = None
                self.status, self.message = "idle", "采集已停止。"
                self.prediction = unavailable("采集已停止。")
        return dict(ok=True)

    def close(self):
        self.closed.set()
        self.stop()
        if self.monitor is not threading.current_thread():
            self.monitor.join(timeout=2)
        if self.training_thread and self.training_thread is not threading.current_thread():
            self.training_thread.join(timeout=10)

    def _run(self, source, stop):
        failure = None
        try:
            for batch in source.stream(stop):
                with self.lock:
                    if stop.is_set() or self.source is not source:
                        break
                    self._consume(batch)
        except Exception as exc:
            failure = exc
        finally:
            # Do not publish an inactive/error state while this source still
            # owns UDP: another workflow could otherwise race its close.
            try:
                source.close()
            except Exception as exc:
                failure = failure or exc
            with self.lock:
                if self.source is source and not stop.is_set():
                    self.status = "error"
                    self.message = f"c-VEP采集失败：{failure}" if failure else "采集源意外结束，请重新连接。"
                    self.stable_since = None
                    self._event(self.message, "error")
                    self._finish("aborted", self.message,
                                 self._acquisition_diagnostics("source_error"))
                    self.prediction = unavailable(self.message, "invalid")

    def _discontinuity(self, reason, diagnostics=None):
        self._finish("aborted", reason, diagnostics)
        self.segment += 1
        self.stable_since = None
        self.sample_clock.reset()
        self._reset_buffer()
        self.prediction = unavailable(reason, "invalid")
        self.message = reason

    def _consume(self, batch):
        now = time.perf_counter()
        data = np.asarray(batch.samples, float)
        if data.ndim != 2 or data.shape[1] != 8 or not np.isfinite(data).all():
            raise ValueError("数据必须是有限的N×8原始微伏样本。")
        if batch.gap:
            diagnostics = self._acquisition_diagnostics("source_gap", now=now, batch=batch)
            self.cursor += max(0, int(batch.missing_samples))
            self._discontinuity("设备数据出现断点，当前实验中止。", diagnostics)
        if not len(data):
            return
        received = finite(getattr(batch, "received_perf_counter", None), "QPC接收时间")
        if received > now + 1e-6:
            raise ValueError("QPC接收时间晚于当前时刻，禁止混用monotonic时钟。")
        if now - received > MAX_PROCESSING_LAG_S:
            diagnostics = self._acquisition_diagnostics("processing_lag", now=now, batch=batch)
            self.cursor += len(data)
            self._discontinuity("数据处理落后于收包超过250ms，当前实验中止。", diagnostics)
            return
        if self.last_received is not None and received - self.last_received > MAX_RECEIPT_GAP_S:
            self.stable_since = None
            self._interrupt("设备数据到达间隔超过1.25秒，当前实验中止。",
                             max(0., received - (self.session_start or received) - MAX_RECEIPT_GAP_S),
                             self._acquisition_diagnostics("receipt_gap", now=now, batch=batch))
        indices = np.arange(self.cursor, self.cursor + len(data), dtype=np.int64)
        timestamps, jumped = self.sample_clock.map(indices, received)
        if jumped:
            self._discontinuity("样本时钟发生突变，当前实验中止。",
                                self._acquisition_diagnostics("sample_clock_jump", now=now, batch=batch))
            timestamps, _ = self.sample_clock.map(indices, received)
        self.cursor += len(data)
        self.last_received, self.last_sample = received, float(timestamps[-1])
        # Keep the accepted predecessor until the trigger snapshot is captured.
        self.last_batch_info = self._batch_info(batch)
        if self.origin is None:
            self.origin = float(timestamps[0])
        if self.stable_since is None:
            self.stable_since = now
        if self.source_kind == "demo":
            data = self._demo_samples(timestamps)
        # A recording's analysis starts at its first retained raw sample, so
        # offline and live causal-filter state have exactly the same origin.
        if self.filter_start is not None:
            selected = timestamps >= self.filter_start
            filtered = np.zeros_like(data)
            filtered[selected] = self.preprocessor.process(data[selected])
            if np.any(selected):
                self.filter_start = None
        else:
            filtered = self.preprocessor.process(data)
        self.raw = np.concatenate((self.raw, data))[-5000:]
        self.filtered = np.concatenate((self.filtered, filtered))[-5000:]
        self.times = np.concatenate((self.times, timestamps))[-5000:]
        self.indices = np.concatenate((self.indices, indices))[-5000:]
        self.quality = (_core().raw_quality(self.raw[-500:]) if len(self.raw) >= 50 else
                        dict(valid=False, reason="等待足够的枕顶通道数据。", warnings=[]))
        self.status, self.message = "running", getattr(self.source, "message", "数据接收中")
        if self.recorder:
            relative = timestamps - self.session_start
            keep = relative >= 0
            self.recorder.write(data[keep], indices[keep], relative[keep], self.segment, received)
        self._infer()
        self._maybe_complete()
        if self.recorder and not self.quality["valid"]:
            self._interrupt("O2/O1/P8/P7信号硬故障，已停止实验并保留原始数据：" + self.quality["reason"],
                            max(0., float(timestamps[0]) - self.session_start))

    def _demo_samples(self, timestamps):
        data = self.rng.normal(0, 1.2, (len(timestamps), 8))
        data += np.sin(timestamps[:, None] * (2 * np.pi * 10)) * .5
        if not self.recorder:
            return data
        bits = np.asarray(self.recorder.manifest["codebook"]["frames"], float) * 2 - 1
        relative = timestamps - self.session_start
        for task in self.recorder.manifest["plan"]:
            if task["phase"] != "task":
                continue
            proof = self.frames_by_trial.get(task["trial_id"], [])
            if not proof:
                continue
            if (relative[-1] < proof[0]["observed_s"] + .08 or
                    relative[0] > self.tails.get(task["trial_id"], float("inf")) + .12):
                continue
            target = task["target"] if task["target"] is not None else self.demo_target
            ons = np.asarray([f["observed_s"] for f in proof])
            codes = np.asarray([f["code_frame"] % 126 for f in proof])
            response = np.zeros(len(timestamps))
            for latency, weight in ((.08, .45), (.10, .35), (.12, .20)):
                positions = np.searchsorted(ons, relative - latency, side="right") - 1
                valid = (positions >= 0) & (relative - latency <= self.tails.get(task["trial_id"], ons[-1] + FRAME_PERIOD))
                response[valid] += weight * bits[target, codes[positions[valid]]]
            data[:, :4] += response[:, None] * np.array([18., 16., 12., 10.])
        return data

    def start_session(self, options):
        try:
            result = self._start_session(options)
        except (ValueError, RuntimeError) as exc:
            with self.lock:
                details = self._acquisition_diagnostics("session_start_rejected")
                details.update(reason=str(exc), time=utc_now())
                self.last_start_failure = details
                self._event("开始刺激被拒绝：" + str(exc), "warning")
                # HTTP can return the same facts without logging the submitted body.
                exc.cvep_start_diagnostics = details
            LOGGER.warning("c-VEP session start rejected: %s", json.dumps(details, ensure_ascii=False))
            raise
        with self.lock:
            self.last_start_failure = None
        return result

    def _start_session(self, options):
        kind = options.get("kind", "calibration")
        if options.get("display_timing_method") != DISPLAY_METHOD:
            raise ValueError("请使用逐帧证据版本页面。")
        if options.get("client_build") != CLIENT_BUILD or "frame_interval_policy" not in options:
            raise ValueError("页面版本已过期，请刷新更新后的c-VEP页面再开始录制。")
        if frame_interval_policy(options["frame_interval_policy"]) != FRAME_INTERVAL_POLICY:
            raise ValueError("页面与后台的帧间隔门槛不一致，请刷新更新后的页面。")
        uncertainty = finite(options.get("initial_clock_sync_uncertainty_s"), "初始同步不确定度", 0, MAX_UNCERTAINTY)
        timing_budget = finite(options.get("software_timing_budget_s"), "总软件时序预算", MAX_UNCERTAINTY, MAX_UNCERTAINTY)
        target_count = _core().validate_target_count(options.get("target_count", 6))
        contract, codebook = _core().get_contract(target_count), _core().get_codebook(target_count)
        plan = make_plan(kind, options.get("trials_per_target", 6), options.get("trial_count", 12), options.get("seed"), target_count=target_count)
        demo_target = integer(options.get("demo_target", 0), "演示注视目标", 0, target_count - 1)
        with self.lifecycle, self.lock:
            now = time.perf_counter()
            if self.source_kind not in ("demo", "device") or self.status != "running":
                raise ValueError("请先开始设备或演示采集。")
            if self.recorder:
                raise ValueError("已有实验正在进行。")
            if (self.last_received is None or now - self.last_received > START_FRESHNESS_S
                    or self.stable_since is None or now - self.stable_since < 2):
                raise ValueError("请等待至少2秒新鲜、无间断的数据。")
            if len(self.raw) < 500:
                raise ValueError("尚未积累2秒原始数据，不能进行录前质量检查。")
            self.quality = _core().raw_quality(self.raw[-500:])
            if not self.quality["valid"]:
                raise ValueError("O2/O1/P8/P7录前质量检查未通过：" + self.quality["reason"])
            if kind == "online" and (self.model is None or self.model.get("source") != self.source_kind):
                raise ValueError("在线识别需要先加载与采集来源一致的个人c-VEP模型。")
            if kind == "online" and self.model["contract"] != contract:
                raise ValueError("在线识别的刺激类别数必须与个人模板一致，请选择对应的2/4/6类模型。")
            recorder = CVEPRecorder(self.recordings, self.source_kind, kind, plan, contract, now,
                                    timing_budget, options.get("subject", ""), codebook=codebook,
                                    initial_clock_sync_uncertainty_s=uncertainty, software_timing_budget_s=timing_budget,
                                    recording_name=options.get("recording_name", ""))
            self.recorder, self.session_start = recorder, now
            self.owner_token = secrets.token_urlsafe(24)
            self._reset_buffer()
            self.filter_start = now
            self.frames_by_trial, self.tails, self.decoded_ends = {}, {}, {}
            self.clock_audits = {}
            self.callback_max_by_trial = {}
            self._timing_invalid_trials = set()
            self.timing_warnings = []
            self.next_frame_index = 0
            self.last_frame_received = None
            self.finish_requested = False
            self.waiting_final_audit = False
            self.demo_target = demo_target
            self.prediction = unavailable("等待完整刺激证据与1.5秒数据窗口。", "warming_up")
            self.predictions.clear()
            self.prediction_invalidations.clear()
            self.prediction_id = 0
            try:
                self.prediction_handle = (self.output / f"{recorder.id}.jsonl").open("x", encoding="utf-8")
            except OSError as exc:
                self._event(f"预测导出不可写：{exc}", "error")
            self._event(f"开始{kind}实验：{recorder.id}")
            return dict(ok=True, session=self._session_state(), owner_token=self.owner_token, server_time=time.perf_counter())

    def frames(self, options):
        with self.lock:
            if not self.recorder or options.get("id") != self.recorder.id:
                return dict(ok=False, ignored=True)
            if not self._is_owner(options):
                raise ValueError("只有启动本次刺激的页面可以提交帧证据。")
            now = time.perf_counter() - self.session_start
            frame_policy = self.recorder.manifest["frame_interval_policy"]
            try:
                if options.get("visible") is not True:
                    raise ValueError("刺激页面不可见。")
                trial = integer(options.get("trial_id"), "刺激试次", 1)
                tasks = [p for p in self.recorder.manifest["plan"] if p["phase"] == "task"]
                if trial > len(tasks) or trial != len(self.tails) + 1:
                    raise ValueError("刺激试次次序不连续。")
                incoming = options.get("frames", [])
                if not isinstance(incoming, list) or len(incoming) > 16 or (not incoming and options.get("tail_s") is None):
                    raise ValueError("每次应提交1～16帧，或末尾显示证据。")
                previous = self.frames_by_trial.get(trial, [])
                try:
                    audit, sync_budget = self._validate_clock_audit(options.get("clock_audit"), trial)
                except ValueError as exc:
                    prior = self.clock_audits.get(trial) or self.clock_audits.get(trial - 1)
                    if prior:
                        raise _RetroactiveTimingError(str(exc), prior["checked_at_s"]) from exc
                    raise
                if trial > 1 and sync_budget + self.callback_max_by_trial.get(trial - 1, 0.) > MAX_UNCERTAINTY + 1e-9:
                    self._timing_warning(trial - 1, "post_clock_budget", "事后时钟复核与回调迟到合计超过8ms。",
                                         sync_s=sync_budget, callback_s=self.callback_max_by_trial.get(trial - 1, 0.))
                declared_sync = finite(options.get("clock_sync_uncertainty_s"), "本批同步预算")
                if abs(declared_sync - sync_budget) > 1e-9:
                    raise ValueError("本批同步预算与时钟复核证据不一致。")
                callback_lags = [finite(f.get("callback_end_lag_s"), "回调提交迟到") for f in incoming]
                if options.get("tail_s") is not None:
                    callback_lags.append(finite(options.get("tail_callback_end_lag_s"), "尾帧提交迟到"))
                callback_lag = max(callback_lags, default=0.)
                declared_lag = finite(options.get("callback_end_lag_s"), "本批回调迟到")
                budget = finite(options.get("clock_uncertainty_s"), "本批总软件时序预算")
                if abs(declared_lag - callback_lag) > 1e-9 or abs(budget - sync_budget - callback_lag) > 1e-9:
                    raise ValueError("本批软件时序预算没有如实包含逐帧回调提交迟到。")
                if sync_budget + callback_lag > self.recorder.manifest["software_timing_budget_s"] + 1e-9:
                    self._timing_warning(trial, "callback_budget", "显示总软件时序预算超过8ms。",
                                         sync_s=sync_budget, callback_s=callback_lag, total_s=budget)
                checked = []
                last = previous[-1]["observed_s"] if previous else None
                for n, frame in enumerate(incoming):
                    index = integer(frame.get("frame_index"), "全局帧序号")
                    code = integer(frame.get("code_frame"), "编码帧序号", 0, FRAMES_PER_TRIAL - 1)
                    onset = finite(frame.get("observed_s"), "帧显示时间")
                    if index != self.next_frame_index + n or code != len(previous) + n:
                        raise ValueError("刺激帧缺失、重复或顺序错误；禁止补造帧。")
                    if now - onset < callback_lags[n] - sync_budget - 1e-9:
                        raise ValueError("帧证据超出保存的时钟预算，早于实际回调提交。")
                    if now - onset > .35:
                        self._timing_warning(trial, "frame_report_age", "帧证据报告迟到超过350ms。",
                                             age_s=now - onset, code_frame=code)
                    if last is not None and onset <= last:
                        raise ValueError("帧显示时间必须严格递增，禁止倒退或重复。")
                    if last is not None and not frame_policy["min_s"] - 1e-9 <= onset - last <= frame_policy["max_s"] + 1e-9:
                        self._timing_warning(trial, "frame_interval", "单帧间隔超出目标范围。",
                                             interval_s=onset - last, code_frame=code)
                    if last is None and onset < (self.tails.get(trial - 1, 0) + 1.95):
                        raise ValueError("试次开始前休息不足2秒。")
                    if last is None and onset < audit["checked_at_s"]:
                        raise ValueError("时钟复核不能晚于本试次首帧。")
                    if last is None and onset - audit["checked_at_s"] > 3:
                        self._timing_warning(trial, "clock_audit_age", "首帧前的时钟复核已超过3秒。",
                                             age_s=onset - audit["checked_at_s"])
                    checked.append(dict(frame_index=index, trial_id=trial, code_frame=code, observed_s=onset,
                                        callback_end_lag_s=callback_lags[n]))
                    last = onset
                tail = options.get("tail_s")
                if tail is not None:
                    tail = finite(tail, "刺激结束帧")
                    if len(previous) + len(checked) != FRAMES_PER_TRIAL or last is None:
                        raise ValueError("尚未收到378个刺激帧，不能确认结束。")
                    if tail <= last or now - tail < callback_lags[-1] - sync_budget - 1e-9:
                        raise ValueError("刺激尾帧缺失或时间无效。")
                    if not frame_policy["min_s"] - 1e-9 <= tail - last <= frame_policy["max_s"] + 1e-9:
                        self._timing_warning(trial, "tail_interval", "最后刺激帧至清屏的间隔超出目标范围。", interval_s=tail - last)
                    if now - tail > .35:
                        self._timing_warning(trial, "frame_report_age", "尾帧证据报告迟到超过350ms。", age_s=now - tail)
                    first = previous[0]["observed_s"] if previous else checked[0]["observed_s"]
                    if abs((tail - first) / FRAMES_PER_TRIAL - FRAME_PERIOD) > frame_policy["mean_tolerance_s"] + 1e-9:
                        self._timing_warning(trial, "mean_interval", "整段刺激的平均间隔偏离60Hz。",
                                             mean_interval_s=(tail - first) / FRAMES_PER_TRIAL)
                details = dict(trial_id=trial, frames=checked, received_s=now, visible=True,
                               clock_uncertainty_s=budget, clock_sync_uncertainty_s=sync_budget,
                               callback_end_lag_s=callback_lag, clock_audit=audit, tail_s=tail)
                if tail is not None:
                    details["tail_callback_end_lag_s"] = callback_lags[-1]
                self.recorder.event("frames", details)
                self.clock_audits[trial] = audit
                self.callback_max_by_trial[trial] = max(self.callback_max_by_trial.get(trial, 0.), callback_lag)
                self.frames_by_trial.setdefault(trial, []).extend(checked)
                self.next_frame_index += len(checked)
                self.last_frame_received = time.perf_counter()
                if tail is not None:
                    self.tails[trial] = tail
                    if len(self.tails) == len(tasks):
                        self.waiting_final_audit = True
                self._infer()
                if tail is not None:
                    self.prediction = unavailable("试次结束，当前处于休息或等待末尾数据。", "rest")
                elif trial in self._timing_invalid_trials:
                    self.prediction = unavailable("当前试次时序不合格；EEG继续保存，等待后续正常试次。", "timing_warning")
                elif not previous:
                    self.prediction = unavailable("新试次积累编码周期与1.5秒数据窗口。", "warming_up")
                self._maybe_complete()
                return dict(ok=True, accepted=True, frames_accepted=len(checked),
                            timing_valid=trial not in self._timing_invalid_trials,
                            timing_warnings=[copy.deepcopy(w) for w in self.timing_warnings if w["trial_id"] == trial])
            except Exception as exc:
                cutoff = (exc.invalid_since if isinstance(exc, _RetroactiveTimingError) else
                          max(0., now - .35, max(self.tails.values(), default=0.)))
                self._interrupt(str(exc), cutoff)
                raise

    def _is_owner(self, options):
        token = options.get("owner_token")
        return isinstance(token, str) and self.owner_token is not None and secrets.compare_digest(token, self.owner_token)

    def _validate_clock_audit(self, value, trial, *, final=False):
        if not isinstance(value, dict) or integer(value.get("trial_id"), "时钟复核试次", 1) != trial:
            raise ValueError("缺少本试次时钟复核。")
        checked = finite(value.get("checked_at_s"), "时钟复核完成时刻")
        delta = finite(value.get("offset_delta_s"), "时钟映射漂移", -float("inf"))
        new = finite(value.get("new_sync_uncertainty_s"), "复核同步不确定度")
        effective = finite(value.get("effective_sync_uncertainty_s"), "冻结映射有效不确定度")
        lower_bound = self.tails.get(trial, 0.) if final else self.tails.get(trial - 1, 0.)
        if abs(effective - abs(delta) - new) > 1e-9 or checked < lower_bound:
            raise ValueError("时钟复核预算或阶段不合法。")
        audit = dict(trial_id=trial, checked_at_s=checked, offset_delta_s=delta,
                     new_sync_uncertainty_s=new, effective_sync_uncertainty_s=effective)
        if not final and trial in self.clock_audits and self.clock_audits[trial] != audit:
            raise ValueError("同一试次中不能替换时钟复核或刺激时间映射。")
        return audit, effective

    def _interrupt(self, reason, invalid_since, diagnostics=None):
        if self.recorder:
            try:
                # The owning page may discover a clock-mapping failure during
                # the next rest and invalidate an earlier trial. Emergency
                # observers instead supply the safe last-tail cutoff explicitly.
                cutoff = max(0., invalid_since)
                self.recorder.event("interruption", dict(invalid_since_s=cutoff, reason=str(reason)))
            finally:
                self._finish("aborted", reason, diagnostics)
        self.prediction = unavailable(reason, "invalid")

    def stop_session(self, options=None):
        options = options or {}
        with self.lock:
            if not self.recorder:
                return dict(ok=True, session=copy.deepcopy(self.last_session))
            if options.get("id") and options["id"] != self.recorder.id:
                return dict(ok=False, ignored=True)
            if options.get("emergency") is True:
                self._interrupt(options.get("reason", "其他页面紧急停止实验。"), max(self.tails.values(), default=0.))
                return dict(ok=True, session=self._session_state())
            if not self._is_owner(options):
                raise ValueError("此页面不拥有本次刺激；如需停止，请使用明确的紧急停止操作。")
            if options.get("reason") == "completed":
                if self.finish_requested:
                    return dict(ok=True, session=self._session_state())
                if len(self.tails) != len([p for p in self.recorder.manifest["plan"] if p["phase"] == "task"]):
                    raise ValueError("全部任务的刺激证据尚未完整，不能标记完成。")
                last_trial = max(self.tails)
                try:
                    audit, effective = self._validate_clock_audit(options.get("final_clock_audit"), last_trial, final=True)
                    if effective + self.callback_max_by_trial.get(last_trial, 0.) > MAX_UNCERTAINTY + 1e-9:
                        self._timing_warning(last_trial, "post_clock_budget", "最终时钟复核加上末试次回调迟到超过8ms。",
                                             sync_s=effective, callback_s=self.callback_max_by_trial.get(last_trial, 0.))
                    received = time.perf_counter() - self.session_start
                    if (audit["checked_at_s"] > self.tails[last_trial] + 3 or
                            audit["checked_at_s"] > received + effective + 1e-9):
                        raise ValueError("最后刺激后的时钟复核超时或晚于实际接收。")
                    self.recorder.event("final_clock_audit", dict(audit, received_s=received))
                except (ValueError, TypeError) as exc:
                    self._timing_warning(last_trial, "final_clock_audit", "最后刺激后的时钟复核缺失或失败。",
                                         error=str(exc), client_error=str(options.get("final_clock_audit_error", ""))[:240])
                self.waiting_final_audit = False
                self.finish_requested = True
                self._maybe_complete()
            else:
                now = time.perf_counter() - self.session_start
                cutoff = finite(options.get("invalid_since_s", now), "失效时刻", 0, now + MAX_UNCERTAINTY)
                self._interrupt(options.get("reason", "用户停止实验。"), cutoff)
            return dict(ok=True, session=self._session_state())

    def _maybe_complete(self):
        if self.recorder and self.finish_requested and self.tails:
            tail = max(self.tails.values())
            if self.last_sample is not None and self.last_sample - self.session_start >= tail:
                self._finish("complete", "全部刺激帧及末尾EEG已接收。")

    def _finish(self, status, reason, diagnostics=None):
        if not self.recorder:
            return
        recorder, self.recorder = self.recorder, None
        self.owner_token = None
        self.last_session = dict(active=False, id=recorder.id, kind=recorder.manifest["kind"],
            subject=recorder.manifest["subject"], recording_name=recorder.manifest.get("recording_name", ""),
            frame_interval_policy=dict(recorder.manifest["frame_interval_policy"]),
            client_build=CLIENT_BUILD, frame_timing_action=FRAME_TIMING_ACTION,
            timing_invalid_trials=sorted(self._timing_invalid_trials), timing_warnings=copy.deepcopy(self.timing_warnings),
            target_count=recorder.manifest["target_count"], plan=recorder.manifest["plan"], start_time=self.session_start, status=status, reason=str(reason), finishing=False)
        if diagnostics is not None:
            diagnostics = dict(diagnostics, reason=str(reason))
            self.last_session["acquisition_diagnostics"] = copy.deepcopy(diagnostics)
            LOGGER.warning("c-VEP acquisition interrupted: %s", json.dumps(diagnostics, ensure_ascii=False))
            try:
                recorder.event("acquisition_diagnostic", diagnostics)
            except Exception as exc:
                # A recorder with a prior write error rejects further events
                # with RuntimeError; diagnostic failure must still reach close.
                self._event(f"中止诊断写入失败：{exc}", "error")
        try:
            recorder.close(status, str(reason))
            loaded = load_session(recorder.path)
            self.last_session["validation"] = dict(status="complete", **loaded["diagnostics"])
        except Exception as exc:
            self.last_session.update(status="error", reason=f"录制保存/验证失败：{exc}",
                                     validation=dict(status="error", error=str(exc)))
            self._event(self.last_session["reason"], "error")
        finally:
            if self.prediction_handle:
                self.prediction_handle.close()
                self.prediction_handle = None
        self._event(reason, "info" if status == "complete" else "warning")
        self.prediction = unavailable("实验已结束；历史结果保留。")

    def _session_state(self):
        if not self.recorder:
            return copy.deepcopy(self.last_session)
        return dict(active=True, id=self.recorder.id, kind=self.recorder.manifest["kind"],
            subject=self.recorder.manifest["subject"], recording_name=self.recorder.manifest.get("recording_name", ""),
            frame_interval_policy=dict(self.recorder.manifest["frame_interval_policy"]),
            client_build=CLIENT_BUILD, frame_timing_action=FRAME_TIMING_ACTION,
            timing_invalid_trials=sorted(self._timing_invalid_trials), timing_warnings=copy.deepcopy(self.timing_warnings),
            target_count=self.recorder.manifest["target_count"],
            start_time=self.session_start, plan=self.recorder.manifest["plan"],
            contract=self.recorder.manifest["contract"], elapsed_s=max(0., time.perf_counter() - self.session_start),
            current_trial=len(self.tails) + 1, finishing=self.finish_requested or self.waiting_final_audit,
            waiting_final_audit=self.waiting_final_audit,
            samples=self.recorder.manifest["sample_count"], completed_trials=len(self.tails))

    def _infer(self):
        if not self.recorder or self.model is None or self.model.get("source") != self.source_kind:
            return
        if self.model["contract"] != self.recorder.manifest["contract"]:
            self.prediction = unavailable("当前模板与本次刺激类别数不一致；校准数据仍正常录制。", "no_model")
            return
        if self.last_received is None or time.perf_counter() - self.last_received > PREDICTION_FRESHNESS_S:
            # Late frame evidence can unlock a buffered EEG window without a
            # new packet. Keep its cursor pending until data is fresh again.
            self.prediction = unavailable("EEG数据暂未到达，等待连续样本；当前不输出新识别结果。")
            return
        relative = self.times - self.session_start
        for trial, proof in self.frames_by_trial.items():
            if trial in self._timing_invalid_trials:
                if trial == len(self.tails) + 1:
                    self.prediction = unavailable("当前试次时序不合格；EEG继续保存，等待后续正常试次。", "timing_warning")
                continue
            if len(proof) < 127 or trial in self.decode_done:
                continue
            frames = list(proof)
            if trial in self.tails:
                frames.append(dict(trial_id=trial, code_frame=378, observed_s=self.tails[trial]))
            coverage = frames[-1]["observed_s"]
            positions = np.flatnonzero((relative >= proof[126]["observed_s"]) & (relative <= coverage))
            if len(positions) < 375:
                continue
            self.decode_starts.setdefault(trial, int(self.indices[positions[0]]))
            start_index = self.decode_starts[trial]
            next_end = self.decoded_ends.get(trial, start_index + 374 - 25) + 25
            last_index = int(self.indices[positions[-1]])
            for index in range(next_end, last_index + 1, 25):
                self.decoded_ends[trial] = index
                begin = int(np.searchsorted(self.indices, index - 374))
                end = int(np.searchsorted(self.indices, index, side="right"))
                if end - begin != 375 or self.indices[begin] != index - 374 or self.indices[end - 1] != index:
                    continue
                result = _core().predict_epoch(self.model, self.filtered[end - 375:end], relative[end - 375:end], frames,
                                               source_kind=self.source_kind, raw_samples=self.raw[end - 375:end])
                self.prediction_id += 1
                item = dict(result, type="prediction", timing_valid=True, id=self.prediction_id, trial_id=trial, start_s=float(relative[end - 375]),
                            end_s=float(relative[end - 1]), source=self.source_kind, server_time=time.perf_counter())
                if trial not in self.tails:
                    self.prediction = item
                self.predictions.append(item)
                self._export_prediction(item)
            if trial in self.tails and len(relative) and relative[-1] >= self.tails[trial]:
                self.decode_done.add(trial)

    def prediction_records(self, after=0):
        with self.lock:
            return dict(records=[copy.deepcopy(p) for p in self.predictions if p["id"] > after],
                        invalidations=[copy.deepcopy(p) for p in self.prediction_invalidations if p["id"] > after],
                        timing_invalid_trials=sorted(self._timing_invalid_trials), latest_id=self.prediction_id)

    def train(self, options):
        identifiers = options.get("ids")
        if not isinstance(identifiers, list) or not identifiers or len(identifiers) > 100:
            raise ValueError("请选择1～100段校准录制。")
        paths = [self._path(self.recordings, identifier) for identifier in dict.fromkeys(identifiers)]
        with self.lock:
            if self.recorder or self.status in ("running", "connecting") or self.training.get("status") == "running":
                raise ValueError("请先停止采集/回放，且等待当前训练完成。")
            self.training = dict(status="running", message="正在审计逐帧证据并按试次划分训练。", metrics=None, diagnostics=None)
            self.training_thread = threading.Thread(target=self._train, args=(paths,), daemon=True, name="CVEP-training")
            self.training_thread.start()
        return dict(ok=True)

    def _train(self, paths):
        diagnostics = None
        try:
            sessions = [load_session(path) for path in paths]
            provenance = [dict(id=s["metadata"]["id"], subject=s["metadata"].get("subject", ""),
                               recording_name=s["metadata"].get("recording_name", "")) for s in sessions]
            diagnostics = dict(per_recording=[dict(**identity, **s["diagnostics"]) for identity, s in zip(provenance, sessions)])
            subjects = {s["metadata"].get("subject", "").strip() for s in sessions if s["metadata"].get("subject", "").strip()}
            unidentified = sum(not s["metadata"].get("subject", "").strip() for s in sessions)
            diagnostics["subject"] = dict(ids=sorted(subjects), unidentified_recordings=unidentified,
                warning="存在未填写受试者编号的录制，请自行核对全部录制来自同一人。" if unidentified else "")
            if len(subjects) > 1:
                raise ValueError("个人c-VEP模板不能混合不同受试者编号。受试者编号表示固定身份；同一人的每轮代号应填写在“录制代号”。请核对所选录制的身份，同一人使用同一受试者编号，不同人分别训练。")
            if any(s["metadata"]["kind"] != "calibration" for s in sessions):
                raise ValueError("训练只能使用有真实提示目标标签的校准录制；在线录制不能当作训练标签。")
            model = _core().train_model(sessions)
            model.setdefault("metadata", {})["subject"] = next(iter(subjects), "")
            model["metadata"]["subject_warning"] = diagnostics["subject"]["warning"]
            model["metadata"]["recordings"] = provenance
            model["metadata"]["recording_names"] = sorted({p["recording_name"] for p in provenance if p["recording_name"]})
            identifier = unique_name("cvep_model", "")
            _core().save_model(model, self.models / f"{identifier}.json")
            with self.lock:
                self.model, self.model_id = model, identifier
                self.training = dict(status="complete", source=model["source"], message="训练完成；测试指标仅代表当前录制。", metrics=model.get("metrics", {}),
                                     diagnostics=diagnostics, model_id=identifier)
        except Exception as exc:
            with self.lock:
                decoder_diagnostics = getattr(exc, "diagnostics", None)
                if decoder_diagnostics is not None:
                    diagnostics = dict(diagnostics or {}, decoder=decoder_diagnostics)
                self.training = dict(status="error", message=str(exc), metrics=None,
                                     diagnostics=diagnostics)
                self._event(f"c-VEP训练失败：{exc}", "error")

    def load_model(self, identifier):
        model = _core().load_model(self._path(self.models, identifier, ".json"))
        with self.lock:
            if self.recorder or self.training.get("status") == "running":
                raise ValueError("实验或训练中不能切换模型。")
            self.model, self.model_id = model, identifier
            self.prediction = unavailable("模型已加载，等待新实验。")
        return dict(ok=True)

    def load_replay(self, identifier):
        with self.lifecycle:
            with self.lock:
                if self.recorder:
                    raise RuntimeError("刺激录制正在进行，不能用回放请求中止它。")
                if self.training.get("status") == "running":
                    raise RuntimeError("正在训练，请完成后再加载回放。")
            loaded = load_session(self._path(self.recordings, identifier))
            self.stop()
            with self.lock:
                self.replay, self.replay_id = loaded, identifier
                self.source_kind = "replay"
                self.replay_position, self.replay_speed, self.replay_playing = 0., 1., False
                self.status, self.message = "idle", "已加载c-VEP原始回放与帧审计；回放不会驱动闪烁刺激。"
                self.prediction = unavailable("回放仅展示录制波形与真实帧证据。")
        return dict(ok=True)

    def replay_control(self, options):
        with self.lock:
            if self.replay is None:
                raise ValueError("请先加载c-VEP录制。")
            action = options.get("action")
            if "speed" in options:
                self.replay_speed = finite(options["speed"], "回放速度", .25, 4)
            if action == "seek":
                self.replay_position = finite(options.get("position_s"), "回放位置", 0, self._replay_duration())
            elif action in ("play", "pause"):
                self.replay_playing = action == "play"
            else:
                raise ValueError("回放操作应为play、pause或seek。")
            self.replay_clock = time.perf_counter()
            self.status = "running" if self.replay_playing else "idle"
        return dict(ok=True)

    def _replay_duration(self):
        return float(self.replay["times"][-1]) if self.replay is not None and len(self.replay["times"]) else 0.

    def _monitor(self):
        while not self.closed.wait(.025):
            try:
                with self.lock:
                    self._monitor_tick(time.perf_counter())
            except Exception as exc:
                with self.lock:
                    self._event(f"c-VEP监控失败：{exc}", "error")
                    self._finish("error", str(exc))

    def _monitor_tick(self, now):
        if self.replay is not None and self.replay_playing:
            self.replay_position = min(self._replay_duration(), self.replay_position + (now - self.replay_clock) * self.replay_speed)
            self.replay_clock = now
            if self.replay_position >= self._replay_duration():
                self.replay_playing, self.status = False, "idle"
        if self.source_kind != "replay" and self.status in ("running", "connecting"):
            age = None if self.last_received is None else now - self.last_received
            if age is not None and age > MAX_RECEIPT_GAP_S:
                self.stable_since = None
                self.quality = dict(valid=False, reason="EEG数据已超时，请重新等待2秒连续新鲜数据。", warnings=[])
                self.prediction = unavailable(self.quality["reason"], "invalid")
                self.status, self.message = "connecting", self.quality["reason"]
            elif age is not None and age > PREDICTION_FRESHNESS_S:
                # Stop showing the previous decision while waiting for a late
                # packet; continuity is checked when the next packet arrives.
                self.message = "EEG数据暂未到达，等待连续样本；当前不输出新识别结果。"
                self.prediction = unavailable(self.message)
        if self.recorder:
            if self.last_received is None or now - self.last_received > MAX_RECEIPT_GAP_S:
                self._interrupt("EEG数据等待超过1.25秒，实验已中止。",
                                max(0., now - self.session_start - MAX_RECEIPT_GAP_S),
                                self._acquisition_diagnostics("receipt_timeout", now=now))
                return
            current = len(self.tails) + 1
            if self.waiting_final_audit:
                last_trial = max(self.tails)
                if now - self.session_start > self.tails[last_trial] + 3:
                    self._timing_warning(last_trial, "final_clock_audit", "最后刺激后的时钟复核超过3秒未完成。",
                                         age_s=now - self.session_start - self.tails[last_trial])
                    self.waiting_final_audit = False
                    self.finish_requested = True
                    self._maybe_complete()
                return
            proof = self.frames_by_trial.get(current, [])
            if proof and now - self.last_frame_received > .35:
                self._timing_warning(current, "frame_report_silence", "超过350ms未收到新帧证据。",
                                     age_s=now - self.last_frame_received)
            elif not proof and not self.finish_requested and now - self.session_start > self.tails.get(current - 1, 0.) + 3.:
                self._timing_warning(current, "next_trial_late", "下一试次超过计划准备时间尚未开始。",
                                     elapsed_since_tail_s=now - self.session_start - self.tails.get(current - 1, 0.))
            else:
                self._maybe_complete()

    def snapshot(self):
        with self.lock:
            now = time.perf_counter()
            origin = self.source_kind
            annotations = []
            if self.replay is not None:
                times = self.replay["times"]
                keep = (times >= max(0., self.replay_position - 5)) & (times <= self.replay_position)
                shown, traces = times[keep], self.replay["samples"][keep]
                annotations = self.replay["trials"]
                origin = self.replay["metadata"]["source"]
            else:
                shown = self.times[-1250:] - (self.session_start if self.recorder else self.origin or 0)
                traces = self.raw[-1250:]
            compatible = self.model is not None and self.model.get("source") == origin
            shown_prediction = self.prediction
            if (self.recorder is not None and self.source_kind != "replay"
                    and (self.last_received is None or now - self.last_received > PREDICTION_FRESHNESS_S)):
                shown_prediction = unavailable("EEG数据暂未到达，等待连续样本；当前不输出新识别结果。")
            model_metadata = self.model.get("metadata", {}) if self.model else {}
            if not isinstance(model_metadata, dict):
                model_metadata = {}
            return dict(server_time=now, clock_domain=CLOCK_DOMAIN, status=self.status, source=self.source_kind,
                message=self.message, device_ip=getattr(self.source, "device_ip", None),
                signal=dict(times=shown.tolist(), traces=traces.T.tolist(), channel_names=list(CHANNELS), annotations=annotations),
                session=self._session_state(), prediction=copy.deepcopy(shown_prediction),
                last_start_failure=copy.deepcopy(self.last_start_failure),
                model=dict(available=self.model is not None, id=self.model_id, source=self.model.get("source") if self.model else None,
                    subject=model_metadata.get("subject", ""), recording_names=copy.deepcopy(model_metadata.get("recording_names", [])),
                    target_count=len(self.model["contract"]["targets"]) if self.model else None,
                    compatible=compatible, reason="" if compatible else "未加载模型或模型来源与当前采集不一致。",
                    metrics=copy.deepcopy(self.model.get("metrics", {})) if self.model else {}),
                training=copy.deepcopy(self.training), telemetry=dict(getattr(self.source, "stats", {})) if self.source else {},
                quality=copy.deepcopy(self.quality),
                events=list(self.events), playback=dict(loaded=self.replay is not None, id=self.replay_id,
                    playing=self.replay_playing, position_s=self.replay_position, speed=self.replay_speed,
                    duration_s=self._replay_duration(), diagnostics=self.replay["diagnostics"] if self.replay else None))
