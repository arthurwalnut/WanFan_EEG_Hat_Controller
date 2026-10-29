"""Fixed eight-channel experiment orchestration, using the old project's ML core.

The device has one owner. Cue time, file annotations and EEG are kept separate
from predictions; no predicted class is ever used as a training label.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .mi_recording import CHANNELS, FS, SessionRecorder, list_sessions, load_session, _atomic_manifest
from .mi_delete import RecordingDeletion
from .mi_library import LibraryMetadata, ModelDeletion
from .mi_test_job import EvaluationJob
from .sources import DemoSource, DeviceSource
from .sample_clock import CLOCK_METHOD, ReceiptSampleClock
from .storage import unique_name, utc_now
from .robot_bridge import RobotBridge, prediction_available

PIPELINE_SRC = Path(__file__).resolve().parents[2] / "EEG_Signal_to_Decision" / "src"
if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))
from bci.hat_pipeline import StreamingPipeline, train_model, save_model, load_model
from bci.hat_pipeline.contract import VERSION, input_contract
from bci.hat_pipeline.dataset import build_dataset, DatasetBuildError
from bci.hat_pipeline.processing import raw_quality

DISPLAY_TIMING_METHOD = "raf-visible-heartbeat-v1"
MAX_PROCESSING_LAG_S = .25
DISPLAY_LEASE_S = .6
PACKET_LOSS_INTERVAL_S = 15.0
SAMPLES_PER_PACKET = 50


def number(value, minimum, maximum, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name}必须是有限数字。")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name}应在 {minimum}～{maximum} 之间。")
    return float(value)


def unavailable(reason, status="unavailable"):
    return {"status": status, "label": None, "confidence": None, "probabilities": [],
            "label_order": ["rest", "left_mi", "right_mi"], "reason": reason}


class MIEngine:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.recordings = self.root / "data" / "mi_recordings"
        self.models = self.root / "data" / "mi_models"
        self.output = self.root / "data" / "mi_predictions"
        for directory in (self.recordings, self.models, self.output):
            directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.lifecycle = threading.RLock()
        self.stop_event = threading.Event()
        self.closed = threading.Event()
        self.worker = None
        self.source = None
        self.source_kind = "demo"
        self.status = "idle"
        self.message = "固定 250 Hz / 8 通道。先连接设备，或用模拟信号检查流程。"
        self.events = deque(maxlen=60)
        self.pipeline = StreamingPipeline()
        self.model = None
        self.model_id = None
        self.model_display_name = None
        self.model_error = None
        self.prediction = unavailable("尚未加载八通道三类模型。", "no_model")
        self.predictions = deque(maxlen=5000)
        self.result_id = 0
        self.prediction_handle = None
        self.prediction_path = None
        self.samples = np.empty((0, 8))
        self.times = np.empty(0)
        self.cursor = 0
        self.segment = 0
        self.anchor = None
        self.sample_clock = ReceiptSampleClock(FS)
        self.session_origin = None
        self.last_sample_at = None
        self.last_sample_time = None
        self.last_receipt_time = None
        self.stream_stable_since = None
        self.recorder = None
        self.record_start = None
        self.last_recording = {"active": False, "id": None, "plan": []}
        self.record_last_time = None
        self.display_last_observed = None
        self.display_last_received = None
        self.record_clock_uncertainty = None
        self.record_packet_loss = None
        self.validation_threads = []
        self.replay = None
        self.replay_id = None
        self.replay_position = 0.0
        self.replay_speed = 1.0
        self.replay_playing = False
        self.replay_index = 0
        self.replay_clock = time.monotonic()
        self.training = {"status": "idle", "message": "请选择录制文件和算法。"}
        self.training_thread = None
        self.recording_deletion = RecordingDeletion(self.recordings)
        self.library = LibraryMetadata(self.recordings, self.models)
        self.model_deletion = ModelDeletion(self.models)
        self.evaluation = EvaluationJob(self.root / "data" / "mi_evaluations")
        self.robot = RobotBridge(log_dir=self.root / "data" / "mi_robot")
        self.monitor = threading.Thread(target=self._monitor, daemon=True, name="MI-monitor")
        self.monitor.start()

    def _event(self, message, level="info"):
        self.events.append({"time": utc_now(), "level": level, "message": message})

    def sessions(self):
        with self.lock:
            return self.library.decorate_recordings(list_sessions(self.recordings))

    def _require_recording_files_idle(self):
        self.evaluation.require_idle()
        if self.recorder or self.status in ("running", "connecting") or (self.worker and self.worker.is_alive()):
            raise RuntimeError("请先停止采集或暂停回放，再管理录制数据和模型。")
        if self.training["status"] == "running" or (self.training_thread and self.training_thread.is_alive()):
            raise RuntimeError("模型正在训练，请等待完成后再管理录制数据和模型。")
        if any(worker.is_alive() for worker in self.validation_threads):
            raise RuntimeError("录制文件正在检查或补齐，请等待完成后再管理数据。")

    def preview_recording_deletion(self, options):
        with self.lock:
            self._require_recording_files_idle()
            preview = self.recording_deletion.preview(options.get("recordings"))
            preview["recordings"] = self.library.decorate_recordings(preview["recordings"])
            return preview

    def update_library(self, options):
        if set(options) - {"kind", "id", "display_name", "favorite"}:
            raise ValueError("名称或收藏请求含未知字段。")
        changes = {key: options[key] for key in ("display_name", "favorite") if key in options}
        if not changes:
            raise ValueError("请提供新名称或收藏状态。")
        with self.lock:
            self._require_recording_files_idle()
            result = self.library.update(options.get("kind"), options.get("id"), **changes)
            if result.get("kind") == "model" and result.get("id") == self.model_id:
                current = next((row for row in self.model_files() if row["id"] == self.model_id), None)
                if current:
                    self.model_display_name = current["display_name"]
            return result

    def preview_model_deletion(self, options):
        with self.lock:
            self._require_recording_files_idle()
            return self.model_deletion.preview(options.get("models"))

    def delete_models(self, options):
        with self.lock:
            self._require_recording_files_idle()
            selected = set(self.model_deletion.selected(options.get("token")))
            result = self.model_deletion.delete(options.get("token"))
            # A partial filesystem failure must not leave a selected, removed
            # model running from memory. Unload after a confirmed attempt.
            if self.model_id in selected:
                self.robot.disarm("已确认删除当前模型，机械臂发送已关闭。")
                self.model = self.model_id = self.model_display_name = None
                self.model_error = None
                self.pipeline.set_model(None)
                self.pipeline.reset()
                self.predictions.clear()
                self.prediction = unavailable("当前模型已卸载，请重新选择模型。", "no_model")
            if self.training.get("model_id") in selected:
                self.training = {"status": "idle", "message": "已处理所选模型的删除，请核对模型列表。"}
            if result.get("deleted"):
                self._event(f"已删除 {len(result['deleted'])} 个模型；录制数据保留。")
            return result

    def delete_recordings(self, options):
        with self.lock:
            self._require_recording_files_idle()
            result = self.recording_deletion.delete(options.get("token"))
            deleted = set(result.get("deleted", []))
            if self.last_recording.get("id") in deleted:
                self.last_recording = {"active": False, "id": None, "plan": []}
            if self.replay_id in deleted:
                self.replay = self.replay_id = None
                self.replay_position, self.replay_index = 0.0, 0
                self.replay_playing = False
                self.pipeline.reset()
                self.samples, self.times = np.empty((0, 8)), np.empty(0)
                self.predictions.clear()
                self.prediction = unavailable("回放录制已删除，请重新选择数据。")
                self._close_predictions()
            if deleted:
                self._event(f"已删除 {len(deleted)} 段录制数据；已有模型保留。")
            return result

    def model_files(self):
        with self.lock:
            return self.library.decorate_models(self._model_files())

    def _model_files(self):
        result = []
        for path in sorted(self.models.glob("*.json"), reverse=True):
            if path.name.endswith(".library.json") or path.is_symlink() or path.stat().st_size > 1_000_000:
                continue
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                model_id = path.stem
                if (self.models / f"{model_id}.joblib").is_file():
                    compatible = meta.get("version") == VERSION and meta.get("input") == input_contract()
                    result.append({**meta, "id": model_id, "compatible": compatible})
            except (ValueError, OSError):
                continue
        return result

    @staticmethod
    def _safe_path(base, identifier, suffix=""):
        if not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier:
            raise ValueError("请选择列表中的文件，不能使用外部路径。")
        path = base / f"{identifier}{suffix}"
        if path.is_symlink() or path.resolve().parent != base.resolve() or not path.exists():
            raise ValueError("文件不存在或不在本项目数据目录。")
        return path

    def _set_pipeline_model(self):
        origin = self.source_kind
        if origin == "replay" and self.replay is not None:
            origin = self.replay["metadata"]["source"]
        self.model_error = None
        try:
            self.pipeline.set_model(self.model, source_kind=origin)
        except ValueError as exc:
            # An incompatible model must not prevent raw capture or recordings.
            # It remains visible in the UI but is detached from inference.
            self.pipeline.set_model(None, source_kind=origin)
            self.model_error = str(exc)

    def start(self, options):
        self.evaluation.require_idle()
        kind = options.get("source", "demo")
        if kind not in ("demo", "device"):
            raise ValueError("请选择模拟或真实设备；录制文件请使用回放区。")
        if options.get("sample_rate", FS) != FS or options.get("channels", 8) != 8:
            raise ValueError("本流程固定为 250 Hz、8 通道。")
        config = {"sample_rate": FS, "channels": 8, "mode": "eeg",
                  "device_ip": options.get("device_ip", "")}
        # Construct first, so invalid IP/mode cannot silently stop an active session.
        source = DeviceSource(config) if kind == "device" else DemoSource(config)
        with self.lifecycle:
            self.stop()
            with self.lock:
                self.source, self.source_kind = source, kind
                self.stop_event = threading.Event()
                self.status = "connecting"
                self.message = "正在连接；不会修改采样率或电极映射。"
                self.samples, self.times = np.empty((0, 8)), np.empty(0)
                self.cursor, self.segment = 0, 0
                self.anchor = self.session_origin = self.last_sample_at = self.last_sample_time = None
                self.last_receipt_time = self.stream_stable_since = None
                self.sample_clock.reset()
                self.replay = None
                self.replay_id = None
                self.pipeline.reset()
                self._set_pipeline_model()
                self.prediction = unavailable("等待完整连续窗口。", "warming_up")
                self.predictions.clear()
                self._open_predictions()
                self.worker = threading.Thread(target=self._run, args=(source, self.stop_event),
                                               daemon=True, name="MI-acquisition")
                self.worker.start()
                self._event(f"开始 {'实机' if kind == 'device' else '模拟'}采集，250 Hz / 8 通道。")
        return {"ok": True}

    def _open_predictions(self):
        self._close_predictions()
        self.prediction_path = self.output / unique_name("predictions", ".jsonl")
        try:
            self.prediction_handle = self.prediction_path.open("x", encoding="utf-8")
        except OSError as exc:
            self.prediction_path = None
            self._event(f"分类导出文件无法创建，采集继续，结果仍可通过页面和 API 读取：{exc}", "error")

    def _close_predictions(self):
        if self.prediction_handle:
            self.prediction_handle.close()
            self.prediction_handle = None

    def stop(self):
        with self.lifecycle:
            with self.lock:
                self.robot.disarm("采集停止，机械臂发送已关闭。")
                self.stop_event.set()
                source, worker = self.source, self.worker
                self.replay_playing = False
                if source:
                    source.close()
                self._finish_recording("aborted", "采集停止，保留已录制内容。")
            if worker and worker is not threading.current_thread():
                worker.join(timeout=5)
                if worker.is_alive():
                    raise RuntimeError("采集线程尚未退出，请稍后重试。")
            with self.lock:
                self.source = None
                self.worker = None
                self.status = "idle"
                self.message = "采集已停止。"
                self.pipeline.reset()
                self.prediction = unavailable("采集已停止。")
                self._close_predictions()
        return {"ok": True}

    def close(self):
        self.closed.set()
        self.evaluation.close()
        self.stop()
        self.robot.close()
        if self.monitor is not threading.current_thread():
            self.monitor.join(timeout=2)
        for worker in self.validation_threads:
            if worker is not threading.current_thread():
                worker.join(timeout=15)

    def _run(self, source, stop):
        try:
            for batch in source.stream(stop):
                with self.lock:
                    if stop.is_set() or self.source is not source:
                        break
                    self._consume_batch(batch)
        except Exception as exc:
            with self.lock:
                if not stop.is_set():
                    self.robot.disarm("采集异常，机械臂发送已关闭。")
                    self.status = "error"
                    self.message = f"采集失败：{exc}"
                    self._event(self.message, "error")
                    self.prediction = unavailable(self.message, "invalid")
                    self._finish_recording("aborted", self.message)
        finally:
            with self.lock:
                if self.source is source:
                    self.robot.disarm("采集源已结束，机械臂发送已关闭。")
            source.close()

    def _consume_batch(self, batch):
        data = np.asarray(batch.samples, dtype=float)
        if data.ndim != 2 or data.shape[1] != 8 or not np.all(np.isfinite(data)):
            raise ValueError("收到的 EEG 不是有限的八通道数据。")
        now = time.monotonic()
        received = getattr(batch, "received_monotonic", None)
        received = now if received is None else float(received)
        if not math.isfinite(received) or received > now + .001:
            raise ValueError("数据包接收时间无效。")
        self.last_receipt_time = received
        if now - received > MAX_PROCESSING_LAG_S:
            self.robot.disarm("数据处理积压，机械臂发送已关闭。")
            self._finish_recording("aborted", "数据处理落后于收包超过 250 ms，录制已中止。请等待数据稳定后重新开始。")
            self.cursor += len(data) + max(0, int(batch.missing_samples))
            self.segment += 1
            self.stream_stable_since = None
            self.sample_clock.reset()
            self.pipeline.reset()
            self.samples, self.times = np.empty((0, 8)), np.empty(0)
            self.last_sample_at = now
            self.prediction = unavailable("收到积压数据，等待新鲜连续信号。", "invalid")
            self.message = "收到积压数据，暂停标注，等待新鲜连续信号。"
            return
        timed_out = self.last_sample_at is not None and now - self.last_sample_at > 1.25
        gap = batch.gap or timed_out
        sequence_gap = (batch.gap and self.source_kind == "device" and batch.sequence is not None
                        and len(data) == SAMPLES_PER_PACKET and batch.missing_samples > 0
                        and batch.missing_samples % SAMPLES_PER_PACKET == 0)
        stop_after_batch = None
        if gap:
            self.robot.disarm("数据间断，机械臂发送已关闭；恢复后需重新启用。")
            self.stream_stable_since = None
            if self.recorder:
                if sequence_gap:
                    stop_after_batch = self._record_packet_gap(batch, received, timed_out)
                else:
                    self._finish_recording("aborted", "录制发生断流或无法确定缺样数量的数据间断，已停止并保存。")
            self.cursor += max(0, int(batch.missing_samples))
            self.segment += 1
            self.anchor = None
            if not sequence_gap:
                self.sample_clock.reset()
            self.pipeline.reset()
            self.prediction = unavailable("数据间断，重新积累连续窗口。", "invalid")
            self._event("数据间断；没有拼接断点两侧的信号。", "warning")
        if not len(data):
            if gap and self.source_kind == "device" and self.status in ("running", "connecting"):
                self.status = "connecting"
                self.message = getattr(self.source, "message", "") or "设备数据中断，正在重新连接。"
            return
        indices = np.arange(self.cursor, self.cursor + len(data), dtype=np.int64)
        sample_times, clock_jump = self.sample_clock.map(
            indices, received, missing_samples=int(batch.missing_samples) if sequence_gap else 0)
        if clock_jump:
            self.robot.disarm("采样时钟重新对齐，机械臂发送已关闭。")
            self.stream_stable_since = None
            self._finish_recording("aborted", "采样时钟与接收时间发生突变，已停止标注录制并保留原始文件。请等待波形稳定后重新录制。")
            self.segment += 1
            self.pipeline.reset()
            self.prediction = unavailable("样本时钟重新对齐，等待新的连续窗口。", "warming_up")
            self.samples, self.times = np.empty((0, 8)), np.empty(0)
            self._event("样本时钟重新对齐接收时钟；未拼接时钟突变两侧的数据。", "warning")
        self.anchor = float(sample_times[-1]) - indices[-1] / FS
        if self.session_origin is None:
            self.session_origin = float(sample_times[0])
        self.cursor += len(data)
        self.last_sample_at = now
        self.last_sample_time = float(sample_times[-1])
        if self.stream_stable_since is None:
            self.stream_stable_since = now
        self.status = "running"
        self.message = getattr(self.source, "message", "正在采集")
        relative = sample_times - self.session_origin
        self.samples = np.concatenate((self.samples, data))[-1250:]
        self.times = np.concatenate((self.times, relative))[-1250:]
        if self.robot.mode == "live":
            # Include the newest raw tail even if fewer than 25 new samples
            # remain after the last classifier endpoint in this batch.
            saturated = bool(np.any(np.abs(data) >= 187500 * .995))
            hard_failure = len(self.samples) >= 375 and not raw_quality(self.samples[-375:])["valid"]
            if saturated or hard_failure:
                self.robot.disarm("当前原始信号未通过硬质量检查，机械臂发送已关闭。")
        if self.recorder:
            record_times = sample_times - self.record_start
            keep = (record_times >= 0) & (record_times < self.recorder.manifest["duration_s"])
            if np.any(keep):
                try:
                    self.recorder.write(data[keep], indices[keep], record_times[keep], self.segment, received)
                    self.record_last_time = float(record_times[keep][-1])
                except Exception as exc:
                    self._finish_recording("aborted", f"录制写入失败：{exc}")
                    self._event(f"录制写入失败：{exc}", "error")
        # Save the received recovery packet as the right-hand real anchor before
        # ending a too-frequent/multiple-packet-loss recording. No future packet
        # is awaited and no interpolated sample enters online classification.
        if stop_after_batch and self.recorder:
            self._finish_recording("aborted", stop_after_batch)
        self._push_predictions(data, indices, relative, self.segment)

    def _record_packet_gap(self, batch, received, timed_out):
        stats = self.record_packet_loss
        previous = stats.get("last_detected_monotonic")
        interval = None if previous is None else received - previous
        missing = int(batch.missing_samples)
        packets = missing // SAMPLES_PER_PACKET
        reason = None
        if timed_out:
            reason = "设备数据超时，已停止录制并保存。"
        elif packets != 1:
            reason = f"一次连续丢失 {packets} 包；仅允许孤立单包丢失，已停止录制并保存。"
        elif interval is not None and interval < PACKET_LOSS_INTERVAL_S:
            reason = "距上次丢包不足 15 秒；已停止录制并保存。"
        gap_id = stats["events"] + 1
        self.recorder.event("packet_loss", {
            "gap_id": gap_id, "missing_samples": missing, "missing_packets": packets,
            "missing_start_index": self.cursor, "next_sample_index": self.cursor + missing,
            "detected_s": max(0, received - self.record_start), "detected_monotonic": received,
            "allowed": reason is None, "interval_since_previous_s": interval,
            "minimum_interval_s": PACKET_LOSS_INTERVAL_S})
        stats.update(events=gap_id, last_gap_id=gap_id, last_detected_monotonic=received,
                     lost_packets=stats["lost_packets"] + packets,
                     missing_samples=stats["missing_samples"] + missing,
                     tolerated_events=stats["tolerated_events"] + int(reason is None))
        if reason is None:
            self._event("丢失 1 包（50 点），录制继续；15 秒内再次丢包将停止。结束后另存均值补齐文件。", "warning")
        return reason

    def _packet_loss_state(self, now):
        stats = self.record_packet_loss or {}
        last = stats.get("last_detected_monotonic")
        return {**{key: value for key, value in stats.items() if key != "last_detected_monotonic"},
                "next_allowed_in_s": max(0, PACKET_LOSS_INTERVAL_S - (now - last)) if last is not None else 0}

    def _push_predictions(self, data, indices, times, segment):
        items = self.pipeline.push(data, indices, times, segment=segment)
        # A received batch may contain an older good epoch followed by a hard
        # failure. Suppress the entire batch's actuation before emitting either.
        if self.robot.mode == "live" and any(not prediction_available(item) for item in items):
            self.robot.disarm("当前数据批次包含不可用窗口，机械臂发送已关闭。")
        for item in items:
            entry = copy.deepcopy(item)
            if self.model_error and entry.get("status") == "no_model":
                entry["reason"] = self.model_error
            entry["window_start"] = entry.get("window_start_time", entry.get("window_start"))
            entry["window_end"] = entry.get("window_end_time", entry.get("window_end"))
            self.result_id += 1
            entry.update({"id": self.result_id, "emitted_at": utc_now(), "model_id": self.model_id})
            quality = entry.get("quality")
            if not entry.get("reason") and isinstance(quality, dict):
                entry["reason"] = quality.get("reason", "")
            self.prediction = entry
            self.predictions.append(entry)
            context = self._robot_context()
            end_time = entry.get("window_end")
            context["window_age_s"] = (time.monotonic() - self.session_origin - end_time
                                        if self.session_origin is not None and isinstance(end_time, (int, float))
                                        else None)
            self.robot.observe(entry, context)
            if self.prediction_handle:
                try:
                    self.prediction_handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                    self.prediction_handle.flush()
                except (OSError, ValueError) as exc:
                    self._close_predictions()
                    self._event(f"分类输出文件写入失败：{exc}", "error")

    def start_recording(self, options):
        duration = number(options.get("duration_s", 210), 1, 3600, "录制时长（秒）")
        if abs(duration * FS - round(duration * FS)) > 1e-6:
            raise ValueError("录制时长必须是 1/250 秒的整数倍。")
        subject = options.get("subject", "")
        if not isinstance(subject, str) or len(subject) > 80 or any(ord(c) < 32 for c in subject):
            raise ValueError("受试者编号最多 80 字，不能包含控制字符。")
        if options.get("display_timing_method") != DISPLAY_TIMING_METHOD:
            raise ValueError("请刷新页面，使用带持续提示确认的新录制流程。")
        uncertainty = number(options.get("clock_uncertainty_s"), 0, .05, "提示时钟同步误差上界（秒）")
        with self.lock:
            if self.source_kind not in ("demo", "device") or self.status != "running":
                raise ValueError("请先连接实时设备或模拟信号并等待数据。")
            if self.last_sample_at is None or time.monotonic() - self.last_sample_at > .75:
                raise ValueError("当前没有新数据，不能开始录制。")
            if self.recorder:
                raise ValueError("录制已经在进行。")
            if self.source_kind == "device" and (self.last_receipt_time is None
                    or time.monotonic() - self.last_receipt_time > MAX_PROCESSING_LAG_S
                    or self.stream_stable_since is None or time.monotonic() - self.stream_stable_since < 2):
                raise ValueError("请等待至少 2 秒新鲜、无间断的设备数据后再录制。")
            self.robot.disarm("正在录制训练数据，机械臂发送已关闭。")
            recorder = SessionRecorder(self.recordings, self.source_kind, duration, subject=subject,
                                       clock_method=CLOCK_METHOD, display_timing_method=DISPLAY_TIMING_METHOD)
            started = time.monotonic()
            try:
                recorder.event("clock_anchor", {"start_monotonic": started,
                                                 "clock_method": CLOCK_METHOD,
                                                 "device_time_estimate": self.last_sample_time,
                                                 "next_sample_index": self.cursor})
                recorder.manifest["packet_loss_policy"] = {
                    "minimum_interval_s": PACKET_LOSS_INTERVAL_S, "max_packets_per_event": 1,
                    "clock": "received_monotonic", "interpolation": "endpoint_channel_mean",
                    "raw_file": "samples.csv", "filled_file": "samples_filled.csv"}
                _atomic_manifest(recorder.path / "manifest.json", recorder.manifest)
            except Exception:
                try:
                    recorder.close(status="error", reason="记录时钟信息失败。")
                except OSError:
                    pass
                raise
            self.recorder, self.record_start = recorder, started
            self.record_last_time = None
            self.display_last_observed = None
            self.display_last_received = None
            self.record_clock_uncertainty = uncertainty
            self.record_packet_loss = {"minimum_interval_s": PACKET_LOSS_INTERVAL_S,
                                       "events": 0, "tolerated_events": 0, "lost_packets": 0,
                                       "missing_samples": 0, "last_gap_id": None,
                                       "last_detected_monotonic": None}
            self._event(f"开始三类提示录制：{self.recorder.id}，{duration:g} 秒。")
            return {"ok": True, "recording": self._recording_state(self.record_start),
                    "server_time": time.monotonic()}

    def acknowledge_cue(self, options):
        with self.lock:
            if not self.recorder or options.get("id") != self.recorder.id:
                return {"ok": False, "ignored": True}
            index = options.get("phase_index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.recorder.plan):
                raise ValueError("提示阶段编号无效。")
            now = time.monotonic() - self.record_start
            onset = number(options.get("onset_s"), -.15, self.recorder.manifest["duration_s"] + 1, "显示时间")
            if onset > now + self.record_clock_uncertainty + 1e-9 or now - onset > .35:
                raise ValueError("提示显示时间不在当前会话的合理范围内。")
            # Browser clock mapping has RTT uncertainty. Preserve its estimate
            # and any correction; an event cannot precede start or its receipt.
            bounded = min(now, max(0.0, onset))
            if bounded != onset:
                self.recorder.event("display_clock_adjustment", {"phase_index": index,
                                    "estimated_onset_s": onset, "bounded_onset_s": bounded,
                                    "received_s": now})
            accepted = self.recorder.acknowledge(index, bounded, now)
            return {"ok": True, "accepted": accepted}

    def acknowledge_display(self, options):
        with self.lock:
            if not self.recorder or options.get("id") != self.recorder.id:
                return {"ok": False, "ignored": True}
            now = time.monotonic() - self.record_start
            observed = number(options.get("observed_s"), 0, self.recorder.manifest["duration_s"] + 1, "提示可见时间")
            uncertainty = number(options.get("clock_uncertainty_s"), 0, .05, "提示时钟误差")
            index = options.get("phase_index")
            if index is not None and (isinstance(index, bool) or not isinstance(index, int)
                                      or not 0 <= index < len(self.recorder.plan)):
                raise ValueError("提示阶段编号无效。")
            if options.get("visible") is not True or observed > now + uncertainty + 1e-9 or now - observed > .35:
                self._interrupt_display("提示可见性确认超时或时间无效。")
                return {"ok": False, "accepted": False}
            if self.display_last_observed is not None and observed <= self.display_last_observed:
                return {"ok": False, "ignored": True}
            if self.display_last_observed is not None and observed - self.display_last_observed > .35:
                self._interrupt_display("提示持续可见性确认发生间断。")
                return {"ok": False, "accepted": False}
            self.recorder.event("display_heartbeat", {"observed_s": observed, "received_s": now,
                "phase_index": index, "clock_uncertainty_s": uncertainty})
            self.display_last_observed, self.display_last_received = observed, time.monotonic()
            return {"ok": True, "accepted": True}

    def _interrupt_display(self, reason, invalid_since=None):
        if not self.recorder:
            return
        now = max(0., time.monotonic() - self.record_start)
        # Never extend a visible interval beyond the last evidence held by the
        # server, even if a stop request itself is delayed or its timestamp bad.
        proven = self.display_last_observed if self.display_last_observed is not None else 0.
        cutoff = max(0., min(now, proven, invalid_since if invalid_since is not None else proven))
        try:
            self.recorder.event("display_interruption", {"invalid_since_s": cutoff, "received_s": now,
                                                        "reason": str(reason)})
        finally:
            self._finish_recording("aborted", str(reason))

    def stop_recording(self, options=None, reason="用户停止录制，已采集内容保留。"):
        with self.lock:
            options = options or {}
            if options.get("id") and (not self.recorder or options["id"] != self.recorder.id):
                return {"ok": False, "ignored": True}
            if "visibility_lost_s" in options:
                cutoff = number(options["visibility_lost_s"], 0, 86401, "提示失效时间")
                self._interrupt_display(str(options.get("reason", reason))[:300], cutoff)
            else:
                self._interrupt_display(reason)
        return {"ok": True}

    def _finish_recording(self, status, reason):
        if self.recorder is None:
            return
        self.last_recording = self._recording_state(time.monotonic())
        recorder, self.recorder = self.recorder, None
        self.last_recording.update({"active": False, "status": status, "reason": reason})
        closed_ok = True
        try:
            recorder.close(status=status, reason=reason)
        except OSError as exc:
            closed_ok = False
            self.last_recording.update({"status": "error", "reason": str(exc)})
            self._event(f"录制关闭失败：{exc}", "error")
        if self.source and hasattr(self.source, "cue"):
            self.source.cue = None
        self._event(reason, "info" if status == "complete" else "warning")
        if closed_ok:
            self.last_recording["validation"] = {"status": "checking", "id": recorder.id,
                                                "gap_fill": {"status": "pending" if (self.record_packet_loss or {}).get("events") else "not_needed"},
                                                "message": "文件已保存，正在检查时序与可训练窗口。"}
            worker = threading.Thread(target=self._validate_recording, args=(recorder.path, recorder.id),
                                      daemon=True, name="MI-recording-check")
            self.validation_threads = [thread for thread in self.validation_threads if thread.is_alive()]
            self.validation_threads.append(worker)
            worker.start()
        else:
            self.last_recording["validation"] = {"status": "error", "id": recorder.id,
                                                "message": "文件关闭失败，不能确认录制有效。"}

    def _validate_recording(self, path, identifier):
        from .mi_gap_export import export_gap_filled
        gap_fill = None
        try:
            # Export after close, outside the acquisition lock. The source CSV
            # Online inputs and the default validation remain real samples only.
            try:
                gap_fill = export_gap_filled(path)
            except Exception as exc:
                gap_fill = {"status": "error", "output_file": None, "errors": [str(exc)]}
            session = load_session(path)
            try:
                dataset = build_dataset([session])
                diagnostics = dataset["metadata"]["dataset_diagnostics"]
            except DatasetBuildError as exc:
                diagnostics = exc.diagnostics
            report = {"status": "checked", "id": identifier, "sample_count": len(session["samples"]),
                      "gap_fill": gap_fill,
                      "display_audit": session["metadata"].get("display_audit", {}),
                      "timestamp_audit": session["metadata"].get("timestamp_audit", {}),
                      "diagnostics": diagnostics,
                      "message": ("已保存并检查：存在可训练试次，详见各类保留数量。" if diagnostics.get("accepted_trials", 0)
                                  else "已保存，但没有可训练试次；请查看时序与质量排除原因。")}
            _atomic_manifest(path / "validation.json", report)
        except Exception as exc:
            report = {"status": "error", "id": identifier, "gap_fill": gap_fill,
                      "message": f"录制检查失败：{exc}"}
        with self.lock:
            if self.last_recording.get("id") == identifier:
                self.last_recording["validation"] = report

    def _recording_state(self, now):
        if self.recorder is None:
            return copy.deepcopy(self.last_recording)
        duration = self.recorder.manifest["duration_s"]
        elapsed = min(duration, max(0, now - self.record_start))
        phase = next((p for p in self.recorder.plan if p["start_s"] <= elapsed < p["end_s"]), None)
        return {"active": True, "id": self.recorder.id, "duration_s": duration,
                "packet_loss": self._packet_loss_state(now),
                "elapsed_s": elapsed, "remaining_s": max(0, duration - elapsed),
                "start_time": self.record_start, "plan": self.recorder.plan,
                "phase": phase["phase"] if phase else "interval",
                "label": phase["label"] if phase else "interval",
                "trial": phase["trial"] if phase else 0}

    def load_replay(self, identifier):
        self.evaluation.require_idle()
        path = self._safe_path(self.recordings, identifier)
        session = load_session(path)
        if not len(session["samples"]):
            raise ValueError("这段录制没有可回放的 EEG 样本。")
        with self.lifecycle:
            self.stop()
            with self.lock:
                self.replay, self.replay_id = session, identifier
                self.source_kind = "replay"
                self.replay_position, self.replay_index, self.replay_speed = 0.0, 0, 1.0
                self.replay_playing = False
                self.status = "idle"
                self.message = "回放已加载；标注来自录制文件。"
                self.pipeline.reset()
                self._set_pipeline_model()
                self.prediction = unavailable("点击播放；跳转后重新积累滤波与分类窗口。", "warming_up")
                self.predictions.clear()
                self._open_predictions()
        return {"ok": True}

    def replay_control(self, options):
        action = options.get("action")
        with self.lock:
            self.evaluation.require_idle()
            if self.replay is None or self.source_kind != "replay":
                raise ValueError("请先加载录制文件。")
            if "speed" in options:
                self.replay_speed = number(options["speed"], .25, 4, "回放倍速")
            if action == "seek":
                position = number(options.get("position_s"), 0, self._replay_duration(), "回放位置")
                self.replay_position = position
                self.replay_index = int(np.searchsorted(self.replay["times"], position, side="left"))
                self.pipeline.reset()
                self.prediction = unavailable("已跳转，重新积累连续窗口。", "warming_up")
            elif action == "play":
                if self.replay_position >= self._replay_duration():
                    self.replay_position, self.replay_index = 0.0, 0
                    self.pipeline.reset()
                self.replay_playing = True
                self.status = "running"
            elif action == "pause":
                self.replay_playing = False
                self.status = "idle"
            else:
                raise ValueError("未知回放操作。")
            self.replay_clock = time.monotonic()
        return {"ok": True}

    def _replay_duration(self):
        if self.replay is None:
            return 0.0
        return min(float(self.replay["metadata"]["duration_s"]), float(self.replay["times"][-1] + 1 / FS))

    def _tick_replay(self, now):
        if not self.replay_playing or self.replay is None:
            return
        self.replay_position = min(self._replay_duration(), self.replay_position + (now - self.replay_clock) * self.replay_speed)
        self.replay_clock = now
        end = int(np.searchsorted(self.replay["times"], self.replay_position, side="right"))
        while self.replay_index < end:
            start = self.replay_index
            stop = min(end, start + 50)
            segments = self.replay["segments"][start:stop]
            changes = np.flatnonzero(segments != segments[0])
            if len(changes):
                stop = start + int(changes[0])
            self._push_predictions(self.replay["samples"][start:stop], self.replay["indices"][start:stop],
                                   self.replay["times"][start:stop], int(segments[0]))
            self.replay_index = stop
        if self.replay_position >= self._replay_duration():
            self.replay_playing = False
            self.status = "idle"
            self.message = "回放结束。"

    def _monitor(self):
        while not self.closed.wait(.02):
            try:
                with self.lock:
                    now = time.monotonic()
                    self.robot.tick(self._robot_context(now))
                    self._tick_replay(now)
                    if self.recorder:
                        state = self._recording_state(now)
                        if self.source and hasattr(self.source, "cue"):
                            self.source.cue = {"left_mi": "a", "right_mi": "b"}.get(state["label"], "rest")
                        last_display = self.display_last_received if self.display_last_received is not None else self.record_start
                        if now - last_display > DISPLAY_LEASE_S:
                            self._interrupt_display("提示可见性心跳中断，已停止录制并排除未确认阶段。")
                        elif self.last_sample_at is None or now - self.last_sample_at > 1.25:
                            self._finish_recording("aborted", "数据超时，录制已中止并保留文件。")
                        elif now - self.record_start >= self.recorder.manifest["duration_s"] + .3:
                            duration = self.recorder.manifest["duration_s"]
                            complete = self.record_last_time is not None and self.record_last_time >= duration - .03
                            self._finish_recording("complete" if complete else "aborted",
                                                   "录制完成。" if complete else "录制末尾缺少数据，保留文件但标记为中止。")
            except Exception as exc:
                with self.lock:
                    self._event(f"会话处理失败：{exc}", "error")
                    self.replay_playing = False
                    self.status = "error"
                    self.robot.disarm("会话处理失败，机械臂发送已关闭。")
                    self.message = str(exc)
                    self._finish_recording("aborted", str(exc))

    def start_evaluation(self, options):
        identifiers = options.get("ids")
        model_id = options.get("model_id")
        if (set(options) - {"ids", "model_id"} or not isinstance(identifiers, list)
                or not 1 <= len(identifiers) <= 50
                or any(not isinstance(item, str) for item in identifiers)):
            raise ValueError("请选择一个模型和 1～50 段录制数据。")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("不要重复选择同一段录制。")
        with self.lock:
            self._require_recording_files_idle()
            if self.replay_playing:
                raise RuntimeError("请先暂停回放再测试模型。")
            models = {item["id"]: item for item in self.model_files()}
            recordings = {item["id"]: item for item in self.sessions()}
            model = models.get(model_id)
            if not model or not model.get("compatible"):
                raise ValueError("请选择兼容的八通道三类模型。")
            if any(item not in recordings for item in identifiers):
                raise ValueError("录制列表已变化，请刷新后重新选择。")
            selected = [recordings[item] for item in identifiers]
            if any(item.get("source") != model.get("source") for item in selected):
                raise ValueError("模型与测试数据的来源须一致，演示和实机不能混用。")
            paths = [self._safe_path(self.recordings, item) for item in identifiers]
            model_path = self._safe_path(self.models, model_id, ".joblib")
            self.robot.disarm("离线模型测试中，机械臂发送已关闭。")
            return self.evaluation.start(model_path, paths,
                                         {key: model.get(key) for key in ("id", "display_name", "feature", "classifier", "source")},
                                         [{key: item.get(key) for key in ("id", "display_name", "subject", "source", "duration_s")}
                                          for item in selected])

    def train(self, options):
        self.evaluation.require_idle()
        identifiers = options.get("ids", [])
        if not isinstance(identifiers, list) or not identifiers or len(identifiers) > 50:
            raise ValueError("请选择 1～50 段带标注录制。")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("不要重复选择同一段录制。")
        paths = [self._safe_path(self.recordings, i) for i in identifiers]
        feature, classifier = options.get("feature", "bandpower"), options.get("classifier", "lda")
        use_filled = options.get("use_filled", False)
        if type(use_filled) is not bool:
            raise ValueError("补齐数据训练选项必须为开或关。")
        if feature not in ("bandpower", "csp", "combined") or classifier not in ("lda", "svm"):
            raise ValueError("请选择频带功率/CSP/组合特征与 LDA/SVM。")
        with self.lock:
            if self.training["status"] == "running":
                raise RuntimeError("训练正在进行，请等待完成。")
            if self.recorder or self.status in ("running", "connecting"):
                raise ValueError("请先停止采集或回放，再训练模型。")
            if any(worker.is_alive() for worker in self.validation_threads):
                raise RuntimeError("录制文件正在检查或补齐，请等待完成后再训练。")
            self.training = {"status": "running", "use_filled": use_filled,
                             "message": "读取标注、按完整试次分组并训练；不会随机拆散重叠窗口。"}
            self.training_thread = threading.Thread(target=self._train_worker, args=(paths, feature, classifier, use_filled),
                                                   daemon=True, name="MI-training")
            self.training_thread.start()
        return {"ok": True}

    def _train_worker(self, paths, feature, classifier, use_filled=False):
        try:
            sessions = [load_session(p, use_filled=use_filled) for p in paths]
            model = train_model(sessions, feature=feature, classifier=classifier)
            model_id = unique_name(f"mi_{feature}_{classifier}", "")
            save_model(model, self.models / f"{model_id}.joblib")
            meta = {"feature": feature, "classifier": classifier, "created": utc_now(),
                    "version": model["version"], "input": model["input"],
                    "diagnostics": model["metadata"].get("dataset_diagnostics", {}),
                    "metrics": model.get("metrics", {}), "source": sessions[0]["metadata"]["source"],
                    "recordings": [p.name for p in paths], "use_filled": use_filled}
            with (self.models / f"{model_id}.json").open("x", encoding="utf-8") as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2, allow_nan=False)
            with self.lock:
                self.training = {"status": "complete", "message": "训练完成。部署模型未使用留出测试集重新拟合。",
                                 "model_id": model_id, "metrics": model.get("metrics", {}),
                                 "diagnostics": meta["diagnostics"], "use_filled": use_filled}
                self._event(f"训练完成并保存：{model_id}")
        except Exception as exc:
            with self.lock:
                self.training = {"status": "error", "message": str(exc),
                                 "diagnostics": getattr(exc, "diagnostics", {}), "use_filled": use_filled}
                self._event(f"训练失败：{exc}", "error")

    def load_model(self, identifier):
        self.evaluation.require_idle()
        listing = next((m for m in self.model_files() if m["id"] == identifier), None)
        if listing is None:
            raise ValueError("请选择本流程保存的模型。旧两类模型不兼容。")
        path = self._safe_path(self.models, identifier, ".joblib")
        model = load_model(path)
        with self.lock:
            if self.recorder:
                raise ValueError("请在录制结束后更换模型。")
            self.robot.disarm("模型已更换，机械臂发送已关闭；请核对模型后重新启用。")
            self.model, self.model_id = model, identifier
            self.model_display_name = listing["display_name"]
            self.pipeline.reset()
            self._set_pipeline_model()
            self.predictions.clear()
            self.prediction = unavailable("模型已加载，等待完整连续窗口。", "warming_up")
            self._event(f"加载八通道模型：{identifier}")
        return {"ok": True}

    def _robot_context(self, now=None):
        now = time.monotonic() if now is None else now
        return {"source": self.source_kind, "status": self.status,
                "model_compatible": self.model is not None and self.model_error is None,
                "model_id": self.model_id, "recording": self.recorder is not None,
                "data_age_s": None if self.last_receipt_time is None else max(0., now - self.last_receipt_time),
                "device_ip": getattr(self.source, "device_ip", None),
                "stream_id": self.session_origin, "segment": self.segment}

    def robot_configure(self, options):
        with self.lock:
            self.evaluation.require_idle()
            return self.robot.configure(options, self._robot_context())

    def robot_heartbeat(self, options):
        with self.lock:
            self.robot.tick(self._robot_context())
            return self.robot.touch(options.get("client_id"))

    def robot_test(self):
        with self.lock:
            self.evaluation.require_idle()
            return self.robot.test_link(self._robot_context())

    def robot_disarm(self, options=None):
        with self.lock:
            detail = options.get("reason") if isinstance(options, dict) else None
            reason = ("页面停发：" + detail[:240] if isinstance(detail, str) and detail.strip()
                      else "用户停止机械臂发送；已执行的动作不会由此撤销。")
            self.robot.disarm(reason)
            return self.robot.snapshot()

    @staticmethod
    def _annotation_runs(times, labels, trials):
        if not len(times):
            return []
        breaks = np.flatnonzero((labels[1:] != labels[:-1]) | (trials[1:] != trials[:-1]) |
                                (np.diff(times) > 1.5 / FS)) + 1
        edges = [0, *map(int, breaks), len(times)]
        return [{"start": float(times[a]), "end": float(times[b - 1] + 1 / FS),
                 "label": str(labels[a]), "trial": int(trials[a])} for a, b in zip(edges[:-1], edges[1:])]

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            recording = self._recording_state(now)
            playback = {"loaded": self.replay is not None, "id": self.replay_id, "playing": self.replay_playing,
                        "position_s": self.replay_position, "duration_s": self._replay_duration(),
                        "speed": self.replay_speed, "label": "interval", "trial": 0}
            annotations = []
            times, data = self.times, self.samples
            if self.source_kind == "replay" and self.replay is not None:
                end = int(np.searchsorted(self.replay["times"], self.replay_position, side="right"))
                start = int(np.searchsorted(self.replay["times"], max(0, self.replay_position - 5)))
                times, data = self.replay["times"][start:end], self.replay["samples"][start:end]
                annotations = self._annotation_runs(times, self.replay["labels"][start:end], self.replay["trials"][start:end])
                if end:
                    playback.update({"label": str(self.replay["labels"][end - 1]), "trial": int(self.replay["trials"][end - 1]),
                                     "timing_valid": bool(self.replay["timing_valid"][end - 1])})
            elif self.recorder and self.session_origin is not None:
                offset = self.record_start - self.session_origin
                annotations = [{"start": p["start_s"] + offset, "end": p["end_s"] + offset,
                                "label": p["label"], "trial": p["trial"]} for p in self.recorder.plan
                               if len(times) and p["end_s"] + offset >= times[0] and p["start_s"] + offset <= times[-1]]
            skip = max(1, len(times) // 1000)
            visible = data[::skip]
            traces = (visible - np.median(visible, axis=0)).T.tolist() if len(visible) else [[] for _ in CHANNELS]
            pred = copy.deepcopy(self.prediction)
            if self.source_kind != "replay" and self.last_sample_at is not None and now - self.last_sample_at > 1.25:
                pred = unavailable("无新数据，旧分类已过期。", "invalid")
            telemetry = dict(getattr(self.source, "stats", {})) if self.source else {}
            telemetry.update({"received_samples": self.cursor, "age_s": None if self.last_receipt_time is None else now - self.last_receipt_time,
                              "stable_s": 0. if self.stream_stable_since is None else now - self.stream_stable_since})
            message = getattr(self.source, "message", self.message) if self.status == "connecting" else self.message
            return {"server_time": now, "status": self.status, "source": self.source_kind, "message": message,
                    "device_ip": getattr(self.source, "device_ip", None) if self.source_kind == "device" else None,
                    "telemetry": telemetry, "channels": list(CHANNELS), "sample_rate": FS,
                    "signal": {"times": times[::skip].tolist(), "traces": traces, "channel_names": list(CHANNELS),
                               "annotation_segments": annotations}, "recording": recording, "playback": playback,
                    "prediction": pred, "predictions": list(self.predictions)[-80:],
                    "predictions_file": str(self.prediction_path) if self.prediction_path else None,
                    "model": {"available": self.model is not None, "id": self.model_id,
                              "display_name": self.model_display_name,
                              "compatible": self.model_error is None, "reason": self.model_error,
                              "metrics": self.model.get("metrics", {}) if self.model else {},
                              "diagnostics": self.model.get("metadata", {}).get("dataset_diagnostics", {}) if self.model else {},
                              "feature": self.model.get("metadata", {}).get("feature") if self.model else None,
                              "classifier": self.model.get("metadata", {}).get("classifier") if self.model else None},
                    "training": copy.deepcopy(self.training), "evaluation": self.evaluation.snapshot(),
                    "robot": self.robot.snapshot(),
                    "events": list(self.events)[-20:]}

    def prediction_records(self, after=0):
        with self.lock:
            return {"records": [p for p in self.predictions if p["id"] > after],
                    "first_available_id": self.predictions[0]["id"] if self.predictions else None,
                    "last_id": self.result_id, "file": str(self.prediction_path) if self.prediction_path else None}
