"""Acquisition lifecycle, quality-gated feedback and trial-based calibration."""
from __future__ import annotations

import copy
import ipaddress
import math
import random
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from . import classifier
from .signal import FEATURE_VERSION, analyze
from .sources import DemoSource, DeviceSource, ReplaySource, read_csv
from .storage import Recorder, Store, utc_now

RATES = (250, 500, 1000, 2000, 4000, 8000)
COUNTS = (2, 4, 6, 8)
WINDOW_SECONDS = 2.0
STALE_SECONDS = 1.25


def integer(value, lo: int, hi: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
        raise ValueError(f"{name}必须为整数。")
    number = int(value)
    if not lo <= number <= hi:
        raise ValueError(f"{name}应在 {lo}～{hi} 之间。")
    return number


def clean_text(value, name: str, length: int = 40) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > length:
        raise ValueError(f"{name}不能为空，且不能超过 {length} 字。")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{name}不能含换行或控制字符。")
    return value.strip()


class Engine:
    def __init__(self, root: Path):
        self.store = Store(root)
        self.lock = threading.RLock()
        self.lifecycle = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = None
        self.source = None
        self.source_kind = "demo"
        self.status = "idle"
        self.message = "选择模拟、已有文件回放或真实设备，然后开始采集。"
        self.config = {"sample_rate": 250, "channels": 8,
                       "channel_names": [f"CH{i+1}" for i in range(8)],
                       "selected_channels": list(range(8)), "mode": "eeg"}
        self.events = deque(maxlen=60)
        self.buffer = np.empty((0, 8))
        self.cursor = 0
        self.last_sample_at = None
        self.last_analysis_sample = -1
        self.analysis = self._empty_analysis("尚未采集")
        self.prediction = self._no_prediction("请先完成个人两类任务校准。")
        self.model = None
        self.feedback_target = "alpha"
        self.feedback_threshold = 0.35
        self.recorder = None
        self.last_recording = {"active": False, "path": "", "samples": 0}
        self.pending_markers = []
        self.calibration = self._empty_calibration()
        self.calibration_plan = []
        self.examples = []
        self.calibration_context = None
        self.calibration_owns_recording = False
        self._trial_windows = {}
        self._last_calibration_phase = None
        self._event("info", "服务已就绪。原始 InBrain 文件只读，所有新文件保存在本项目 data 下。")

    @staticmethod
    def _empty_analysis(reason):
        return {"quality": {"valid": False, "summary": reason, "channels": []},
                "signal": {"freqs": [], "psd": [], "bands": {}, "dominant_band": "—"}, "features": None}

    @staticmethod
    def _no_prediction(reason):
        return {"label": None, "state": "unavailable", "score": None, "strength": 0.0, "reason": reason}

    @staticmethod
    def _empty_calibration():
        return {"phase": "idle", "label": "", "trial": 0, "total_trials": 0,
                "remaining_s": 0, "accepted_windows": 0, "rejected_windows": 0,
                "progress": 0, "message": "设置两类任务，按提示采集个人校准试次。"}

    def _event(self, level, message):
        self.events.append({"time": utc_now(), "level": level, "message": message})
        if self.recorder:
            try:
                self.recorder.event("message", {"level": level, "message": message, "sample_index": self.cursor})
            except OSError as exc:
                self._stop_recording()
                self.events.append({"time": utc_now(), "level": "error", "message": f"录制事件写入失败，已停止录制：{exc}"})
                if self._calibrating():
                    self._abort_calibration("录制写入失败，校准已中止。")

    def context(self):
        return {"source_kind": self.source_kind, "sample_rate": self.config["sample_rate"],
                "channels": self.config["channels"], "channel_names": self.config["channel_names"],
                "selected_channels": self.config["selected_channels"],
                "feature_version": FEATURE_VERSION, "window_seconds": WINDOW_SECONDS}

    def _compatible(self):
        if not self.model:
            return False, "尚未训练或载入个人模型。"
        if self.config["mode"] != "eeg":
            return False, "阻抗模式不执行分类。"
        if self.model.get("context") != self.context():
            return False, "模型的来源、采样率或通道配置不匹配；模拟模型不能用于真实设备。"
        return True, "模型配置与当前采集一致。"

    def _validate_config(self, options):
        if not isinstance(options, dict):
            raise ValueError("采集参数应为 JSON 对象。")
        kind = options.get("source", "demo")
        if kind not in ("demo", "device", "replay"):
            raise ValueError("未知数据来源。")
        fs = integer(options.get("sample_rate", 250), 250, 8000, "采样率")
        n = integer(options.get("channels", 8), 2, 8, "通道数")
        if fs not in RATES or n not in COUNTS:
            raise ValueError("请选择文档支持的采样率和 2/4/6/8 通道。")
        mode = options.get("mode", "eeg")
        if mode not in ("eeg", "impedance"):
            raise ValueError("未知采集模式。")
        replay_path = None
        metadata = None
        if kind == "replay":
            replay_path = self.store.resolve_replay(options.get("replay_id", ""))
            metadata = read_csv(replay_path)
            if any(mode != "eeg" for mode in metadata.get("modes", [])):
                raise ValueError("此文件包含阻抗或未知模式；不能按普通 EEG 回放分析。请另录纯 EEG 会话。")
            inferred = float(metadata["sample_rate"])
            fs = min(RATES, key=lambda rate: abs(rate - inferred))
            if not math.isclose(inferred, fs, rel_tol=1e-4, abs_tol=1e-6):
                raise ValueError("回放采样率与支持值不符，不能直接截断或重解释采样率。")
            n = int(metadata["samples"].shape[1])
            if fs not in RATES or n not in COUNTS:
                raise ValueError("回放文件的采样率或通道数不受支持。")
            # A CSV can contain both modes; it is never used as calibration ground truth.
            mode = "eeg"
        names = options.get("channel_names")
        if names is None or (kind == "replay" and len(names) != n):
            names = metadata.get("channel_names") if metadata else [f"CH{i+1}" for i in range(n)]
        if not isinstance(names, list) or len(names) != n:
            raise ValueError("通道名称数量应与通道数一致。")
        names = [clean_text(name, "通道名称", 24) for name in names]
        if len(set(names)) != len(names):
            raise ValueError("通道名称不能重复。")
        selected = options.get("selected_channels", list(range(n)))
        if not isinstance(selected, list) or not selected:
            raise ValueError("至少选择一个用于反馈与识别的通道。")
        selected = [integer(i, 0, n - 1, "通道编号") for i in selected]
        if len(selected) != len(set(selected)):
            raise ValueError("识别通道不能重复。")
        config = {"sample_rate": fs, "channels": n, "channel_names": names,
                  "selected_channels": selected, "mode": mode}
        if kind == "device":
            try:
                address = ipaddress.ip_address(options.get("device_ip", ""))
            except ValueError as exc:
                raise ValueError("请输入设备的 IPv4 地址，或先搜索设备。") from exc
            if address.version != 4 or address.is_unspecified or address.is_multicast or not (address.is_private or address.is_link_local):
                raise ValueError("设备地址应为本地网络的 IPv4 地址。")
            config["device_ip"] = str(address)
        if kind == "demo":
            state = options.get("demo_state", "rest")
            if state not in ("rest", "a", "b", "saturated", "flat", "noisy"):
                raise ValueError("未知模拟状态。")
            config["demo_state"] = state
        return kind, config, replay_path

    def start(self, options):
        with self.lifecycle:
            kind, config, path = self._validate_config(options)
            self.stop()
            source = {"demo": DemoSource, "device": DeviceSource}.get(kind)
            instance = source(config) if source else ReplaySource(path)
            if kind == "demo":
                instance.state = config["demo_state"]
            with self.lock:
                self.config, self.source_kind, self.source = config, kind, instance
                self.status = "connecting"
                self.message = "连接与初始化中…" if kind == "device" else "准备数据源…"
                self.stop_event = threading.Event()
                self.buffer = np.empty((0, config["channels"]))
                self.cursor = 0
                self.last_sample_at = None
                self.last_analysis_sample = -1
                self.analysis = self._empty_analysis("等待完整的 2 秒连续数据窗。")
                self.prediction = self._no_prediction("等待新数据。")
                self.calibration = self._empty_calibration()
                self.calibration_plan, self.examples = [], []
                self.pending_markers = []
                self.worker = threading.Thread(target=self._run, args=(instance, self.stop_event),
                                               name="EEG-acquisition", daemon=True)
                self.worker.start()
                self._event("info", f"开始 {kind} 数据源，{config['sample_rate']} SPS，{config['channels']} 通道。")
            return {"ok": True}

    def stop(self):
        with self.lifecycle:
            with self.lock:
                worker, source = self.worker, self.source
                self.stop_event.set()
                self.status = "stopped"
                self.prediction = self._no_prediction("采集已停止。")
                self.analysis = self._empty_analysis("采集已停止。")
                if self._calibrating():
                    self._abort_calibration("采集停止，校准已中止。")
            if source:
                source.close()
            if worker and worker is not threading.current_thread():
                worker.join(timeout=5)
                if worker.is_alive():
                    raise RuntimeError("旧采集线程未退出，暂不启动新会话；请稍后重试。")
            with self.lock:
                self.worker = None
                self._stop_recording()
                self.message = "采集已停止。"
            return {"ok": True}

    def _run(self, source, stop):
        failed = False
        try:
            for batch in source.stream(stop):
                if stop.is_set():
                    break
                with self.lock:
                    self._consume(batch)
        except Exception as exc:
            failed = True
            with self.lock:
                self.status = "error"
                self.message = str(exc)
                self._event("error", str(exc))
        finally:
            source.close()
            with self.lock:
                if not failed:
                    self.status = "stopped"
                    self.message = "回放结束。" if self.source_kind == "replay" and not stop.is_set() else "采集已停止。"
                self.analysis = self._empty_analysis(self.message)
                self.prediction = self._no_prediction(self.message)
                if self._calibrating():
                    self._abort_calibration("数据源结束或异常，校准已中止。")
                self._stop_recording()

    def _consume(self, batch):
        n, fs = self.config["channels"], self.config["sample_rate"]
        data = np.asarray(batch.samples, dtype=float)
        if data.ndim != 2 or data.shape[1] != n:
            raise ValueError("数据源通道数与配置不一致。")
        if self.source_kind == "device" and getattr(self.source, "mode", None) != self.config["mode"]:
            self._invalidate("等待采集线程应用模式切换，过渡数据不进入分析和录制。")
            return
        if self.last_sample_at is not None and time.monotonic() - self.last_sample_at > STALE_SECONDS:
            self._invalidate("数据超时后恢复，重新积累连续窗。")
            if self._calibrating():
                self._abort_calibration("校准期间发生数据超时，请重新校准。")
        if batch.gap:
            self.cursor += max(0, int(batch.missing_samples))
            self._invalidate("数据流不连续，重新积累 2 秒连续窗。")
            self._event("warning", f"数据间断：已知缺失 {batch.missing_samples} 个采样时刻。")
            if self._calibrating():
                self._abort_calibration("校准期间发生数据间断，请重新校准。")
        if not len(data):
            return
        self.last_sample_at = time.monotonic()
        self.status = "running"
        self.message = getattr(self.source, "message", "") or "正在采集。"
        if self.recorder:
            marker = " | ".join(self.pending_markers)
            try:
                self.recorder.write(batch, self.cursor, self.config["mode"], self._annotation, marker)
            except OSError as exc:
                self._stop_recording()
                self._event("error", f"录制写入失败，数据采集继续但文件已关闭：{exc}")
                if self._calibrating():
                    self._abort_calibration("录制写入失败，校准已中止。")
            self.pending_markers.clear()
        self.buffer = np.concatenate((self.buffer, data))[-4 * fs:]
        self.cursor += len(data)
        # A new 200 ms flat tail must not continue rewarding the valid older window.
        tail_length = max(2, round(.2 * fs))
        tail = self.buffer[-tail_length:, self.config["selected_channels"]]
        if (self.config["mode"] == "eeg" and len(tail) == tail_length and np.isfinite(tail).all()
                and not np.any(np.abs(tail) >= 187500 * .995)
                and np.any(np.ptp(tail, axis=0) < .5)):
            if self._calibrating():
                self._abort_calibration("所选通道出现持续平线，校准已中止。")
            self._invalidate("所选通道最近 200 ms 持续平线，暂停反馈与识别。")
            return
        if self._calibrating():
            self._advance_calibration()
        # New clipping/nonfinite input invalidates feedback immediately, before the FFT cadence.
        selected = data[:, self.config["selected_channels"]]
        if not np.isfinite(selected).all() or np.any(np.abs(selected) >= 187500 * .995):
            self.analysis = self._empty_analysis("检测到饱和或非有限数据，暂停反馈与识别。")
            self.prediction = self._no_prediction(self.analysis["quality"]["summary"])
        if self.cursor - self.last_analysis_sample >= max(1, fs // 4) or self.last_analysis_sample < 0:
            self.last_analysis_sample = self.cursor
            if len(self.buffer) >= int(WINDOW_SECONDS * fs):
                self.analysis = analyze(self.buffer[-int(WINDOW_SECONDS * fs):], fs,
                                        self.config["channel_names"], self.config["selected_channels"], self.config["mode"])
                self._predict()
            else:
                self.analysis = self._empty_analysis(f"连续数据 {len(self.buffer)/fs:.1f}/2.0 秒，积累中。")
                self.prediction = self._no_prediction("等待完整数据窗。")

    def _invalidate(self, reason):
        if self.recorder:
            self.recorder.pending_discontinuity = True
        self.buffer = np.empty((0, self.config["channels"]))
        self.last_analysis_sample = -1
        self.analysis = self._empty_analysis(reason)
        self.prediction = self._no_prediction(reason)

    def _predict(self):
        if not self.analysis["quality"]["valid"] or self.analysis.get("features") is None:
            self.prediction = self._no_prediction(self.analysis["quality"]["summary"])
            return
        if self._calibrating():
            self.prediction = self._no_prediction("校准采集中，暂停在线分类。")
            return
        ok, reason = self._compatible()
        self.prediction = classifier.predict(self.model, self.analysis["features"]) if ok else self._no_prediction(reason)

    def set_demo(self, state):
        if state not in ("rest", "a", "b", "saturated", "flat", "noisy"):
            raise ValueError("未知模拟状态。")
        with self.lock:
            if self.source_kind != "demo" or not self.source:
                raise ValueError("当前不是模拟数据源。")
            if self._calibrating():
                raise ValueError("校准期间由试次控制模拟状态，请先中止校准。")
            self.source.state = state
            self.config["demo_state"] = state
            self._invalidate("模拟状态已改变，重新积累数据。")
            return {"ok": True}

    def set_mode(self, mode):
        if mode not in ("eeg", "impedance"):
            raise ValueError("未知采集模式。")
        with self.lock:
            if self.status != "running" or self.source_kind == "replay":
                raise ValueError("请先启动模拟或设备采集；回放不支持硬件模式切换。")
            if self._calibrating():
                self._abort_calibration("切换模式，校准已中止。")
            if self.source_kind == "device":
                self.source.set_mode(mode)
            self.config["mode"] = mode
            self._invalidate("模式切换，等待完整新数据窗；设备没有独立模式确认包。")
            self._event("info", f"模式请求：{mode}。")
            return {"ok": True}

    def set_feedback(self, target, threshold):
        if target not in ("alpha", "beta", "class_b"):
            raise ValueError("未知反馈目标。")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("反馈阈值应在 0 和 1 之间。")
        with self.lock:
            self.feedback_target, self.feedback_threshold = target, float(threshold)
        return {"ok": True}

    def start_recording(self):
        with self.lock:
            if self.status != "running":
                raise ValueError("请先启动采集并等待数据。")
            if self.recorder:
                raise ValueError("当前已在录制。")
            self.recorder = Recorder(self.store, copy.deepcopy(self.config), self.source_kind, self.cursor)
            self._event("info", f"录制开始：{self.recorder.name}")
            return {"ok": True, "name": self.recorder.name}

    def _stop_recording(self):
        if self.recorder:
            recorder = self.recorder
            self.recorder = None
            self.last_recording = {"active": False, "path": recorder.name, "samples": recorder.samples}
            try:
                if self.pending_markers:
                    recorder.event("unattached_markers", {"labels": list(self.pending_markers)})
                recorder.close()
            except OSError as exc:
                # Detach before cleanup so failed logging can never recurse into a broken writer.
                try:
                    recorder.close()
                except OSError:
                    pass
                self.events.append({"time": utc_now(), "level": "error", "message": f"录制文件关闭时发生 I/O 错误：{exc}"})
        self.pending_markers.clear()

    def stop_recording(self):
        with self.lock:
            if self._calibrating():
                raise ValueError("校准正在自动录制，请先中止校准。")
            self._stop_recording()
        return {"ok": True}

    def marker(self, label):
        label = clean_text(label, "事件标记", 80)
        with self.lock:
            if not self.recorder:
                raise ValueError("请先开始录制。")
            self.pending_markers.append(label)
            self.recorder.event("software_marker_requested", {"label": label, "next_sample_index": self.cursor})
            self._event("info", f"软件标记：{label}（附到下一收到的样本）。")
        return {"ok": True}

    def _calibrating(self):
        return self.calibration["phase"] in ("rest", "cue", "active")

    def start_calibration(self, options):
        labels = options.get("labels", ["左手想象", "右手想象"])
        if not isinstance(labels, list) or len(labels) != 2:
            raise ValueError("请设置两个不同的类别名称。")
        labels = [clean_text(label, "类别名称") for label in labels]
        if labels[0] == labels[1]:
            raise ValueError("两类名称不能相同。")
        trials = integer(options.get("trials_per_class", 8), 4, 20, "每类试次数")
        active = integer(options.get("trial_seconds", 6), 4, 20, "任务时长")
        rest = integer(options.get("rest_seconds", 3), 1, 20, "休息时长")
        cue = integer(options.get("cue_seconds", 1), 1, 5, "准备时长")
        with self.lock:
            if self.status != "running" or self.config["mode"] != "eeg":
                raise ValueError("请先启动普通 EEG 采集。")
            if self.source_kind == "replay":
                raise ValueError("回放不能根据当前提示生成真实任务标签；请用实时设备重新校准。")
            if self._calibrating():
                raise ValueError("校准已在进行。")
            if self.source_kind == "demo" and getattr(self.source, "state", "rest") in ("saturated", "flat", "noisy"):
                raise ValueError("请先选择正常模拟状态。")
            if not self.analysis["quality"]["valid"]:
                raise ValueError("当前信号质量不合格，请等到所选通道质量通过后校准。")
            fs = self.config["sample_rate"]
            self.calibration = self._empty_calibration()
            self.calibration.update({"phase": "rest", "total_trials": trials * 2, "labels": labels,
                                     "start_sample": self.cursor, "message": "校准开始，请跟随提示。"})
            self.examples, self.calibration_plan, self._trial_windows = [], [], {}
            self.calibration_context = copy.deepcopy(self.context())
            position = self.cursor
            randomizer = random.Random(time.time_ns())
            order = []
            for _ in range(trials):
                pair = [0, 1]
                randomizer.shuffle(pair)
                order.extend(pair)
            for trial, label_index in enumerate(order, 1):
                for phase, seconds in (("rest", rest), ("cue", cue), ("active", active)):
                    self.calibration_plan.append({"phase": phase, "start": position, "end": position + seconds * fs,
                                                  "trial": trial, "label": labels[label_index], "label_index": label_index})
                    position += seconds * fs
            self.calibration["end_sample"] = position
            self.calibration_owns_recording = self.recorder is None
            try:
                if not self.recorder:
                    self.start_recording()
                if not self.recorder:
                    raise OSError("无法建立校准录制文件。")
                self.recorder.event("calibration_plan", {"labels": labels, "context": self.calibration_context,
                                                        "phases": self.calibration_plan,
                                                        "timing_note": "sample clock scheduled browser cue; not hardware synchronized"})
            except OSError:
                self._abort_calibration("校准录制初始化失败，本次校准已中止。")
                raise
            self._last_calibration_phase = None
            self.prediction = self._no_prediction("校准采集中。")
            self._advance_calibration()
            self._event("info", f"开始两类校准，每类 {trials} 个试次；任务窗不重叠，整试次留出验证。")
            return {"ok": True}

    def _annotation(self, sample):
        if self.calibration["phase"] not in ("rest", "cue", "active", "complete"):
            return "", ""
        for phase in self.calibration_plan:
            if phase["start"] <= sample < phase["end"]:
                return (phase["label"], phase["trial"]) if phase["phase"] == "active" else (phase["phase"], phase["trial"])
        return "", ""

    def _advance_calibration(self):
        fs = self.config["sample_rate"]
        window = int(WINDOW_SECONDS * fs)
        buffer_start = self.cursor - len(self.buffer)
        # Extract only complete, nonoverlapping windows wholly within one active trial.
        for phase in self.calibration_plan:
            if phase["phase"] != "active" or phase["start"] >= self.cursor:
                continue
            accepted = self._trial_windows.get(phase["trial"], 0)
            end = phase["start"] + (accepted + 1) * window
            while end <= min(self.cursor, phase["end"]):
                start = end - window
                if start >= buffer_start:
                    result = analyze(self.buffer[start-buffer_start:end-buffer_start], fs,
                                     self.config["channel_names"], self.config["selected_channels"], "eeg")
                    if result["quality"]["valid"] and result.get("features") is not None:
                        self.examples.append({"features": result["features"], "label": phase["label"],
                                              "trial_id": phase["trial"], "start_sample": start, "end_sample": end})
                        self.calibration["accepted_windows"] += 1
                    else:
                        self.calibration["rejected_windows"] += 1
                else:
                    self.calibration["rejected_windows"] += 1
                accepted += 1
                self._trial_windows[phase["trial"]] = accepted
                end += window
        start, end = self.calibration["start_sample"], self.calibration["end_sample"]
        self.calibration["progress"] = min(1, max(0, (self.cursor - start) / (end - start)))
        if self.cursor >= end:
            self.calibration.update({"phase": "complete", "remaining_s": 0, "progress": 1,
                                     "message": "校准采集完成。点击训练，查看按完整试次留出的评估结果。"})
            if hasattr(self.source, "cue"):
                self.source.cue = None
            name = self.store.save_json("calibration", {"context": self.calibration_context,
                                        "labels": self.calibration["labels"], "examples": self.examples,
                                        "plan": self.calibration_plan, "created": utc_now()})
            self._event("info", f"校准结束：{len(self.examples)} 个有效窗，特征及试次保存在 {name}。")
            if self.calibration_owns_recording:
                self._stop_recording()
            return
        phase = next(p for p in self.calibration_plan if p["start"] <= self.cursor < p["end"])
        self.calibration.update({"phase": phase["phase"], "trial": phase["trial"], "label": phase["label"],
                                 "remaining_s": round((phase["end"] - self.cursor) / fs, 1)})
        key = (phase["trial"], phase["phase"])
        if key != self._last_calibration_phase:
            self._last_calibration_phase = key
            if hasattr(self.source, "cue"):
                self.source.cue = ("a" if phase["label_index"] == 0 else "b") if phase["phase"] in ("cue", "active") else None
            if self.recorder:
                self.recorder.event("trial_phase", {"phase": phase["phase"], "label": phase["label"],
                                                    "trial": phase["trial"], "scheduled_sample": phase["start"]})

    def _abort_calibration(self, reason):
        self.calibration.update({"phase": "aborted", "message": reason, "remaining_s": 0})
        if self.source and hasattr(self.source, "cue"):
            self.source.cue = None
        self._event("warning", reason)
        if self.calibration_owns_recording:
            self._stop_recording()

    def abort_calibration(self):
        with self.lock:
            if self._calibrating():
                self._abort_calibration("用户中止校准；已录制的文件保留。")
        return {"ok": True}

    def train(self):
        with self.lock:
            if self.calibration["phase"] != "complete":
                raise ValueError("请先完成完整校准；中止的试次不会用于训练。")
            if self.context() != self.calibration_context:
                raise ValueError("采集配置已改变，请重新校准。")
            model = classifier.train_model(self.examples, self.calibration["labels"], self.calibration_context)
            classifier.validate_model(model)
            self.model = model
            self._event("info", "个人模型训练完成，请查看留出评估；小样本校准不代表独立实验有效性。")
            self._predict()
            return {"ok": True, "metrics": model.get("metrics", {})}

    def save_model(self):
        with self.lock:
            if not self.model:
                raise ValueError("暂无可保存的模型。")
            name = self.store.save_json("model", self.model)
            self._event("info", f"模型已保存：{name}")
            return {"ok": True, "name": name}

    def load_model(self, name):
        model = self.store.load_model(name)
        classifier.validate_model(model)
        with self.lock:
            if self._calibrating():
                raise ValueError("请先结束或中止校准。")
            self.model = model
            self.prediction = self._no_prediction("模型已载入，等待下一有效窗。")
            self._event("info", f"载入模型：{name}。推理前将检查来源与通道兼容性。")
        return {"ok": True}

    def snapshot(self):
        with self.lock:
            age = None if self.last_sample_at is None else max(0, time.monotonic() - self.last_sample_at)
            expired = self.status != "running" or age is None or age > STALE_SECONDS
            if self.status == "running" and age is not None and age > STALE_SECONDS:
                self._invalidate("无新数据，重新等待连续数据窗。")
            if expired and self._calibrating():
                self._abort_calibration("数据超时或断开，校准已中止。")
            quality = copy.deepcopy(self.analysis["quality"])
            prediction = copy.deepcopy(self.prediction)
            if expired:
                quality.update({"valid": False, "summary": "无新数据，反馈已暂停。"})
                prediction = self._no_prediction("无新数据，禁止沿用旧预测。")
            signal = copy.deepcopy(self.analysis["signal"])
            fs = self.config["sample_rate"]
            skip = max(1, len(self.buffer) // 800)
            preview = self.buffer[::skip]
            signal["times"] = ((np.arange(len(self.buffer))[::skip] - len(self.buffer) + 1) / fs).tolist()
            # Plot is centered for readability; recordings and gates always retain raw microvolts.
            if len(preview):
                safe = np.where(np.isfinite(preview), preview, 0)
                signal["traces"] = (safe - np.median(safe, axis=0)).T.tolist()
            else:
                signal["traces"] = []
            signal["plot_note"] = "波形仅显示去中值偏置；录制和质量检测使用原始微伏值。"
            compatible, reason = self._compatible()
            feedback = {"target": self.feedback_target, "threshold": self.feedback_threshold,
                        "value": None, "active": False, "status": "信号未通过质量门控。"}
            if quality["valid"] and not expired and self.config["mode"] == "eeg" and not self._calibrating():
                value = prediction.get("score") if self.feedback_target == "class_b" else signal.get("bands", {}).get(self.feedback_target)
                if self.feedback_target == "class_b" and prediction.get("state") != "predicted":
                    value = None
                if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
                    feedback.update({"value": float(value), "active": value >= self.feedback_threshold,
                                     "status": "达到目标" if value >= self.feedback_threshold else "未达目标"})
                else:
                    feedback["status"] = "模型未给出可用分类，反馈暂停。"
            stats = dict(getattr(self.source, "stats", {})) if self.source else {}
            stats.update({"received_samples": self.cursor, "age_s": round(age, 3) if age is not None else None})
            recording = {"active": True, "path": self.recorder.name, "samples": self.recorder.samples} if self.recorder else self.last_recording
            # Connection/recovery progress can change before any data batch arrives.
            # Terminal states retain the engine's final error or stop explanation.
            message = self.message
            if self.source_kind == "device" and self.status in ("connecting", "running"):
                message = getattr(self.source, "message", "") or message
            return {"status": self.status, "source": self.source_kind, "mode": self.config["mode"],
                    "demo_state": getattr(self.source, "state", "rest"), "message": message,
                    "config": copy.deepcopy(self.config), "telemetry": stats, "quality": quality,
                    "signal": signal, "feedback": feedback, "prediction": prediction,
                    "model": {"available": self.model is not None, "labels": self.model.get("labels", []) if self.model else [],
                              "context": self.model.get("context") if self.model else None,
                              "metrics": self.model.get("metrics", {}) if self.model else {},
                              "compatible": compatible, "reason": reason},
                    "calibration": copy.deepcopy(self.calibration), "recording": dict(recording),
                    "events": list(self.events)[-25:], "impedances": self.analysis.get("impedances", []),
                    "window_seconds": WINDOW_SECONDS, "packet_seconds": 50 / fs}
