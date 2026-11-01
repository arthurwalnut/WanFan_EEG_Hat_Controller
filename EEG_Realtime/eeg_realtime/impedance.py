"""Live electrode contact estimates using the vendor's unfiltered RMS method.

The protocol v1.0 section 12.5 specifies the current and calibration formula.
InBrain DataProcessor::processBufferedData uses a 500-sample minimum window,
an absolute peak above 170000 uV as its saturation guard, and a half-second
update interval. Values are contact-quality estimates, not calibrated ohms.
"""
from __future__ import annotations

import copy
import ipaddress
import math
import threading
import time

import numpy as np

from .sources import DeviceSource

CHANNEL_NAMES = ("O2", "O1", "P8", "P7", "C4", "C3", "FP1", "FP2")
SAMPLE_RATE = 250
WINDOW_SAMPLES = 500
UPDATE_SAMPLES = SAMPLE_RATE // 2
STALE_SECONDS = 1.2
SATURATION_UV = 170000.0
UPPER_KOHM = 250.0


def estimate_impedance(samples):
    """Return eight per-channel estimates from a complete raw microvolt window.

    Mean removal is the only signal processing: EEG filters, average reference,
    and detrending would change the vendor's impedance calculation.
    """
    data = np.asarray(samples, dtype=float)
    if data.shape != (WINDOW_SAMPLES, len(CHANNEL_NAMES)):
        raise ValueError("阻抗计算需要 500 × 8 个原始微伏采样。")
    result = []
    for index, name in enumerate(CHANNEL_NAMES):
        channel = data[:, index]
        rms = None
        reason = "ok"
        if not np.isfinite(channel).all():
            reason = "nonfinite"
        else:
            centered = channel - channel.mean()
            rms = float(np.sqrt(np.mean(centered * centered)))
            if float(np.max(np.abs(channel))) > SATURATION_UV:
                reason = "saturated"
            elif rms < .2:
                reason = "low_rms"
        kohm = UPPER_KOHM if reason != "ok" else min(UPPER_KOHM, rms * math.sqrt(2) / 24 * 1.32)
        if reason == "ok" and kohm >= UPPER_KOHM:
            reason = "upper_limit"
        result.append({"name": name, "kohm": round(kohm, 3),
                       "abnormal": reason != "ok", "reason": reason,
                       "rms_uv": round(rms, 5) if rms is not None else None})
    return result


class ImpedanceEngine:
    """A separate device session; never writes EEG recordings or predictions."""

    def __init__(self):
        self.lock = threading.RLock()
        self.lifecycle = threading.RLock()
        self.worker = None
        self.source = None
        self.stop_event = threading.Event()
        self.status = "idle"
        self.device_ip = ""
        self.message = "连接 EEG Hat 后开始检查电极接触阻抗。"
        self.buffer = np.empty((0, len(CHANNEL_NAMES)))
        self.channels = []
        self.last_sample_at = None
        self.last_estimate_at = None
        self.samples_since_update = 0

    @staticmethod
    def validate_options(options):
        """Validate before the caller stops a previous EEG session."""
        if not isinstance(options, dict):
            raise ValueError("阻抗参数应为 JSON 对象。")
        if options.get("source", "device") != "device":
            raise ValueError("阻抗检查仅支持真实设备。")
        if (options.get("sample_rate", SAMPLE_RATE) != SAMPLE_RATE or
                options.get("channels", 8) != 8 or options.get("mode", "impedance") != "impedance"):
            raise ValueError("阻抗检查固定为 250 Hz、8 通道、阻抗模式。")
        value = options.get("device_ip")
        if not isinstance(value, str):
            raise ValueError("请输入设备的 IPv4 地址，或先搜索设备。")
        try:
            address = ipaddress.ip_address(value.strip())
        except ValueError as exc:
            raise ValueError("请输入有效的设备 IPv4 地址。") from exc
        if (address.version != 4 or address.is_unspecified or address.is_multicast or
                address.is_reserved or not (address.is_private or address.is_link_local)):
            raise ValueError("设备地址应为本地网络的 IPv4 地址。")
        return {"device_ip": str(address), "sample_rate": SAMPLE_RATE,
                "channels": len(CHANNEL_NAMES), "mode": "impedance"}

    def _clear_window(self):
        self.buffer = np.empty((0, len(CHANNEL_NAMES)))
        self.channels = []
        self.last_estimate_at = None
        self.samples_since_update = 0

    def start(self, options):
        config = self.validate_options(options)
        with self.lifecycle:
            self.stop()
            source = DeviceSource(config)
            with self.lock:
                self.source = source
                self.device_ip = config["device_ip"]
                self.status = "connecting"
                self.message = "正在连接并发送阻抗启动命令；等待完整数据窗。"
                self.last_sample_at = None
                self._clear_window()
                self.stop_event = threading.Event()
                self.worker = threading.Thread(target=self._run, args=(source, self.stop_event),
                                               daemon=True, name="EEG-impedance")
                self.worker.start()
        return {"ok": True}

    def stop(self):
        with self.lifecycle:
            with self.lock:
                self.stop_event.set()
                source, worker = self.source, self.worker
                self.status = "stopped"
                self.message = "阻抗检查已停止。"
                self._clear_window()
            if source is not None:
                source.close()
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=5)
                if worker.is_alive():
                    raise RuntimeError("阻抗采集线程尚未退出，请稍后重试。")
            with self.lock:
                self.worker = None
                self.status = "stopped"
                self.message = "阻抗检查已停止。"
                self._clear_window()
        return {"ok": True}

    def close(self):
        self.stop()

    def _run(self, source, stop):
        failed = False
        try:
            for batch in source.stream(stop):
                with self.lock:
                    if stop.is_set():
                        break
                    self._consume(batch)
        except Exception as exc:
            failed = True
            with self.lock:
                self.status = "error"
                self.message = str(exc)
        finally:
            source.close()
            with self.lock:
                if not failed:
                    self.status = "stopped"
                    self.message = "阻抗检查已停止。"
                self._clear_window()

    def _consume(self, batch):
        data = np.asarray(batch.samples, dtype=float)
        if data.ndim != 2 or data.shape[1] != len(CHANNEL_NAMES):
            raise ValueError("阻抗数据的通道数与配置不一致。")
        if batch.gap:
            self._clear_window()
            self.message = "数据流中断，重新积累完整阻抗窗口。"
        if not len(data):
            return
        if getattr(self.source, "mode", None) != "impedance":
            self._clear_window()
            self.message = "等待采集线程应用阻抗模式。"
            return
        now = time.monotonic()
        received = batch.received_monotonic
        # Use socket receipt time, not the time a delayed consumer catches up.
        if (received is None or not math.isfinite(received) or received > now + .05 or
                now - received > STALE_SECONDS):
            self._clear_window()
            self.message = "数据已过期，等待新数据。"
            return
        if self.last_sample_at is not None:
            # Windows monotonic timestamps can be quantized to the same tick
            # for consecutive recv calls; packet sequence checks live in source.
            if received < self.last_sample_at:
                self._clear_window()
                self.message = "接收时间不连续，等待新数据。"
                return
            if received - self.last_sample_at > STALE_SECONDS:
                self._clear_window()
        self.last_sample_at = received
        self.status = "running"
        self.buffer = np.concatenate((self.buffer, data))[-WINDOW_SAMPLES:]
        self.samples_since_update += len(data)
        if len(self.buffer) < WINDOW_SAMPLES:
            self.message = "正在积累 2 秒连续阻抗数据。"
        elif not self.channels or self.samples_since_update >= UPDATE_SAMPLES:
            self.channels = estimate_impedance(self.buffer)
            self.last_estimate_at = received
            self.samples_since_update = 0
            self.message = "阻抗数据接收中；模式按本机命令维护，协议无 ACK。"

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            age = None if self.last_sample_at is None else max(0., now - self.last_sample_at)
            estimate_age = None if self.last_estimate_at is None else max(0., now - self.last_estimate_at)
            expired = age is None or age > STALE_SECONDS
            if expired and len(self.buffer):
                self._clear_window()
                self.message = "无新数据，等待完整阻抗窗口。"
            fresh = (self.status == "running" and not expired and bool(self.channels)
                     and estimate_age is not None and estimate_age <= STALE_SECONDS)
            stats = dict(getattr(self.source, "stats", {}))
            stats["age_s"] = round(age, 3) if age is not None else None
            stats["estimate_age_s"] = round(estimate_age, 3) if estimate_age is not None else None
            message = self.message
            if self.status == "connecting":
                message = getattr(self.source, "message", "") or message
            active = (self.status in ("connecting", "running") or
                      (self.worker is not None and self.worker.is_alive()))
            return {"status": self.status, "active": active, "mode": "impedance", "device_ip": self.device_ip,
                    "message": message, "channels": copy.deepcopy(self.channels) if fresh else [],
                    "telemetry": stats, "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE,
                    "window_samples": len(self.buffer), "required_samples": WINDOW_SAMPLES,
                    "stale_seconds": STALE_SECONDS, "fresh": fresh}
