"""Opt-in MI prediction-to-UDP bridge. No EEG, robot SDK, or retry queue.

Only newly observed 25-sample-spaced prediction endpoints advance the vote
history. The history is diagnostic only: every available prediction maps
directly to a command, including uncertain predictions. Confidence remains
visible but does not gate output. UDP success is not robot completion.
"""
from collections import deque
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import math
from pathlib import Path
import queue
import socket
import threading
import time
import uuid


PROTOCOL = "eeg-mi-robot-v1"
SOURCE = "EEG_Realtime_MI"
FS, HOP, WINDOW_SIZE, THRESHOLD = 250, 25, 15, 1
# The cap delivers 50 samples per packet (~0.2 s), so a 0.25 s receipt
# timeout gave only 50 ms scheduling/network tolerance. Check packet liveness
# separately from the age of the exact classifier endpoint being submitted.
LEASE_S, DATA_FRESH_S, WINDOW_FRESH_S = 2.0, 1.0, .5
ACK_STATUSES = {"accepted", "completed", "failed", "busy", "duplicate", "ignored", "test", "dry_run"}
LABELS = {"rest": "neutral", "right_mi": "blink", "left_mi": "clench"}


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def prediction_available(entry):
    if not isinstance(entry, dict):
        return False
    quality = entry.get("quality")
    confidence = entry.get("confidence")
    return (entry.get("status") in ("predicted", "uncertain")
            and isinstance(entry.get("label"), str) and entry["label"] in LABELS
            and _number(confidence) and 0 <= confidence <= 1
            and not (isinstance(quality, dict) and quality.get("valid") is False))


def _host(value):
    if not isinstance(value, str):
        raise ValueError("机械臂地址必须是 IPv4 单播地址。")
    try:
        addr = ipaddress.IPv4Address(value.strip())
    except ipaddress.AddressValueError as exc:
        raise ValueError("机械臂地址必须是 IPv4 单播地址。") from exc
    if addr.is_unspecified or addr.is_multicast or addr.is_reserved or int(addr) >> 24 == 0:
        raise ValueError("机械臂地址不能是广播、多播或保留地址。")
    # This application uses ordinary /24 local robot networks; do not target
    # their network/broadcast endpoints even though IPv4 alone has no netmask.
    if addr.packed[-1] in (0, 255):
        raise ValueError("请填写具体机器人 IPv4 地址，不能使用 .0 或 .255 地址。")
    return str(addr)


def _port(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("UDP 端口必须是 1～65535 的整数。")
    return value


class RobotBridge:
    def __init__(self, root=None, *, log_dir=None, clock=None, socket_factory=None, start_watchdog=True):
        self._clock = clock or time.monotonic
        self._socket_factory = socket_factory or socket.socket
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._socket = None
        self._context = {}
        self._context_at = self._clock()
        self._history = deque(maxlen=WINDOW_SIZE)
        self._signature = None
        self._last_end = None
        self._last_entry_id = None
        self._last_time = None
        self._owner = None
        self._lease = 0.0
        self.mode, self.host, self.port = "off", "", 5005
        self.session_id = uuid.uuid4().hex
        self._seq = 0
        self.sent_count = self.test_count = 0
        self.last_send = self.last_ack = self.last_action_ack = None
        self.last_stop = self.last_prediction = None
        self.delivery_status = "not_sent"
        self.reason, self.error = "输出尚未启用。", None
        self._sent = {}
        self._counts = dict(neutral=0, blink=0, clench=0, unknown=0)
        self.command = "WAITING"
        self.window_start_s = self.window_end_s = None
        self.log_error = None
        self._log_queue = queue.Queue(maxsize=2048)
        self._log_thread = None
        self._watchdog = None
        self._directory = Path(log_dir) if log_dir is not None else (Path(root) / "data" / "robot_bridge" if root is not None else None)
        self.log_file = None
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)
            config = self._directory / "address.json"
            if config.is_file():
                try:
                    saved = json.loads(config.read_text(encoding="utf-8"))
                    self.host, self.port = _host(saved["host"]), _port(saved["port"])
                except (OSError, ValueError, KeyError, TypeError):
                    self.reason = "保存的地址无效，请重新配置。"
            self.log_file = str(self._directory / ("robot_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8] + ".jsonl"))
            self._log_thread = threading.Thread(target=self._write_logs, daemon=True, name="MI-robot-log")
            self._log_thread.start()
        if start_watchdog:
            self._watchdog = threading.Thread(target=self._watch, daemon=True, name="MI-robot-watchdog")
            self._watchdog.start()

    def _log(self, kind, **details):
        if self.log_file is None:
            return
        event = dict(type=kind, at=datetime.now(timezone.utc).isoformat(), **details)
        try:
            self._log_queue.put_nowait(event)
        except queue.Full:
            self.log_error = "机械臂日志队列已满，部分诊断未写入。"

    def _write_logs(self):
        try:
            with open(self.log_file, "x", encoding="utf-8") as handle:
                while not self._closed.is_set() or not self._log_queue.empty():
                    try:
                        event = self._log_queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
        except (OSError, ValueError) as exc:
            self.log_error = str(exc)

    def _watch(self):
        while not self._closed.wait(.05):
            self.tick()

    def _update_context(self, context):
        if context is not None:
            if not isinstance(context, dict):
                raise ValueError("机械臂上下文必须为对象。")
            self._context = dict(context)
            self._context_at = self._clock()

    def _eligibility(self, *, for_entry=False):
        c = self._context
        if c.get("source") != "device":
            return "仅真实设备实时数据可发送动作；演示和回放只可预览。"
        if c.get("status") != "running":
            return "真实设备尚未处于采集状态。"
        if c.get("recording") is not False:
            return "录制期间不发送机械臂动作。"
        if c.get("model_compatible") is not True or not c.get("model_id"):
            return "需要已加载且与真实设备兼容的 MI 模型。"
        age = c.get("data_age_s")
        elapsed = max(0., self._clock() - self._context_at)
        if not _number(age) or not 0 <= age + elapsed <= DATA_FRESH_S:
            return "超过 1 秒未收到新 EEG 数据，动作输出停止。"
        if self.host and self.host == str(c.get("device_ip", "")).strip():
            return "机械臂目标地址不能是 EEG 帽地址。"
        if for_entry:
            age = c.get("window_age_s")
            if not _number(age) or not 0 <= age + elapsed <= WINDOW_FRESH_S:
                return "预测窗口延迟超过 0.5 秒或时间无效，动作输出停止。"
        return None

    def _clear_votes(self):
        self._history.clear()
        self._signature = self._last_end = self._last_entry_id = self._last_time = None
        self._counts = dict(neutral=0, blink=0, clench=0, unknown=0)
        self.command = "WAITING"
        self.window_start_s = self.window_end_s = None
        self.last_prediction = None

    def _close_socket(self):
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    def _disarm(self, reason):
        was_active = self.mode != "off"
        was_live = self.mode == "live"
        self.mode, self._owner, self._lease = "off", None, 0.
        self.reason = str(reason)
        self._clear_votes()
        if was_live:
            elapsed = max(0., self._clock() - self._context_at)
            self.last_stop = dict(reason=self.reason, at=datetime.now(timezone.utc).isoformat(),
                **{key: (self._context[key] + elapsed if _number(self._context.get(key)) else None)
                   for key in ("data_age_s", "window_age_s")})
        # Retain the socket briefly for completion ACKs already in flight.
        if was_active:
            self._log("disarmed", reason=self.reason, details=self.last_stop if was_live else None)

    def disarm(self, reason="用户停止机械臂输出。"):
        with self._lock:
            self._disarm(reason)
            return self.snapshot()

    def configure(self, options, context=None):
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("机械臂桥接已关闭。")
            # Even a rejected replacement configuration ends the old lease.
            self._disarm("输出配置已更改。")
            if not isinstance(options, dict):
                raise ValueError("机械臂配置必须为对象。")
            self._update_context(context)
            mode = options.get("mode", "off")
            if mode not in ("off", "preview", "live"):
                raise ValueError("机械臂模式必须为 off、preview 或 live。")
            host = options.get("host", self.host)
            host = _host(host) if host else ""
            port = _port(options.get("port", self.port))
            if mode == "live" and not host:
                raise ValueError("请先填写机器人 IPv4 地址。")
            if host and host == str(self._context.get("device_ip", "")).strip():
                raise ValueError("机械臂目标地址不能是 EEG 帽地址。")
            owner = options.get("client_id")
            if mode == "live" and (not isinstance(owner, str) or not owner.strip() or len(owner) > 160):
                raise ValueError("启用发送需要当前页面的 client_id。")
            self.host, self.port = host, port
            if mode == "live":
                reason = self._eligibility()
                if reason:
                    raise ValueError(reason)
            if self._directory is not None and host:
                path = self._directory / "address.json"
                temporary = self._directory / "address.json.tmp"
                temporary.write_text(json.dumps(dict(host=host, port=port)), encoding="utf-8")
                temporary.replace(path)
            self._close_socket()
            self._sent.clear()
            self.session_id = uuid.uuid4().hex
            self.last_send = self.last_ack = self.last_action_ack = None
            self.delivery_status, self.error = "not_sent", None
            self.mode = mode
            self._owner = owner.strip() if mode == "live" else None
            self._lease = self._clock() + LEASE_S if mode == "live" else 0.
            self.reason = {"off": "动作输出已关闭。", "preview": "逐窗预览预测映射，不发送动作。", "live": "已启用逐窗发送，等待新的有效分类结果；置信度不参与筛选。"}[mode]
            self._log("configured", mode=mode, host=host, port=port, session_id=self.session_id)
            return self.snapshot()

    def touch(self, client_id):
        with self._lock:
            self.tick()
            if self.mode == "live" and isinstance(client_id, str) and client_id == self._owner:
                self._lease = self._clock() + LEASE_S
            return self.snapshot()

    def tick(self, context=None):
        with self._lock:
            self._update_context(context)
            if self.mode == "live":
                reason = "发送页面心跳已过期，动作输出停止。" if self._clock() >= self._lease else self._eligibility()
                if reason:
                    self._disarm(reason)
            self._receive_acks()

    def observe(self, entry, context=None):
        with self._lock:
            self.tick(context)
            if self.mode == "off":
                return self.snapshot()
            if not isinstance(entry, dict):
                raise ValueError("MI 预测必须为对象。")
            if not prediction_available(entry):
                reason = "当前信号或模型窗口不可用，已清空机械臂统计；等待新的连续有效数据。"
                if self.mode == "live":
                    self._disarm(reason)
                else:
                    self._clear_votes()
                    self.reason = reason
                return self.snapshot()
            if self.mode == "live":
                reason = self._eligibility(for_entry=True)
                if reason:
                    self._disarm(reason)
                    return self.snapshot()
            end = entry.get("window_end_index")
            time_end = entry.get("window_end_time", entry.get("window_end"))
            if isinstance(end, bool) or not isinstance(end, int) or end < 0 or not _number(time_end):
                self._disarm("预测缺少有效样本索引或时间，动作输出停止。")
                return self.snapshot()
            signature = (self._context.get("source"), self._context.get("stream_id"),
                         self._context.get("model_id"), entry.get("segment"))
            if self._signature is not None and signature != self._signature:
                if self.mode == "live":
                    self._disarm("数据源、模型或连续数据段已改变，请重新启用发送。")
                    return self.snapshot()
                self._clear_votes()
            if self._last_end is not None:
                if end == self._last_end:
                    return self.snapshot()  # A repeated HTTP/event read contributes no vote.
                if end < self._last_end:
                    self._log("ignored_prediction", reason="out_of_order", end_index=end)
                    return self.snapshot()
                if end != self._last_end + HOP or abs(time_end - self._last_time - HOP / FS) > .01:
                    if self.mode == "live":
                        self._disarm("预测窗口不连续，请等待新数据后重新启用发送。")
                        return self.snapshot()
                    self._clear_votes()
            self._signature = signature
            self._last_end, self._last_time = end, float(time_end)
            self._last_entry_id = entry.get("id")
            label = LABELS[entry["label"]]
            self.last_prediction = {key: entry.get(key) for key in ("label", "confidence", "accepted", "status")}
            self._history.append((end, label))
            self._counts = {key: sum(label == key for _, label in self._history) for key in self._counts}
            self.window_start_s = float(time_end) - WINDOW_SIZE * HOP / FS
            self.window_end_s = float(time_end)
            self.command = {"blink": "BLINK", "clench": "CLENCH", "neutral": "NEUTRAL"}[label]
            self._log("decision", prediction_id=self._last_entry_id, end_index=end, mode=self.mode,
                      command=self.command, counts=dict(self._counts), prediction=dict(self.last_prediction),
                      decision_policy="direct_prediction", window_start_s=self.window_start_s, window_end_s=self.window_end_s)
            if self.mode == "live" and self.command in ("BLINK", "CLENCH"):
                self._send(self.command)
            return self.snapshot()

    def _send(self, command):
        # Nonblocking sendto; no command is queued or retried, including EWOULDBLOCK.
        now = datetime.now(timezone.utc)
        self._seq += 1
        payload = dict(protocol=PROTOCOL, session_id=self.session_id, request_id=f"{self.session_id}:{self._seq}",
                       seq=self._seq, cmd=command, source=SOURCE, sent_at=now.isoformat(timespec="milliseconds"),
                       expires_at=(now + timedelta(seconds=2)).isoformat(timespec="milliseconds"),
                       neutral_count=self._counts["neutral"], blink_count=self._counts["blink"], clench_count=self._counts["clench"],
                       window_start_s=self.window_start_s, window_end_s=self.window_end_s)
        if command != "TEST":
            payload.update(decision_policy="direct_prediction", prediction_id=self._last_entry_id,
                           prediction=dict(self.last_prediction))
        if command == "TEST":
            payload.update(message="Manual UDP link check; receiver must not run an action")
        try:
            if self._socket is None:
                self._socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
                self._socket.setblocking(False)
            data = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
            count = self._socket.sendto(data, (self.host, self.port))
            if count != len(data):
                raise OSError("UDP datagram was not fully submitted")
        except (OSError, ValueError) as exc:
            self.error, self.delivery_status = str(exc), "send_failed"
            self._log("send_failed", command=command, reason=str(exc), seq=self._seq)
            self._disarm("UDP 发送失败，已关闭输出；需要人工重新启用。")
            self._close_socket()
            return False
        self.sent_count += command != "TEST"
        self.test_count += command == "TEST"
        self._sent[self._seq] = dict(cmd=command, request_id=payload["request_id"], status="sent_unconfirmed")
        self._trim_ack_tracking()
        self.last_send = dict(payload=payload, target=f"{self.host}:{self.port}", status="sent_unconfirmed")
        self.delivery_status = "sent_unconfirmed"
        self._log("udp_submitted", payload=payload, target=self.last_send["target"], execution_confirmed=False)
        return True

    def _trim_ack_tracking(self):
        # Busy traffic must not evict an accepted long-running action before its
        # terminal ACK. The receiver normally accepts at most one active action.
        active = [seq for seq, row in self._sent.items() if row["status"] == "accepted"]
        for seq in active[:-8]:
            del self._sent[seq]
            self._log("ack_tracking_limit", seq=seq, reason="more_than_eight_unfinished_accepted_actions")
        recent = [seq for seq, row in self._sent.items() if row["status"] != "accepted"]
        for seq in recent[:-256]:
            del self._sent[seq]

    def test_link(self, context=None):
        with self._lock:
            self._update_context(context)
            if self._closed.is_set():
                raise RuntimeError("机械臂桥接已关闭。")
            if not self.host:
                raise ValueError("请先填写机器人 IPv4 地址。")
            if self.host == str(self._context.get("device_ip", "")).strip():
                raise ValueError("不能向 EEG 帽发送机器人 TEST 包。")
            self._send("TEST")
            return self.snapshot()

    def _receive_acks(self):
        if self._socket is None:
            return
        for _ in range(16):
            try:
                data, address = self._socket.recvfrom(8192)
            except BlockingIOError:
                return
            except OSError as exc:
                self.error = str(exc)
                self._disarm("UDP 接收状态异常，已停止动作输出。")
                self._close_socket()
                return
            if address != (self.host, self.port):
                continue
            try:
                ack = json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite ACK")))
            except (ValueError, UnicodeDecodeError):
                continue
            if (not isinstance(ack, dict) or ack.get("type") != "robot_ack" or ack.get("protocol") != PROTOCOL
                    or ack.get("session_id") != self.session_id or type(ack.get("seq")) is not int
                    or ack["seq"] not in self._sent or not isinstance(ack.get("status"), str)
                    or ack["status"] not in ACK_STATUSES):
                continue
            sent = self._sent[ack["seq"]]
            if ack.get("cmd") != sent["cmd"] or ack.get("request_id", sent["request_id"]) != sent["request_id"]:
                continue
            # A delayed accepted ACK cannot undo an already observed terminal result.
            if sent["status"] in ("completed", "failed", "dry_run"):
                continue
            sent["status"] = ack["status"]
            self._trim_ack_tracking()
            self._log("receiver_ack", ack=ack)
            if self.last_ack is None or ack["seq"] >= self.last_ack.get("seq", -1):
                self.last_ack = dict(ack)
            if ack["status"] in ("accepted", "completed", "failed", "dry_run") and (
                    self.last_action_ack is None or ack["seq"] >= self.last_action_ack.get("seq", -1)):
                self.last_action_ack = dict(ack)
            if self.last_send and ack["seq"] == self.last_send["payload"]["seq"]:
                self.delivery_status = ack["status"]
            if ack["status"] == "failed":
                self.error = str(ack.get("reason", "接收端动作失败。"))
                self._disarm("接收端报告动作失败，已关闭发送；需要人工检查。")

    def snapshot(self):
        with self._lock:
            reason = self._eligibility()
            return dict(mode=self.mode, host=self.host, port=self.port, armed=self.mode == "live",
                        eligible=reason is None, ineligible_reason=reason, owner_client_id=self._owner,
                        lease_remaining_s=max(0., self._lease - self._clock()) if self.mode == "live" else 0.,
                        session_id=self.session_id, window_count=len(self._history), window_size=WINDOW_SIZE, threshold=THRESHOLD,
                        decision_policy="direct_prediction", data_timeout_s=DATA_FRESH_S, window_timeout_s=WINDOW_FRESH_S,
                        last_stop=dict(self.last_stop) if self.last_stop else None,
                        last_prediction=dict(self.last_prediction) if self.last_prediction else None,
                        counts=dict(self._counts), command=self.command, window_start_s=self.window_start_s, window_end_s=self.window_end_s,
                        last_entry_id=self._last_entry_id, last_end_index=self._last_end, sent_count=self.sent_count, test_count=self.test_count,
                        last_send=json.loads(json.dumps(self.last_send)), last_ack=json.loads(json.dumps(self.last_ack)),
                        last_action_ack=json.loads(json.dumps(self.last_action_ack)),
                        delivery_status=self.delivery_status, error=self.error, reason=self.reason, log_file=self.log_file, log_error=self.log_error)

    def close(self):
        with self._lock:
            self._disarm("机械臂桥接已关闭。")
            self._closed.set()
            self._close_socket()
        if self._watchdog and self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=1)
        if self._log_thread and self._log_thread is not threading.current_thread():
            self._log_thread.join(timeout=1)
