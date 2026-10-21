"""Cancellable live, simulated and recorded EEG sources. No device is contacted on import."""
from dataclasses import dataclass, field
from collections import deque
import csv
import errno
import io
import ipaddress
import math
from pathlib import Path
import re
import select
import socket
import threading
import time

import numpy as np

from .protocol import CRCError, PacketError, SAMPLE_RATES, parse_packet
from .network_discovery import DiscoveryInterface, get_discovery_interfaces


def _stats():
    return dict(packets=0, lost_packets=0, crc_errors=0, bad_packets=0,
                reordered_packets=0, reconnects=0)


@dataclass
class Batch:
    samples: np.ndarray
    sequence: int | None = None
    gap: bool = False
    missing_samples: int = 0
    received_at: float = field(default_factory=time.time)
    received_monotonic: float | None = None
    # High-resolution QPC domain used by frame-timed c-VEP. Keep the older
    # monotonic stamp for MI; the two clocks must never be mixed implicitly.
    received_perf_counter: float | None = None


def _wait(stop, closed, seconds):
    deadline = time.monotonic() + max(0, seconds)
    while not stop.is_set() and not closed.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        stop.wait(min(remaining, 0.05))
    return True


def discover_devices(timeout=2.0, device_ip=None, *, port=3333, reply_port=None):
    """Discover on local interfaces, plus observed neighbours; never subnet-scan.

    Production discovery has its own UDP 3334 reply port. EEG reception retains
    exclusive UDP 3333. Tests may explicitly use reply_port=0 or another port.
    The source address/port and exact identity reply are checked before listing.
    """
    if device_ip is not None and not isinstance(device_ip, str):
        raise ValueError("设备 IP 必须是 IPv4 地址。")
    address_text = device_ip.strip() if device_ip else ""
    address = None
    if address_text:
        try:
            address = ipaddress.ip_address(address_text)
        except ValueError as exc:
            raise ValueError("请输入有效的设备 IPv4 地址。") from exc
        if (address.version != 4 or address.is_unspecified or address.is_multicast
                or not (address.is_private or address.is_link_local)):
            raise ValueError("设备地址应为本地网络的 IPv4 地址。")
    port = int(port)
    reply_port = (3334 if port == 3333 else 0) if reply_port is None else int(reply_port)
    if not 1 <= port <= 65535 or not 0 <= reply_port <= 65535 or reply_port == port:
        raise ValueError("发现请求端口与回复端口必须有效且互不相同。")
    if not math.isfinite(float(timeout)) or float(timeout) < 0:
        raise ValueError("发现超时必须是有限非负数。")
    # Explicit loopback is for local diagnostics/tests, never probe actual NICs.
    if address is not None and address.is_loopback:
        interfaces = [DiscoveryInterface(str(address), 32, 0)]
    else:
        interfaces = get_discovery_interfaces()
    probes = []
    found = set()
    try:
        entries = interfaces or [None]
        for interface in entries:
            targets = []
            network = interface.network if interface else None
            if address is not None:
                targets.append(str(address))
            if interface is not None:
                targets.extend(interface.neighbors[:32])
                if network.prefixlen <= 30:
                    targets.append(str(network.broadcast_address))
                bind_host = interface.address
            else:
                bind_host = ""
                targets.append("255.255.255.255")
            targets = tuple(dict.fromkeys(targets))
            if not targets:
                continue
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                sock.bind((bind_host, reply_port))
                sock.setblocking(False)
            except OSError as exc:
                sock.close()
                if reply_port and (exc.errno in (errno.EADDRINUSE, errno.EACCES, 10048, 10013)
                                   or getattr(exc, "winerror", None) in (10048, 10013)):
                    raise RuntimeError(f"发现回复 UDP {reply_port} 被占用或禁止绑定；请关闭其他发现程序后重试。") from exc
                # A disappeared/down interface must not hide the other NICs.
                continue
            probes.append((sock, network, targets))
        if not probes:
            raise RuntimeError("没有可绑定的 IPv4 发现接口，请检查电脑热点或网络连接。")
        deadline = time.monotonic() + max(0, float(timeout))
        next_send = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                for sock, _, targets in probes:
                    for destination in targets:
                        try:
                            sock.sendto(b"WHO_IS_EEG?", (destination, port))
                        except OSError:
                            continue  # Independent targets/interfaces survive.
                next_send = now + 0.4
            try:
                ready, _, _ = select.select([entry[0] for entry in probes], [], [],
                                             min(.15, max(0, deadline - time.monotonic())))
            except OSError:
                continue
            for sock, network, _ in probes:
                if sock not in ready:
                    continue
                try:
                    data, peer = sock.recvfrom(1024)
                except OSError:
                    # One unreachable target can cause Windows UDP RESET;
                    # keep listening for the remaining identity responses.
                    continue
                if data != b"I_AM_EEG_DEVICE" or peer[1] != port:
                    continue
                try:
                    peer_address = ipaddress.IPv4Address(peer[0])
                except ValueError:
                    continue
                if peer_address.is_unspecified or peer_address.is_multicast:
                    continue
                if network is not None and peer_address not in network and peer_address != address:
                    continue
                if network is not None and network.prefixlen <= 30 and peer_address in (network.network_address, network.broadcast_address):
                    continue
                found.add(str(peer_address))
    finally:
        for sock, _, _ in probes:
            sock.close()
    return sorted(found, key=ipaddress.IPv4Address)


class DemoSource:
    """Synthetic mu/beta patterns for UI/calibration exercises, not real EEG evidence."""
    def __init__(self, config):
        self.sample_rate = int(config.get("sample_rate", 250))
        self.channels = int(config.get("channels", 8))
        if self.sample_rate not in SAMPLE_RATES or not 1 <= self.channels <= 8:
            raise ValueError("Unsupported demo sample rate/channel count")
        self.state = "rest"
        self.cue = None
        self.stats = _stats()
        self.message = "SIMULATION / 模拟信号"
        self._closed = threading.Event()
        self._rng = np.random.default_rng(20260919)

    def close(self):
        self._closed.set()

    def stream(self, stop_event):
        sample = 0
        seq = 0
        next_due = time.monotonic()
        while not stop_event.is_set() and not self._closed.is_set():
            state = self.state if self.state in {"saturated", "flat", "noisy"} else (self.cue or self.state)
            t = (sample + np.arange(50)) / self.sample_rate
            phases = np.arange(self.channels) * 0.23
            # Interleaved left/right channels change in opposite directions.
            lateral = np.where(np.arange(self.channels) % 2 == 0, 1.0, -1.0)
            contrast = {"a": 0.75, "b": -0.75}.get(state, 0.0)
            alpha = 14 * (1 + contrast * lateral)
            beta = 6 * (1 - contrast * lateral * 0.75)
            samples = (np.sin(2 * np.pi * 10 * t[:, None] + phases) * alpha
                       + np.sin(2 * np.pi * 20 * t[:, None] + phases * 0.4) * beta
                       + self._rng.normal(0, 1.2, (50, self.channels)))
            if state == "saturated":
                samples.fill(187500.0)
            elif state == "flat":
                samples.fill(0)
            elif state == "noisy":
                samples += self._rng.normal(0, 120, samples.shape)
            self.stats["packets"] += 1
            self.message = f"SIMULATION / 模拟信号: {state}"
            # A task cue changes the simulated signal, not the continuity of its
            # sample clock. Marking cues as packet gaps aborts valid calibration.
            yield Batch(samples, seq, received_monotonic=time.monotonic(),
                        received_perf_counter=time.perf_counter())
            sample += 50
            seq += 1
            next_due += 50 / self.sample_rate
            if _wait(stop_event, self._closed, next_due - time.monotonic()):
                return


def read_csv(path):
    """Read InBrain microvolt CSV; preserve values and reject invalid numeric data."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    rate = None
    content = []
    for line in lines:
        if line.lstrip().startswith("#"):
            match = re.search(r"SampleRate\s*:\s*([0-9.]+)", line, re.IGNORECASE)
            if match:
                rate = float(match.group(1))
        elif line.strip():
            content.append(line)
    reader = csv.DictReader(io.StringIO("\n".join(content)))
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)):
        raise ValueError("CSV contains duplicate column names")
    channels = sorted((name for name in fields if re.fullmatch(r"CH[1-8]", name)), key=lambda n: int(n[2:]))
    if not channels:
        raise ValueError("CSV must contain CH1..CH8 channel columns in microvolts")
    rows = []
    times = []
    modes = set()
    discontinuities = []
    for index, row in enumerate(reader, 1):
        if None in row:
            raise ValueError(f"Too many CSV fields at sample {index}")
        try:
            values = [float(row[name]) for name in channels]
            timestamp = float(row["Time"]) if "Time" in fields else None
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid numeric CSV data at sample {index}") from exc
        if not all(math.isfinite(v) for v in values) or (timestamp is not None and not math.isfinite(timestamp)):
            raise ValueError(f"Non-finite CSV data at sample {index}")
        rows.append(values)
        if "Discontinuity" in fields:
            discontinuity = row.get("Discontinuity", "").strip()
            if discontinuity not in ("", "0", "1"):
                raise ValueError(f"Invalid Discontinuity at sample {index}")
            if discontinuity == "1":
                discontinuities.append(index - 1)
        if "Mode" in fields:
            modes.add(row.get("Mode", "").strip().lower())
        if timestamp is not None:
            times.append(timestamp)
    if not rows:
        raise ValueError("CSV contains no samples")
    times_array = np.asarray(times, dtype=float) if times else None
    if rate is None and times_array is not None and len(times_array) > 1:
        differences = np.diff(times_array)
        positive = differences[differences > 0]
        if len(positive):
            rate = 1.0 / float(np.median(positive))
    if rate is None or not math.isfinite(rate) or rate <= 0 or rate > 32000:
        raise ValueError("CSV requires a valid # SampleRate or inferable Time column")
    return dict(samples=np.asarray(rows, dtype=float), sample_rate=rate,
                channel_names=channels, times=times_array, modes=sorted(modes), discontinuities=discontinuities)


class ReplaySource:
    def __init__(self, path, speed=1.0):
        self.data = read_csv(path)
        self.sample_rate = self.data["sample_rate"]
        self.channels = len(self.data["channel_names"])
        self.speed = float(speed)
        if not math.isfinite(self.speed) or self.speed <= 0 or self.speed > 100:
            raise ValueError("Replay speed must be in (0, 100]")
        self.stats = _stats()
        self.message = "REPLAY / CSV 文件回放"
        self._closed = threading.Event()

    def close(self):
        self._closed.set()

    def stream(self, stop_event):
        data = self.data["samples"]
        times = self.data["times"]
        # Split at every discontinuity so no Batch can hide an interior gap.
        boundaries = np.array([], dtype=int)
        if times is not None and len(times) > 1:
            boundaries = np.flatnonzero(np.abs(np.diff(times) - 1 / self.sample_rate) > 0.25 / self.sample_rate) + 1
        boundaries = np.array(sorted(set(map(int, boundaries)) | set(self.data.get("discontinuities", []))), dtype=int)
        gaps = set(map(int, boundaries))
        segment_ends = list(map(int, boundaries)) + [len(data)]
        index = 0
        sequence = 0
        next_due = time.monotonic()
        for end in segment_ends:
            while index < end and not stop_event.is_set() and not self._closed.is_set():
                stop = min(end, index + 50)
                gap = index in gaps
                missing = 0
                if gap and index > 0 and times is not None:
                    missing = max(0, round((times[index] - times[index - 1]) * self.sample_rate) - 1)
                self.stats["packets"] += 1
                yield Batch(data[index:stop].copy(), sequence, gap, missing)
                next_due += (stop - index) / self.sample_rate / self.speed
                index = stop
                sequence += 1
                if _wait(stop_event, self._closed, next_due - time.monotonic()):
                    return
            if stop_event.is_set() or self._closed.is_set():
                return
        self.message = "REPLAY FINISHED / 文件回放结束"


class DeviceSource:
    def __init__(self, config):
        self.sample_rate = int(config.get("sample_rate", 250))
        self.channels = int(config.get("channels", 8))
        if self.sample_rate not in SAMPLE_RATES or not 1 <= self.channels <= 8:
            raise ValueError("Unsupported device sample rate/channel count")
        self.device_ip = str(config.get("device_ip", "")).strip()
        if not self.device_ip:
            raise ValueError("Device IP is required; discover devices first")
        self.port = int(config.get("port", 3333))
        self.bind_host = str(config.get("bind_host", "0.0.0.0"))
        self.data_timeout = max(0.1, float(config.get("data_timeout", 1.2)))
        self.max_reconnects = max(0, int(config.get("max_reconnects", 5)))
        self.command_interval = max(0.1, float(config.get("command_interval", 0.2)))
        self.connect_timeout = max(0.1, float(config.get("connect_timeout", 2.0)))
        self.stats = _stats()
        self.message = "等待连接设备"
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._requested_mode = "eeg"
        self._version = 0
        self.mode = "eeg"
        self.set_mode(config.get("mode", "eeg"))
        self._tcp = None
        self._udp = None
        self._peer_ip = None
        self._receiver_thread = None
        self._receiver_error = None
        self._receive_queue = deque()
        self._receive_lock = threading.Lock()
        self._receive_ready = threading.Event()
        self._drain_requested = threading.Event()
        self._drain_completed = threading.Event()
        self._overflow_pending = False
        self._queue_limit = max(1, int(config.get("receiver_queue_limit", 16)))
        self.stats.update(receiver_queue_depth=0, receiver_queue_overflows=0,
                          receiver_dropped_packets=0, receiver_max_queue_depth=0)

    def set_mode(self, mode):
        if mode not in {"eeg", "impedance"}:
            raise ValueError("mode must be eeg or impedance")
        with self._lock:
            if self._requested_mode != mode:
                self._requested_mode = mode
                self._version += 1

    def close(self):
        self._closed.set()
        for sock in (self._tcp, self._udp):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
        self._receive_ready.set()
        self._drain_completed.set()
        worker = self._receiver_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)

    def _gap(self):
        return Batch(np.empty((0, self.channels)), gap=True)

    def _drain(self):
        # Only the receiver reads the socket. Wait for its drain barrier before
        # issuing START so both the OS buffer and our queue exclude old data.
        if self._receiver_error is not None:
            raise OSError(f"UDP receiver failed: {self._receiver_error}")
        self._drain_completed.clear()
        self._drain_requested.set()
        if not self._drain_completed.wait(1.0):
            raise TimeoutError("UDP receiver did not acknowledge the drain barrier")
        if self._closed.is_set():
            raise OSError("Device source closed during UDP drain")
        if self._receiver_error is not None:
            raise OSError(f"UDP receiver failed: {self._receiver_error}")

    def _receive_datagrams(self):
        """Receive independently of decoding/model/disk work and stamp at recv.

        The queue is deliberately bounded. Overflow is a separate sticky event,
        emitted before any payload validation so a bad packet cannot hide loss.
        """
        try:
            while not self._closed.is_set():
                if self._drain_requested.is_set():
                    # Bound draining even if another process floods this port.
                    for _ in range(2048):
                        ready, _, _ = select.select([self._udp], [], [], 0)
                        if not ready:
                            break
                        self._udp.recvfrom(65535)
                    ready, _, _ = select.select([self._udp], [], [], 0)
                    if ready:
                        # Exhausting the bound is not proof of an empty socket.
                        # Refuse START instead of admitting residual old-mode data.
                        raise OSError("UDP drain limit reached while socket remained readable")
                    with self._receive_lock:
                        self._receive_queue.clear()
                        self._overflow_pending = False
                        self.stats["receiver_queue_depth"] = 0
                        self._receive_ready.clear()
                    self._drain_requested.clear()
                    self._drain_completed.set()
                    continue
                ready, _, _ = select.select([self._udp], [], [], .05)
                if not ready:
                    continue
                payload, peer = self._udp.recvfrom(65535)
                received_perf_counter = time.perf_counter()
                received_monotonic = time.monotonic()
                received_at = time.time()
                if self._drain_requested.is_set() or peer[0] != self._peer_ip:
                    continue
                with self._receive_lock:
                    if len(self._receive_queue) >= self._queue_limit:
                        self.stats["receiver_queue_overflows"] += 1
                        self.stats["receiver_dropped_packets"] += len(self._receive_queue)
                        self._receive_queue.clear()
                        self._overflow_pending = True
                    self._receive_queue.append((payload, peer, received_monotonic, received_at, received_perf_counter))
                    depth = len(self._receive_queue)
                    self.stats["receiver_queue_depth"] = depth
                    self.stats["receiver_max_queue_depth"] = max(self.stats["receiver_max_queue_depth"], depth)
                    self._receive_ready.set()
        except (OSError, ValueError) as exc:
            if not self._closed.is_set():
                self._receiver_error = exc
        finally:
            self._receive_ready.set()
            self._drain_completed.set()

    def _next_datagram(self):
        self._receive_ready.wait(.05)
        with self._receive_lock:
            overflow = self._overflow_pending
            self._overflow_pending = False
            datagram = self._receive_queue.popleft() if self._receive_queue else None
            self.stats["receiver_queue_depth"] = len(self._receive_queue)
            if not self._receive_queue and not self._overflow_pending:
                self._receive_ready.clear()
        if self._receiver_error is not None:
            raise OSError(f"UDP receiver failed: {self._receiver_error}")
        return datagram, overflow

    def _command(self, command):
        self._tcp.sendall(command)

    def _start_mode(self, stop, configure=False, restart=False):
        if restart:
            self._command(b"\x5a\x03")
            if _wait(stop, self._closed, self.command_interval):
                return None
        self._drain()
        if configure:
            self._command(bytes((0x5A, 0x01, SAMPLE_RATES[self.sample_rate], (1 << self.channels) - 1)))
            if _wait(stop, self._closed, self.command_interval):
                return None
            self._drain()
        with self._lock:
            mode, version = self._requested_mode, self._version
        self._command(b"\x5a\x05" if mode == "impedance" else b"\x5a\x02")
        self.mode = mode
        self.message = f"已发送 {mode.upper()} 启动命令，等待有效 UDP 数据（协议无 ACK）"
        return version

    def stream(self, stop_event):
        failures = 0
        try:
            self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self._udp.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                self._udp.bind((self.bind_host, self.port))
            except OSError as exc:
                raise OSError(f"Cannot bind UDP {self.port}; close InBrain/other EEG receivers: {exc}") from exc
            self._udp.setblocking(False)
            self._receiver_thread = threading.Thread(target=self._receive_datagrams,
                                                     daemon=True, name="EEG-UDP-receiver")
            self._receiver_thread.start()
            while not stop_event.is_set() and not self._closed.is_set():
                yield self._gap()
                try:
                    self.message = f"连接 {self.device_ip}:{self.port}"
                    self._tcp = socket.create_connection((self.device_ip, self.port), timeout=self.connect_timeout)
                    self._tcp.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    peer_ip = self._tcp.getpeername()[0]
                    self._peer_ip = peer_ip
                    version = self._start_mode(stop_event, configure=True)
                    if version is None:
                        return
                    last_sequence = None
                    last_valid = time.monotonic()
                    attempted_restart = False
                    while not stop_event.is_set() and not self._closed.is_set():
                        with self._lock:
                            requested_version = self._version
                        if requested_version != version:
                            yield self._gap()
                            version = self._start_mode(stop_event, restart=True)
                            if version is None:
                                return
                            last_sequence = None
                            last_valid = time.monotonic()
                            attempted_restart = False
                        if time.monotonic() - last_valid > self.data_timeout:
                            yield self._gap()
                            if attempted_restart:
                                raise TimeoutError("No valid EEG packets after STOP/START recovery")
                            self.message = "有效 UDP 数据超时，尝试 STOP/START 恢复"
                            version = self._start_mode(stop_event, restart=True)
                            if version is None:
                                return
                            last_sequence = None
                            last_valid = time.monotonic()
                            attempted_restart = True
                        ready, _, _ = select.select([self._tcp], [], [], 0)
                        if self._tcp in ready:
                            if self._tcp.recv(1024) == b"":
                                raise ConnectionError("TCP connection closed by device")
                        datagram, overflow = self._next_datagram()
                        if overflow:
                            self.message = "UDP 接收队列溢出，已丢弃积压数据并标记断流"
                            yield self._gap()
                        if datagram is not None:
                            payload, peer, received_monotonic, received_at, received_perf_counter = datagram
                            if peer[0] != peer_ip:
                                continue
                            try:
                                packet = parse_packet(payload, self.channels)
                            except CRCError:
                                self.stats["crc_errors"] += 1
                                continue
                            except PacketError:
                                self.stats["bad_packets"] += 1
                                continue
                            with self._lock:
                                if self._version != version:
                                    continue
                            missing_packets = 0
                            if last_sequence is not None:
                                delta = (packet.sequence - last_sequence) & 0xFFFFFFFF
                                if delta == 0 or delta >= 0x80000000:
                                    self.stats["reordered_packets"] += 1
                                    continue
                                missing_packets = delta - 1
                            last_sequence = packet.sequence
                            last_valid = time.monotonic()
                            attempted_restart = False
                            self.stats["packets"] += 1
                            self.stats["lost_packets"] += missing_packets
                            self.message = f"设备数据接收中 / {self.mode.upper()}（模式由本机命令维护）"
                            yield Batch(packet.samples, packet.sequence, missing_packets > 0, missing_packets * 50,
                                        received_at=received_at, received_monotonic=received_monotonic,
                                        received_perf_counter=received_perf_counter)
                except (OSError, ValueError) as exc:
                    if stop_event.is_set() or self._closed.is_set():
                        return
                    self.message = f"设备会话中断: {exc}"
                    yield self._gap()
                    if stop_event.is_set() or self._closed.is_set():
                        return
                    if failures >= self.max_reconnects:
                        raise ConnectionError(f"Device reconnect limit reached ({self.max_reconnects}): {exc}") from exc
                    failures += 1
                    self.stats["reconnects"] += 1
                finally:
                    if self._tcp is not None:
                        try:
                            self._command(b"\x5a\x03")
                        except OSError:
                            pass
                        self._tcp.close()
                        self._tcp = None
                if _wait(stop_event, self._closed, min(0.5 * 2 ** (failures - 1), 4.0)):
                    return
        finally:
            self.close()
