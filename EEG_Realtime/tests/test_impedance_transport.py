"""Loopback device integration tests; no EEG hardware or external network used."""
from pathlib import Path
import queue
import socket
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_realtime.protocol import UV_PER_COUNT, encode_packet
from eeg_realtime.sources import DeviceSource


class LoopbackDevice:
    def __init__(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        self.listener.listen(1)
        self.listener.settimeout(.1)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.commands = queue.Queue()
        self.closed = threading.Event()
        self.connection = None
        self.worker = threading.Thread(target=self._listen, daemon=True)
        self.worker.start()

    def _listen(self):
        try:
            while not self.closed.is_set():
                try:
                    self.connection, _ = self.listener.accept()
                    break
                except socket.timeout:
                    pass
            if self.connection is None:
                return
            self.connection.settimeout(.1)
            pending = bytearray()
            while not self.closed.is_set():
                try:
                    data = self.connection.recv(1024)
                except socket.timeout:
                    continue
                if not data:
                    return
                pending.extend(data)
                while len(pending) >= 2:
                    length = 4 if pending[1] == 1 else 2
                    if len(pending) < length:
                        break
                    self.commands.put(bytes(pending[:length]))
                    del pending[:length]
        except OSError:
            if not self.closed.is_set():
                self.commands.put(b"socket-error")

    def send(self, sequence, raw=123, corrupt=False):
        packet = bytearray(encode_packet(sequence, np.full((50, 8), raw, dtype=np.int32)))
        if corrupt:
            packet[-1] ^= 1
        self.udp.sendto(packet, ("127.0.0.1", self.port))

    def close(self):
        self.closed.set()
        self.listener.close()
        if self.connection is not None:
            self.connection.close()
        self.udp.close()
        self.worker.join(2)


class DeviceModeTransportTests(unittest.TestCase):
    def setUp(self):
        self.device = LoopbackDevice()
        self.source = None
        self.worker = None
        self.stop = threading.Event()
        self.batches = queue.Queue()
        self.errors = queue.Queue()

    def tearDown(self):
        self.stop.set()
        if self.source is not None:
            self.source.close()
        if self.worker is not None:
            self.worker.join(2)
        self.device.close()

    def start(self, mode="eeg", timeout=3):
        self.source = DeviceSource({
            "device_ip": "127.0.0.1", "bind_host": "127.0.0.1",
            "port": self.device.port, "sample_rate": 250, "channels": 8,
            "mode": mode, "data_timeout": timeout, "command_interval": .1,
            "connect_timeout": 1, "max_reconnects": 0,
        })

        def consume():
            try:
                for batch in self.source.stream(self.stop):
                    self.batches.put(batch)
            except Exception as exc:
                self.errors.put(exc)

        self.worker = threading.Thread(target=consume, daemon=True)
        self.worker.start()
        self.command(b"\x5a\x01\x96\xff")
        self.command(b"\x5a\x05" if mode == "impedance" else b"\x5a\x02")

    def command(self, expected):
        self.assertEqual(self.device.commands.get(timeout=3), expected)

    def data(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            batch = self.batches.get(timeout=max(.01, deadline - time.monotonic()))
            if len(batch.samples):
                return batch
        self.fail("No valid samples from loopback device")

    def test_modes_crc_sequence_reset_and_old_queue_drain(self):
        self.start()
        self.device.send(100, raw=-1000)
        first = self.data()
        self.assertEqual(first.sequence, 100)
        np.testing.assert_allclose(first.samples, -1000 * UV_PER_COUNT)

        self.source.set_mode("impedance")
        self.command(b"\x5a\x03")
        # A late packet from the old session arrives during STOP settling.
        self.device.send(101, raw=2222)
        self.command(b"\x5a\x05")
        self.device.send(0, raw=3333, corrupt=True)
        self.device.send(0, raw=4444)
        impedance = self.data()
        self.assertEqual(impedance.sequence, 0)
        np.testing.assert_allclose(impedance.samples, 4444 * UV_PER_COUNT)
        self.assertEqual(self.source.mode, "impedance")
        self.assertEqual(self.source.stats["crc_errors"], 1)
        self.assertEqual(self.source.stats["lost_packets"], 0)
        self.assertEqual(self.source.stats["reordered_packets"], 0)

        self.source.set_mode("eeg")
        self.command(b"\x5a\x03")
        self.command(b"\x5a\x02")
        self.device.send(0, raw=-5555)
        eeg = self.data()
        self.assertEqual(eeg.sequence, 0)
        np.testing.assert_allclose(eeg.samples, -5555 * UV_PER_COUNT)
        self.assertEqual(self.source.mode, "eeg")
        self.assertEqual(self.source.stats["packets"], 3)
        self.assertTrue(self.errors.empty())

    def test_timeout_restarts_requested_mode_and_accepts_sequence_zero(self):
        self.start(mode="impedance", timeout=.3)
        self.device.send(20)
        self.assertEqual(self.data().sequence, 20)
        self.command(b"\x5a\x03")
        self.command(b"\x5a\x05")
        self.device.send(0)
        self.assertEqual(self.data().sequence, 0)
        self.assertEqual(self.source.stats["lost_packets"], 0)
        self.assertEqual(self.source.stats["reordered_packets"], 0)
        self.assertTrue(self.errors.empty())

    def test_stop_event_sends_stop_and_releases_transport(self):
        self.start(mode="impedance")
        self.stop.set()
        self.command(b"\x5a\x03")
        self.worker.join(2)
        self.assertFalse(self.worker.is_alive())
        self.assertTrue(self.errors.empty())
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind(("127.0.0.1", self.device.port))
        finally:
            probe.close()


if __name__ == "__main__":
    unittest.main()
