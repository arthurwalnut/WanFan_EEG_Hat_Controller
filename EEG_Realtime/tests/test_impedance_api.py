"""HTTP integration of both controller entry points without real hardware."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener
from unittest.mock import Mock

from eeg_realtime.engine import Engine
from eeg_realtime.server import EEGServer
from eeg_realtime.mi_server import MIServer


class FakeMI:
    def __init__(self):
        self.lock = threading.RLock()
        self.status = "idle"
        self.source_kind = "device"
        self.source = SimpleNamespace(device_ip="192.168.137.10")
        self.worker = self.recorder = None
        self.evaluation = SimpleNamespace(active=lambda: False)
        self.training = {"status": "idle"}
        self.stops = 0
        self.starts = []

    def stop(self):
        self.stops += 1
        self.status = "idle"
        return {"ok": True}

    close = stop

    def start(self, options):
        self.starts.append(options)
        self.status = "running"
        return {"ok": True}


class APIContract:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mi = FakeMI()
        self.legacy = Engine(Path(self.tmp.name)) if self.server_class is EEGServer else None
        self.server = self.server_class(("127.0.0.1", 0), self.legacy or self.mi)
        if self.legacy:
            self.server.mi_engine = self.mi
        self.imp = self.server.impedance_engine
        self.calls = []
        # Replace hardware acquisition only: validation/HTTP/locks remain real.
        self.imp.start = self.fake_start
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = build_opener(ProxyHandler({}))

    def fake_start(self, options):
        self.calls.append(dict(options))
        self.imp.device_ip = options["device_ip"]
        self.imp.status = "connecting"
        return {"ok": True}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        if self.legacy:
            self.legacy.stop()
        self.thread.join(2)
        self.tmp.cleanup()

    def request(self, path, body=None, token=True):
        headers, data = {}, None
        if body is not None:
            data = json.dumps(body).encode()
            headers = {"Content-Type": "application/json", "Origin": self.url}
            if token:
                headers["X-EEG-Token"] = self.server.csrf_token
        try:
            response = self.opener.open(Request(self.url + path, data=data, headers=headers), timeout=5)
        except HTTPError as error:
            response = error
        with response:
            payload = response.read()
            return response.status, (json.loads(payload) if "json" in response.headers.get("Content-Type", "") else payload)

    def start(self, **extra):
        return self.request("/api/impedance/start", {
            "source": "device", "device_ip": "192.168.137.10", **extra})

    def test_read_only_page_state_and_styles(self):
        code, state = self.request("/api/impedance/state")
        self.assertEqual(code, 200)
        self.assertFalse(state["fresh"])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.request("/impedance.css")[0], 200)
        code, page = self.request("/")
        self.assertEqual(code, 200)
        self.assertIn(b'"impedanceTab"', page)

    def test_requires_csrf_and_valid_input(self):
        self.assertEqual(self.request("/api/impedance/start", {}, token=False)[0], 403)
        self.assertEqual(self.start(device_ip="bad-address")[0], 400)
        self.assertEqual(self.start(source="demo")[0], 400)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.mi.stops, 0)

    def test_shutdown_cleans_other_workflows_if_impedance_close_fails(self):
        original_close = self.imp.close
        self.imp.close = Mock(side_effect=RuntimeError("worker timeout"))
        try:
            with self.assertRaisesRegex(RuntimeError, "worker timeout"):
                self.server.server_close()
            self.assertEqual(self.server.socket.fileno(), -1)
            self.assertEqual(self.mi.stops, 1)
        finally:
            self.imp.close = original_close

    def test_explicit_switch_and_return_to_eeg(self):
        self.mi.status = "running"
        self.assertEqual(self.start()[0], 409)
        self.assertEqual(self.start(switch_from_mi=True, device_ip="192.168.137.11")[0], 409)
        self.assertEqual(self.mi.stops, 0)
        self.assertEqual(self.start(switch_from_mi=True)[0], 200)
        self.assertEqual(self.mi.stops, 1)
        self.assertEqual(self.mi.status, "idle")
        self.assertEqual(self.request("/api/impedance/eeg", {})[0], 200)
        self.assertEqual(self.mi.starts, [{"source": "device", "device_ip": "192.168.137.10"}])
        self.assertNotIn(self.imp.status, ("running", "connecting"))

    def test_recording_and_replay_are_preserved(self):
        self.mi.status = "running"
        self.mi.recorder = object()
        self.assertEqual(self.start(switch_from_mi=True)[0], 409)
        self.mi.recorder = None
        self.mi.source_kind = "replay"
        self.assertEqual(self.start(switch_from_mi=True)[0], 409)
        self.assertEqual(self.mi.stops, 0)
        self.assertEqual(self.calls, [])

    def test_active_impedance_blocks_competing_actions(self):
        self.assertEqual(self.start()[0], 200)
        # Install sentinels for eager action-map construction in existing code.
        for name in ("start_recording", "load_replay", "replay_control", "start_evaluation",
                     "robot_configure", "robot_test", "stop_recording", "acknowledge_cue",
                     "acknowledge_display", "train", "preview_recording_deletion", "delete_recordings",
                     "update_library", "preview_model_deletion", "delete_models", "load_model",
                     "robot_heartbeat", "robot_disarm"):
            setattr(self.mi, name, Mock(side_effect=AssertionError("competing action executed")))
        for path, body in (("start", {}), ("record/start", {}), ("replay/load", {}),
                           ("replay/control", {"action": "play"}), ("evaluation/start", {}),
                           ("robot/test", {}), ("robot/configure", {})):
            with self.subTest(path=path):
                self.assertEqual(self.request("/api/mi/" + path, body)[0], 409)
        self.assertEqual(self.start()[0], 409)
        self.assertEqual(self.request("/api/impedance/stop", {})[0], 200)


class FullServerTests(APIContract, unittest.TestCase):
    server_class = EEGServer

    def test_legacy_and_visual_ownership(self):
        self.legacy.status = "running"
        self.assertEqual(self.start()[0], 409)
        self.legacy.status = "stopped"
        self.assertEqual(self.start()[0], 200)
        self.assertEqual(self.request("/api/start", {"source": "demo"})[0], 409)
        self.assertEqual(self.request("/api/cvep/start", {"source": "demo"})[0], 409)
        self.assertEqual(self.request("/api/ssvep/start", {"source": "demo"})[0], 409)


class StandaloneServerTests(APIContract, unittest.TestCase):
    server_class = MIServer


if __name__ == "__main__":
    unittest.main()
