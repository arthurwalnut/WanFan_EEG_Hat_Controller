"""Vendor RMS estimates and session freshness; no physical device required."""
import math
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from eeg_realtime.impedance import ImpedanceEngine, estimate_impedance
from eeg_realtime.sources import Batch


def wave(rms=240., offset=10000.):
    # Integer cycles make expected de-meaned RMS analytic.
    t = np.arange(500) / 250
    return np.tile((offset + rms * math.sqrt(2) * np.sin(2 * np.pi * 8 * t))[:, None], (1, 8))


class EstimateTests(unittest.TestCase):
    def test_vendor_formula_and_dc_removal(self):
        result = estimate_impedance(wave())
        self.assertEqual([x["name"] for x in result], ["O2", "O1", "P8", "P7", "C4", "C3", "FP1", "FP2"])
        for item in result:
            self.assertAlmostEqual(item["kohm"], 240 * math.sqrt(2) / 24 * 1.32, places=3)
            self.assertEqual(item["rms_uv"], 240.)
            self.assertFalse(item["abnormal"])

    def test_independent_channels_and_guards(self):
        data = wave()
        data[:, 0] = 100.
        data[0, 1] = 170000.01
        data[0, 2] = -170000.01
        data[0, 3] = np.nan
        data[:, 4] = wave(rms=5000., offset=0)[:, 4]
        result = estimate_impedance(data)
        self.assertEqual([x["reason"] for x in result[:5]],
                         ["low_rms", "saturated", "saturated", "nonfinite", "upper_limit"])
        self.assertTrue(all(x["kohm"] == 250 for x in result[:5]))
        self.assertFalse(result[5]["abnormal"])

    def test_exact_thresholds(self):
        data = np.tile(np.where(np.arange(500) % 2, .2, -.2)[:, None], (1, 8))
        self.assertFalse(estimate_impedance(data)[0]["abnormal"])
        data[0, 0] = 170000.
        self.assertNotEqual(estimate_impedance(data)[0]["reason"], "saturated")
        data[0, 0] += 1
        self.assertEqual(estimate_impedance(data)[0]["reason"], "saturated")

    def test_requires_complete_window(self):
        for data in (np.zeros((499, 8)), np.zeros((500, 7)), np.zeros((501, 8))):
            with self.assertRaises(ValueError):
                estimate_impedance(data)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = ImpedanceEngine()
        self.engine.source = SimpleNamespace(mode="impedance", stats={}, close=lambda: None)
        self.engine.status = "connecting"

    def consume(self, samples=None, *, stamp=None, gap=False):
        if samples is None:
            samples = wave()
        self.engine._consume(Batch(samples, gap=gap, received_monotonic=time.monotonic() if stamp is None else stamp))

    def test_no_values_before_full_window(self):
        self.consume(wave()[:450])
        state = self.engine.snapshot()
        self.assertFalse(state["fresh"])
        self.assertEqual(state["window_samples"], 450)
        self.assertEqual(state["channels"], [])
        self.consume(wave()[450:])
        self.assertTrue(self.engine.snapshot()["fresh"])

    def test_gap_clears_old_values_and_rebuilds_window(self):
        self.consume()
        self.consume(wave()[:50], gap=True)
        state = self.engine.snapshot()
        self.assertEqual(state["channels"], [])
        self.assertEqual(state["window_samples"], 50)
        self.consume(wave()[50:])
        self.assertTrue(self.engine.snapshot()["fresh"])

    def test_empty_gap_clears_values_immediately(self):
        self.consume()
        self.consume(np.empty((0, 8)), gap=True)
        self.assertEqual(self.engine.snapshot()["channels"], [])

    def test_receipt_time_expires_values_and_rejects_delayed_batches(self):
        self.consume()
        self.engine.last_sample_at -= 2
        self.assertFalse(self.engine.snapshot()["fresh"])
        self.assertEqual(self.engine.snapshot()["window_samples"], 0)
        self.consume(stamp=time.monotonic() - 2)
        self.assertEqual(self.engine.snapshot()["channels"], [])
        self.consume()
        self.assertTrue(self.engine.snapshot()["fresh"])

    def test_old_or_wrong_mode_batches_never_retain_values(self):
        self.consume()
        self.consume(stamp=self.engine.last_sample_at - .1)
        self.assertEqual(self.engine.snapshot()["channels"], [])
        self.engine.source.mode = "eeg"
        self.consume()
        self.assertEqual(self.engine.snapshot()["window_samples"], 0)

    def test_stop_clears_and_preserves_target(self):
        self.engine.device_ip = "192.168.137.2"
        self.consume()
        self.engine.stop()
        state = self.engine.snapshot()
        self.assertEqual(state["device_ip"], "192.168.137.2")
        self.assertEqual(state["channels"], [])
        self.assertEqual(state["status"], "stopped")

    def test_timed_out_stop_still_reports_active_worker_for_retry(self):
        release = threading.Event()
        worker = threading.Thread(target=lambda: release.wait(2), daemon=True)
        self.engine.worker = worker
        worker.start()
        try:
            with patch.object(worker, "join"):
                with self.assertRaises(RuntimeError):
                    self.engine.stop()
            state = self.engine.snapshot()
            self.assertTrue(state["active"])
            self.assertEqual(state["channels"], [])
        finally:
            release.set()
            worker.join(2)
            self.engine.stop()
        self.assertFalse(self.engine.snapshot()["active"])

    def test_snapshot_does_not_share_mutable_channels(self):
        self.consume()
        state = self.engine.snapshot()
        state["channels"][0]["kohm"] = 99
        self.assertNotEqual(self.engine.snapshot()["channels"][0]["kohm"], 99)

    def test_estimate_age_is_separate_from_latest_packet_age(self):
        with patch("eeg_realtime.impedance.time.monotonic", return_value=100.):
            self.consume(stamp=100.)
        with patch("eeg_realtime.impedance.time.monotonic", return_value=101.):
            self.consume(wave()[:50], stamp=101.)
            state = self.engine.snapshot()
            self.assertEqual(state["telemetry"]["age_s"], 0.)
            self.assertEqual(state["telemetry"]["estimate_age_s"], 1.)
            self.assertEqual(state["stale_seconds"], 1.2)
            self.assertTrue(state["fresh"])
        with patch("eeg_realtime.impedance.time.monotonic", return_value=101.21):
            state = self.engine.snapshot()
            self.assertFalse(state["fresh"])
            self.assertEqual(state["channels"], [])

    def test_same_clock_tick_does_not_discard_a_sequence_checked_batch(self):
        with patch("eeg_realtime.impedance.time.monotonic", return_value=100.):
            self.consume(wave()[:450], stamp=100.)
            self.consume(wave()[450:], stamp=100.)
            self.assertTrue(self.engine.snapshot()["fresh"])

    def test_invalid_configuration_does_not_stop_existing_session(self):
        self.consume()
        for options in ({"device_ip": "8.8.8.8"}, {"device_ip": "not-an-ip"}, {"device_ip": "::1"},
                        {"device_ip": "0.0.0.0"}, {"device_ip": "192.168.1.2", "source": "demo"},
                        {"device_ip": "192.168.1.2", "mode": "eeg"}):
            with self.assertRaises(ValueError):
                self.engine.start(options)
            self.assertTrue(self.engine.snapshot()["fresh"])

    def test_valid_config_has_only_fixed_device_settings(self):
        self.assertEqual(ImpedanceEngine.validate_options({"device_ip": " 192.168.1.2 ", "port": 123}),
                         {"device_ip": "192.168.1.2", "sample_rate": 250, "channels": 8, "mode": "impedance"})

    def test_worker_starts_only_on_explicit_start_and_closes_after_error(self):
        entered = threading.Event()
        released = threading.Event()
        closed = threading.Event()

        class FakeSource:
            mode = "impedance"
            stats = {}
            message = "fake source"

            def __init__(self, config):
                self.config = config

            def stream(self, stop):
                entered.set()
                yield Batch(wave(), received_monotonic=time.monotonic())
                released.wait(2)
                raise OSError("test connection lost")

            def close(self):
                closed.set()

        with patch("eeg_realtime.impedance.DeviceSource", FakeSource):
            engine = ImpedanceEngine()
            self.assertFalse(entered.is_set())
            engine.start({"device_ip": "192.168.1.2"})
            self.assertTrue(entered.wait(1))
            released.set()
            engine.worker.join(2)
            self.assertEqual(engine.snapshot()["status"], "error")
            self.assertEqual(engine.snapshot()["channels"], [])
            self.assertTrue(closed.is_set())
            engine.close()


if __name__ == "__main__":
    unittest.main()
