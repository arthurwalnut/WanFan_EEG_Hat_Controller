"""Causal filtering and frame-phase validation shared by training and live use."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos
from .contract import CHANNELS, FS, SELECTED, FRAMES_PER_CYCLE, SAMPLES_PER_CYCLE


def as_samples(samples):
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(CHANNELS):
        raise ValueError("c-VEP samples must be n x 8 uV in the contract electrode order")
    return x


class ContinuousPreprocessor:
    def __init__(self):
        self.sos = np.vstack([tf2sos(*iirnotch(50., 30., fs=FS)),
                              butter(4, [4., 40.], btype="bandpass", fs=FS, output="sos")])
        self.reset()

    def reset(self):
        self._zi = None

    def process(self, samples):
        x = as_samples(samples)
        if not np.isfinite(x).all():
            raise ValueError("Causal filter requires finite data; reset after a corrupt segment")
        if not len(x):
            return x.copy()
        if self._zi is None:
            self._zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self._zi = sosfilt(self.sos, x, axis=0, zi=self._zi)
        return y


def raw_quality(samples):
    x = as_samples(samples)
    reasons, warnings = [], []
    for index in SELECTED:
        a = x[:, index]
        name = CHANNELS[index]
        if not len(a) or not np.isfinite(a).all():
            reasons.append(name + ": nonfinite")
            continue
        if np.any(np.abs(a) >= 187500 * .995):
            reasons.append(name + ": saturated")
        centered = a - a.mean()
        if np.ptp(a) < .5 or np.std(a) < .1 or (len(a) >= 50 and
                np.any(np.ptp(np.lib.stride_tricks.sliding_window_view(a, 50), axis=-1) < .5)):
            reasons.append(name + ": flat")
        if abs(a.mean()) > 10000 or np.std(centered) > 150 or np.max(np.abs(centered)) > 500:
            warnings.append(name + ": large raw offset/amplitude")
    return {"valid": not reasons, "reason": "; ".join(reasons) or "raw engineering checks passed",
            "warnings": warnings}


def frame_phase(times, frames, *, require_warmup=True):
    """Return unwrapped phase in 250-Hz cycle bins using actual frame intervals.

    A following frame boundary must already have been observed, so interpolation
    never extrapolates a frame that the renderer has not confirmed. Tail 378 is
    a boundary only. No lag search is performed on held-out or live data.
    """
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("invalid sample times")
    if len(frames) < 2:
        raise ValueError("missing frame evidence")
    try:
        ft = np.asarray([f["observed_s"] for f in frames], dtype=float)
        cf_raw = np.asarray([f["code_frame"] for f in frames], dtype=float)
        trial_ids = {f["trial_id"] for f in frames}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid frame schema") from exc
    if len(trial_ids) != 1 or not np.isfinite(ft).all() or not np.isfinite(cf_raw).all():
        raise ValueError("frame evidence crosses trials or is nonfinite")
    if np.any(cf_raw != np.floor(cf_raw)) or np.any(np.diff(cf_raw) != 1):
        raise ValueError("missing/duplicated/out-of-order code frame")
    if cf_raw[0] < 0 or cf_raw[-1] > 3 * FRAMES_PER_CYCLE:
        raise ValueError("code frame outside trial")
    dt = np.diff(ft)
    if np.any(dt < .010) or np.any(dt > .025) or abs(np.mean(dt) - 1 / 60) > .001:
        raise ValueError("frame timing does not support the 60 Hz contract")
    if t[0] < ft[0] - 1e-9 or t[-1] > ft[-1] + 1e-9:
        raise ValueError("frame evidence does not bracket the EEG window")
    phase_frames = np.interp(t, ft, cf_raw)
    if require_warmup and phase_frames[0] < FRAMES_PER_CYCLE - 1e-7:
        raise ValueError("first cycle is warmup")
    return phase_frames * SAMPLES_PER_CYCLE / FRAMES_PER_CYCLE
