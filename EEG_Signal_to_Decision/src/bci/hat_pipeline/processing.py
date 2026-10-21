"""Shared causal preprocessing and streaming epochs. No device or UI side effects."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from .contract import CHANNELS, FS, HOP, LABELS, QUALITY_VERSION, WARMUP, WINDOW


def _samples(samples):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(CHANNELS):
        raise ValueError("Expected samples with shape (n, 8), in the fixed electrode order")
    return x


class ContinuousPreprocessor:
    """Average reference, causal 50/100-Hz notches, 1-40-Hz bandpass.

    State is preserved between chunks. Reset at recording/segment discontinuities;
    neither epoch boundaries nor label changes reset this continuous filter.
    Input and output are microvolts. No filtfilt or independently filtered epochs.
    """
    def __init__(self):
        notches = [tf2sos(*iirnotch(hz, Q=30.0, fs=FS)) for hz in (50.0, 100.0)]
        self.sos = np.vstack([*notches, butter(4, [1, 40], btype="bandpass", fs=FS, output="sos")])
        self.reset()

    def reset(self):
        self._zi = None

    def process(self, samples):
        x = _samples(samples)
        if not np.isfinite(x).all():
            raise ValueError("Preprocessor requires finite input")
        if not len(x):
            return x.copy()
        referenced = x - x.mean(axis=1, keepdims=True)
        if self._zi is None:
            # A zero initial state turns electrode DC into a startup transient.
            # Steady-state initialization uses only the first observed frame,
            # preserving causality and invariance to input chunk size.
            self._zi = sosfilt_zi(self.sos)[:, :, None] * referenced[0][None, None, :]
        filtered, self._zi = sosfilt(self.sos, referenced, axis=0, zi=self._zi)
        return filtered


def raw_quality(samples):
    """Preserve hard device fault checks; raw amplitude is advisory only.

    The legacy pipeline did not reject epochs by DC, RMS, peak or sample step.
    In particular, removable DC and mains interference must not make every
    otherwise continuous, annotated epoch ineligible for training. Warnings
    describe raw data, not the quality of the filtered EEG or its MI content.
    """
    x = _samples(samples)
    channels, reasons, warnings = [], [], []
    for i, name in enumerate(CHANNELS):
        raw = x[:, i]
        channel_warnings = []
        dc = None
        if not len(raw) or not np.isfinite(raw).all():
            state, rms, peak, saturation = "nonfinite", None, None, None
        else:
            saturation = float(np.mean(np.abs(raw) >= 187500 * .995))
            centered = np.clip(raw, -1e7, 1e7) - np.clip(raw, -1e7, 1e7).mean()
            rms = float(np.sqrt(np.mean(centered ** 2)))
            peak = float(np.max(np.abs(centered)))
            dc = float(np.clip(raw, -1e7, 1e7).mean())
            if abs(dc) > 10000:
                channel_warnings.append("dc_offset")
            if rms > 150 or peak > 500 or np.max(np.abs(np.diff(raw)), initial=0) > 250:
                channel_warnings.append("raw_amplitude")
            if saturation:
                state = "saturated"
            elif (np.ptp(raw) < .5 or rms < .1 or
                  (len(raw) >= FS // 5 and np.any(np.ptp(np.lib.stride_tricks.sliding_window_view(raw, FS // 5), axis=-1) < .5))):
                state = "flat"
            else:
                state = "ok"
        channels.append({"name": name, "state": state, "rms_uv": rms,
                         "peak_uv": peak, "dc_uv": dc, "saturated_fraction": saturation,
                         "warnings": channel_warnings})
        warnings.extend({"channel": name, "reason": reason} for reason in channel_warnings)
        if state != "ok":
            reasons.append(f"{name}: {state}")
    return {"valid": not reasons, "reason": "; ".join(reasons) or "raw signal passed engineering checks",
            "channels": channels, "warnings": warnings, "policy": QUALITY_VERSION}


class StreamingPipeline:
    def __init__(self, model=None, source_kind=None):
        self.preprocessor = ContinuousPreprocessor()
        self.model = None
        self.source_kind = source_kind
        self.reset()
        if model is not None:
            self.set_model(model, source_kind=source_kind)

    def set_model(self, model, source_kind=None):
        from .model import validate_model
        if source_kind is not None:
            if source_kind not in ("device", "demo"):
                raise ValueError("source_kind must be device or demo")
            self.source_kind = source_kind
        if model is not None:
            validate_model(model, source_kind=self.source_kind)
        self.model = model
        self.reset()

    def reset(self):
        self.preprocessor.reset()
        self._raw = np.empty((0, 8))
        self._filtered = np.empty((0, 8))
        self._indices = np.empty(0, dtype=np.int64)
        self._times = np.empty(0)
        self._last_index = None
        self._last_time = None
        self._segment = None
        self._window_position = 0
        self._processed = 0
        self._valid_after = WARMUP

    def _block(self, x, indices, times, segment):
        # Corrupt values remain in the raw gate; bounded substitutes prevent NaN
        # poisoning filter state. The subsequent recovery interval is rejected.
        safe = np.clip(np.nan_to_num(x, nan=0., posinf=375000., neginf=-375000.), -375000., 375000.)
        filtered = self.preprocessor.process(safe)
        self._raw = np.concatenate([self._raw, x])
        self._filtered = np.concatenate([self._filtered, filtered])
        self._indices = np.concatenate([self._indices, indices])
        self._times = np.concatenate([self._times, times])
        self._processed += len(x)
        out = []
        while len(self._raw) >= WINDOW:
            bad_rows = np.any(~np.isfinite(self._raw[:WINDOW]) | (np.abs(self._raw[:WINDOW]) >= 187500 * .995), axis=1)
            if bad_rows.any():
                self._valid_after = max(self._valid_after, self._window_position + int(np.flatnonzero(bad_rows)[-1]) + 1 + WARMUP)
            quality = raw_quality(self._raw[:WINDOW])
            status = "ready" if quality["valid"] else "invalid"
            reason = quality["reason"]
            if quality["valid"] and self._window_position < self._valid_after:
                status, reason = "warming_up", "Waiting for causal filter warmup/recovery"
            row = {"window_start": int(self._indices[0]), "window_end": int(self._indices[WINDOW - 1]),
                   "window_start_index": int(self._indices[0]), "window_end_index": int(self._indices[WINDOW - 1]),
                   "window_start_time": float(self._times[0]), "window_end_time": float(self._times[WINDOW - 1]),
                   "segment": int(segment), "label": None, "confidence": None, "probabilities": [],
                   "label_order": list(LABELS), "status": status, "quality": quality, "reason": reason, "accepted": False}
            out.append((row, self._filtered[:WINDOW].T.copy()))
            self._raw, self._filtered = self._raw[HOP:], self._filtered[HOP:]
            self._indices, self._times = self._indices[HOP:], self._times[HOP:]
            self._window_position += HOP
        self._last_index, self._last_time, self._segment = int(indices[-1]), float(times[-1]), int(segment)
        return out

    def _push_windows(self, samples, indices, times, segment=0):
        x = _samples(samples)
        raw_indices = np.asarray(indices)
        t = np.asarray(times, dtype=float)
        if raw_indices.ndim != 1 or t.ndim != 1 or len(raw_indices) != len(x) or len(t) != len(x):
            raise ValueError("indices/times must be one-dimensional and match samples")
        if not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("Sample indices must be integers")
        index = raw_indices.astype(np.int64, copy=False)
        if not np.isfinite(t).all() or np.any(index < 0):
            raise ValueError("Sample times must be finite and indices nonnegative")
        if isinstance(segment, bool) or not isinstance(segment, (int, np.integer)):
            raise ValueError("segment must be an integer")
        if not len(x):
            return []
        if (self._segment is not None and segment != self._segment) or (self._last_index is not None and
                (index[0] != self._last_index + 1 or abs((t[0] - self._last_time) - 1 / FS) > .25 / FS)):
            self.reset()
        boundaries = np.flatnonzero((np.diff(index) != 1) | (np.abs(np.diff(t) - 1 / FS) > .25 / FS)) + 1
        out, start = [], 0
        for end in list(boundaries) + [len(x)]:
            if start:
                self.reset()
            out.extend(self._block(x[start:end], index[start:end], t[start:end], segment))
            start = int(end)
        return out

    def push(self, samples, indices, times, segment=0):
        from .model import predict_epoch
        records = []
        for row, epoch in self._push_windows(samples, indices, times, segment):
            if row["status"] == "ready":
                if self.model is None:
                    row.update(status="no_model", reason="No trained eight-channel model loaded")
                else:
                    row.update(predict_epoch(self.model, epoch, source_kind=self.source_kind))
            records.append(row)
        return records
