"""Explainable spectral features and conservative engineering quality gates.

Input is unfiltered microvolts. Gates are engineering heuristics, not a clinical
assessment. Spectral feedback is descriptive and does not identify mental states.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

FEATURE_VERSION = "log-mu-beta-hann-detrend-v1"
WINDOW_SECONDS = 2.0
ADC_FULL_SCALE_UV = 187500.0
FEATURE_BANDS = ((8.0, 13.0), (13.0, 30.0))
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0),
         "alpha": (8.0, 13.0), "beta": (13.0, 30.0),
         "gamma": (30.0, 45.0)}


def _finite_round(value: float, digits: int = 5) -> float | None:
    return round(float(value), digits) if math.isfinite(float(value)) else None


def _power(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    # Half-open bands ensure adjoining frequency bins are counted only once.
    mask = (freqs >= low) & (freqs < high)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
    return np.sum(psd[mask], axis=0) * df


def analyze(samples: np.ndarray, sample_rate: int, channel_names: list[str],
            selected_channels: list[int], mode: str = "eeg") -> dict[str, Any]:
    """Analyze a complete trailing window, preserving selected channel order.

    Each selected channel contributes natural-log mu and beta power (uV squared).
    Bad/unavailable channels outside the selection remain visible but do not
    invalidate a fixed-channel classifier. A selected bad channel blocks features.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate < 100:
        raise ValueError("sample_rate must be an integer of at least 100 Hz")
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(channel_names) or not 1 <= x.shape[1] <= 32:
        raise ValueError("samples must have shape (n, len(channel_names)), with 1..32 channels")
    if (not selected_channels or len(set(selected_channels)) != len(selected_channels)
            or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                   or i < 0 or i >= x.shape[1] for i in selected_channels)):
        raise ValueError("selected_channels must be distinct valid zero-based indices")
    n, c = x.shape
    empty_signal = {"freqs": [], "psd": [], "bands": {k: 0.0 for k in BANDS},
                    "dominant_band": "unavailable", "frequency_resolution_hz": None,
                    "window_seconds": n / sample_rate}
    if n < 2:
        return {"quality": {"valid": False, "summary": "等待完整 2 秒数据窗", "channels": []},
                "signal": empty_signal, "features": None, "impedances": [],
                "feature_version": FEATURE_VERSION}

    finite_channels = np.all(np.isfinite(x), axis=0)
    # Replacing a nonfinite channel is only to produce safe diagnostics. Its gate
    # remains invalid and it can never contribute classifier features.
    safe = np.where(np.isfinite(x), x, 0.0)
    means = safe.mean(axis=0)
    centered = safe - means
    t = np.arange(n, dtype=float) - (n - 1) / 2
    detrended = centered - t[:, None] * ((t @ centered) / (t @ t))[None, :]
    hann = np.hanning(n)
    norm = sample_rate * float(np.sum(hann * hann))
    if norm <= 0:
        return {"quality": {"valid": False, "summary": "等待完整 2 秒数据窗", "channels": []},
                "signal": empty_signal, "features": None, "impedances": [],
                "feature_version": FEATURE_VERSION}
    spectrum = np.fft.rfft(detrended * hann[:, None], axis=0)
    psd = (spectrum.real ** 2 + spectrum.imag ** 2) / norm
    psd[1:-1 if n % 2 == 0 else None] *= 2
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    wide_power = _power(psd, freqs, 1, min(100, sample_rate / 2))
    line_power = _power(psd, freqs, 48, 52) + _power(psd, freqs, 58, 62)
    channels = []
    for i in range(c):
        rms = float(np.sqrt(np.mean(centered[:, i] ** 2)))
        peak = float(np.max(np.abs(centered[:, i])))
        saturated_pct = float(np.mean(np.abs(safe[:, i]) >= ADC_FULL_SCALE_UV * 0.995) * 100)
        line_ratio = float(line_power[i] / max(wide_power[i], 1e-20))
        if not finite_channels[i]:
            state = "nonfinite"
        elif saturated_pct > 0:
            state = "saturated"
        elif float(np.ptp(safe[:, i])) < 0.5 or rms < 0.1:
            state = "flat"
        elif abs(float(means[i])) > 10000 or rms > 150 or peak > 500 or np.max(np.abs(np.diff(safe[:, i]))) > 250:
            state = "artifact"
        elif line_ratio > 0.65:
            state = "line_noise"
        else:
            state = "ok"
        channels.append({"name": str(channel_names[i]), "state": state,
                         "rms_uv": _finite_round(rms), "peak_uv": _finite_round(peak),
                         "saturated_pct": _finite_round(saturated_pct),
                         "line_noise_ratio": _finite_round(line_ratio)})
    reasons = [f"{channels[i]['name']}: {channels[i]['state']}" for i in selected_channels
               if channels[i]["state"] != "ok"]
    if n < math.ceil(sample_rate * WINDOW_SECONDS):
        reasons.insert(0, "等待完整 2 秒数据窗")
    if mode != "eeg":
        reasons.insert(0, "非 EEG 采集模式，暂停识别")
    valid = not reasons
    # Always return finite JSON values, including corrupted replay inputs.
    selected_psd = psd[:, selected_channels].mean(axis=1)
    display_mask = freqs <= min(80.0, sample_rate / 2)
    band_powers = {name: float(_power(selected_psd, freqs, lo, hi))
                   for name, (lo, hi) in BANDS.items()}
    total = sum(band_powers.values())
    bands = {name: float(value / total) if total > 1e-20 else 0.0
             for name, value in band_powers.items()}
    features = None
    if valid:
        features = [float(np.log(max(float(_power(psd[:, i], freqs, lo, hi)), 1e-12)))
                    for i in selected_channels for lo, hi in FEATURE_BANDS]
    impedances = []
    if mode == "impedance" and n >= math.ceil(sample_rate * WINDOW_SECONDS):
        for channel in channels:
            rms = channel["rms_uv"]
            abnormal = (channel["state"] in ("nonfinite", "saturated") or rms is None or rms < .2)
            value = 250.0 if abnormal else min(250.0, rms * math.sqrt(2) / 24 * 1.32)
            impedances.append({"kohm": round(value, 3), "abnormal": abnormal or value >= 250,
                               "method": "vendor RMS estimate; 24 nA, calibration factor 1.32"})
    return {"quality": {"valid": valid, "summary": "信号通过工程质量检查" if valid else "; ".join(reasons),
                        "channels": channels},
            "signal": {"freqs": freqs[display_mask].tolist(),
                       "psd": np.nan_to_num(selected_psd[display_mask], nan=0, posinf=0, neginf=0).tolist(),
                       "bands": bands if all(math.isfinite(v) for v in bands.values()) else {k: 0.0 for k in BANDS},
                       "dominant_band": max(bands, key=bands.get) if valid and total > 1e-20 else "unavailable",
                       "frequency_resolution_hz": sample_rate / n, "window_seconds": n / sample_rate},
            "features": features, "impedances": impedances, "feature_version": FEATURE_VERSION}
