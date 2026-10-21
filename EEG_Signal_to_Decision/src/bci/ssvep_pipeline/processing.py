"""Shared causal filter bank and stable ordinary CCA; no supervised weights."""
from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from .contract import BANDS, CHANNELS, FREQUENCIES, FS, HARMONICS, SELECTED, WEIGHTS, WINDOW, filter_sos


def as_samples(samples):
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(CHANNELS):
        raise ValueError("SSVEP input must be N x 8 uV in contract channel order")
    return x


def raw_quality(samples):
    x = as_samples(samples)
    reasons, warnings = [], []
    for channel in SELECTED:
        a, name = x[:, channel], CHANNELS[channel]
        if not len(a) or not np.isfinite(a).all():
            reasons.append(name + ": nonfinite")
            continue
        if np.any(np.abs(a) >= 187500 * .995):
            reasons.append(name + ": saturated")
        if np.ptp(a) < .5 or np.std(a) < .1 or (len(a) >= 50 and
                np.any(np.ptp(np.lib.stride_tricks.sliding_window_view(a, 50), axis=-1) < .5)):
            reasons.append(name + ": flat")
        centered = a - a.mean()
        if abs(a.mean()) > 10000 or np.std(centered) > 150 or np.max(np.abs(centered)) > 500:
            warnings.append(name + ": large raw offset/amplitude")
    return dict(valid=not reasons, reason="; ".join(reasons) or "raw engineering checks passed", warnings=warnings)


class ContinuousPreprocessor:
    """N x 8 raw uV -> N x 5 banks x 8 channels. Reset at every source gap."""
    def __init__(self):
        self.sos = filter_sos()
        self.reset()

    def reset(self):
        self._states = [None] * BANDS

    def process(self, samples):
        x = as_samples(samples)
        if not np.isfinite(x).all():
            raise ValueError("Filter requires finite raw data; reset after a corrupt segment")
        if not len(x):
            return np.empty((0, BANDS, len(CHANNELS)))
        result = []
        for i, sos in enumerate(self.sos):
            if self._states[i] is None:
                self._states[i] = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
            y, self._states[i] = sosfilt(sos, x, axis=0, zi=self._states[i])
            result.append(y)
        return np.stack(result, axis=1)


def _orthogonal_basis(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("CCA requires finite two-dimensional sample matrices")
    x = x - x.mean(axis=0)
    scale = np.sqrt(np.mean(x * x, axis=0))
    active = scale > 1e-12
    if not np.any(active):
        return np.empty((len(x), 0))
    x = x[:, active] / scale[active]
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, s > max(float(s[0]) * 1e-8, 1e-12)]


def canonical_correlation(x, y):
    """Largest ordinary CCA correlation via principal angles of column spaces.

    SVD rank truncation handles duplicate/redundant channels without inverse
    covariance matrices or iterative fitting. It is not a stored classifier.
    """
    if len(x) != len(y):
        raise ValueError("CCA sample counts differ")
    ux, uy = _orthogonal_basis(x), _orthogonal_basis(y)
    if not ux.shape[1] or not uy.shape[1]:
        return 0.
    return float(np.clip(np.linalg.svd(ux.T @ uy, compute_uv=False)[0], 0., 1.))


def reference_signals(frequency, samples=WINDOW):
    t = np.arange(samples, dtype=float) / FS
    return np.column_stack([function(2 * np.pi * frequency * harmonic * t)
                            for harmonic in range(1, HARMONICS + 1)
                            for function in (np.sin, np.cos)])


_REFERENCE_BASES = tuple(_orthogonal_basis(reference_signals(f)) for f in FREQUENCIES)


def fbcca_scores(filtered):
    x = np.asarray(filtered, dtype=float)
    if x.shape != (WINDOW, BANDS, len(CHANNELS)) or not np.isfinite(x).all():
        raise ValueError("FBCCA epoch must be 375 x 5 x 8 finite filtered samples")
    correlations = np.zeros((BANDS, len(FREQUENCIES)))
    for band in range(BANDS):
        ux = _orthogonal_basis(x[:, band, SELECTED])
        if not ux.shape[1]:
            continue
        for target, uy in enumerate(_REFERENCE_BASES):
            correlations[band, target] = np.clip(np.linalg.svd(ux.T @ uy, compute_uv=False)[0], 0., 1.)
    # Chen et al., Eq. (6): square each correlation before weighted summation.
    raw = np.asarray(WEIGHTS) @ (correlations ** 2)
    return raw / sum(WEIGHTS), correlations
