"""Fixed, independently versioned four-target SSVEP acquisition/decoder contract."""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from scipy.signal import cheb1ord, cheby1, iirnotch, tf2sos

FS = 250
CHANNELS = ("O2", "O1", "P8", "P7", "C4", "C3", "FP1", "FP2")
SELECTED = (0, 1, 2, 3)
FREQUENCIES = (8., 10., 12., 14.)
TARGETS = (0, 1, 2, 3)
WINDOW, HOP = 375, 25
BANDS = 5
HARMONICS = 5
WARMUP_SECONDS = .5
TASK_SECONDS, REST_SECONDS = 6., 2.
VERSION = 1
KIND = "ssvep_fbcca"
STIMULUS_METHOD = "timestamp-sine-luminance-v1"
PASS_LOW = (6., 14., 22., 30., 38.)
STOP_LOW = (4., 10., 16., 24., 32.)
WEIGHTS = tuple(float(n ** -1.25 + .25) for n in range(1, BANDS + 1))


@lru_cache(maxsize=1)
def _designed_sos():
    """Return independent causal banks including a common 50Hz notch.

    Chebyshev band edges/order recipe follows Nakanishi's public FBCCA filter
    bank. Continuous causal SOS replaces that example's epoch-wise filtfilt.
    """
    notch = tf2sos(*iirnotch(50., 30., fs=FS))
    banks = []
    for low, stop in zip(PASS_LOW, STOP_LOW):
        order, critical = cheb1ord([low, 90.], [stop, 100.], 3., 40., fs=FS)
        banks.append(np.vstack((notch, cheby1(order, .5, critical, btype="bandpass", fs=FS, output="sos"))))
    return tuple(banks)


def filter_sos():
    # Design once; callers receive copies so their filter state/configuration
    # cannot alter the coefficients used for later contract validation.
    return [s.copy() for s in _designed_sos()]


def validate_target_count(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value != 4:
        raise ValueError("This SSVEP protocol requires exactly 4 targets")
    return 4


def get_config(target_count=4):
    validate_target_count(target_count)
    return dict(frequencies=list(FREQUENCIES), target_count=4, task_seconds=TASK_SECONDS,
                rest_seconds=REST_SECONDS, stimulus_method=STIMULUS_METHOD,
                warmup_seconds=WARMUP_SECONDS, window_seconds=WINDOW / FS, hop_seconds=HOP / FS)


def get_contract(target_count=4):
    validate_target_count(target_count)
    return dict(kind=KIND, version=VERSION, sample_rate=FS, channel_names=list(CHANNELS),
                units="uV", selected_channels=[CHANNELS[i] for i in SELECTED], targets=list(TARGETS),
                frequencies=list(FREQUENCIES), stimulus_method=STIMULUS_METHOD,
                task_seconds=TASK_SECONDS, rest_seconds=REST_SECONDS, warmup_seconds=WARMUP_SECONDS,
                window_samples=WINDOW, hop_samples=HOP, harmonics=HARMONICS,
                filter_bank=dict(family="Chebyshev I", passbands=[[lo, 90.] for lo in PASS_LOW],
                    stopbands=[[lo, 100.] for lo in STOP_LOW], order_pass_loss_db=3., stop_loss_db=40.,
                    ripple_db=.5, notch_hz=50., notch_q=30., application="continuous causal SOS",
                    reference="no rereference", resampling="none"),
                cca="centered, per-column scaled, rank-truncated SVD canonical correlation; relative rank tolerance 1e-8",
                aggregation="sum(weight * largest_canonical_correlation**2) / sum(weight)",
                weights=list(WEIGHTS), weight_exponent=1.25, weight_offset=.25,
                score_type="normalized_fbcca_score_not_probability",
                timing="250Hz sample coordinates; actual stimulus onset/tail and loader-validated display evidence; no photon timing claim")


def validate_contract(value):
    # JSON canonical comparison also distinguishes True from 1 and rejects
    # additional fields instead of silently adopting a different decoder.
    import json
    try:
        equal = json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(get_contract(), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        equal = False
    if not equal:
        raise ValueError("SSVEP contract mismatch; MI/c-VEP models and other frequencies are incompatible")


def target_count_from_contract(value):
    validate_contract(value)
    return 4

