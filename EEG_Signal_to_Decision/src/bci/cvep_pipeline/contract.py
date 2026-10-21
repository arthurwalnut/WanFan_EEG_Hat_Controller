"""Explicit local c-VEP protocol; not the unpublished ArmBCIsys codebook."""
from __future__ import annotations

import hashlib
import json
import numpy as np

FS = 250
CHANNELS = ("O2", "O1", "P8", "P7", "C4", "C3", "FP1", "FP2")
SELECTED = (0, 1, 2, 3)
TARGETS = tuple(range(6))
SUPPORTED_TARGET_COUNTS = (2, 4, 6)
REFRESH_HZ = 60
BITS = 63
FRAMES_PER_BIT = 2
FRAMES_PER_CYCLE = BITS * FRAMES_PER_BIT
CYCLES_PER_TRIAL = 3
SAMPLES_PER_CYCLE = 525
WINDOW = 375
HOP = 25
VERSION = 1


def validate_target_count(count):
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count not in SUPPORTED_TARGET_COUNTS:
        raise ValueError("c-VEP target_count must be one of 2, 4, 6")
    return int(count)


def get_codebook(count=6):
    count = validate_target_count(count)
    # x^6+x+1; all 63 nonzero LFSR states. Decimation 5 is a preferred
    # degree-6 pair, independently checked below instead of assumed by name.
    sequence = [1] * 6
    for n in range(63):
        sequence.append(sequence[n + 1] ^ sequence[n])
    a = np.asarray(sequence[:63], dtype=np.int64)
    b = a[(5 * np.arange(63)) % 63]
    correlations = {int((2 * a - 1) @ np.roll(2 * b - 1, k)) for k in range(63)}
    if len({tuple(sequence[i:i + 6]) for i in range(63)}) != 63 or correlations != {-17, -1, 15}:
        raise RuntimeError("Gold preferred-pair construction failed")
    bits = np.asarray([a ^ np.roll(b, k) for k in range(count)], dtype=np.uint8)
    payload = {"bits": bits.tolist(), "frames_per_bit": FRAMES_PER_BIT,
               "refresh_hz": REFRESH_HZ}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "frames": np.repeat(bits, FRAMES_PER_BIT, axis=1).tolist(),
            "hash": digest, "construction": f"s[n+6]=s[n+1] XOR s[n], seed=111111; b[n]=s[5n mod63]; target k=s XOR roll(b,k), k=0..{count - 1}",
            "provenance": "Local explicit protocol; original paper's exact stimulus bits were not supplied",
            "preferred_pair_correlations": sorted(correlations)}


def get_contract(count=6):
    count = validate_target_count(count)
    return {"kind": "cvep_personal_template", "version": VERSION,
            "sample_rate": FS, "channel_names": list(CHANNELS), "units": "uV",
            "selected_channels": [CHANNELS[i] for i in SELECTED],
            "targets": list(range(count)), "refresh_hz": REFRESH_HZ,
            "bits_per_cycle": BITS, "frames_per_bit": FRAMES_PER_BIT,
            "frames_per_cycle": FRAMES_PER_CYCLE, "cycles_per_trial": CYCLES_PER_TRIAL,
            "warmup_cycles": 1, "window_samples": WINDOW, "hop_samples": HOP,
            "codebook_hash": get_codebook(count)["hash"],
            "preprocessing": "continuous causal Butterworth4 4-40Hz + 50Hz notch Q30; no rereference; uV; no resampling",
            "decoder": "per-channel normalized phase-template correlation; equal posterior-channel weights",
            "timing": "actual frame boundaries, software estimated EEG times; no hardware synchronization claim"}


def target_count_from_contract(contract):
    """Validate the full canonical contract, including historical six targets."""
    if not isinstance(contract, dict) or not isinstance(contract.get("targets"), list):
        raise ValueError("c-VEP contract/codebook mismatch: missing target list")
    targets = contract["targets"]
    count = validate_target_count(len(targets))
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in targets)
            or targets != list(range(count)) or contract != get_contract(count)):
        raise ValueError("c-VEP contract/codebook mismatch")
    return count
