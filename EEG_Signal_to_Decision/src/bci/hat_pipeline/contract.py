"""The explicit, versioned input contract of the eight-channel EEG Hat decoder."""
from __future__ import annotations

CHANNELS = ("O2", "O1", "P8", "P7", "C4", "C3", "FP1", "FP2")
FS = 250
WINDOW = 375
HOP = 25
LABELS = ("rest", "left_mi", "right_mi")
WARMUP = 2 * FS
VERSION = 3
PREPROCESS_VERSION = "hat8-average-notch50and100q30-butter4-1to40-causal-sos-steadyinit-v2"
QUALITY_VERSION = "hat8-hard-faults-only-amplitude-warnings-v2"
FEATURE_VERSION = "legacy8to30-multitaper-nw4-log10-csp6-v2"


def input_contract():
    return {"sample_rate": FS, "channel_names": list(CHANNELS), "units": "uV",
            "window_samples": WINDOW, "hop_samples": HOP,
            "preprocessing": PREPROCESS_VERSION, "warmup_samples": WARMUP,
            "quality_policy": QUALITY_VERSION, "feature_extraction": FEATURE_VERSION}


def check_recording_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Recording metadata must be an object")
    if metadata.get("source") not in ("device", "demo"):
        raise ValueError("Recording source must explicitly be device or demo")
    if metadata.get("sample_rate") != FS or metadata.get("channel_names") != list(CHANNELS) or metadata.get("units") != "uV":
        raise ValueError("Recording must be 250 Hz, uV, in O2 O1 P8 P7 C4 C3 FP1 FP2 order")
    identifier = metadata.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Recording requires an explicit unique session id")
