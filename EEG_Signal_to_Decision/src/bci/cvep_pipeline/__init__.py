"""Independent c-VEP calibration/decoding; deliberately incompatible with MI."""
from .contract import (FS, CHANNELS, SELECTED, TARGETS, REFRESH_HZ, FRAMES_PER_BIT,
                       FRAMES_PER_CYCLE, CYCLES_PER_TRIAL, WINDOW, HOP,
                       get_contract, get_codebook, SUPPORTED_TARGET_COUNTS,
                       validate_target_count, target_count_from_contract)
from .processing import ContinuousPreprocessor, raw_quality, frame_phase
from .model import build_dataset, train_model, predict_epoch, save_model, load_model, validate_model

__all__ = ["FS", "CHANNELS", "SELECTED", "TARGETS", "REFRESH_HZ", "FRAMES_PER_BIT",
           "FRAMES_PER_CYCLE", "CYCLES_PER_TRIAL", "WINDOW", "HOP", "get_contract",
           "get_codebook", "SUPPORTED_TARGET_COUNTS", "validate_target_count", "target_count_from_contract",
           "ContinuousPreprocessor", "raw_quality", "frame_phase",
           "build_dataset", "train_model", "predict_epoch", "save_model", "load_model", "validate_model"]
