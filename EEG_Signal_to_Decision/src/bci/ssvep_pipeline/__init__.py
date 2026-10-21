"""Independent four-frequency FBCCA; incompatible with MI and c-VEP models."""
from .contract import (FS, CHANNELS, SELECTED, FREQUENCIES, TARGETS, WINDOW, HOP,
                       BANDS, WARMUP_SECONDS, get_contract, get_config,
                       validate_target_count, target_count_from_contract)
from .processing import ContinuousPreprocessor, raw_quality, canonical_correlation, fbcca_scores
from .model import build_dataset, train_model, validate_model, predict_epoch, save_model, load_model

__all__ = ["FS", "CHANNELS", "SELECTED", "FREQUENCIES", "TARGETS", "WINDOW", "HOP", "BANDS",
           "WARMUP_SECONDS", "get_contract", "get_config", "validate_target_count", "target_count_from_contract",
           "ContinuousPreprocessor", "raw_quality", "canonical_correlation", "fbcca_scores",
           "build_dataset", "train_model", "validate_model", "predict_epoch", "save_model", "load_model"]
