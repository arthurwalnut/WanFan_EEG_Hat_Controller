"""Eight-channel EEG Hat training/inference, isolated from the legacy MNE path."""
from .contract import CHANNELS, FS, WINDOW, HOP, LABELS, input_contract
from .processing import ContinuousPreprocessor, StreamingPipeline, raw_quality
from .dataset import build_dataset
from .model import train_model, predict_epoch, validate_model, save_model, load_model

__all__ = ["CHANNELS", "FS", "WINDOW", "HOP", "LABELS", "input_contract",
           "ContinuousPreprocessor", "StreamingPipeline", "raw_quality", "build_dataset",
           "train_model", "predict_epoch", "validate_model", "save_model", "load_model"]

