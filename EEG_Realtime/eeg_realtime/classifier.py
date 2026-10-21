"""Personal two-class regularized LDA with trial-grouped holdout validation.

Models are JSON data, never executable pickle. Scores indicate model direction,
not calibrated probabilities, diagnoses, or validated mental-state measurements.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .signal import FEATURE_VERSION

MODEL_VERSION = 1
MIN_TRIALS_PER_CLASS = 4
MIN_WINDOWS_PER_TRIAL = 2
VALIDATION_THRESHOLD = 0.70
MIN_STRENGTH = 0.25


def _fit(x: np.ndarray, y: np.ndarray) -> dict:
    mean = x.mean(axis=0)
    # Small floors avoid exploding harmless sensor/rounding variation.
    scale = np.maximum(x.std(axis=0), 0.15)
    z = (x - mean) / scale
    centroids = np.vstack([z[y == k].mean(axis=0) for k in (0, 1)])
    residuals = z - centroids[y]
    covariance = residuals.T @ residuals / max(1, len(z) - 2)
    target = max(float(np.trace(covariance) / x.shape[1]), 0.05)
    covariance = 0.8 * covariance + np.eye(x.shape[1]) * (0.2 * target + 1e-4)
    weights = np.linalg.solve(covariance, centroids[1] - centroids[0])
    bias = -float((centroids[1] + centroids[0]) @ weights) / 2
    margins = z @ weights + bias
    margin_scale = max(float(np.median(np.abs(margins))), 1e-3)
    distances = np.sqrt(np.mean(residuals ** 2, axis=1))
    return {"mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(),
            "bias": bias, "centroids": centroids.tolist(), "margin_scale": margin_scale,
            "ood_radius": max(2.5, float(np.quantile(distances, 0.99)) * 3.0)}


def _raw_margin(parameters: dict, x: np.ndarray) -> np.ndarray:
    return ((x - np.asarray(parameters["mean"])) / np.asarray(parameters["scale"])) @ np.asarray(parameters["weights"]) + parameters["bias"]


def train_model(examples: list[dict], labels: list[str], context: dict) -> dict[str, Any]:
    """Hold out the latest quarter of whole trials within each class.

    The fold scaler and classifier see training trials only. After evaluation,
    deployment parameters are refitted using all examples; holdout metrics describe
    the earlier split, not an independent evaluation of those refitted parameters.
    """
    if not isinstance(labels, list) or len(labels) != 2 or any(not isinstance(s, str) or not s.strip() or len(s) > 80 for s in labels) or labels[0] == labels[1]:
        raise ValueError("Exactly two distinct nonempty labels are required")
    if not isinstance(context, dict):
        raise ValueError("context must be a JSON object")
    try:
        context_copy = json.loads(json.dumps(context, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("context must contain only finite JSON data") from exc
    if context_copy.get("feature_version", FEATURE_VERSION) != FEATURE_VERSION:
        raise ValueError("Incompatible feature version")
    context_copy["feature_version"] = FEATURE_VERSION
    if not isinstance(examples, list) or not examples:
        raise ValueError("No calibration examples")
    rows, targets, trial_ids = [], [], []
    trial_labels, trial_counts = {}, {}
    dimension = None
    for example in examples:
        if not isinstance(example, dict) or example.get("label") not in labels:
            raise ValueError("Each example must have one of the two labels")
        trial = example.get("trial_id")
        if isinstance(trial, bool) or not isinstance(trial, int) or trial < 0:
            raise ValueError("trial_id must be a nonnegative integer")
        try:
            row = np.asarray(example.get("features"), dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid features") from exc
        if row.ndim != 1 or not 1 <= row.size <= 128 or not np.all(np.isfinite(row)):
            raise ValueError("features must be a finite nonempty vector of at most 128 values")
        if dimension is None:
            dimension = row.size
        if row.size != dimension:
            raise ValueError("All feature vectors must have identical dimensions")
        target = labels.index(example["label"])
        if trial in trial_labels and trial_labels[trial] != target:
            raise ValueError("A trial cannot contain both labels")
        trial_labels[trial] = target
        trial_counts[trial] = trial_counts.get(trial, 0) + 1
        rows.append(row); targets.append(target); trial_ids.append(trial)
    if any(count < MIN_WINDOWS_PER_TRIAL for count in trial_counts.values()):
        raise ValueError("Each trial needs at least two independent 2-second feature windows")
    by_class = [sorted(trial for trial, label in trial_labels.items() if label == k) for k in (0, 1)]
    if any(len(trials) < MIN_TRIALS_PER_CLASS for trials in by_class):
        raise ValueError("Calibration needs at least four complete trials per class (two windows per trial)")
    test_ids = sorted(t for trials in by_class for t in trials[-max(1, math.ceil(len(trials) * 0.25)):])
    train_ids = sorted(set(trial_ids) - set(test_ids))
    x, y = np.vstack(rows), np.asarray(targets)
    is_test = np.isin(trial_ids, test_ids)
    fold_parameters = _fit(x[~is_test], y[~is_test])
    predicted = (_raw_margin(fold_parameters, x[is_test]) >= 0).astype(int)
    actual = y[is_test]
    confusion = [[int(np.sum((actual == a) & (predicted == p))) for p in (0, 1)] for a in (0, 1)]
    recall = [confusion[k][k] / sum(confusion[k]) for k in (0, 1)]
    balanced = float(np.mean(recall))
    # Whole-trial majority voting prevents long trials from dominating this metric.
    test_window_trials = np.asarray(trial_ids)[is_test]
    trial_correct = []
    for trial in test_ids:
        votes = predicted[test_window_trials == trial]
        fraction = float(votes.mean())
        trial_correct.append(fraction != 0.5 and int(fraction > 0.5) == trial_labels[trial])
    usable = balanced >= VALIDATION_THRESHOLD and float(np.mean(trial_correct)) >= VALIDATION_THRESHOLD
    metrics = {"holdout_accuracy": float(np.mean(predicted == actual)), "balanced_accuracy": balanced,
               "trial_accuracy": float(np.mean(trial_correct)), "confusion_matrix": confusion,
               "train_trials": train_ids, "test_trials": test_ids,
               "train_windows": int(np.sum(~is_test)), "test_windows": int(np.sum(is_test)),
               "usable": usable,
               "reason": "通过初步留出试次检查；仍需独立实验验证" if usable else "留出试次表现未达预设 0.70 门槛，暂停在线分类",
               "validation_threshold": VALIDATION_THRESHOLD,
               "split": "last_quarter_trials_per_class", "scaler_fit": "training_trials_only",
               "deployment_refit": "all_calibration_trials_after_holdout_evaluation",
               "limitations": "少量个人校准试次，仅初步检验；相邻窗并非独立试次；不代表跨天或跨人准确率"}
    model = {"model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
             "kind": "regularized_lda_two_class", "labels": list(labels),
             "context": context_copy, "feature_count": int(dimension),
             "parameters": _fit(x, y), "metrics": metrics,
             "score_kind": "class_1_direction_score_not_probability",
             "min_strength": MIN_STRENGTH}
    validate_model(model)
    return model


def validate_model(model: dict) -> bool:
    """Reject malformed or incompatible JSON models before any prediction."""
    if not isinstance(model, dict):
        raise ValueError("Model must be a JSON object")
    try:
        encoded = json.dumps(model, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Model contains nonfinite or non-JSON data") from exc
    if len(encoded) > 1_000_000:
        raise ValueError("Model is unexpectedly large")
    if model.get("model_version") != MODEL_VERSION or model.get("feature_version") != FEATURE_VERSION or model.get("kind") != "regularized_lda_two_class":
        raise ValueError("Unsupported model or feature version")
    labels = model.get("labels")
    if not isinstance(labels, list) or len(labels) != 2 or any(not isinstance(v, str) or not v.strip() or len(v) > 80 for v in labels) or labels[0] == labels[1]:
        raise ValueError("Invalid model labels")
    n = model.get("feature_count")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 128:
        raise ValueError("Invalid feature count")
    p = model.get("parameters")
    if not isinstance(p, dict):
        raise ValueError("Missing model parameters")
    for key, shape in (("mean", (n,)), ("scale", (n,)), ("weights", (n,)), ("centroids", (2, n))):
        try:
            vector = np.asarray(p.get(key), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid parameter {key}") from exc
        if vector.shape != shape or not np.all(np.isfinite(vector)) or np.any(np.abs(vector) > 1e10):
            raise ValueError(f"Invalid parameter {key}")
        if key == "scale" and np.any(vector < 1e-6):
            raise ValueError("Invalid standardization scale")
    for key in ("bias", "margin_scale", "ood_radius"):
        value = p.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e10:
            raise ValueError(f"Invalid parameter {key}")
        if key != "bias" and value <= 0:
            raise ValueError(f"Invalid parameter {key}")
    if not isinstance(model.get("context"), dict) or model["context"].get("feature_version") != FEATURE_VERSION:
        raise ValueError("Missing compatible model context")
    metrics = model.get("metrics")
    if not isinstance(metrics, dict) or type(metrics.get("usable")) is not bool:
        raise ValueError("Missing validation metrics")
    for key in ("holdout_accuracy", "balanced_accuracy", "trial_accuracy"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("Invalid validation metrics")
    strength = model.get("min_strength")
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not MIN_STRENGTH <= strength <= 1:
        raise ValueError("Invalid rejection threshold")
    if model.get("score_kind") != "class_1_direction_score_not_probability":
        raise ValueError("Invalid score semantics")
    return True


def predict(model: dict, features: list[float]) -> dict[str, Any]:
    validate_model(model)
    result = {"label": None, "score": 0.5, "strength": 0.0,
              "reason": "", "state": "uncertain", "score_kind": model["score_kind"]}
    try:
        x = np.asarray(features, dtype=float)
    except (TypeError, ValueError):
        result["reason"] = "特征无效"; return result
    if x.shape != (model["feature_count"],) or not np.all(np.isfinite(x)):
        result["reason"] = "特征维度不符或包含非有限值"; return result
    metrics = model["metrics"]
    if not metrics["usable"] or metrics["balanced_accuracy"] < VALIDATION_THRESHOLD or metrics["trial_accuracy"] < VALIDATION_THRESHOLD:
        result["reason"] = "模型未通过留出试次验证"; return result
    p = model["parameters"]
    z = (x - np.asarray(p["mean"])) / np.asarray(p["scale"])
    distance = float(np.min(np.sqrt(np.mean((z - np.asarray(p["centroids"])) ** 2, axis=1))))
    if np.any(np.abs(z) > 10.0) or distance > p["ood_radius"]:
        result["reason"] = "特征超出个人校准分布，请重新校准"; return result
    margin = float(z @ np.asarray(p["weights"]) + p["bias"])
    # This bounded direction score is intentionally NOT a calibrated probability.
    score = 1.0 / (1.0 + math.exp(-float(np.clip(margin / p["margin_scale"], -40, 40))))
    strength = abs(2.0 * score - 1.0)
    result.update(score=score, strength=strength)
    if strength < model["min_strength"]:
        result["reason"] = "两类模型得分接近，暂不确定"; return result
    result.update(label=model["labels"][int(score >= 0.5)], state="predicted",
                  reason="个人校准模型方向得分，非概率")
    return result
