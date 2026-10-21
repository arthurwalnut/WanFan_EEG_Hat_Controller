"""Grouped train/calibration/test fitting and interchangeable Hat8 estimators."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import scipy
from scipy.fft import rfft, rfftfreq
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.signal.windows import dpss
from scipy.special import logsumexp, softmax
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .contract import FEATURE_VERSION, FS, LABELS, VERSION, WINDOW, input_contract
from .dataset import DatasetBuildError, build_dataset

BANDS = ((8., 30.),)
FEATURE_WIDTHS = {"bandpower": 8, "csp": 6, "combined": 14}
CONFIDENCE_METHOD = "heldout_class_trial_balanced_temperature_scaling"


class TemperatureCalibrator(BaseEstimator):
    """Fit one positive scale on held-out trials; never relearn class boundaries.

    Each class contributes one third of the loss, each trial within a class has
    equal weight, and windows divide their trial's weight. Total weight is one,
    so duplicating all windows in a trial does not change its influence. This
    estimates confidence under an equal-class target prior, not class prevalence.
    """
    def __init__(self, min_temperature=.05, max_temperature=100.):
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    @staticmethod
    def _check_scores(scores):
        values = np.asarray(scores, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("Temperature calibration requires finite three-class scores")
        return values

    @staticmethod
    def _trial_weights(y, groups):
        truth, group_ids = np.asarray(y), np.asarray(groups)
        if (truth.ndim != 1 or group_ids.ndim != 1 or len(truth) != len(group_ids)
                or not np.array_equal(np.unique(truth), np.arange(3))
                or any(not isinstance(group, str) or not group for group in group_ids)):
            raise ValueError("Calibration requires all three classes and explicit trial groups")
        unique_groups = np.unique(group_ids)
        group_labels = {}
        for group in unique_groups:
            labels = np.unique(truth[group_ids == group])
            if len(labels) != 1:
                raise ValueError("A calibration trial must contain one class")
            group_labels[group] = int(labels[0])
        counts = np.asarray([sum(label == k for label in group_labels.values()) for k in range(3)])
        weights = np.zeros(len(truth), dtype=float)
        for group, label in group_labels.items():
            mask = group_ids == group
            weights[mask] = 1. / (3 * counts[label] * int(mask.sum()))
        return weights, counts

    def fit(self, scores, y, groups):
        values = self._check_scores(scores)
        truth = np.asarray(y)
        weights, trial_counts = self._trial_weights(truth, groups)
        if len(values) != len(truth):
            raise ValueError("Calibration scores and labels must have matching lengths")
        if (not np.isfinite([self.min_temperature, self.max_temperature]).all()
                or not 0 < self.min_temperature < self.max_temperature):
            raise ValueError("Temperature bounds must be positive and finite")

        def objective(log_temperature):
            scaled = (values - values.max(axis=1, keepdims=True)) / np.exp(log_temperature)
            losses = logsumexp(scaled, axis=1) - scaled[np.arange(len(truth)), truth.astype(int)]
            return float(weights @ losses)

        result = minimize_scalar(objective,
                                 bounds=(np.log(self.min_temperature), np.log(self.max_temperature)),
                                 method="bounded", options={"xatol": 1e-7, "maxiter": 200})
        if not result.success or not np.isfinite(result.fun):
            raise ValueError("Held-out temperature calibration failed")
        self.temperature_ = float(np.exp(result.x))
        self.classes_ = np.arange(3)
        self.fit_groups_ = sorted(np.unique(groups).tolist())
        self.fit_trials_per_class_ = trial_counts.tolist()
        self.fit_windows_ = int(len(values))
        self.class_weight_sums_ = [float(weights[truth == k].sum()) for k in range(3)]
        self.weighted_log_loss_ = float(result.fun)
        return self

    def predict_proba(self, scores):
        values = self._check_scores(scores)
        temperature = getattr(self, "temperature_", None)
        if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
                or not np.isfinite(temperature) or temperature <= 0
                or not np.array_equal(getattr(self, "classes_", None), np.arange(3))):
            raise ValueError("Temperature calibrator is not fitted or is invalid")
        # Subtract before dividing, avoiding positive overflow for huge scores.
        with np.errstate(over="ignore", under="ignore"):
            probabilities = softmax((values - values.max(axis=1, keepdims=True)) / temperature, axis=1)
        # Positive scaling preserves order mathematically. Preserve it also if
        # floating-point rounding makes almost-equal probabilities tie.
        winners = values.argmax(axis=1)
        changed = np.flatnonzero(probabilities.argmax(axis=1) != winners)
        for row in changed:
            probabilities[row, winners[row]] = probabilities[row].max() + 4 * np.finfo(float).eps
            probabilities[row] /= probabilities[row].sum()
        return probabilities


def _feature_spec(kind):
    """Persist spectral conventions separately from the uV waveform contract."""
    return {"version": FEATURE_VERSION, "kind": kind, "width": FEATURE_WIDTHS[kind],
            "bandpower": {"enabled": kind in ("bandpower", "combined"),
                          "band_hz_inclusive": [8., 30.], "method": "multitaper",
                          "time_half_bandwidth": 4., "max_tapers": 8,
                          "periodic_tapers": True, "taper_norm": 2,
                          "low_bias_eigenvalue_gt": .9, "adaptive": False,
                          "remove_dc": True, "normalization": "length",
                          "input_units": "uV", "spectral_units": "V",
                          "spectral_input_dtype": "float32", "aggregation": "frequency_mean",
                          "log_base": 10, "power_floor": 1e-12},
            "csp": {"enabled": kind in ("csp", "combined"), "components": 6,
                    "method": "hat8_one_vs_rest_trace_normalized_shrinkage",
                    "shrinkage": .05, "log_base": "e", "input_units": "uV"}}


@lru_cache(maxsize=1)
def _multitaper_parameters():
    # Match MNE psd_array_multitaper defaults used by legacy ml_features.py:
    # NW=4, periodic L2-normalized DPSS, low_bias=True, adaptive=False.
    tapers, eigenvalues = dpss(WINDOW, 4., 8, sym=False, norm=2, return_ratios=True)
    keep = eigenvalues > .9
    tapers, eigenvalues = tapers[keep], eigenvalues[keep]
    frequencies = rfftfreq(WINDOW, 1 / FS)
    mask = (frequencies >= BANDS[0][0]) & (frequencies <= BANDS[0][1])
    return tapers, np.sqrt(eigenvalues)[None, None, :, None], mask


def _legacy_bandpower(x):
    """Legacy 8-30-Hz log10 mean spectrum, without an MNE dependency.

    Input remains uV at the pipeline boundary. Spectral input is converted to
    float32 V, matching the old XDF loader/transformer and its 1e-12 power floor.
    MNE's 'length' normalization does not divide the taper-normalized spectrum
    by sample rate; this is a frequency mean, not integrated band power.
    """
    tapers, weights, mask = _multitaper_parameters()
    output = np.empty((len(x), 8), dtype=float)
    # Bound temporary FFT storage for long recordings; epochs are independent.
    for start in range(0, len(x), 64):
        volts = np.asarray(x[start:start + 64] * 1e-6, dtype=np.float32)
        centered = volts - volts.mean(axis=-1, keepdims=True)
        spectra = rfft(centered[..., None, :] * tapers, axis=-1)
        spectra[..., 0] /= np.sqrt(2.)
        if WINDOW % 2 == 0:
            spectra[..., -1] /= np.sqrt(2.)
        weighted = spectra[..., mask] * weights
        psd = (weighted * weighted.conj()).real.sum(axis=-2)
        psd *= 2 / np.square(weights).sum(axis=-2)
        output[start:start + 64] = np.log10(np.maximum(psd.mean(axis=-1), 1e-12))
    return output


class HatFeatures(BaseEstimator, TransformerMixin):
    """Log bandpower and/or one-vs-rest regularized CSP fitted only on train.

    CSP follows the covariance/one-vs-rest design of legacy bci.ml_features.CSP,
    reimplemented here without MNE and with trace-normalized shrinkage covariances.
    """
    def __init__(self, kind="bandpower", n_components=6):
        self.kind = kind
        self.n_components = n_components

    @staticmethod
    def _check(X):
        x = np.asarray(X, dtype=float)
        if x.ndim != 3 or x.shape[1:] != (8, WINDOW) or not np.isfinite(x).all():
            raise ValueError("Feature input must be finite (n, 8, 375) filtered uV epochs")
        return x

    def fit(self, X, y=None):
        x = self._check(X)
        if self.kind not in ("bandpower", "csp", "combined"):
            raise ValueError("feature must be bandpower, csp or combined")
        self.filters_ = None
        if self.kind in ("csp", "combined"):
            labels = np.asarray(y)
            if labels.shape != (len(x),) or not np.array_equal(np.unique(labels), np.arange(3)):
                raise ValueError("CSP training requires all three configured classes")
            centered = x - x.mean(axis=-1, keepdims=True)
            covariance = np.einsum("nct,ndt->ncd", centered, centered) / WINDOW
            traces = np.trace(covariance, axis1=1, axis2=2)
            covariance /= np.maximum(traces[:, None, None], 1e-12)
            covariance = .95 * covariance + .05 * np.eye(8)[None] / 8
            filters = []
            for label in range(3):
                positive, negative = covariance[labels == label].mean(axis=0), covariance[labels != label].mean(axis=0)
                values, vectors = eigh(positive, positive + negative)
                # Two extreme spatial patterns per class, six total.
                filters.extend([vectors[:, np.argmax(values)], vectors[:, np.argmin(values)]])
            self.filters_ = np.asarray(filters)
        self.n_features_in_ = 8
        self.feature_version_ = FEATURE_VERSION
        return self

    def transform(self, X):
        x = self._check(X)
        outputs = []
        if self.kind in ("bandpower", "combined"):
            outputs.append(_legacy_bandpower(x))
        if self.kind in ("csp", "combined"):
            if self.filters_ is None:
                raise ValueError("CSP feature transformer has not been fitted")
            projected = np.einsum("kc,nct->nkt", self.filters_, x)
            outputs.append(np.log(np.maximum(projected.var(axis=-1), 1e-12)))
        return np.concatenate(outputs, axis=1)


def _split_dataset(dataset):
    y, groups, sessions = dataset["y"], dataset["groups"], dataset["session_ids"]
    group_labels = {}
    for group in np.unique(groups):
        labels = np.unique(y[groups == group])
        if len(labels) != 1:
            raise ValueError(f"One trial contains multiple task labels: {group}")
        group_labels[str(group)] = int(labels[0])
    if any(sum(value == label for value in group_labels.values()) < 6 for label in range(3)):
        counts = [sum(value == label for value in group_labels.values()) for label in range(3)]
        raise DatasetBuildError(
            "每类至少需要 6 个有效完整试次；当前静息 %d、左手 %d、右手 %d。请补充录制后共同训练。" % tuple(counts),
            dataset["metadata"]["dataset_diagnostics"])
    # The generated recording IDs encode capture time. Do not let UI selection
    # order decide which recordings are the unseen calibration/test sets.
    session_order = sorted(set(sessions.tolist()))
    split = None
    # Prefer whole sessions whenever all three resulting sets contain all classes.
    if len(session_order) >= 3:
        count = max(1, len(session_order) // 5)
        session_sets = (session_order[:-2 * count], session_order[-2 * count:-count], session_order[-count:])
        candidate = [np.isin(sessions, values) for values in session_sets]
        if all(np.array_equal(np.unique(y[mask]), np.arange(3)) for mask in candidate):
            split, scope = candidate, "whole_session_holdout"
    if split is None:
        rng = np.random.default_rng(20260920)
        assigned = [[], [], []]
        for label in range(3):
            available = sorted(group for group, value in group_labels.items() if value == label)
            rng.shuffle(available)
            count = max(1, len(available) // 5)
            assigned[0].extend(available[:-2 * count]); assigned[1].extend(available[-2 * count:-count]); assigned[2].extend(available[-count:])
        split = [np.isin(groups, values) for values in assigned]
        scope = "whole_trial_holdout_within_available_sessions"
    if any(not mask.any() or not np.array_equal(np.unique(y[mask]), np.arange(3)) for mask in split):
        raise ValueError("Each train/calibration/test split must contain all three classes")
    if np.any(sum(mask.astype(int) for mask in split) != 1):
        raise ValueError("Internal split overlap")
    return split, scope


def _scores(pipeline, X):
    scores = np.asarray(pipeline.decision_function(X), dtype=float)
    if (scores.shape != (len(X), 3) or not np.isfinite(scores).all()
            or not np.array_equal(pipeline.classes_, np.arange(3))):
        raise ValueError("Classifier must return finite scores for all three classes")
    return scores


def _metrics(y, probabilities, groups):
    predicted = np.argmax(probabilities, axis=1)
    group_true, group_pred = [], []
    for group in np.unique(groups):
        mask = groups == group
        group_true.append(int(y[mask][0]))
        group_pred.append(int(np.argmax(probabilities[mask].mean(axis=0))))
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "trial_balanced_accuracy": float(balanced_accuracy_score(group_true, group_pred)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
            "trial_confusion_matrix": confusion_matrix(group_true, group_pred, labels=[0, 1, 2]).tolist(),
            "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
            "brier_score": float(np.mean(np.sum((np.eye(3)[y] - probabilities) ** 2, axis=1)))}


def train_model(sessions, feature="bandpower", classifier="lda"):
    if feature not in ("bandpower", "csp", "combined") or classifier not in ("lda", "svm"):
        raise ValueError("Use feature bandpower/csp/combined and classifier lda/svm")
    dataset = build_dataset(sessions)
    X, y, groups = dataset["X"], dataset["y"], dataset["groups"]
    masks, scope = _split_dataset(dataset)
    train, calibration, test = masks
    estimator = (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto") if classifier == "lda" else
                 SVC(C=1., kernel="linear", class_weight="balanced", decision_function_shape="ovr", break_ties=True, random_state=42))
    pipeline = Pipeline([("features", HatFeatures(feature)), ("scaler", StandardScaler()), ("classifier", estimator)])
    pipeline.fit(X[train], y[train])
    # Only calibration groups determine the positive temperature. Equal class
    # and trial weights cannot relearn the classifier's label boundaries.
    calibrator = TemperatureCalibrator().fit(_scores(pipeline, X[calibration]),
                                            y[calibration], groups[calibration])
    probabilities = calibrator.predict_proba(_scores(pipeline, X[test]))
    report = _metrics(y[test], probabilities, groups[test])
    report["usable"] = report["balanced_accuracy"] >= .5 and report["trial_balanced_accuracy"] >= .5
    report.update({"scope": scope, "holdout_accuracy": report["accuracy"],
                   "validation_threshold": .5, "chance_level": 1 / 3,
                   "reason": "Passed preliminary grouped holdout" if report["usable"] else "Grouped holdout below the fixed engineering threshold",
                   "deployment_refit": False, "rejected_windows": dataset["rejected"],
                   "limitations": "Small personal dataset; grouped windows are not independent trials; no cross-subject or clinical validation"})
    independent_sessions = scope == "whole_session_holdout"
    report["evaluation_isolation"] = {
        "level": "session" if independent_sessions else "trial",
        "independent_sessions": independent_sessions,
        "shared_filter_history": not independent_sessions,
        "split_order": "session_id_chronological",
        "assignment": "whole_session_chronological" if independent_sessions else "fixed_seed_stratified_trial",
        "note": ("按录制时间顺序留出整次录制；训练、校准、测试不共享录制或连续滤波状态。"
                 "同一人的同日录制仍不代表跨日期或跨被试泛化。" if independent_sessions else
                 "探索性试次留出：三个集合的试次不重叠，但可能来自同次录制并共享连续滤波历史；"
                 "不属于完全独立录制验证。至少补足能各含三类的三组独立录制可采用整录制留出。")}
    report["split_metrics"] = {}
    for name, mask in zip(("train", "calibration", "test"), masks):
        report[name + "_groups"] = sorted(np.unique(groups[mask]).tolist())
        report[name + "_sessions"] = sorted(np.unique(dataset["session_ids"][mask]).tolist())
        report[name + "_windows"] = int(mask.sum())
        report[name + "_trials_per_class"] = [int(len(np.unique(groups[mask & (y == label)]))) for label in range(3)]
        split_probabilities = probabilities if name == "test" else calibrator.predict_proba(_scores(pipeline, X[mask]))
        split_report = _metrics(y[mask], split_probabilities, groups[mask])
        for prefix, key in (("", "confusion_matrix"), ("trial_", "trial_confusion_matrix")):
            matrix = np.asarray(split_report[key], dtype=float)
            totals = matrix.sum(axis=1)
            split_report[prefix + "prediction_counts"] = matrix.sum(axis=0).astype(int).tolist()
            split_report[prefix + "recall_per_class"] = np.divide(np.diag(matrix), totals,
                out=np.zeros(3, dtype=float), where=totals > 0).tolist()
        split_report["label_order"] = list(LABELS)
        report["split_metrics"][name] = split_report
    features = pipeline.named_steps["scaler"].transform(pipeline.named_steps["features"].transform(X[train]))
    centers = np.stack([features[y[train] == label].mean(axis=0) for label in range(3)])
    residual = np.sqrt(np.mean((features - centers[y[train]]) ** 2, axis=1))
    model = {"format": "eeg-hat-eight-channel", "version": VERSION,
             "metadata": {**dataset["metadata"], "created": datetime.now(timezone.utc).isoformat(),
                          "feature": feature, "classifier": classifier,
                          "feature_spec": _feature_spec(feature),
                          "confidence_calibration": {
                              "method": CONFIDENCE_METHOD, "temperature": calibrator.temperature_,
                              "label_preserving": True, "target_class_prior": [1 / 3] * 3,
                              "weighting": "equal_class_then_equal_trial_then_equal_window",
                              "fit_scope": "calibration_groups_only",
                              "temperature_bounds": [calibrator.min_temperature, calibrator.max_temperature]},
                          "libraries": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}},
             "input": input_contract(), "source": dataset["metadata"]["source"], "label_order": list(LABELS),
             "pipeline": pipeline, "calibrator": calibrator, "metrics": report,
             "confidence_method": CONFIDENCE_METHOD,
             "confidence_note": "Positive temperature fitted on separate class/trial-balanced calibration data preserves the classifier label; small-sample confidence is not a guarantee",
             "min_confidence": .55, "min_margin": .10,
             "ood_centers": centers, "ood_radius": max(4., float(np.quantile(residual, .99)) * 3)}
    validate_model(model)
    return model


def validate_model(model, source_kind=None):
    if not isinstance(model, dict) or model.get("format") != "eeg-hat-eight-channel" or model.get("version") != VERSION:
        raise ValueError("模型处理版本不兼容；请用当前版本重新训练八通道模型。原录制文件仍可使用。")
    if model.get("input") != input_contract() or model.get("label_order") != list(LABELS):
        raise ValueError("Model channel order, units, filter, sample rate, epoch or label contract is incompatible")
    if model.get("source") not in ("device", "demo") or (source_kind is not None and model["source"] != source_kind):
        raise ValueError("Demo and device models cannot be interchanged")
    metadata = model.get("metadata", {})
    if metadata.get("source") != model["source"] or metadata.get("input") != model["input"] or metadata.get("feature") not in ("bandpower", "csp", "combined") or metadata.get("classifier") not in ("lda", "svm"):
        raise ValueError("Missing or inconsistent model metadata")
    if metadata.get("feature_spec") != _feature_spec(metadata["feature"]):
        raise ValueError("Model feature extraction specification is incompatible; retrain with the current version")
    trained_sklearn = metadata.get("libraries", {}).get("sklearn")
    if not isinstance(trained_sklearn, str) or trained_sklearn.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise ValueError("Model sklearn version differs from this runtime; retrain or load with its original environment")
    if not isinstance(model.get("pipeline"), Pipeline) or not isinstance(model.get("calibrator"), TemperatureCalibrator):
        raise ValueError("Unsupported estimator or confidence calibrator")
    calibrator = model["calibrator"]
    temperature = getattr(calibrator, "temperature_", None)
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Temperature must be positive and finite")
    calibration_spec = metadata.get("confidence_calibration", {})
    if (calibration_spec.get("method") != CONFIDENCE_METHOD
            or calibration_spec.get("temperature") != temperature
            or calibration_spec.get("label_preserving") is not True
            or calibration_spec.get("target_class_prior") != [1 / 3] * 3
            or calibration_spec.get("weighting") != "equal_class_then_equal_trial_then_equal_window"
            or calibration_spec.get("fit_scope") != "calibration_groups_only"
            or calibration_spec.get("temperature_bounds") != [calibrator.min_temperature, calibrator.max_temperature]
            or not np.isfinite([calibrator.min_temperature, calibrator.max_temperature]).all()
            or not 0 < calibrator.min_temperature <= temperature <= calibrator.max_temperature):
        raise ValueError("Missing or inconsistent label-preserving calibration metadata")
    steps = model["pipeline"].named_steps
    expected_width = FEATURE_WIDTHS[metadata["feature"]]
    classifier_type = LinearDiscriminantAnalysis if metadata["classifier"] == "lda" else SVC
    if (list(steps) != ["features", "scaler", "classifier"] or not isinstance(steps["features"], HatFeatures)
            or steps["features"].kind != metadata["feature"] or not isinstance(steps["scaler"], StandardScaler)
            or getattr(steps["features"], "feature_version_", None) != FEATURE_VERSION
            or not isinstance(steps["classifier"], classifier_type)):
        raise ValueError("Estimator pipeline disagrees with model metadata")
    if metadata["classifier"] == "svm" and (steps["classifier"].decision_function_shape != "ovr" or not steps["classifier"].break_ties):
        raise ValueError("SVM prediction must use the same tie-breaking as its calibrated scores")
    scaler = steps["scaler"]
    if (np.shape(getattr(scaler, "mean_", None)) != (expected_width,) or
            np.shape(getattr(scaler, "scale_", None)) != (expected_width,) or
            not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_ <= 0)):
        raise ValueError("Invalid fitted standardizer")
    if not np.array_equal(getattr(model["pipeline"], "classes_", None), np.arange(3)) or not np.array_equal(getattr(model["calibrator"], "classes_", None), np.arange(3)):
        raise ValueError("Model class IDs do not match rest/left_mi/right_mi")
    if model.get("confidence_method") != CONFIDENCE_METHOD:
        raise ValueError("Unknown confidence semantics")
    for key in ("min_confidence", "min_margin"):
        value = model.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid rejection thresholds")
    centers = np.asarray(model.get("ood_centers"), dtype=float)
    if centers.shape != (3, expected_width) or not np.isfinite(centers).all():
        raise ValueError("Invalid training distribution metadata")
    if not isinstance(model.get("ood_radius"), (int, float)) or not np.isfinite(model["ood_radius"]) or model["ood_radius"] <= 0:
        raise ValueError("Invalid distribution rejection threshold")
    metrics = model.get("metrics", {})
    if type(metrics.get("usable")) is not bool or metrics.get("deployment_refit") is not False:
        raise ValueError("Missing grouped validation evidence")
    split_sets = []
    for split in ("train", "calibration", "test"):
        values = metrics.get(split + "_groups")
        if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
            raise ValueError("Missing split provenance")
        split_sets.append(set(values))
    if any(split_sets[a] & split_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Model split provenance contains overlapping trials")
    if (getattr(calibrator, "fit_groups_", None) != sorted(split_sets[1])
            or getattr(calibrator, "fit_trials_per_class_", None) != metrics.get("calibration_trials_per_class")
            or getattr(calibrator, "fit_windows_", None) != metrics.get("calibration_windows")):
        raise ValueError("Temperature calibration provenance does not match the held-out groups")
    return True


def predict_epoch(model, epoch, source_kind=None):
    validate_model(model, source_kind=source_kind)
    result = {"label": None, "confidence": None, "probabilities": [], "label_order": list(LABELS),
              "status": "invalid", "reason": "Invalid filtered epoch", "confidence_method": model["confidence_method"], "accepted": False}
    x = np.asarray(epoch, dtype=float)
    if x.shape != (8, WINDOW) or not np.isfinite(x).all():
        return result
    pipeline = model["pipeline"]
    z = pipeline.named_steps["scaler"].transform(pipeline.named_steps["features"].transform(x[None]))[0]
    if not np.isfinite(z).all():
        return result
    distance = float(np.min(np.sqrt(np.mean((z - model["ood_centers"]) ** 2, axis=1))))
    outside = bool(np.any(np.abs(z) > 12) or distance > model["ood_radius"])
    p = model["calibrator"].predict_proba(_scores(pipeline, x[None]))[0]
    if not np.isfinite(p).all():
        return result
    top = int(np.argmax(p)); confidence = float(p[top]); ordered = np.sort(p)
    result.update(status="uncertain", probabilities=p.tolist(), confidence=confidence,
                  label=LABELS[top], candidate_label=LABELS[top], reason="Confidence is below the fixed rejection threshold")
    if outside:
        result["reason"] = "Epoch is outside the training feature distribution; tentative result only"
    elif not model["metrics"]["usable"]:
        result["reason"] = "Model did not pass preliminary grouped holdout validation"
    elif confidence >= model["min_confidence"] and float(ordered[-1] - ordered[-2]) >= model["min_margin"]:
        result.update(status="predicted", accepted=True, reason="Valid personal model prediction")
    return result


def save_model(model, path):
    validate_model(model)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous model packages.
    with target.open("xb") as handle:
        joblib.dump(model, handle)
    return target


def load_model(path, source_kind=None):
    # Joblib is only for locally trained/trusted model files, never network uploads.
    model = joblib.load(Path(path))
    validate_model(model, source_kind=source_kind)
    return model
