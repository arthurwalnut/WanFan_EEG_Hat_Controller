"""Personal time-locked template baseline with whole-trial holdouts.

No MI features, neural-network weights, or paper performance claims are reused.
Correlation scores and margin are not probabilities. This closed-set baseline
has no validated no-attention detector; accepted means calibration thresholds
passed, never proof of intention or permission to move a physical actuator.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .contract import (FS, CHANNELS, SELECTED, TARGETS, WINDOW, HOP, VERSION,
                       FRAMES_PER_CYCLE, SAMPLES_PER_CYCLE, get_contract,
                       validate_target_count, target_count_from_contract)
from .processing import as_samples, ContinuousPreprocessor, frame_phase, raw_quality


def _normalize(x):
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=0, keepdims=True)
    return centered / np.maximum(np.sqrt(np.mean(centered ** 2, axis=0, keepdims=True)), 1e-10)


def _phase_template(samples, phase):
    p = np.asarray(phase) % SAMPLES_PER_CYCLE
    left = np.floor(p).astype(int)
    fraction = p - left
    sums = np.zeros((SAMPLES_PER_CYCLE, len(SELECTED)))
    counts = np.zeros(SAMPLES_PER_CYCLE)
    for index, weight in ((left, 1 - fraction), ((left + 1) % SAMPLES_PER_CYCLE, fraction)):
        np.add.at(sums, index, samples * weight[:, None])
        np.add.at(counts, index, weight)
    good = counts > 1e-8
    if good.mean() < .95:
        raise ValueError("insufficient complete code-cycle phase coverage")
    out = sums / np.maximum(counts[:, None], 1e-10)
    for ch in range(len(SELECTED)):
        out[~good, ch] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), out[good, ch], period=SAMPLES_PER_CYCLE)
    return _normalize(out)


def _correlations(templates, samples, phase):
    x = _normalize(samples)
    p = np.asarray(phase) % SAMPLES_PER_CYCLE
    left = np.floor(p).astype(int)
    fraction = p - left
    scores = []
    for template in templates:
        reference = template[left] * (1 - fraction[:, None]) + template[(left + 1) % SAMPLES_PER_CYCLE] * fraction[:, None]
        scores.append(float(np.mean(x * _normalize(reference))))
    return np.clip(scores, -1, 1)


def _session_id(session, index):
    return str(session.get("metadata", {}).get("id", f"session_{index}"))


def build_dataset(sessions):
    """Validate timing first, continuously filter each segment, then cut epochs.

    Per-trial templates use each post-warmup raw sample once, not overlapping
    windows. All derived windows retain their parent session/trial identity.
    """
    if not isinstance(sessions, (list, tuple)) or not sessions:
        raise ValueError("At least one c-VEP recording is required")
    trials, rejected, sources, seen, subjects = [], [], set(), set(), set()
    unknown_subject = False
    target_count = None
    for si, session in enumerate(sessions):
        meta = session.get("metadata", {})
        subject = meta.get("subject", "")
        if not isinstance(subject, str) or len(subject) > 80 or any(ord(c) < 32 for c in subject):
            raise ValueError("Invalid subject identifier")
        subject = subject.strip()
        if subject:
            subjects.add(subject)
        else:
            unknown_subject = True
        if len(subjects) > 1:
            raise ValueError("Personal c-VEP calibration cannot mix different subject identifiers")
        if meta.get("kind", "calibration") != "calibration":
            raise ValueError("Only labeled calibration recordings can train a personal c-VEP model")
        source = meta.get("source")
        if source not in ("device", "demo"):
            raise ValueError("Recording source must be device or demo")
        sources.add(source)
        count = target_count_from_contract(meta.get("contract"))
        if "target_count" in meta and validate_target_count(meta["target_count"]) != count:
            raise ValueError("Recording target_count differs from its contract")
        if target_count is not None and count != target_count:
            raise ValueError("Different c-VEP target counts cannot be mixed")
        target_count = count
        if meta.get("channel_names", list(CHANNELS)) != list(CHANNELS):
            raise ValueError("Recording electrode order mismatch")
        x = as_samples(session["samples"])
        t = np.asarray(session["times"], dtype=float)
        indices = np.asarray(session["indices"])
        segments = np.asarray(session["segments"])
        if any(a.shape != (len(x),) for a in (t, indices, segments)) or not np.isfinite(t).all():
            raise ValueError("Malformed recording sample coordinates")
        if len(x) < WINDOW:
            continue
        if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(segments.dtype, np.integer):
            raise ValueError("Sample indices and segments must be integers")
        boundaries = np.flatnonzero((np.diff(indices) != 1) | (np.diff(segments) != 0) |
                                   (np.diff(t) <= 0) | (np.diff(t) > .02)) + 1
        filtered = np.empty_like(x)
        run = np.zeros(len(x), dtype=int)
        start = 0
        for run_id, end in enumerate([*boundaries, len(x)]):
            # Hard faults are rejected below; substitutions prevent propagation
            # into otherwise finite channels. No cross-channel rereferencing.
            safe = np.clip(np.nan_to_num(x[start:end], nan=0., posinf=375000., neginf=-375000.), -375000., 375000.)
            filtered[start:end] = ContinuousPreprocessor().process(safe)
            run[start:end] = run_id
            start = int(end)
        sid = _session_id(session, si)
        for trial in session.get("trials", []):
            tid = trial.get("trial_id")
            key = f"{sid}:{tid}"
            if key in seen:
                raise ValueError("Duplicate recording/trial identity: " + key)
            seen.add(key)
            try:
                if not trial.get("valid", False):
                    quality_detail = (trial.get("quality") or {}).get("reason", "")
                    raise ValueError("recording trial invalid (timing/quality): " + str(trial.get("reasons", [])) +
                                     ("; " + quality_detail if quality_detail else ""))
                target = trial["target"]
                if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)) or target not in range(target_count):
                    raise ValueError("invalid target label")
                frames = [dict(f) for f in session.get("frames", []) if f.get("trial_id") == tid]
                if not frames or frames[0].get("code_frame") != 0 or frames[-1].get("code_frame") not in (377, 378):
                    raise ValueError("incomplete stimulus frame sequence")
                if frames[-1]["code_frame"] == 377:
                    frames.append({"trial_id": tid, "code_frame": 378, "observed_s": float(trial["tail_s"])})
                if len(frames) != 379:
                    raise ValueError("task does not have 378 consecutive displayed frames")
                onset, tail = float(frames[0]["observed_s"]), float(frames[-1]["observed_s"])
                full = np.flatnonzero((t >= onset) & (t < tail))
                if len(full) < int(.95 * FS * (tail - onset)) or len(full) < 1400:
                    raise ValueError("incomplete EEG task coverage")
                if t[full[0]] - onset > .02 or tail - t[full[-1]] > .02 or len(set(run[full])) != 1:
                    raise ValueError("EEG task has a boundary gap or discontinuity")
                quality = raw_quality(x[full])
                if not quality["valid"]:
                    raise ValueError(quality["reason"])
                keep = full[t[full] >= float(frames[126]["observed_s"])]
                phase = frame_phase(t[keep], frames)
                y = filtered[keep][:, SELECTED]
                template = _phase_template(y, phase)
                windows = [{"samples": y[j:j + WINDOW].copy(), "phase": phase[j:j + WINDOW].copy(),
                            "start_index": int(indices[keep[j]]), "end_index": int(indices[keep[j + WINDOW - 1]])}
                           for j in range(0, len(keep) - WINDOW + 1, HOP)]
                if not windows:
                    raise ValueError("no post-warmup full epochs")
                created = str(meta.get("created_at", meta.get("created_utc", meta.get("start_utc", sid))))
                trials.append({"key": key, "session_id": sid, "trial_id": tid, "target": int(target),
                               "sort_key": (created, sid, onset, str(tid)),
                               "template": template, "windows": windows, "quality": quality})
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append({"key": key, "target": trial.get("target"), "reason": str(exc)})
    if len(sources) != 1:
        raise ValueError("Demo and device recordings cannot be mixed")
    return {"trials": trials, "source": next(iter(sources)), "rejected": rejected,
            "target_count": target_count, "contract": get_contract(target_count),
            "subject": next(iter(subjects), None), "subject_verified": bool(subjects) and not unknown_subject,
            "accepted_trials": len(trials), "accepted_windows": sum(len(t["windows"]) for t in trials)}


def _split(trials, target_count=6):
    target_count = validate_target_count(target_count)
    result = {"train": [], "calibration": [], "test": []}
    for target in range(target_count):
        rows = sorted((t for t in trials if t["target"] == target),
                      key=lambda row: row.get("sort_key", (row["session_id"], row["session_id"], 0., str(row["trial_id"]))))
        if len(rows) < 6:
            raise ValueError(f"Target {target} needs at least 6 complete valid trials; got {len(rows)}")
        # Acquisition timestamps (stable identity fallback) determine order;
        # selecting the same sessions in reverse cannot change the holdout.
        # Entire parent trials stay intact, including all overlapping windows.
        holdout = max(1, len(rows) // 5)
        result["train"].extend(rows[:-2 * holdout])
        result["calibration"].extend(rows[-2 * holdout:-holdout])
        result["test"].extend(rows[-holdout:])
    return result


def _evaluate(templates, trials):
    count = validate_target_count(len(templates))
    confusion = np.zeros((count, count), dtype=int)
    trial_confusion = np.zeros((count, count), dtype=int)
    details = []
    for trial in trials:
        scores = np.asarray([_correlations(templates, w["samples"], w["phase"]) for w in trial["windows"]])
        predicted = scores.argmax(axis=1)
        for candidate in predicted:
            confusion[trial["target"], candidate] += 1
        trial_prediction = int(scores.mean(axis=0).argmax())
        trial_confusion[trial["target"], trial_prediction] += 1
        for row, candidate in zip(scores, predicted):
            ordered = np.sort(row)
            details.append({"key": trial["key"], "target": trial["target"], "candidate": int(candidate),
                            "correlation": float(row[candidate]), "margin": float(ordered[-1] - ordered[-2])})
    per_class = np.diag(confusion) / np.maximum(1, confusion.sum(axis=1))
    return {"window_accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
            "balanced_accuracy": float(per_class.mean()), "confusion_matrix": confusion.tolist(),
            "trial_accuracy": float(np.trace(trial_confusion) / max(1, trial_confusion.sum())),
            "trial_confusion_matrix": trial_confusion.tolist(), "window_count": int(confusion.sum()),
            "trial_count": len(trials), "target_count": count, "label_order": list(range(count))}, details


def _fit_thresholds(trials, rows, target_count=6):
    target_count = validate_target_count(target_count)
    correct, represented = [], set()
    for trial in trials:
        values = [r for r in rows if r["key"] == trial["key"] and r["candidate"] == r["target"]]
        if values:
            correct.append([float(np.median([r[name] for r in values])) for name in ("correlation", "margin")])
            represented.add(trial["target"])
    learned = len(correct) >= target_count and represented == set(range(target_count))
    if learned:
        thresholds = {"correlation": max(.05, float(np.quantile(np.asarray(correct)[:, 0], .1)) * .5),
                      "margin": max(.01, float(np.quantile(np.asarray(correct)[:, 1], .1)) * .5)}
        method = "heldout correct-trial median quantile with fixed floors; heuristic, not probability calibration"
        warnings = []
    else:
        thresholds = {"correlation": .2, "margin": .05}
        method = "fixed heuristic fallback (correlation=0.2, margin=0.05); not learned from calibration"
        warnings = ["Too few correct calibration trials or missing correctly decoded target; fixed thresholds have no calibrated reliability guarantee"]
    return thresholds, {"threshold_method": method, "threshold_trials": len(correct),
                        "threshold_targets": sorted(represented), "threshold_fallback": not learned,
                        "warnings": warnings}


def _acceptance_metrics(rows, thresholds):
    accepted = [r for r in rows if r["correlation"] >= thresholds["correlation"] and r["margin"] >= thresholds["margin"]]
    coverage = len(accepted) / max(1, len(rows))
    return {"coverage": float(coverage), "rejection_rate": float(1 - coverage),
            "accepted_windows": len(accepted), "rejected_windows": len(rows) - len(accepted),
            "accepted_accuracy": float(np.mean([r["candidate"] == r["target"] for r in accepted])) if accepted else None}


def train_model(sessions):
    dataset = build_dataset(sessions)
    count = dataset["target_count"]
    try:
        split = _split(dataset["trials"], count)
    except ValueError as exc:
        exc.diagnostics = {"accepted_trials": dataset["accepted_trials"],
                           "target_count": count,
                           "per_target_valid_trials": {str(k): sum(t["target"] == k for t in dataset["trials"]) for k in range(count)},
                           "rejected": dataset["rejected"]}
        raise
    templates = np.asarray([_normalize(np.mean([t["template"] for t in split["train"] if t["target"] == k], axis=0)) for k in range(count)])
    calibration_metrics, rows = _evaluate(templates, split["calibration"])
    # Equal trial influence: each trial contributes one median among its correct
    # calibration windows. There are no test-derived thresholds or lag searches.
    thresholds, threshold_report = _fit_thresholds(split["calibration"], rows, count)
    calibration_metrics.update(_acceptance_metrics(rows, thresholds))
    training_metrics, train_rows = _evaluate(templates, split["train"])
    training_metrics.update(_acceptance_metrics(train_rows, thresholds))
    test_metrics, test_rows = _evaluate(templates, split["test"])
    test_metrics.update(_acceptance_metrics(test_rows, thresholds))
    test_metrics.update(**threshold_report)
    if not dataset["subject_verified"]:
        test_metrics["warnings"].append("Some recordings have no subject identifier; same-person provenance is unverified")
    test_metrics.update(calibration=calibration_metrics, training=training_metrics, evaluation_split="test",
                        split_isolation="whole trials; sessions may occur in multiple splits",
                        interpretation="overlapping-window accuracy is descriptive; independent units are trials; no cross-person claim",
                        no_attention_validated=False)
    model = {"kind": "cvep_personal_template", "version": VERSION, "source": dataset["source"],
             "contract": get_contract(count), "target_count": count, "templates": templates.tolist(), "thresholds": thresholds,
             "metrics": test_metrics, "split": {name: [t["key"] for t in values] for name, values in split.items()},
             "metadata": {"score_type": "correlation_not_probability", "personal_calibration": True,
                          "subject": dataset["subject"], "subject_verified": dataset["subject_verified"],
                          "template_fit_trials": [t["key"] for t in split["train"]],
                          "threshold_fit_trials": [t["key"] for t in split["calibration"]],
                          "rejected_trials": dataset["rejected"], "accepted_trials": dataset["accepted_trials"]}}
    validate_model(model)
    return model


def validate_model(model, source_kind=None, target_count=None):
    if not isinstance(model, dict) or model.get("kind") != "cvep_personal_template" or model.get("version") != VERSION:
        raise ValueError("This is not a supported c-VEP model; MI models are incompatible")
    count = target_count_from_contract(model.get("contract"))
    if "target_count" in model and validate_target_count(model["target_count"]) != count:
        raise ValueError("c-VEP model target_count differs from its contract")
    if target_count is not None and validate_target_count(target_count) != count:
        raise ValueError("c-VEP model target count mismatch")
    if model.get("source") not in ("demo", "device") or (source_kind is not None and model["source"] != source_kind):
        raise ValueError("c-VEP model source mismatch")
    template = np.asarray(model.get("templates"), dtype=float)
    if template.shape != (count, SAMPLES_PER_CYCLE, len(SELECTED)) or not np.isfinite(template).all():
        raise ValueError("Invalid c-VEP template shape or values")
    if np.any(np.std(template, axis=1) < 1e-8):
        raise ValueError("Degenerate c-VEP templates")
    threshold = model.get("thresholds", {})
    for key, limit in (("correlation", 1.), ("margin", 2.)):
        value = threshold.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= limit:
            raise ValueError("Invalid c-VEP rejection threshold")
    split = model.get("split", {})
    groups = [set(split.get(name, [])) for name in ("train", "calibration", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Model split contains trial leakage")
    return True


def predict_epoch(model, samples, times, frames, *, source_kind=None, raw_samples=None, target_count=None):
    empty = {"label": None, "candidate_target": None, "accepted": False,
             "correlations": [], "scores": [], "margin": None, "confidence": None,
             "score_type": "correlation_not_probability", "label_order": list(TARGETS)}
    try:
        if target_count is not None:
            empty["label_order"] = list(range(validate_target_count(target_count)))
        elif isinstance(model, dict) and "contract" in model:
            empty["label_order"] = list(range(target_count_from_contract(model["contract"])))
        if model is None:
            return {**empty, "status": "no_model", "reason": "No personal c-VEP model loaded"}
        validate_model(model, source_kind=source_kind, target_count=target_count)
        empty["label_order"] = list(range(target_count_from_contract(model["contract"])))
        x = as_samples(samples)
        if x.shape != (WINDOW, 8) or not np.isfinite(x[:, SELECTED]).all():
            raise ValueError("c-VEP epoch must contain 375 finite samples")
        if raw_samples is None:
            raise ValueError("Raw epoch is required for quality checks")
        raw = as_samples(raw_samples)
        if raw.shape != x.shape:
            raise ValueError("Raw/filtered epoch shape mismatch")
        quality = raw_quality(raw)
        if not quality["valid"]:
            return {**empty, "status": "invalid", "reason": quality["reason"], "quality": quality}
        t = np.asarray(times, dtype=float)
        if len(t) != WINDOW or np.any(np.diff(t) > .02):
            raise ValueError("Epoch sample timing discontinuity")
        phase = frame_phase(t, frames)
        if np.any(np.std(x[:, SELECTED], axis=0) < 1e-8):
            raise ValueError("Filtered epoch is degenerate")
        scores = _correlations(np.asarray(model["templates"]), x[:, SELECTED], phase)
        candidate = int(scores.argmax())
        ranked = np.sort(scores)
        margin = float(ranked[-1] - ranked[-2])
        reasons = []
        if scores[candidate] < model["thresholds"]["correlation"]:
            reasons.append("low template correlation")
        if margin < model["thresholds"]["margin"]:
            reasons.append("ambiguous target margin")
        accepted = not reasons
        return {**empty, "label": candidate if accepted else None, "candidate_target": candidate,
                "accepted": accepted, "correlations": scores.tolist(), "scores": scores.tolist(),
                "margin": margin, "status": "predicted" if accepted else "uncertain",
                "reason": "; ".join(reasons) or "template thresholds passed; no-attention state not validated",
                "quality": quality}
    except (ValueError, TypeError, KeyError) as exc:
        return {**empty, "status": "invalid", "reason": str(exc)}


def save_model(model, path):
    validate_model(model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_model(path, source_kind=None, target_count=None):
    path = Path(path)
    if path.stat().st_size > 5_000_000:
        raise ValueError("c-VEP model exceeds supported JSON size")
    def reject_constant(value):
        raise ValueError("Nonfinite JSON value in c-VEP model: " + value)
    model = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    validate_model(model, source_kind=source_kind, target_count=target_count)
    return model
