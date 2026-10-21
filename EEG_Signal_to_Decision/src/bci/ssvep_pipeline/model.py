"""FBCCA reference decoder plus whole-trial threshold calibration/evaluation.

No supervised classifier or subject template is fitted. Only the rejection
thresholds use calibration labels. Test trials never determine parameters.
Scores are squared canonical correlations, not probabilities of correctness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contract import (BANDS, CHANNELS, FREQUENCIES, FS, HOP, KIND, SELECTED, TARGETS,
                       TASK_SECONDS, VERSION, WARMUP_SECONDS, WINDOW, filter_sos,
                       get_contract, validate_contract, validate_target_count)
from .processing import ContinuousPreprocessor, as_samples, fbcca_scores, raw_quality


def _valid_times(times, count):
    t = np.asarray(times, dtype=float)
    if t.shape != (count,) or not np.isfinite(t).all() or (count > 1 and
            np.any(np.abs(np.diff(t) - 1 / FS) > .25 / FS + 1e-9)):
        raise ValueError("SSVEP window timing must be continuous at 250Hz")
    return t


def build_dataset(sessions):
    if not isinstance(sessions, (list, tuple)) or not sessions:
        raise ValueError("At least one labeled SSVEP calibration recording is required")
    rows, rejected, seen, sources, subjects = [], [], set(), set(), set()
    unidentified = False
    for session in sessions:
        meta = session["metadata"]
        validate_contract(meta.get("contract"))
        if meta.get("kind") != "calibration":
            raise ValueError("Only labeled SSVEP calibration recordings can be evaluated")
        source = meta.get("source")
        if source not in ("device", "demo"):
            raise ValueError("SSVEP source must be device or demo")
        sources.add(source)
        if len(sources) != 1:
            raise ValueError("Demo and device recordings cannot be mixed")
        subject = meta.get("subject", "")
        if not isinstance(subject, str) or len(subject) > 80 or any(ord(c) < 32 for c in subject):
            raise ValueError("Invalid subject identifier")
        subject = subject.strip()
        if subject:
            subjects.add(subject)
        else:
            unidentified = True
        if len(subjects) > 1:
            raise ValueError("Different subject identities cannot be mixed")
        sid = meta.get("id")
        if not isinstance(sid, str) or not sid or ":" in sid:
            raise ValueError("A stable recording id is required for trial grouping")
        if meta.get("channel_names") != list(CHANNELS):
            raise ValueError("SSVEP channel order mismatch")
        if "target_count" in meta:
            validate_target_count(meta["target_count"])
        x = as_samples(session["samples"])
        t = np.asarray(session["times"], dtype=float)
        indices, segments = np.asarray(session["indices"]), np.asarray(session["segments"])
        if any(a.shape != (len(x),) for a in (t, indices, segments)) or not np.isfinite(t).all():
            raise ValueError("Malformed recording sample coordinates")
        if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(segments.dtype, np.integer):
            raise ValueError("Sample indices and segments must be integers")
        boundaries = np.flatnonzero((np.diff(indices) != 1) | (np.diff(segments) != 0) |
                                    (np.abs(np.diff(t) - 1 / FS) > .25 / FS + 1e-9)) + 1
        bank = np.empty((len(x), BANDS, len(CHANNELS)))
        run = np.zeros(len(x), dtype=int)
        first = 0
        for run_id, end in enumerate([*boundaries, len(x)]):
            # Only hard-good trials are retained below. Corrupt raw values must
            # not contaminate otherwise independent channels or future runs.
            safe = np.clip(np.nan_to_num(x[first:end], nan=0., posinf=375000., neginf=-375000.), -375000., 375000.)
            bank[first:end] = ContinuousPreprocessor().process(safe)
            run[first:end] = run_id
            first = int(end)
        for trial in session.get("trials", []):
            tid = trial.get("trial_id")
            if isinstance(tid, bool) or not isinstance(tid, (int, np.integer)) or tid < 1:
                raise ValueError("Invalid trial identity")
            key = f"{sid}:{tid}"
            if key in seen:
                raise ValueError("Duplicate recording/trial identity: " + key)
            seen.add(key)
            try:
                if trial.get("valid") is not True:
                    raise ValueError("recording trial invalid: " + str(trial.get("reasons", [])))
                target = trial["target"]
                if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)) or target not in TARGETS:
                    raise ValueError("Invalid frequency target label")
                onset, tail = float(trial["onset_s"]), float(trial["tail_s"])
                if not np.isfinite([onset, tail]).all() or abs(tail - onset - TASK_SECONDS) > .25:
                    raise ValueError("Incomplete six-second stimulus trial")
                full = np.flatnonzero((t >= onset) & (t < tail))
                if (len(full) < int(.95 * FS * (tail - onset)) or not len(full) or
                        t[full[0]] > onset + 1.25 / FS or t[full[-1]] < tail - 1.25 / FS or
                        len(np.unique(run[full])) != 1):
                    raise ValueError("Incomplete EEG trial coverage or sample discontinuity")
                quality = raw_quality(x[full])
                if not quality["valid"]:
                    raise ValueError(quality["reason"])
                keep = full[t[full] >= onset + WARMUP_SECONDS - 1e-9]
                windows = []
                for j in range(0, len(keep) - WINDOW + 1, HOP):
                    positions = keep[j:j + WINDOW]
                    score, _ = fbcca_scores(bank[positions])
                    windows.append(dict(scores=score.tolist(), start_index=int(indices[positions[0]]),
                                        end_index=int(indices[positions[-1]])))
                if not windows:
                    raise ValueError("No complete post-onset SSVEP windows")
                created = str(meta.get("created_utc", meta.get("created_at", sid)))
                rows.append(dict(key=key, session_id=sid, trial_id=int(tid), target=int(target),
                                 sort_key=(created, sid, onset, int(tid)), windows=windows, quality=quality))
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append(dict(key=key, target=trial.get("target"), reason=str(exc)))
    return dict(trials=rows, rejected=rejected, accepted_trials=len(rows),
                accepted_windows=sum(len(r["windows"]) for r in rows), target_count=4,
                source=next(iter(sources)), subject=next(iter(subjects), ""),
                subject_verified=bool(subjects) and not unidentified)


def _split(trials):
    result = dict(train=[], calibration=[], test=[])
    for target in TARGETS:
        rows = sorted((row for row in trials if row["target"] == target), key=lambda row: row["sort_key"])
        if len(rows) < 6:
            raise ValueError(f"Target {target} needs at least 6 complete valid trials; got {len(rows)}")
        test_count = max(1, len(rows) // 3)
        result["calibration"].extend(rows[:-test_count])
        result["test"].extend(rows[-test_count:])
    return result


def _evaluate(trials, thresholds=None):
    confusion = np.zeros((4, 4), dtype=int)
    trial_confusion = np.zeros((4, 4), dtype=int)
    details = []
    for trial in trials:
        scores = np.asarray([w["scores"] for w in trial["windows"]])
        candidates = scores.argmax(axis=1)
        for candidate, score in zip(candidates, scores):
            confusion[trial["target"], candidate] += 1
            ordered = np.sort(score)
            details.append(dict(key=trial["key"], target=trial["target"], candidate=int(candidate),
                                score=float(score[candidate]), margin=float(ordered[-1] - ordered[-2])))
        trial_confusion[trial["target"], scores.mean(axis=0).argmax()] += 1
    metrics = dict(window_accuracy=float(np.trace(confusion) / max(1, confusion.sum())),
                   balanced_accuracy=float(np.mean(np.diag(confusion) / np.maximum(1, confusion.sum(axis=1)))),
                   trial_accuracy=float(np.trace(trial_confusion) / max(1, trial_confusion.sum())),
                   confusion_matrix=confusion.tolist(), trial_confusion_matrix=trial_confusion.tolist(),
                   window_count=int(confusion.sum()), trial_count=len(trials), label_order=list(TARGETS),
                   frequencies=list(FREQUENCIES), target_count=4)
    if thresholds is not None:
        accepted = [r for r in details if r["score"] >= thresholds["score"] and r["margin"] >= thresholds["margin"]]
        coverage = len(accepted) / max(1, len(details))
        metrics.update(coverage=float(coverage), rejection_rate=float(1 - coverage),
                       accepted_windows=len(accepted), rejected_windows=len(details) - len(accepted),
                       accepted_accuracy=float(np.mean([r["candidate"] == r["target"] for r in accepted])) if accepted else None)
    return metrics, details


def _fit_thresholds(trials, rows):
    values, targets = [], set()
    for trial in trials:
        correct = [r for r in rows if r["key"] == trial["key"] and r["candidate"] == r["target"]]
        if correct:
            values.append([float(np.median([r[name] for r in correct])) for name in ("score", "margin")])
            targets.add(trial["target"])
    learned = len(values) >= 4 and targets == set(TARGETS)
    thresholds = dict(score=.2, margin=.03)
    if learned:
        thresholds = dict(score=max(.1, float(np.quantile(np.asarray(values)[:, 0], .1)) * .5),
                          margin=max(.01, float(np.quantile(np.asarray(values)[:, 1], .1)) * .5))
    return thresholds, dict(threshold_fallback=not learned, threshold_trials=len(values),
        threshold_targets=sorted(targets),
        threshold_method=("correct-calibration-trial median quantile with fixed floors; heuristic, not probability calibration"
                          if learned else "fixed heuristic fallback score=0.2 margin=0.03; insufficient correct calibration targets"),
        warnings=[] if learned else ["Rejection thresholds have insufficient calibration evidence; fixed fallback used"])


def train_model(sessions):
    """Calibrate rejection and report held-out evaluation; fit no classifier."""
    dataset = build_dataset(sessions)
    try:
        split = _split(dataset["trials"])
    except ValueError as exc:
        exc.diagnostics = dict(accepted_trials=dataset["accepted_trials"], accepted_windows=dataset["accepted_windows"],
            per_target_valid_trials={str(k): sum(t["target"] == k for t in dataset["trials"]) for k in TARGETS},
            rejected=dataset["rejected"])
        raise
    _, rows = _evaluate(split["calibration"])
    thresholds, threshold_report = _fit_thresholds(split["calibration"], rows)
    calibration, _ = _evaluate(split["calibration"], thresholds)
    metrics, _ = _evaluate(split["test"], thresholds)
    metrics.update(**threshold_report, calibration=calibration,
        training=dict(not_applicable=True, trial_count=0, window_count=0, reason="FBCCA has no supervised classifier fit"),
        evaluation_split="test", split_isolation="whole trials; sessions may occur in multiple splits",
        classifier_fit="none; analytical sine/cosine references and fixed filter-bank/weights", training_free=True,
        interpretation="overlapping windows are descriptive; independent units are trials; no cross-person claim",
        no_attention_validated=False)
    if not dataset["subject_verified"]:
        metrics["warnings"].append("Some recordings have no subject identity; same-person provenance is unverified")
    model = dict(kind=KIND, version=VERSION, source=dataset["source"], contract=get_contract(), target_count=4,
        filter_sos=[s.tolist() for s in filter_sos()], thresholds=thresholds, metrics=metrics,
        split={name: [t["key"] for t in rows] for name, rows in split.items()},
        metadata=dict(subject=dataset["subject"], subject_verified=dataset["subject_verified"],
                      score_type="normalized_fbcca_score_not_probability", classifier_fit="none",
                      threshold_fit_trials=[r["key"] for r in split["calibration"]],
                      accepted_trials=dataset["accepted_trials"], rejected_trials=dataset["rejected"],
                      references=["https://doi.org/10.1088/1741-2560/12/4/046008",
                                  "https://github.com/mnakanishi/TRCA-SSVEP/blob/master/src/filterbank.m"],
                      adaptation="4 posterior Hat channels; 5 causal Chebyshev banks; 5 harmonics; normalized squared-CCA score"))
    validate_model(model)
    return model


def validate_model(model, source_kind=None, target_count=None):
    if not isinstance(model, dict) or model.get("kind") != KIND or type(model.get("version")) is not int or model["version"] != VERSION:
        raise ValueError("Not a supported SSVEP FBCCA model")
    validate_contract(model.get("contract"))
    validate_target_count(model.get("target_count"))
    if target_count is not None:
        validate_target_count(target_count)
    if model.get("source") not in ("device", "demo") or (source_kind is not None and source_kind != model["source"]):
        raise ValueError("SSVEP model source mismatch")
    stored = model.get("filter_sos")
    expected = filter_sos()
    if not isinstance(stored, list) or len(stored) != BANDS:
        raise ValueError("SSVEP filter-bank coefficients missing")
    for coefficients, reference in zip(stored, expected):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != reference.shape or not np.isfinite(coefficients).all() or not np.allclose(coefficients, reference, rtol=1e-10, atol=1e-12):
            raise ValueError("SSVEP filter coefficients differ from the fixed contract")
    thresholds = model.get("thresholds", {})
    for name in ("score", "margin"):
        value = thresholds.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid SSVEP rejection threshold")
    split = model.get("split")
    if not isinstance(split, dict) or set(split) != {"train", "calibration", "test"} or split["train"] != []:
        raise ValueError("FBCCA must not claim a supervised training split")
    for name in ("calibration", "test"):
        group = split[name]
        if not isinstance(group, list) or not group or any(not isinstance(k, str) or not k for k in group) or len(set(group)) != len(group):
            raise ValueError("Invalid SSVEP trial split")
    if set(split["calibration"]) & set(split["test"]):
        raise ValueError("Calibration/test trial leakage")
    if model.get("metadata", {}).get("threshold_fit_trials") != split["calibration"]:
        raise ValueError("Threshold provenance differs from calibration split")


def predict_epoch(model, samples, times, *, source_kind=None, raw_samples=None, target_count=None):
    empty = dict(label=None, candidate_target=None, accepted=False, scores=[], correlations=[],
                 margin=None, confidence=None, score_type="normalized_fbcca_score_not_probability",
                 label_order=list(TARGETS), frequencies=list(FREQUENCIES), frequency=None)
    try:
        if model is None:
            return dict(empty, status="no_model", reason="Load an SSVEP evaluation/calibration model")
        validate_model(model, source_kind=source_kind, target_count=target_count)
        _valid_times(times, WINDOW)
        if raw_samples is None:
            raise ValueError("Raw window required for signal-quality checks")
        raw = as_samples(raw_samples)
        if raw.shape != (WINDOW, len(CHANNELS)):
            raise ValueError("Raw/filtered SSVEP window shape mismatch")
        quality = raw_quality(raw)
        if not quality["valid"]:
            return dict(empty, status="invalid", reason=quality["reason"], quality=quality)
        scores, correlations = fbcca_scores(samples)
        candidate = int(scores.argmax())
        ordered = np.sort(scores)
        margin = float(ordered[-1] - ordered[-2])
        reasons = []
        if scores[candidate] < model["thresholds"]["score"]:
            reasons.append("low FBCCA score")
        if margin < model["thresholds"]["margin"]:
            reasons.append("ambiguous frequency margin")
        accepted = not reasons
        return dict(empty, label=candidate if accepted else None, candidate_target=candidate,
                    frequency=FREQUENCIES[candidate] if accepted else None, candidate_frequency=FREQUENCIES[candidate],
                    accepted=accepted, scores=scores.tolist(), correlations=correlations.tolist(), margin=margin,
                    status="predicted" if accepted else "uncertain", quality=quality,
                    reason="; ".join(reasons) or "heuristic thresholds passed; no-attention state not validated")
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        return dict(empty, status="invalid", reason=str(exc))


def save_model(model, path):
    validate_model(model)
    Path(path).write_text(json.dumps(model, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_model(path, source_kind=None):
    path = Path(path)
    if path.stat().st_size > 5_000_000:
        raise ValueError("SSVEP model JSON is unexpectedly large")
    model = json.loads(path.read_text(encoding="utf-8"))
    validate_model(model, source_kind=source_kind)
    return model

