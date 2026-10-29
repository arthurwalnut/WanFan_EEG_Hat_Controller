"""Fixed-model evaluation of raw MI recordings without fitting or live I/O.

Only one recording and a bounded chunk of overlapping epochs are held at a
time. Window rows are emitted to the caller; aggregates retain trial-level
probability sums rather than all epochs or all predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bci.hat_pipeline.contract import FS, HOP, LABELS, WINDOW, input_contract
from bci.hat_pipeline.model import validate_model
from bci.hat_pipeline.processing import StreamingPipeline

from .mi_recording import load_session


class EvaluationCancelled(RuntimeError):
    """The caller cancelled an offline evaluation between bounded chunks."""


def _counts():
    return {"windows": 0, "predicted_windows": 0, "scored_windows": 0,
            "candidate_windows": 0, "outside_complete_task": 0,
            "warmup_rejected": 0, "quality_rejected": 0, "prediction_rejected": 0,
            "uncertain_windows": 0, "accepted_windows": 0, "scored_trials": 0,
            "planned_trials": 0, "exclusion_reasons": {}}


def _matrix_metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    total = int(matrix.sum())
    classes = []
    for index, label in enumerate(LABELS):
        precision = float(correct[index] / predicted[index]) if predicted[index] else None
        recall = float(correct[index] / support[index]) if support[index] else None
        denominator = int(support[index] + predicted[index])
        classes.append({"label": label, "support": int(support[index]),
                        "predicted": int(predicted[index]), "precision": precision,
                        "recall": recall,
                        "f1": float(2 * correct[index] / denominator) if denominator else None})
    recalls = [item["recall"] for item in classes if item["recall"] is not None]
    return {"total": total, "correct": int(correct.sum()),
            "accuracy": float(correct.sum() / total) if total else None,
            "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
            "confusion_matrix": matrix.tolist(), "per_class": classes}


class _Aggregate:
    def __init__(self):
        self.counts = _counts()
        self.matrix = np.zeros((3, 3), dtype=np.int64)
        self.trials = {}
        self.confidence = self.log_loss = self.brier = 0.0

    def add(self, row):
        counts = self.counts
        counts["windows"] += 1
        if row["prediction_available"]:
            counts["predicted_windows"] += 1
        counts["uncertain_windows"] += int(row["status"] == "uncertain")
        counts["accepted_windows"] += int(row.get("accepted") is True)
        if row["task_eligible"]:
            counts["candidate_windows"] += 1
        else:
            counts["outside_complete_task"] += 1
        reason = row["exclusion_reason"]
        if reason is not None:
            counts["exclusion_reasons"][reason] = counts["exclusion_reasons"].get(reason, 0) + 1
        if row["task_eligible"] and not row["scoreable"]:
            key = ("warmup_rejected" if reason == "filter_warmup" else
                   "quality_rejected" if reason == "hard_quality" else "prediction_rejected")
            counts[key] += 1
        if not row["scoreable"]:
            return
        counts["scored_windows"] += 1
        truth, predicted = LABELS.index(row["true_label"]), LABELS.index(row["label"])
        probabilities = np.asarray(row["probabilities"], dtype=float)
        self.matrix[truth, predicted] += 1
        self.confidence += float(row["confidence"])
        self.log_loss -= float(np.log(np.clip(probabilities[truth], np.finfo(float).eps, 1.0)))
        self.brier += float(np.sum((probabilities - np.eye(3)[truth]) ** 2))
        key = (row["recording_id"], row["trial"])
        group = self.trials.setdefault(key, {"truth": truth, "probabilities": np.zeros(3), "windows": 0})
        if group["truth"] != truth:
            raise ValueError("同一个任务试次包含不同标签，不能计算试次分数。")
        group["probabilities"] += probabilities
        group["windows"] += 1

    def finish(self):
        trial_matrix = np.zeros((3, 3), dtype=np.int64)
        rows = []
        for (identifier, trial), group in sorted(self.trials.items()):
            probabilities = group["probabilities"] / group["windows"]
            predicted, truth = int(np.argmax(probabilities)), group["truth"]
            trial_matrix[truth, predicted] += 1
            rows.append({"recording_id": identifier, "trial": trial, "true_label": LABELS[truth],
                         "predicted_label": LABELS[predicted], "probabilities": probabilities.tolist(),
                         "confidence": float(probabilities[predicted]), "windows": group["windows"],
                         "correct": predicted == truth})
        self.counts["scored_trials"] = len(rows)
        n = self.counts["scored_windows"]
        metrics = {"window": _matrix_metrics(self.matrix), "trial": _matrix_metrics(trial_matrix),
                   "mean_confidence": self.confidence / n if n else None,
                   "log_loss": self.log_loss / n if n else None,
                   "brier_score": self.brier / n if n else None}
        return metrics, rows


def _provenance(model, recording_ids, trial_rows):
    selected = set(recording_ids)
    scored_groups = {f"{item['recording_id']}:trial:{item['trial']}" for item in trial_rows}
    metadata = model.get("metadata", {})
    development = {value for value in metadata.get("session_ids", []) if isinstance(value, str)}
    by_split, known_splits = {}, []
    for split in ("train", "calibration", "test"):
        metrics = model.get("metrics", {})
        groups = {value for value in metrics.get(split + "_groups", []) if isinstance(value, str)}
        sessions = {value for value in metrics.get(split + "_sessions", []) if isinstance(value, str)}
        sessions.update(value.rsplit(":trial:", 1)[0] for value in groups if ":trial:" in value)
        known_splits.append(bool(sessions))
        development.update(sessions)
        by_split[split] = {"recording_ids": sorted(selected & sessions),
                           "trial_groups": sorted(scored_groups & groups)}
    overlap = sorted(selected & development)
    provenance_available = all(known_splits)
    message = ("所选录制包含模型原开发数据，详见训练、校准和原测试来源；本次结果不能称为全新独立测试。"
               if overlap else "按录制编号未发现与模型原开发数据重叠；此检查未比较复制文件的信号内容。"
               if provenance_available else "原模型来源记录不完整，无法确认所选录制是否与原数据集重叠。")
    return {"has_overlap": bool(overlap), "recording_ids_disjoint": not overlap if provenance_available or overlap else None,
            "provenance_available": provenance_available,
            "overlap_recording_ids": overlap, "by_split": by_split, "message": message}


def _annotate(row, session):
    indices = session["indices"]
    lo = int(np.searchsorted(indices, row["window_start_index"]))
    hi = lo + WINDOW
    if (hi > len(indices) or int(indices[lo]) != row["window_start_index"]
            or int(indices[hi - 1]) != row["window_end_index"]
            or np.any(np.diff(indices[lo:hi]) != 1)):
        raise ValueError("预测窗口与原始样本索引不一致，已停止测试。")
    labels, trials = session["labels"][lo:hi], session["trials"][lo:hi]
    same_label = bool(np.all(labels == labels[0]))
    same_trial = bool(np.all(trials == trials[0]))
    true_label = str(labels[0]) if same_label else None
    trial = int(trials[0]) if same_trial and trials[0] > 0 else None
    if not same_label:
        exclusion = "mixed_labels"
    elif true_label not in LABELS:
        exclusion = "interval_or_unlabelled"
    elif trial is None:
        exclusion = "mixed_or_missing_trial"
    elif not np.all(session["timing_valid"][lo:hi]):
        exclusion = "task_timing_invalid"
    else:
        exclusion = None
    eligible = exclusion is None
    probabilities = np.asarray(row.get("probabilities", []), dtype=float)
    prediction_available = (row.get("label") in LABELS and probabilities.shape == (3,)
                            and np.isfinite(probabilities).all() and np.all(probabilities >= 0)
                            and np.isclose(probabilities.sum(), 1.0)
                            and row.get("confidence") is not None
                            and np.isfinite(row["confidence"]))
    if eligible and not prediction_available:
        exclusion = ("filter_warmup" if row["status"] == "warming_up" else
                     "hard_quality" if not (row.get("quality") or {}).get("valid", False)
                     else "prediction_unavailable")
    row.update(recording_id=session["metadata"]["id"], true_label=true_label, trial=trial,
               task_eligible=eligible, prediction_available=bool(prediction_available),
               scoreable=eligible and bool(prediction_available), exclusion_reason=exclusion)
    row["correct"] = row["label"] == true_label if row["scoreable"] else None
    return row


def evaluate_recordings(model, recording_paths, *, on_row=None, on_progress=None, should_cancel=None):
    """Return JSON-safe aggregates and emit each window through ``on_row``.

    The fixed model supplies all probabilities, including tentative predictions.
    Labels affect scoring only. Samples remain raw; the training fill option does
    not apply here. Each file resets the causal filter, as in training/replay.
    ``should_cancel`` is checked before each load and every <=250-sample chunk.
    """
    validate_model(model)
    paths = sorted((Path(path) for path in recording_paths), key=lambda path: path.name)
    if not paths:
        raise ValueError("请至少选择一条录制数据。")
    total, recordings, seen = _Aggregate(), [], set()

    def cancelled():
        if should_cancel is not None and should_cancel():
            raise EvaluationCancelled("模型测试已取消。")

    for index, path in enumerate(paths):
        cancelled()
        session = load_session(path, use_filled=False)
        metadata = session["metadata"]
        identifier = metadata["id"]
        if identifier in seen:
            raise ValueError("同一录制不能重复参与测试。")
        seen.add(identifier)
        validate_model(model, source_kind=metadata["source"])
        if np.asarray(session.get("is_interpolated", []), dtype=bool).any():
            raise ValueError("模型测试仅使用原始实测数据。")
        local = _Aggregate()
        planned = sum(item.get("phase") == "task" and item.get("label") in LABELS
                      for item in metadata.get("plan", []) if isinstance(item, dict))
        local.counts["planned_trials"] = int(planned)
        total.counts["planned_trials"] += int(planned)
        pipeline = StreamingPipeline(model=model, source_kind=metadata["source"])
        n = len(session["samples"])
        boundaries = np.flatnonzero(np.diff(session["segments"]) != 0) + 1
        start = 0
        for end in list(boundaries) + [n]:
            for chunk in range(start, int(end), FS):
                cancelled()
                stop = min(chunk + FS, int(end))
                rows = pipeline.push(session["samples"][chunk:stop], session["indices"][chunk:stop],
                                     session["times"][chunk:stop], int(session["segments"][chunk]))
                for row in rows:
                    row = _annotate(row, session)
                    total.add(row)
                    local.add(row)
                    if on_row is not None:
                        on_row(row)
                if on_progress is not None:
                    on_progress({"recording_id": identifier, "recording_index": index + 1,
                                 "recording_count": len(paths), "samples_done": stop,
                                 "sample_count": n, "windows": total.counts["windows"],
                                 "progress": (index + (stop / n if n else 1.0)) / len(paths)})
            start = int(end)
        metrics, _ = local.finish()
        recordings.append({"id": identifier, "subject": metadata.get("subject", ""),
                           "source": metadata["source"], "sample_count": n,
                           "duration_s": float(session["times"][-1] - session["times"][0] + 1 / FS) if n else 0.0,
                           "counts": local.counts, "metrics": metrics,
                           "timestamp_audit": metadata.get("timestamp_audit", {}),
                           "display_audit": metadata.get("display_audit", {})})
        # Release the previous raw recording before loading the next CSV.
        del session, pipeline
    cancelled()
    metrics, trials = total.finish()
    metadata = model.get("metadata", {})
    return {"schema_version": 1, "label_order": list(LABELS), "input": input_contract(),
            "window_seconds": WINDOW / FS, "hop_seconds": HOP / FS, "use_filled": False,
            "model_info": {"feature": metadata.get("feature"), "classifier": metadata.get("classifier"),
                           "source": model["source"], "created": metadata.get("created"),
                           "original_holdout_usable": model["metrics"]["usable"]},
            "counts": total.counts, "metrics": metrics, "per_recording": recordings,
            "trials": trials, "provenance": _provenance(model, seen, trials),
            "limitations": ["分数仅使用完整、标注时序有效且通过预热与硬质量检查的任务窗口；间隔和跨任务窗口不计分。",
                            "低置信度或训练分布外的暂定预测仍计入分数，不按置信度筛除错误预测。",
                            "试次预测取该试次可评分窗口的平均概率；重叠窗口不是独立样本。",
                            "平衡准确率只平均本次实际存在类别的召回率；缺失类别的召回率为空。",
                            "模型、特征变换和概率校准均保持原参数；本次不重新拟合。"]}
