"""Bounded background jobs and paged artifacts for fixed-model MI evaluation."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

from .storage import unique_name, utc_now


CSV_FIELDS = ["recording_id", "window_start_time", "window_end_time",
              "window_start_index", "window_end_index", "segment", "trial",
              "true_label", "label", "confidence", "prob_rest", "prob_left_mi",
              "prob_right_mi", "status", "scoreable", "exclusion_reason", "reason",
              "quality_warnings"]


class EvaluationJob:
    def __init__(self, output):
        self.output = Path(output)
        self.lock = threading.RLock()
        self.thread = None
        self.cancel_event = threading.Event()
        self.state = {"status": "idle", "id": None, "progress": 0., "row_count": 0,
                      "message": "选择已训练模型和录制数据后开始测试。"}
        self.report = None
        self.directory = None
        self.offsets = []

    def active(self):
        with self.lock:
            return bool(self.thread and self.thread.is_alive())

    def require_idle(self):
        if self.active():
            raise RuntimeError("模型测试正在进行，请等待完成或取消后再操作。")

    def snapshot(self, *, report=False):
        with self.lock:
            value = copy.deepcopy(self.state)
            if report:
                value["report"] = copy.deepcopy(self.report)
            return value

    def start(self, model_path, paths, model_info, recording_info):
        with self.lock:
            self.require_idle()
            identifier = unique_name("evaluation", "")
            self.output.mkdir(parents=True, exist_ok=True)
            directory = self.output / identifier
            directory.mkdir()
            self.directory, self.offsets, self.report = directory, [], None
            self.cancel_event = threading.Event()
            self.state = {"status": "running", "id": identifier, "progress": 0.,
                          "row_count": 0, "message": "正在读取模型与录制数据。",
                          "created": utc_now(), "model": model_info, "recordings": recording_info}
            self.thread = threading.Thread(target=self._run, args=(model_path, paths),
                                           daemon=True, name="MI-model-test")
            self.thread.start()
            return {"ok": True, "id": identifier}

    def cancel(self, identifier):
        with self.lock:
            self._check_id(identifier)
            if self.active():
                self.cancel_event.set()
                self.state["message"] = "正在取消测试，已写入的窗口记录保留。"
            return {"ok": True}

    def close(self):
        self.cancel_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=15)

    def _run(self, model_path, paths):
        directory = self.directory
        count, offsets = 0, []
        try:
            from bci.hat_pipeline.model import load_model
            from .mi_evaluation import evaluate_recordings, EvaluationCancelled
            model = load_model(model_path)
            with (directory / "windows.jsonl").open("wb") as rows, (directory / "windows.csv").open("w", encoding="utf-8-sig", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writeheader()

                def on_row(row):
                    nonlocal count
                    if count % 50 == 0:
                        offsets.append(rows.tell())
                    rows.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                    values = {key: row.get(key) for key in CSV_FIELDS}
                    probabilities = row.get("probabilities") or []
                    for index, key in enumerate(("prob_rest", "prob_left_mi", "prob_right_mi")):
                        values[key] = probabilities[index] if len(probabilities) == 3 else None
                    values["quality_warnings"] = json.dumps((row.get("quality") or {}).get("warnings", []), ensure_ascii=False)
                    # Spreadsheet-safe text, while the JSONL preserves exact strings.
                    for key, value in values.items():
                        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                            values[key] = "'" + value
                    writer.writerow(values)
                    count += 1

                def progress(value):
                    with self.lock:
                        self.state["row_count"] = count
                        if isinstance(value, dict):
                            self.state["progress"] = float(value.get("progress", 0.))
                            self.state["message"] = value.get("message", "正在逐窗测试。")
                        else:
                            self.state["progress"] = float(value)
                            self.state["message"] = "正在逐窗测试。"

                report = evaluate_recordings(model, paths, on_row=on_row, on_progress=progress,
                                             should_cancel=self.cancel_event.is_set)
            if self.cancel_event.is_set():
                raise EvaluationCancelled()
            with self.lock:
                report.update(id=self.state["id"], created=self.state["created"],
                              model=self.state["model"], recordings=self.state["recordings"])
            (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            with self.lock:
                self.offsets, self.report = offsets, report
                self.state.update(status="complete", progress=1., row_count=count, message="测试完成。结果来自本次所选录制，模型未重新训练。")
        except Exception as exc:
            with self.lock:
                self.offsets = offsets
                if self.cancel_event.is_set():
                    self.state.update(status="cancelled", row_count=count, message="测试已取消；未完成的窗口不计入最终评估。")
                else:
                    self.state.update(status="error", row_count=count, message=f"测试失败：{exc}")
        finally:
            try:
                (directory / "status.json").write_text(json.dumps(self.snapshot(), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            except OSError:
                pass

    def _check_id(self, identifier):
        if not isinstance(identifier, str) or not identifier or identifier != self.state.get("id"):
            raise ValueError("测试结果已变化，请刷新当前测试页面。")

    def windows(self, identifier, offset=0, limit=50):
        if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 100:
            raise ValueError("分页参数无效。")
        with self.lock:
            self._check_id(identifier)
            if self.state["status"] == "running":
                raise ValueError("测试完成后可查看全部窗口。")
            total = self.state["row_count"]
            rows = []
            if offset < total:
                with (self.directory / "windows.jsonl").open("rb") as handle:
                    handle.seek(self.offsets[offset // 50])
                    for _ in range(offset % 50):
                        handle.readline()
                    for _ in range(min(limit, total - offset)):
                        rows.append(json.loads(handle.readline()))
            return {"id": identifier, "total": total, "offset": offset, "limit": limit, "rows": rows}

    def export(self, identifier, format):
        with self.lock:
            self._check_id(identifier)
            if self.state["status"] != "complete":
                raise ValueError("测试完成后可导出最终结果。")
            if format not in ("csv", "json"):
                raise ValueError("导出格式应为 csv 或 json。")
            path = self.directory / ("windows.csv" if format == "csv" else "report.json")
            return path, f"{identifier}.{format}"


def handle_evaluation_get(handler, job):
    """Common read/download routes for the combined and standalone servers."""
    parsed = urlsplit(handler.path)
    query = parse_qs(parsed.query)
    if parsed.path == "/api/mi/evaluation":
        handler._json(job.snapshot(report=True))
    elif parsed.path == "/api/mi/evaluation/windows":
        handler._json(job.windows(query.get("id", [None])[0],
                                  int(query.get("offset", ["0"])[0]),
                                  int(query.get("limit", ["50"])[0])))
    elif parsed.path == "/api/mi/evaluation/export":
        format = query.get("format", [""])[0]
        path, name = job.export(query.get("id", [None])[0], format)
        with path.open("rb") as source:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/csv; charset=utf-8" if format == "csv" else "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(os.fstat(source.fileno()).st_size))
            handler.send_header("Content-Disposition", f'attachment; filename="{name}"')
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("X-Content-Type-Options", "nosniff")
            handler.end_headers()
            while block := source.read(65536):
                handler.wfile.write(block)
    else:
        return False
    return True
