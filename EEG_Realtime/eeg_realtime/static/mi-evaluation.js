"use strict";

// Fixed-model offline evaluation has its own state and never loads the live model.
(() => {
  const $ = (id) => document.getElementById(id);
  const LABELS = { rest: "静息", left_mi: "左手想象", right_mi: "右手想象", interval: "间隔", unknown: "未知" };
  const ORDER = ["rest", "left_mi", "right_mi"];
  const STATUS = { idle: "待开始", running: "测试中", complete: "已完成", cancelled: "已取消", error: "测试失败" };
  const REASONS = { outside_complete_task: "完整有效任务范围外", mixed_labels: "跨越不同标注", interval_or_unlabelled: "间隔或没有标注", mixed_or_missing_trial: "跨试次或缺少试次编号", task_timing_invalid: "任务时序检查未通过", filter_warmup: "滤波预热", warmup: "滤波预热", warming_up: "滤波预热", quality_rejected: "信号质量未通过", hard_quality: "信号质量未通过", invalid_prediction: "未产生有效预测", prediction_rejected: "未产生有效预测", prediction_unavailable: "未产生有效预测", nonfinite: "非有限值", saturated: "饱和", flat: "平线", dc_offset: "原始直流偏置较大", raw_amplitude: "原始幅度较大" };
  const state = { api: null, models: [], recordings: [], modelId: "", selected: new Set(), job: {}, workflow: {}, online: false, blocked: false, busy: false, report: null, loadedReportId: null, loadingReport: false, reportRetryAt: 0, offset: 0, total: 0, loadingPage: false, pageSequence: 0, initialized: false };
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const pct = (value) => finite(value) ? `${(value * 100).toFixed(1)}%` : "—";
  const number = (value) => finite(value) ? value.toLocaleString() : "—";
  const decimal = (value, digits = 2) => finite(value) ? value.toFixed(digits) : "—";
  const label = (value) => LABELS[value] || value || "—";
  const source = (value) => ({ demo: "演示数据", device: "真实设备" })[value] || value || "—";
  const feature = (value) => ({ csp: "CSP", bandpower: "频带功率", combined: "CSP + 频带功率" })[value] || value || "—";
  const recordingName = (row) => row.display_name && row.display_name !== row.id ? row.display_name : row.subject || row.id;
  const modelName = (row) => row.display_name || row.id;
  const running = () => state.job.status === "running";
  const element = (tag, text, css) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (css) node.className = css; return node; };
  const textReason = (value) => REASONS[value] || state.api?.reasonText(value) || value || "—";
  const reasonCounts = (values) => Object.entries(values || {}).map(([key, value]) => `${textReason(key)} ${number(value)}`).join("；") || "无";
  const reportRows = () => state.job.recordings || [];
  const nameForId = (id) => { const row = reportRows().find((item) => item.id === id) || state.recordings.find((item) => item.id === id); return row ? recordingName(row) : id; };

  function error(message = "") { $("evaluationError").textContent = message; $("evaluationError").hidden = !message; }
  function workspaceBusy() {
    const value = state.workflow;
    return value.recording?.active || ["running", "connecting"].includes(value.status) || value.playback?.playing || value.training?.status === "running" || value.recording?.validation?.status === "checking";
  }
  function canEdit() { return state.online && !state.blocked && !state.busy && !running() && !workspaceBusy(); }

  async function perform(callback) {
    if (state.busy || !state.api) return;
    state.busy = true; error(); controls();
    try { await state.api.action(callback); }
    finally { state.busy = false; controls(); }
  }
  function selectedModel() { return state.models.find((row) => row.id === state.modelId); }
  function modelMatches(row, query) { return `${modelName(row)} ${row.id} ${feature(row.feature)} ${row.feature || ""} ${row.classifier || ""} ${source(row.source)}`.toLocaleLowerCase().includes(query); }
  function recordingMatches(row, query) { return `${recordingName(row)} ${row.id} ${row.subject || ""} ${row.created_utc || ""} ${source(row.source)}`.toLocaleLowerCase().includes(query); }

  function star(kind, row) {
    const name = kind === "model" ? modelName(row) : recordingName(row);
    const button = element("button", row.favorite ? "★" : "☆", `favorite-button${row.favorite ? " is-favorite" : ""}`);
    button.type = "button"; button.dataset.favorite = "true"; button.dataset.itemId = row.id;
    button.setAttribute("aria-label", `${row.favorite ? "取消收藏" : "收藏"}：${name}`);
    button.setAttribute("aria-pressed", String(Boolean(row.favorite)));
    button.addEventListener("click", () => perform(async () => {
      await state.api.request("/api/mi/library/update", { kind, id: row.id, favorite: !row.favorite });
      await state.api.refresh();
      const list = $(kind === "model" ? "evaluationModels" : "evaluationRecordings");
      [...list.querySelectorAll("button")].find((item) => item.dataset.itemId === row.id)?.focus({ preventScroll: true });
    }));
    return button;
  }
  function lists() {
    const modelQuery = $("evaluationModelSearch").value.trim().toLocaleLowerCase();
    const visibleModels = state.models.filter((row) => modelMatches(row, modelQuery));
    const models = $("evaluationModels"); models.replaceChildren();
    $("evaluationModelCount").textContent = `显示 ${visibleModels.length} / ${state.models.length} 个模型`;
    if (!visibleModels.length) models.append(element("p", state.models.length ? "没有匹配的模型。" : "训练后模型会出现在这里。", "empty-state"));
    for (const row of visibleModels) {
      const item = element("div", undefined, "session-item library-item");
      const selection = element("label", undefined, "library-selection"), input = element("input");
      input.type = "radio"; input.name = "evaluation-model"; input.value = row.id; input.checked = state.modelId === row.id; input.dataset.compatible = String(row.compatible !== false);
      input.addEventListener("change", () => { state.modelId = row.id; controls(); });
      const info = element("span", undefined, "session-info");
      info.append(element("b", modelName(row)), element("small", `${source(row.source)} · ${feature(row.feature)} + ${String(row.classifier || "—").toUpperCase()}${row.compatible === false ? " · 旧版需重训" : ""} · ${row.id}`));
      selection.append(input, info); item.append(selection, star("model", row)); models.append(item);
    }
    const query = $("evaluationRecordingSearch").value.trim().toLocaleLowerCase();
    const visible = state.recordings.filter((row) => recordingMatches(row, query));
    const recordings = $("evaluationRecordings"); recordings.replaceChildren();
    $("evaluationRecordingCount").textContent = `显示 ${visible.length} / ${state.recordings.length} 次录制`;
    if (!visible.length) recordings.append(element("p", state.recordings.length ? "没有匹配的录制。" : "录制后数据会出现在这里。", "empty-state"));
    for (const row of visible) {
      const item = element("div", undefined, "session-item library-item");
      const selection = element("label", undefined, "library-selection"), input = element("input");
      input.type = "checkbox"; input.value = row.id; input.checked = state.selected.has(row.id);
      input.addEventListener("change", () => {
        if (input.checked && state.selected.size >= 50) { input.checked = false; error("一次最多选择 50 次录制。"); return; }
        if (input.checked) state.selected.add(row.id); else state.selected.delete(row.id);
        error(); controls();
      });
      const info = element("span", undefined, "session-info");
      info.append(element("b", `${recordingName(row)} · ${decimal(row.duration_s, 0)} 秒`), element("small", `受试者：${row.subject || "未填编号"} · ${source(row.source)} · ${row.id}`));
      selection.append(input, info); item.append(selection, star("recording", row)); recordings.append(item);
    }
    controls();
  }

  function controls() {
    const editable = canEdit(), model = selectedModel(), inProgress = running();
    for (const id of ["evaluationModelSearch", "evaluationRecordingSearch", "refreshEvaluationButton"]) $(id).disabled = !editable;
    for (const input of $("evaluationModels").querySelectorAll("input")) input.disabled = !editable || input.dataset.compatible === "false";
    for (const input of $("evaluationRecordings").querySelectorAll("input")) input.disabled = !editable;
    for (const list of [$("evaluationModels"), $("evaluationRecordings")]) for (const button of list.querySelectorAll("button")) button.disabled = !editable;
    $("evaluationClearSelection").disabled = !editable || !state.selected.size;
    $("startEvaluationButton").disabled = !editable || !model || model.compatible === false || !state.selected.size;
    $("cancelEvaluationButton").disabled = !inProgress || state.busy || !state.online;
    const query = $("evaluationRecordingSearch").value.trim().toLocaleLowerCase();
    const hidden = [...state.selected].filter((id) => !recordingMatches(state.recordings.find((row) => row.id === id) || { id }, query)).length;
    $("evaluationSelectedCount").textContent = `已选 ${state.selected.size} / 50 次录制${hidden ? `（${hidden} 次被搜索隐藏）` : ""}`;
    $("evaluationModelChoice").textContent = model ? `已选：${model.favorite ? "★ " : ""}${modelName(model)}` : "尚未选择模型。";
    $("evaluationStatus").textContent = STATUS[state.job.status] || "待开始";
    $("evaluationStatus").className = `tag${inProgress || state.job.status === "complete" ? " active" : state.job.status === "error" ? " error" : ""}`;
    $("evaluationProgress").hidden = !inProgress;
    $("evaluationProgress").value = finite(state.job.progress) ? Math.max(0, Math.min(1, state.job.progress)) : 0;
    const idleMessage = workspaceBusy() ? "先停止采集或暂停回放，并等待训练、录后检查结束，再开始测试。" : "选择一个模型和 1–50 次录制。";
    $("evaluationMessage").textContent = !state.online ? "本地服务未连接，正在等待恢复。" : state.job.status && state.job.status !== "idle" ? `${state.job.message || STATUS[state.job.status]}${inProgress ? ` · ${pct(state.job.progress)}` : ""}` : idleMessage;
    $("evaluationPreviousPage").disabled = state.loadingPage || state.offset <= 0 || !state.online;
    $("evaluationNextPage").disabled = state.loadingPage || state.offset + 50 >= state.total || !state.online;
  }

  function table(headers, rows, className = "diagnostic-table") {
    const node = element("table", undefined, className), head = element("thead"), heading = element("tr"), body = element("tbody");
    headers.forEach((value) => heading.append(element("th", value))); head.append(heading); node.append(head);
    for (const values of rows) { const row = element("tr"); values.forEach((value) => row.append(element("td", value))); body.append(row); }
    node.append(body); return node;
  }
  function matrix(containerId, title, values, order) {
    const container = $(containerId); container.replaceChildren(element("h3", title), element("p", "行：原标注 · 列：预测", "matrix-caption"));
    if (!Array.isArray(values) || values.length !== 3 || values.some((row) => !Array.isArray(row) || row.length !== 3)) { container.append(element("p", "暂无可评分结果。", "empty-state")); return; }
    const node = table(["原标注 / 预测", ...order.map(label)], values.map((row, i) => [label(order[i]), ...row.map(number)]), "confusion-table");
    [...node.querySelectorAll("tbody tr")].forEach((row, index) => row.children[index + 1].classList.add("correct")); container.append(node);
  }
  function provenance(report) {
    const target = $("evaluationProvenance"), value = report.provenance || {};
    target.replaceChildren();
    const heading = value.has_overlap ? "所选录制与该模型原有数据集存在重叠，不能作为新的独立测试。" : value.recording_ids_disjoint ? "所选录制编号未出现在模型原有数据集中。" : "模型原有数据来源记录不完整，无法确认测试数据是否独立。";
    target.append(element("p", heading, value.has_overlap || !value.recording_ids_disjoint ? "diagnostic-warning" : "evaluation-provenance-ok"));
    if (value.message) target.append(element("p", value.message, "field-note"));
    if (value.overlap_recording_ids?.length) target.append(element("p", `重叠录制：${value.overlap_recording_ids.map((id) => `${nameForId(id)}（${id}）`).join("；")}`, "field-note"));
    const splits = value.by_split || {};
    const details = element("details", undefined, "diagnostic-details"); details.append(element("summary", "查看与原训练、校准和测试集的重叠"));
    for (const [key, name] of [["train", "训练集"], ["calibration", "置信度校准集"], ["test", "原留出测试集"]]) {
      const info = splits[key] || {}, ids = info.recording_ids || [], groups = info.trial_groups || [];
      const item = element("div", undefined, "diagnostic-recording"); item.append(element("b", name), element("p", ids.length ? ids.join("；") : "本次所选录制未与该分组重叠。"));
      if (groups.length) item.append(element("p", `试次分组：${groups.join("；")}`)); details.append(item);
    }
    target.append(details);
  }
  function renderReport(report) {
    state.report = report;
    $("evaluationResults").hidden = false; $("evaluationWindows").hidden = false;
    const model = state.job.model || {}, records = reportRows(), metrics = report.metrics || {}, counts = report.counts || {};
    const order = report.label_order || ORDER;
    $("evaluationResultContext").textContent = `${model.display_name || model.id || "已保存模型"} · ${records.length} 次录制 · 本次固定模型测试`;
    provenance(report);
    const cards = $("evaluationMetricCards"); cards.replaceChildren();
    for (const [name, value] of [["窗口准确率", metrics.window?.accuracy], ["窗口平衡准确率", metrics.window?.balanced_accuracy], ["试次准确率", metrics.trial?.accuracy], ["试次平衡准确率", metrics.trial?.balanced_accuracy], ["平均置信度", metrics.mean_confidence]]) {
      const card = element("div", undefined, "metric"); card.append(element("span", name), element("b", pct(value))); cards.append(card);
    }
    $("evaluationScoreNote").textContent = `评分窗口 ${number(metrics.window?.total ?? counts.scored_windows)}，评分试次 ${number(metrics.trial?.total ?? counts.scored_trials)}。仅完整任务内、通过时序与信号检查的窗口用于评分；低确定性预测仍计入。平衡准确率只平均实际出现的类别；重叠窗口不是独立试次。`;
    matrix("evaluationWindowMatrix", "窗口混淆矩阵", metrics.window?.confusion_matrix, order);
    matrix("evaluationTrialMatrix", "试次混淆矩阵", metrics.trial?.confusion_matrix, order);
    const classRows = [];
    for (const [key, name] of [["window", "窗口"], ["trial", "试次"]]) for (const row of metrics[key]?.per_class || []) classRows.push([name, label(row.label), number(row.support), number(row.predicted), pct(row.precision), pct(row.recall), pct(row.f1)]);
    $("evaluationClassMetrics").replaceChildren(table(["单位", "类别", "真实数量", "预测数量", "Precision", "Recall", "F1"], classRows));
    const counter = $("evaluationCounts"); counter.replaceChildren();
    counter.append(element("p", `全部窗口 ${number(counts.windows)} · 有预测 ${number(counts.predicted_windows)} · 可评分 ${number(counts.scored_windows)} · 低确定性 ${number(counts.uncertain_windows)}`, "field-note"));
    counter.append(element("p", `未评分原因：${reasonCounts(counts.exclusion_reasons)}`, "field-note"));
    const perRecording = [];
    for (const row of report.per_recording || []) perRecording.push([nameForId(row.id), row.subject || "—", `${number(row.counts?.scored_windows)} / ${number(row.counts?.windows)}`, `${number(row.counts?.scored_trials)} / ${number(row.counts?.planned_trials)}`, pct(row.metrics?.window?.accuracy), pct(row.metrics?.window?.balanced_accuracy), pct(row.metrics?.trial?.accuracy), reasonCounts(row.counts?.exclusion_reasons)]);
    $("evaluationRecordingMetrics").replaceChildren(table(["录制", "受试者", "评分 / 全部窗口", "评分 / 计划试次", "窗口准确率", "窗口平衡准确率", "试次准确率", "未评分原因"], perRecording, "diagnostic-table evaluation-recording-table"));
    const inputs = $("evaluationInputs"); inputs.replaceChildren();
    inputs.append(element("p", `模型：${model.display_name || model.id || "—"}（${model.id || "—"}）`, "field-note"));
    const info = report.model_info || {};
    inputs.append(element("p", `特征：${feature(info.feature)} · 分类器：${String(info.classifier || "—").toUpperCase()} · 来源：${source(info.source)}`, "field-note"));
    for (const row of records) inputs.append(element("p", `${recordingName(row)} · 受试者 ${row.subject || "未填编号"} · ${row.id}`, "field-note"));
    const extra = $("evaluationLimitations"); extra.replaceChildren();
    extra.append(element("p", "本页分数来自上述新选择的录制；训练页分数仍属于模型原来的留出测试。置信度是概率估计，不是正确率。"));
    if (finite(metrics.log_loss) || finite(metrics.brier_score)) extra.append(element("p", `概率评估：Log loss ${decimal(metrics.log_loss, 4)} · Brier score ${decimal(metrics.brier_score, 4)}`));
    const limitations = Array.isArray(report.limitations) ? report.limitations : report.limitations ? [report.limitations] : [];
    limitations.forEach((value) => extra.append(element("p", String(value))));
    for (const [id, format] of [["evaluationExportJson", "json"], ["evaluationExportCsv", "csv"]]) $(id).href = `/api/mi/evaluation/export?id=${encodeURIComponent(state.job.id)}&format=${format}`;
  }

  function windowNote(row) {
    const parts = [row.scoreable ? "计入评分" : `未评分：${textReason(row.exclusion_reason)}`];
    if (row.status === "uncertain") parts.push("低确定性");
    const warnings = row.quality?.warnings || [];
    for (const warning of warnings) parts.push(typeof warning === "string" ? textReason(warning) : `${warning.channel || ""} ${textReason(warning.reason || warning.code || warning.kind)}`.trim());
    if (row.reason) parts.push(textReason(row.reason));
    return [...new Set(parts)].join("；");
  }
  async function loadPage(offset = 0) {
    const id = state.job.id;
    if (!id || state.job.status !== "complete") return;
    const sequence = ++state.pageSequence; state.loadingPage = true; controls();
    const rowsNode = $("evaluationWindowRows"); rowsNode.replaceChildren();
    const waiting = element("tr"), cell = element("td", "正在读取窗口…", "empty-state"); cell.colSpan = 9; waiting.append(cell); rowsNode.append(waiting);
    try {
      const data = await state.api.request(`/api/mi/evaluation/windows?id=${encodeURIComponent(id)}&offset=${offset}&limit=50`);
      if (sequence !== state.pageSequence || id !== state.job.id) return;
      state.offset = data.offset; state.total = data.total; rowsNode.replaceChildren();
      for (const row of data.rows || []) {
        const tr = element("tr");
        const description = `${nameForId(row.recording_id)}${row.trial ? ` / ${row.trial}` : ""}`;
        const probs = row.probabilities || [];
        tr.append(element("td", description), element("td", `${decimal(row.window_start_time)}–${decimal(row.window_end_time)}`), element("td", label(row.true_label)), element("td", label(row.label)), element("td", pct(row.confidence)), ...ORDER.map((kind) => element("td", pct(probs[(state.report?.label_order || ORDER).indexOf(kind)]))), element("td", windowNote(row)));
        if (row.scoreable && row.true_label !== row.label) tr.classList.add("evaluation-wrong");
        if (!row.scoreable) tr.classList.add("evaluation-unscored");
        tr.firstChild.title = row.recording_id; rowsNode.append(tr);
      }
      if (!data.rows?.length) { const tr = element("tr"), td = element("td", "没有窗口记录。", "empty-state"); td.colSpan = 9; tr.append(td); rowsNode.append(tr); }
      $("evaluationWindowCount").textContent = `${number(data.total)} 条`;
      $("evaluationPagePosition").textContent = data.total ? `${number(data.offset + 1)}–${number(Math.min(data.offset + 50, data.total))} / ${number(data.total)} 条 · 每页 50 条` : "0 条";
    } catch (failure) { if (sequence === state.pageSequence) error(`窗口读取失败：${failure.message}`); }
    finally { if (sequence === state.pageSequence) { state.loadingPage = false; controls(); } }
  }
  async function fetchReport() {
    if (state.loadingReport || !state.online || Date.now() < state.reportRetryAt) return;
    const id = state.job.id; state.loadingReport = true;
    try {
      const value = await state.api.request("/api/mi/evaluation");
      if (id !== state.job.id || value.id !== id || value.status !== "complete" || !value.report) return;
      state.job = { ...state.job, ...value }; state.loadedReportId = id; renderReport(value.report); error();
      await loadPage(0);
    } catch (failure) { state.reportRetryAt = Date.now() + 5000; error(`结果读取失败：${failure.message}`); }
    finally { state.loadingReport = false; }
  }
  function render(workflow, online, blocked) {
    state.workflow = workflow || {}; state.online = online; state.blocked = blocked;
    const incoming = workflow?.evaluation || { status: "idle" };
    if (incoming.id !== state.job.id || incoming.status !== "complete") {
      if (incoming.id !== state.job.id || ["running", "idle", "cancelled", "error"].includes(incoming.status)) {
        state.report = null; state.loadedReportId = null; state.pageSequence++; state.loadingPage = false;
        $("evaluationResults").hidden = true; $("evaluationWindows").hidden = true;
      }
    }
    state.job = { ...state.job, ...incoming };
    if (incoming.status === "complete" && incoming.id && state.loadedReportId !== incoming.id) void fetchReport();
    controls();
  }
  function updateLibrary(models, recordings) {
    state.models = models; state.recordings = recordings;
    if (!models.some((row) => row.id === state.modelId)) state.modelId = "";
    state.selected = new Set([...state.selected].filter((id) => recordings.some((row) => row.id === id)));
    lists();
  }
  function initialize(api) {
    state.api = api;
    if (state.initialized) return;
    state.initialized = true;
    $("evaluationModelSearch").addEventListener("input", lists); $("evaluationRecordingSearch").addEventListener("input", lists);
    $("refreshEvaluationButton").addEventListener("click", () => perform(api.refresh));
    $("evaluationClearSelection").addEventListener("click", () => { state.selected.clear(); lists(); });
    $("startEvaluationButton").addEventListener("click", () => perform(async () => {
      const job = await api.request("/api/mi/evaluation/start", { model_id: state.modelId, ids: [...state.selected] });
      state.job = job; state.report = null; state.loadedReportId = null; state.reportRetryAt = 0;
      $("evaluationResults").hidden = true; $("evaluationWindows").hidden = true;
    }));
    $("cancelEvaluationButton").addEventListener("click", () => perform(async () => { await api.request("/api/mi/evaluation/cancel", { id: state.job.id }); }));
    $("evaluationPreviousPage").addEventListener("click", () => { void loadPage(Math.max(0, state.offset - 50)); });
    $("evaluationNextPage").addEventListener("click", () => { void loadPage(state.offset + 50); });
  }
  window.MIEvaluationUI = { initialize, updateLibrary, render };
})();
