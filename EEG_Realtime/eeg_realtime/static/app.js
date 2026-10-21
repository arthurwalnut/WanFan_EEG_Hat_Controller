"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {token: null, files: [], models: [], state: null, ready: false, hydrated: false, connected: false, busy: new Set(), polling: false, errors: 0, lastStateAt: 0, connectionNotice: false, eventKey: "", metricsKey: "", channelNames: Array.from({length: 8}, (_, i) => `CH${i + 1}`), selected: new Set(Array.from({length: 8}, (_, i) => i))};
  const channelColors = ["#227d7e", "#4b81a2", "#97846a", "#6b99a0", "#697cb4", "#839d59", "#a57890", "#b28b63"];
  const bandDefinitions = [["delta", "δ · 1–4 Hz"], ["theta", "θ · 4–8 Hz"], ["alpha", "α/μ · 8–13"], ["beta", "β · 13–30"], ["gamma", "γ · 30+ Hz"]];
  const phaseLabels = {idle: "等待校准", rest: "静息准备", cue: "任务提示", active: "正在执行", complete: "校准采集完成", aborted: "校准已终止", error: "校准异常"};
  const stateLabels = {good: "良好", valid: "良好", ok: "良好", saturated: "饱和", flat: "恒定信号", noisy: "高幅噪声", artifact: "伪迹 / 高幅异常", nonfinite: "数值无效", invalid: "无效", missing: "无数据", line_noise: "工频干扰", insufficient: "数据不足", disconnected: "断流"};
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const clamp = (value, low = 0, high = 1) => Math.min(high, Math.max(low, value));
  const number = (value, digits = 1, suffix = "") => finite(value) ? value.toFixed(digits) + suffix : "—";
  const integer = (value) => finite(value) ? Math.round(value).toLocaleString("zh-CN") : "0";
  const write = (id, value) => { $(id).textContent = value == null ? "—" : String(value); };
  const node = (tag, text, className) => { const el = document.createElement(tag); if (text != null) el.textContent = String(text); if (className) el.className = className; return el; };
  const tag = (id, text, tone = "") => { write(id, text); $(id).className = `tag ${tone}`; };
  const active = () => ["running", "connecting"].includes(ui.state?.status);
  const calibrating = () => ["rest", "cue", "active"].includes(ui.state?.calibration?.phase);

  function notice(message, error = false) {
    ui.connectionNotice = false;
    $("notice").hidden = !message;
    $("notice").className = error ? "notice error" : "notice";
    write("notice", message || "");
  }

  async function request(path, body) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), path === "/api/state" ? 2000 : path.endsWith("/train") ? 60000 : 12000);
    try {
      const options = {signal: controller.signal, cache: "no-store", credentials: "same-origin"};
      if (body !== undefined) {
        if (!ui.token) throw new Error("本地会话尚未就绪，请刷新页面后重试。");
        options.method = "POST";
        options.headers = {"Content-Type": "application/json", "X-EEG-Token": ui.token};
        options.body = JSON.stringify(body);
      }
      const response = await fetch(path, options);
      let result;
      try { result = await response.json(); } catch (_) { throw new Error(`服务响应无法读取（HTTP ${response.status}）`); }
      if (!response.ok) {
        if (response.status === 403) { ui.ready = false; connection(false); }
        throw new Error(result.error || `请求失败（HTTP ${response.status}）`);
      }
      return result;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("请求超时，请检查本地服务与设备状态。");
      throw error;
    } finally { clearTimeout(timeout); }
  }

  async function action(id, path, body = {}, success = "") {
    if (ui.busy.has(id)) return null;
    ui.busy.add(id); syncControls();
    try {
      const result = await request(path, body);
      if (success) notice(success); else notice("");
      return result;
    } catch (error) { notice(error.message || "操作失败", true); return null; }
    finally { ui.busy.delete(id); syncControls(); }
  }

  function drawChannelSettings() {
    const count = Number($("channel-count").value);
    const container = $("channel-settings"); container.replaceChildren();
    for (let i = 0; i < count; i++) {
      const row = node("div", null, "channel-item");
      const check = node("input"); check.type = "checkbox"; check.id = `channel-check-${i}`; check.checked = ui.selected.has(i); check.setAttribute("aria-label", `CH${i + 1} 参与识别`);
      check.addEventListener("change", () => { if (check.checked) ui.selected.add(i); else ui.selected.delete(i); updateSelectedCount(); });
      const label = node("label", `CH${i + 1}`); label.htmlFor = `channel-name-${i}`;
      const name = node("input"); name.id = `channel-name-${i}`; name.maxLength = 24; name.value = ui.channelNames[i]; name.placeholder = `CH${i + 1} / 实际电极`;
      name.addEventListener("input", () => { ui.channelNames[i] = name.value; });
      row.append(check, label, name); container.append(row);
    }
    updateSelectedCount();
  }

  function updateSelectedCount() { const count = Number($("channel-count").value); write("selected-count", `${[...ui.selected].filter((i) => i < count).length} 个已选`); }

  function sourceChanged() {
    const source = $("source").value;
    $("device-options").hidden = source !== "device";
    $("replay-options").hidden = source !== "replay";
    $("demo-panel").hidden = source !== "demo";
    write("source-help", {demo: "用模拟节律和状态检查软件流程；真实识别效果需实机验证。", device: "连接 ESP32 + ADS1299 采集板。仅在点击开始采集后连接设备。", replay: "按原采样率回放历史信号，可查看分析与质量检查结果；回放不采集校准标签。"}[source]);
    if (source === "replay") replayChanged();
    syncControls();
  }

  function replayChanged() {
    const file = ui.files.find((item) => String(item.id) === $("replay-file").value);
    if (!file) { write("replay-description", "没有可用的回放文件。"); return; }
    write("replay-description", `${file.channels ?? "—"} 通道 · ${file.sample_rate ?? "—"} Hz · ${number(file.seconds, 1)} 秒`);
    if (file.sample_rate != null) $("sample-rate").value = String(file.sample_rate);
    if (file.channels != null && file.channels <= 8) { $("channel-count").value = String(file.channels); drawChannelSettings(); }
  }

  function fillModels(models) {
    const current = $("saved-models").value;
    ui.models = Array.isArray(models) ? models : [];
    $("saved-models").replaceChildren();
    const placeholder = node("option", ui.models.length ? "选择已保存的个人模型" : "没有已保存模型"); placeholder.value = ""; $("saved-models").append(placeholder);
    for (const model of ui.models) { const option = node("option", model.name); option.value = model.name; $("saved-models").append(option); }
    if (ui.models.some((model) => model.name === current)) $("saved-models").value = current;
    syncControls();
  }

  async function bootstrap() {
    const data = await request("/api/bootstrap");
    if (ui.token && ui.token !== data.csrf_token) { ui.hydrated = false; ui.eventKey = ""; ui.metricsKey = ""; }
    ui.token = data.csrf_token; ui.files = Array.isArray(data.files) ? data.files : []; ui.ready = true;
    if (!ui.state) {
      $("sample-rate").value = String(data.defaults?.sample_rate || 250);
      $("channel-count").value = String(data.defaults?.channels || 8);
      drawChannelSettings();
    }
    const selectedFile = $("replay-file").value;
    $("replay-file").replaceChildren();
    for (const file of ui.files) { const option = node("option", file.name); option.value = String(file.id); $("replay-file").append(option); }
    if (ui.files.some((file) => String(file.id) === selectedFile)) $("replay-file").value = selectedFile;
    fillModels(data.models); sourceChanged();
  }

  function syncControls() {
    const running = active(); const state = ui.state || {}; const enabled = ui.ready && ui.connected;
    const busy = ui.busy; const cal = calibrating(); const normal = state.mode !== "impedance";
    $("source-settings").disabled = running || busy.has("start") || busy.has("discover") || !ui.ready;
    $("device-ip").disabled = !enabled || running || busy.has("start") || busy.has("discover");
    $("sample-rate").disabled = $("source").value === "replay";
    $("channel-count").disabled = $("source").value === "replay";
    $("acquisition-mode").disabled = $("source").value !== "device";
    $("start").disabled = !enabled || running || busy.has("start") || busy.has("discover") || ($("source").value === "replay" && !ui.files.length);
    $("stop").disabled = !enabled || !running || busy.has("stop");
    $("discover").disabled = !enabled || busy.has("discover") || busy.has("start") || running;
    for (const button of $("devices").querySelectorAll("button")) button.disabled = !enabled || busy.has("start") || busy.has("discover") || running || $("source").value !== "device";
    $("switch-mode").disabled = !enabled || state.status !== "running" || state.source !== "device" || cal || busy.has("switch-mode");
    write("switch-mode", normal ? "切换为阻抗检测" : "切换为正常 EEG");
    $("demo-state").disabled = !enabled || cal || busy.has("demo-state");
    $("record-start").disabled = !enabled || state.status !== "running" || state.recording?.active || busy.has("record-start");
    $("record-stop").disabled = !enabled || !state.recording?.active || busy.has("record-stop");
    $("add-marker").disabled = !enabled || state.status !== "running" || !state.recording?.active || busy.has("add-marker");
    $("calibration-settings").disabled = cal || busy.has("calibration-start") || busy.has("calibration-train");
    $("calibration-start").disabled = !enabled || state.status !== "running" || !state.quality?.valid || !normal || state.source === "replay" || cal || busy.has("calibration-start") || busy.has("calibration-train");
    $("calibration-abort").disabled = !enabled || !cal || busy.has("calibration-abort");
    $("calibration-train").disabled = !enabled || state.calibration?.phase !== "complete" || !normal || busy.has("calibration-train") || cal;
    write("calibration-train", busy.has("calibration-train") ? "正在训练…" : "训练个人模型");
    $("model-save").disabled = !enabled || !state.model?.available || busy.has("model-save");
    $("model-load").disabled = !enabled || !$("saved-models").value || cal || busy.has("model-load");
    $("feedback-target").disabled = !enabled || busy.has("feedback");
    $("feedback-threshold").disabled = !enabled || busy.has("feedback");
  }

  function connection(online) {
    ui.connected = online;
    $("connection-dot").className = online ? "online" : "offline";
    write("connection-text", online ? "本地服务已连接" : "本地服务连接中断");
    syncControls();
  }

  function resetRealtime(reason = "连接中断，反馈已暂停") {
    write("feedback-value", "—"); write("feedback-label", "反馈已暂停");
    $("ring-progress").style.strokeDashoffset = "408.407";
    tag("feedback-status", "已暂停", "warn");
    write("prediction-label", "暂不可识别"); write("prediction-reason", reason);
    write("prediction-score", "方向得分 — · 非概率"); $("score-dot").style.display = "none";
    write("quality-summary", "等待新数据"); $("quality-summary").className = "bad";
    write("quality-detail", reason); write("data-age", "—");
    $("waveform-empty").hidden = false;
  }

  function renderQuality(quality = {}, impedances) {
    write("quality-summary", quality.valid ? "信号可用" : "反馈已暂停");
    $("quality-summary").className = quality.valid ? "good" : "bad";
    write("quality-detail", quality.summary || "正在积累有效信号窗口");
    const channels = Array.isArray(quality.channels) ? quality.channels : [];
    const body = $("quality-rows"); body.replaceChildren();
    if (!channels.length) { const row = node("tr"); const cell = node("td", "等待有效采集窗口", "empty-cell"); cell.colSpan = 7; row.append(cell); body.append(row); return; }
    channels.forEach((channel, i) => {
      const row = node("tr"); row.append(node("td", channel.name || `CH${i + 1}`));
      const state = node("td"); state.append(node("span", stateLabels[channel.state] || channel.state || "待检查", `tag ${["good", "valid", "ok"].includes(channel.state) ? "good" : "warn"}`)); row.append(state);
      row.append(node("td", number(channel.rms_uv)), node("td", number(channel.peak_uv)), node("td", number(channel.saturated_pct, 1, "%")), node("td", finite(channel.line_noise_ratio) ? number(channel.line_noise_ratio * 100, 1, "%") : "—"));
      const impedance = Array.isArray(impedances) ? impedances[i] : impedances?.[channel.name];
      const impedanceValue = typeof impedance === "object" && impedance !== null ? (impedance.kohm ?? (finite(impedance.ohms) ? impedance.ohms / 1000 : null)) : impedance;
      row.append(node("td", finite(impedanceValue) ? number(impedanceValue, 1, " kΩ") : "—")); body.append(row);
    });
  }

  function renderFeedback(feedback = {}, prediction = {}, quality = {}, state) {
    const fresh = finite(state.telemetry?.age_s) && state.telemetry.age_s < 3;
    const valid = state.status === "running" && state.mode !== "impedance" && quality.valid && fresh;
    const value = valid && finite(feedback.value) ? clamp(feedback.value) : null;
    write("feedback-value", value === null ? "—" : `${Math.round(value * 100)}%`);
    write("feedback-label", value === null ? "等待有效信号" : feedback.active ? "已达到目标" : "向目标靠近");
    $("ring-progress").style.strokeDashoffset = String(408.407 * (1 - (value || 0)));
    $("ring-progress").style.stroke = value !== null && feedback.active ? "#137f79" : "#80b7af";
    tag("feedback-status", value === null ? "已暂停" : feedback.active ? "达到目标" : "反馈中", value === null ? "warn" : "good");
    if (feedback.status) write("feedback-explanation", feedback.status);
    const usePrediction = valid && prediction.state !== "unavailable";
    write("prediction-label", usePrediction && prediction.label ? prediction.label : prediction.state === "uncertain" && valid ? "暂不确定" : "暂不可识别");
    write("prediction-reason", prediction.reason || (valid ? "完成个人模型训练后启用" : "信号质量不满足条件，在线分类已暂停"));
    const score = usePrediction && finite(prediction.score) ? clamp(prediction.score) : null;
    $("score-dot").style.display = score === null ? "none" : "block";
    if (score !== null) $("score-dot").style.left = `${score * 100}%`;
    write("prediction-score", `方向得分 ${score === null ? "—" : score.toFixed(3)} · 非概率`);
  }

  function renderCalibration(calibration = {}) {
    const phase = calibration.phase || "idle";
    write("cue-phase", phaseLabels[phase] || phase);
    write("cue-label", phase === "rest" ? "静息，保持身体放松" : phase === "complete" ? "采集完成，可以训练" : calibration.label || (phase === "idle" ? "准备好，跟随提示" : phaseLabels[phase] || "等待提示"));
    write("cue-countdown", ["rest", "cue", "active"].includes(phase) ? `${Math.max(0, Math.ceil(calibration.remaining_s || 0))} s` : phase === "complete" ? "✓" : "—");
    $("cue-box").className = `cue-box ${phase === "active" ? "active" : ""}`;
    let progress = finite(calibration.progress) ? calibration.progress : 0; if (progress > 1) progress /= 100;
    $("calibration-progress").style.width = `${clamp(progress) * 100}%`;
    const details = [];
    if (calibration.total_trials) details.push(`试次 ${calibration.trial || 0} / ${calibration.total_trials}`);
    if (calibration.accepted_windows != null) details.push(`有效窗口 ${calibration.accepted_windows}`);
    if (calibration.message) details.push(calibration.message);
    write("calibration-detail", details.join(" · ") || "请先开始采集并确认信号质量");
  }

  function renderModel(model = {}) {
    tag("model-badge", model.available ? model.compatible === false ? "配置不匹配" : "个人模型就绪" : "尚无模型", model.available ? model.compatible === false ? "warn" : "good" : "");
    const labels = model.labels || [$("label-a").value, $("label-b").value];
    write("score-a", labels[0] || "类别 A"); write("score-b", labels[1] || "类别 B");
    write("model-warning", model.reason || "模型与采样率、通道配置及模式绑定，配置变化后需重新核对或校准。");
    const metrics = model.metrics; const hasMetrics = model.available && metrics && Object.keys(metrics).length > 0; $("model-results").hidden = !hasMetrics;
    const key = JSON.stringify({metrics, labels, context: model.context});
    if (!hasMetrics || key === ui.metricsKey) return;
    ui.metricsKey = key; $("model-metrics").replaceChildren(); $("confusion-matrix").replaceChildren();
    const entries = [];
    for (const [keys, label, percentage] of [[["holdout_accuracy", "accuracy", "test_accuracy", "validation_accuracy"], "留出窗口准确率", true], [["balanced_accuracy"], "平衡准确率", true], [["trial_accuracy"], "留出试次准确率", true], [["train_trials", "n_train_trials"], "训练试次", false], [["test_trials", "n_test_trials", "validation_trials"], "验证试次", false], [["train_windows", "n_train"], "训练窗口", false], [["test_windows", "n_test"], "验证窗口", false]]) {
      const keyName = keys.find((k) => metrics[k] != null);
      if (keyName) entries.push([label, percentage ? number(metrics[keyName] * 100, 1, "%") : Array.isArray(metrics[keyName]) ? metrics[keyName].length : metrics[keyName]]);
    }
    if (!entries.length) entries.push(["验证状态", metrics.message || "模型已训练"]);
    for (const [label, value] of entries) { const item = node("div", null, "model-metric"); item.append(node("strong", value), node("span", label)); $("model-metrics").append(item); }
    const context = model.context;
    write("model-context", typeof context === "string" ? context : context ? `${context.source_kind === "demo" ? "模拟校准" : "个人校准"}${context.sample_rate ? ` · ${context.sample_rate} Hz` : ""}` : "按试次划分训练与验证");
    const matrix = metrics.confusion_matrix || metrics.confusion;
    if (Array.isArray(matrix) && matrix.length >= 2) {
      const table = node("table"); const head = node("thead"); const hr = node("tr"); hr.append(node("th", "真实 ↓ / 预测 →"), node("th", labels[0]), node("th", labels[1])); head.append(hr); table.append(head); const body = node("tbody");
      matrix.slice(0, 2).forEach((values, i) => { const row = node("tr"); row.append(node("th", labels[i])); if (Array.isArray(values)) values.slice(0, 2).forEach((value, j) => row.append(node("td", value, i === j ? "diagonal" : ""))); body.append(row); }); table.append(body); $("confusion-matrix").append(table);
    }
  }

  function renderEvents(events = []) {
    if (!Array.isArray(events)) return;
    const key = JSON.stringify(events); if (key === ui.eventKey) return; ui.eventKey = key;
    if (!events.length) { $("event-list").replaceChildren(node("li", "采集、质量变化、标记与校准事件会显示在这里。", "event-empty")); return; }
    $("event-list").replaceChildren();
    events.slice(-60).reverse().forEach((event) => {
      const row = node("li", null, ["error", "warning"].includes(event.level) ? event.level : "");
      let time = event.time; if (finite(time)) time = new Date(time < 1e12 ? time * 1000 : time).toLocaleTimeString("zh-CN", {hour12: false});
      else if (typeof time === "string" && time.includes("T")) { const date = new Date(time); if (!Number.isNaN(date.getTime())) time = date.toLocaleTimeString("zh-CN", {hour12: false}); }
      row.append(node("time", time || "—"), node("span", event.message || "")); $("event-list").append(row);
    });
  }

  function renderBands(bands = {}, dominant) {
    bandDefinitions.forEach(([name]) => {
      let value = bands[name]; if (value && typeof value === "object") value = value.relative ?? value.ratio ?? value.power;
      const bounded = finite(value) ? clamp(value) : 0;
      $(`band-${name}`).style.width = `${bounded * 100}%`;
      write(`band-value-${name}`, finite(value) ? `${Math.round(bounded * 100)}%` : "—");
    });
    const band = bandDefinitions.find(([name]) => name === dominant);
    tag("dominant-band", band ? `${band[1].split(" · ")[0]} 主导` : "等待分析");
  }

  function canvasContext(id) {
    const canvas = $(id); const rect = canvas.getBoundingClientRect(); const scale = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, rect.width); const height = Math.max(1, rect.height);
    if (canvas.width !== Math.round(width * scale) || canvas.height !== Math.round(height * scale)) { canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale); }
    const ctx = canvas.getContext("2d"); ctx.setTransform(scale, 0, 0, scale, 0, 0); ctx.clearRect(0, 0, width, height);
    ctx.font = '10px "Segoe UI", "Microsoft YaHei", sans-serif'; return {ctx, width, height};
  }

  function drawWaveform(signal = {}, config = {}) {
    const {ctx, width, height} = canvasContext("waveform"); const traces = signal.traces || []; const names = config.channel_names || [];
    const left = 54; const right = width - 11; const top = 9; const bottom = height - 23; const plotWidth = Math.max(1, right - left);
    const rows = traces.length || config.channels || 8; const rowHeight = (bottom - top) / rows;
    ctx.strokeStyle = "#edf2f2"; ctx.lineWidth = 1;
    for (let x = 0; x <= 6; x++) { const px = left + plotWidth * x / 6; ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke(); }
    for (let channel = 0; channel < rows; channel++) {
      const center = top + (channel + .5) * rowHeight; ctx.strokeStyle = "#edf2f2"; ctx.beginPath(); ctx.moveTo(left, center); ctx.lineTo(right, center); ctx.stroke();
      const name = names[channel] || `CH${channel + 1}`; ctx.fillStyle = channelColors[channel % channelColors.length]; ctx.fillText(name.length > 7 ? `${name.slice(0, 6)}…` : name, 0, center + 3);
      const trace = traces[channel]; if (!Array.isArray(trace) || trace.length < 2) continue;
      const step = Math.max(1, Math.floor(trace.length / 100)); const values = []; for (let j = 0; j < trace.length; j += step) if (finite(trace[j])) values.push(trace[j]);
      values.sort((a, b) => a - b); const middle = values[Math.floor(values.length / 2)] || 0;
      const high = values[Math.floor(values.length * .95)] || 0; const low = values[Math.floor(values.length * .05)] || 0;
      const amplitude = Math.max(25, Math.abs(high - middle), Math.abs(low - middle));
      ctx.save(); ctx.beginPath(); ctx.rect(left, center - rowHeight * .45, plotWidth, rowHeight * .9); ctx.clip();
      ctx.strokeStyle = channelColors[channel % channelColors.length]; ctx.lineWidth = 1.15; ctx.beginPath(); let started = false;
      trace.forEach((value, index) => { if (!finite(value)) { started = false; return; } const x = left + index / (trace.length - 1) * plotWidth; const y = center - clamp((value - middle) / amplitude, -1.3, 1.3) * rowHeight * .32; if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; } }); ctx.stroke(); ctx.restore();
      ctx.textAlign = "right"; ctx.fillStyle = "#8ca09f"; ctx.font = '8px "Segoe UI", sans-serif'; ctx.fillText(`±${Math.round(amplitude)} μV`, right, center - rowHeight * .32); ctx.textAlign = "left"; ctx.font = '10px "Segoe UI", "Microsoft YaHei", sans-serif';
    }
    const times = signal.times || []; const duration = times.length > 1 && finite(times[times.length - 1]) && finite(times[0]) ? times[times.length - 1] - times[0] : 0;
    ctx.fillStyle = "#9badb5"; ctx.textAlign = "left"; ctx.fillText(duration > 0 ? `${number(-duration, 1)} s` : "时间 / s", left, height - 5); ctx.textAlign = "right"; ctx.fillText("现在", right, height - 5); ctx.textAlign = "left";
    $("waveform-empty").hidden = traces.some((trace) => Array.isArray(trace) && trace.length > 1) && ui.connected;
  }

  function drawSpectrum(signal = {}) {
    const {ctx, width, height} = canvasContext("spectrum"); const frequencies = Array.isArray(signal.freqs) ? signal.freqs : []; let powers = Array.isArray(signal.psd) ? signal.psd : [];
    if (Array.isArray(powers[0])) { const rows = powers; powers = frequencies.map((_, index) => rows.reduce((sum, row) => sum + (finite(row[index]) ? row[index] : 0), 0) / rows.length); }
    const left = 35; const right = width - 10; const top = 13; const bottom = height - 23; const maxHz = Math.min(60, frequencies[frequencies.length - 1] || 60); const plotWidth = right - left; const plotHeight = bottom - top;
    ctx.fillStyle = "#f0f8f5"; ctx.fillRect(left + 8 / maxHz * plotWidth, top, 5 / maxHz * plotWidth, plotHeight);
    ctx.strokeStyle = "#ecf1f2"; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) { const y = top + i * plotHeight / 4; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke(); }
    ctx.fillStyle = "#96a6ae"; ctx.textAlign = "center";
    for (let hz = 0; hz <= maxHz; hz += 10) { const x = left + hz / maxHz * plotWidth; ctx.fillText(String(hz), x, height - 6); }
    ctx.textAlign = "right"; ctx.fillText("Hz", width - 1, height - 6); ctx.textAlign = "left"; ctx.fillText("dB", 0, 10);
    const points = frequencies.map((hz, i) => [hz, finite(powers[i]) && powers[i] > 0 ? 10 * Math.log10(powers[i]) : null]).filter(([hz, power]) => hz >= 1 && hz <= maxHz && power !== null);
    if (points.length < 2) return;
    const ys = points.map((point) => point[1]); const min = Math.floor(Math.min(...ys) / 10) * 10; const max = Math.max(min + 10, Math.ceil(Math.max(...ys) / 10) * 10);
    ctx.fillStyle = "#95a6ad"; ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) ctx.fillText(String(Math.round(max - (max - min) * i / 4)), left - 7, top + plotHeight * i / 4 + 3);
    ctx.textAlign = "left"; ctx.strokeStyle = "#2b8d85"; ctx.lineWidth = 1.7; ctx.beginPath();
    points.forEach(([hz, power], index) => { const x = left + hz / maxHz * plotWidth; const y = bottom - (power - min) / (max - min) * plotHeight; if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke();
  }

  function render(state) {
    ui.state = state;
    if (!ui.hydrated) {
      ui.hydrated = true;
      if (state.source && state.config) {
        $("source").value = state.source; $("sample-rate").value = String(state.config.sample_rate || 250); $("channel-count").value = String(state.config.channels || 8);
        (state.config.channel_names || []).forEach((name, i) => { ui.channelNames[i] = name; });
        if (Array.isArray(state.config.selected_channels)) ui.selected = new Set(state.config.selected_channels);
        $("acquisition-mode").value = state.mode || "eeg"; if (state.config.device_ip) $("device-ip").value = state.config.device_ip;
        if (state.config.replay_id) $("replay-file").value = state.config.replay_id;
        drawChannelSettings(); sourceChanged();
      }
      if (state.feedback?.target) $("feedback-target").value = state.feedback.target;
      if (finite(state.feedback?.threshold)) { $("feedback-threshold").value = String(Math.round(state.feedback.threshold * 100)); write("threshold-value", `${$("feedback-threshold").value}%`); }
    }
    const statusNames = {idle: "等待开始", connecting: "连接中", running: "正在采集", stopped: "已停止", error: "采集异常"};
    const sourceNames = {demo: "模拟数据", device: "实机采集", replay: "文件回放"};
    write("session-status", statusNames[state.status] || state.status || "等待开始"); write("session-message", `${state.source === "device" && state.config?.device_ip ? `实际连接目标：${state.config.device_ip}。` : ""}${state.message || (state.mode === "impedance" ? "当前为阻抗检测模式" : "信号持续通过质量门控后才开启反馈")}`);
    write("source-badge", sourceNames[state.source] || "尚未开始"); $("source-badge").className = `pill ${state.source === "demo" ? "warn" : state.source === "device" ? "good" : ""}`;
    const config = state.config || {};
    write("waveform-info", `${config.channels || 8} CH / ${config.sample_rate || 250} Hz${state.mode === "impedance" ? " · 阻抗" : ""}`);
    const telemetry = state.telemetry || {};
    $("data-age").replaceChildren(document.createTextNode(number(telemetry.age_s, 2)), node("em", "s"));
    write("sample-count", `${integer(telemetry.received_samples)} 个样本`);
    write("packet-errors", `${integer(telemetry.lost_packets)} / ${integer(telemetry.crc_errors)}`);
    write("packet-count", `${integer(telemetry.packets)} 包 · 异常 ${integer(telemetry.bad_packets)} · 重连 ${integer(telemetry.reconnects)}`);
    renderQuality(state.quality, state.impedances); renderFeedback(state.feedback, state.prediction, state.quality, state); renderCalibration(state.calibration); renderModel(state.model); renderEvents(state.events);
    renderBands(state.signal?.bands, state.signal?.dominant_band); drawWaveform(state.signal, config); drawSpectrum(state.signal);
    const recording = state.recording || {};
    tag("recording-badge", recording.active ? "● 录制中" : "未录制", recording.active ? "bad" : "");
    write("recording-info", recording.path ? `${recording.active ? "正在保存" : "已保存"} ${integer(recording.samples)} 样本 · ${recording.path}` : "录制文件保存在本项目的输出目录。");
    if (active() && state.source === "demo" && state.demo_state && !ui.busy.has("demo-state") && document.activeElement !== $("demo-state")) $("demo-state").value = state.demo_state;
    syncControls();
  }

  async function poll() {
    if (ui.polling) return;
    ui.polling = true;
    try {
      if (!ui.ready) await bootstrap();
      const state = await request("/api/state");
      ui.lastStateAt = Date.now();
      connection(true); render(state); ui.errors = 0;
      if (ui.connectionNotice) notice("");
    } catch (error) {
      ui.errors++; ui.ready = false; connection(false); resetRealtime(error.message || "无法连接本地服务");
      if (ui.errors === 1) { notice("本地服务连接中断，已清除实时分类与反馈。页面会自动重试。", true); ui.connectionNotice = true; }
    } finally { ui.polling = false; window.setTimeout(poll, ui.errors ? Math.min(3000, 500 * ui.errors) : 250); }
  }

  function guardFreshness() {
    if (!ui.lastStateAt || !ui.connected) return;
    const responseAge = Date.now() - ui.lastStateAt;
    const sampleAge = ui.state?.telemetry?.age_s;
    if (responseAge > 2000) resetRealtime("状态更新超时，等待本地服务的新数据");
    else if (active() && finite(sampleAge) && sampleAge + responseAge / 1000 >= 3) resetRealtime("采集数据已过期，等待新数据");
  }

  async function refreshLists() {
    try { await bootstrap(); }
    catch (error) { notice(`操作已完成，但列表刷新失败：${error.message}`, true); }
  }

  // Keep the same explicit selection policy in the eight-channel workspace.
  function resolveDeviceAddress(entries, preferred) {
    const devices = [...new Set((Array.isArray(entries) ? entries : []).map((entry) => typeof entry === "string" ? entry : entry?.ip).filter((ip) => typeof ip === "string" && ip.trim()))];
    if (devices.length === 1) return { devices, ip: devices[0], kind: "unique" };
    if (devices.length > 1) return { devices, ip: devices.includes(preferred) ? preferred : "", kind: "multiple" };
    return { devices, ip: preferred, kind: preferred ? "manual" : "missing" };
  }

  async function discoverDevice(forConnection = false) {
    const preferred = $("device-ip").value.trim();
    const results = $("devices"); results.textContent = "正在确认设备地址…";
    let result;
    try { result = await request("/api/discover", { device_ip: preferred }); }
    catch (error) { results.textContent = `设备发现未完成：${error.message}`; throw error; }
    const selection = resolveDeviceAddress(result.devices, preferred);
    $("device-ip").value = selection.ip;
    let message;
    if (selection.kind === "unique") {
      message = preferred && preferred !== selection.ip ? `设备地址已从 ${preferred} 更新为 ${selection.ip}（唯一身份回复）。` : `已确认并选用设备 ${selection.ip}。`;
    } else if (selection.kind === "multiple") {
      message = selection.ip ? `发现多个设备，已确认当前选择 ${selection.ip}。` : "发现多个设备，请点击下方地址选择后再开始采集；尚未选用任何设备。";
    } else if (selection.kind === "manual") {
      message = forConnection ? `未收到身份回复，正尝试手动地址 ${selection.ip}。` : `未收到身份回复，已保留手动地址 ${selection.ip}；点击开始采集可尝试直连。`;
    } else message = "未收到设备身份回复。请检查帽子与网络，或填写已知设备 IP 后再开始采集。";
    const messageNode = node("span", message, "help");
    results.replaceChildren(messageNode);
    for (const ip of selection.devices) {
      const button = node("button", ip, "button subtle"); button.type = "button";
      button.addEventListener("click", () => {
        if (active() || ui.busy.has("start") || ui.busy.has("discover") || !ui.ready || !ui.connected || $("source").value !== "device") return;
        $("device-ip").value = ip;
        messageNode.textContent = `已选用设备 ${ip}，开始采集前会再次确认地址。`;
        notice(messageNode.textContent);
      });
      results.append(button);
    }
    syncControls(); notice(message, !selection.ip);
    return selection.ip;
  }

  async function startAcquisition() {
    if (active() || ui.busy.has("start") || ui.busy.has("discover") || !ui.ready || !ui.connected) return;
    const source = $("source").value; const channels = Number($("channel-count").value);
    const selected = [...ui.selected].filter((index) => index < channels).sort((a, b) => a - b);
    if (!selected.length) { notice("请至少选择一个参与分析的通道。", true); return; }
    const names = ui.channelNames.slice(0, channels).map((name, index) => name.trim() || `CH${index + 1}`);
    ui.busy.add("start"); syncControls();
    try {
      const ip = source === "device" ? await discoverDevice(true) : "";
      if (source === "device" && !ip) return;
      await request("/api/start", {source, sample_rate: Number($("sample-rate").value), channels, channel_names: names, selected_channels: selected, device_ip: ip, replay_id: $("replay-file").value, demo_state: $("demo-state").value, mode: source === "device" ? $("acquisition-mode").value : "eeg"});
    } catch (error) { notice(error.message || "操作失败", true); }
    finally { ui.busy.delete("start"); syncControls(); }
  }

  function bindControls() {
    $("source").addEventListener("change", sourceChanged);
    $("channel-count").addEventListener("change", drawChannelSettings);
    $("replay-file").addEventListener("change", replayChanged);
    $("saved-models").addEventListener("change", syncControls);
    $("start").addEventListener("click", startAcquisition);
    $("stop").addEventListener("click", async () => { const result = await action("stop", "/api/stop"); if (result) { resetRealtime("采集已停止"); await refreshLists(); } });
    $("discover").addEventListener("click", async () => {
      if (active() || ui.busy.has("start") || ui.busy.has("discover") || !ui.ready || !ui.connected) return;
      ui.busy.add("discover"); syncControls();
      try { await discoverDevice(); }
      catch (error) { notice(error.message || "发现失败", true); }
      finally { ui.busy.delete("discover"); syncControls(); }
    });
    $("demo-state").addEventListener("change", () => { if (ui.state?.source === "demo" && active()) action("demo-state", "/api/demo", {state: $("demo-state").value}); });
    $("switch-mode").addEventListener("click", () => action("switch-mode", "/api/mode", {mode: ui.state?.mode === "impedance" ? "eeg" : "impedance"}));
    const setFeedback = () => action("feedback", "/api/feedback", {target: $("feedback-target").value, threshold: Number($("feedback-threshold").value) / 100});
    $("feedback-target").addEventListener("change", setFeedback);
    $("feedback-threshold").addEventListener("input", () => write("threshold-value", `${$("feedback-threshold").value}%`));
    $("feedback-threshold").addEventListener("change", setFeedback);
    $("record-start").addEventListener("click", () => action("record-start", "/api/record/start"));
    $("record-stop").addEventListener("click", async () => { const result = await action("record-stop", "/api/record/stop", {}, "录制已结束，文件保存在本项目输出目录。"); if (result) await refreshLists(); });
    const marker = async () => { const label = $("marker-label").value.trim(); if (!label) { notice("请填写事件标记内容。", true); return; } const result = await action("add-marker", "/api/marker", {label}); if (result) $("marker-label").value = ""; };
    $("add-marker").addEventListener("click", marker); $("marker-label").addEventListener("keydown", (event) => { if (event.key === "Enter" && !$("add-marker").disabled) marker(); });
    $("calibration-start").addEventListener("click", () => { const labels = [$("label-a").value.trim(), $("label-b").value.trim()]; if (!labels[0] || !labels[1] || labels[0] === labels[1]) { notice("请填写两个不同的任务标签。", true); return; } action("calibration-start", "/api/calibration/start", {labels, trials_per_class: Number($("trials").value), trial_seconds: Number($("trial-seconds").value), rest_seconds: 3, cue_seconds: 1}); });
    $("calibration-abort").addEventListener("click", () => action("calibration-abort", "/api/calibration/abort"));
    $("calibration-train").addEventListener("click", () => action("calibration-train", "/api/calibration/train", {}, "训练完成。请检查按试次留出验证结果，并用独立试验验证真实效果。"));
    $("model-save").addEventListener("click", async () => { const result = await action("model-save", "/api/model/save"); if (result) { notice(`模型已保存：${result.name || "个人模型"}`); if (result.name && !ui.models.some((model) => model.name === result.name)) fillModels([...ui.models, {name: result.name}]); } });
    $("model-load").addEventListener("click", () => action("model-load", "/api/model/load", {name: $("saved-models").value}, "个人模型已载入。请确认当前电极与采样配置匹配。"));
    let resizeTimer; window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { drawWaveform(ui.state?.signal, ui.state?.config); drawSpectrum(ui.state?.signal); }, 100); });
  }

  bandDefinitions.forEach(([name, label]) => { const row = node("div", null, "band-row"); const track = node("div", null, "band-track"); const fill = node("div"); fill.id = `band-${name}`; track.append(fill); const value = node("strong", "—"); value.id = `band-value-${name}`; row.append(node("span", label), track, value); $("band-bars").append(row); });
  drawChannelSettings(); bindControls(); sourceChanged(); drawWaveform(); drawSpectrum(); window.setInterval(guardFreshness, 250); poll();
})();
