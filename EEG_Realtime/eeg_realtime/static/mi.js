"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const CHANNELS = ["O2", "O1", "P8", "P7", "C4", "C3", "FP1", "FP2"];
  const LABELS = { rest: "静息", left_mi: "左手运动想象", right_mi: "右手运动想象", interval: "十字间隔", unknown: "未知 / 无效标注" };
  const SHORT_LABELS = { rest: "静息", left_mi: "左手想象", right_mi: "右手想象", interval: "间隔", unknown: "未知" };
  const SOURCES = { demo: "演示数据", device: "真实设备", replay: "录制回放" };
  const STATUS_LABELS = { ready: "有效", ok: "有效", predicted: "有效", valid: "有效", uncertain: "候选预测 · 低确定性", low_confidence: "低置信", invalid: "无效", unavailable: "不可用", no_model: "未加载模型", warming_up: "积累窗口", rejected: "已拒绝", artifact: "信号干扰", idle: "待开始", running: "训练中", complete: "已完成", error: "错误", stopped: "已停止", aborted: "提前结束", recording: "录制中" };
  const COLORS = { left_mi: "#d3e9e6", right_mi: "#dbe7f6", rest: "#e3ece1", interval: "#edf0f2", unknown: "#e4e8eb" };
  const REASONS = {
    "No trained eight-channel model loaded": "尚未加载训练好的八通道模型。",
    "Waiting for causal filter warmup/recovery": "正在积累滤波预热或间断恢复所需的连续信号。",
    "raw signal passed engineering checks": "原始信号通过当前信号质量检查。",
    "Invalid filtered epoch": "当前滤波窗口无效，暂不输出有效识别。",
    "Confidence is below the fixed rejection threshold": "置信度或类别得分差距低于原诊断门槛；保留候选类别供参考。",
    "Epoch is outside the training feature distribution; tentative result only": "当前窗口特征超出训练数据分布；保留候选类别与得分供参考。",
    "Model did not pass preliminary grouped holdout validation": "模型未通过按组留出验证；保留候选类别与得分供参考。",
    "Valid personal model prediction": "个人模型给出有效预测。模型置信度不等同于实际分类准确率。",
    "Passed preliminary grouped holdout": "通过初步按组留出验证。",
    "Grouped holdout below the fixed engineering threshold": "按组留出验证结果未达到预设门槛。",
  };
  const SCOPES = { whole_session_holdout: "按完整录制划分训练、校准与验证", whole_trial_holdout_within_available_sessions: "在现有录制内按完整试次划分训练、校准与验证" };
  const QUALITY_REASONS = { dc_offset: "原始直流偏置较大", raw_amplitude: "原始幅度较大", nonfinite: "非有限值", saturated: "饱和", flat: "平线", filter_warmup: "滤波预热", hard_quality: "硬质量检查", outside_complete_task: "完整有效任务范围外" };
  // Explicit allowance for browser timestamp quantization up to 1 ms. This is
  // an engineering timing budget, not a bound on display or hardware latency.
  const BROWSER_CLOCK_QUANTIZATION_BUDGET_S = .001;
  const app = {
    token: "", state: null, sessions: [], models: [], selected: new Set(), busy: false,
    polling: null, online: false, offset: null, bestRtt: Infinity, clockSetAt: 0,
    recordingOffset: null, cueKey: "", acked: new Set(), recordRequested: false,
    focusLost: false, safetyStopping: false, seeking: false, lastStateAt: 0, lastServerTime: -Infinity,
    lastRecordId: null, previousTraining: "idle", historyKey: "", eventsKey: "",
    metricsKey: "", diagnosticsKey: "", waveDirty: true, waveAutoAmplitude: null, waveAutoUpdated: 0, noticeTimer: null, safetyMessage: "",
    recordingMode: false, restoreFocus: null, inertNodes: [], presentedCue: null,
    clockUncertainty: null, clockResolution: null, lastFrameAt: null, lastVisibleElapsed: null,
    displayInFlight: false, lastDisplayElapsed: -Infinity, lastDisplayPhase: undefined,
  };
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const label = (value, short = false) => (short ? SHORT_LABELS : LABELS)[value] || (value ? String(value) : "—");
  const fmt = (value, digits = 1) => finite(value) ? value.toFixed(digits) : "—";
  const percent = (value) => finite(value) ? `${Math.round(clamp(value, 0, 1) * 100)}%` : "—";
  const isRecording = () => Boolean(app.state?.recording?.active);
  const isTraining = () => app.state?.training?.status === "running";
  const isLive = () => app.state?.status === "running" && ["demo", "device"].includes(app.state?.source);
  const sourceText = (source) => SOURCES[source] || source || "—";

  function reasonText(value) {
    if (typeof value !== "string") return "";
    if (REASONS[value]) return REASONS[value];
    const states = { nonfinite: "存在非有限数值", saturated: "信号饱和", flat: "信号过平", artifact: "信号干扰" };
    return value.replace(/\b(O2|O1|P8|P7|C4|C3|FP1|FP2): (nonfinite|saturated|flat|artifact)\b/g, (_, channel, state) => `${channel}：${states[state]}`);
  }

  function setRecordingMode(active) {
    if (app.recordingMode === active) return;
    app.recordingMode = active;
    document.body.classList.toggle("recording-mode", active);
    $("focusRecordControls").hidden = !active;
    $("cueCard").setAttribute("role", active ? "dialog" : "region");
    if (active) {
      app.restoreFocus = document.activeElement;
      $("cueCard").setAttribute("aria-modal", "true");
      app.inertNodes = [...document.querySelectorAll(".topbar, .page-heading, .tabs, .control-stack, #replayPane, #trainingPane, #robotPane, .waveform-card, .online-grid, .event-panel, footer")].map((node) => [node, node.inert]);
      for (const [node] of app.inertNodes) node.inert = true;
      // Element focus stays in this window; no Fullscreen API or window focus change.
      if (document.hasFocus()) $("focusStopRecordButton").focus({ preventScroll: true });
    } else {
      $("cueCard").removeAttribute("aria-modal");
      for (const [node, previous] of app.inertNodes) node.inert = previous;
      app.inertNodes = [];
      app.presentedCue = null;
      if (document.hasFocus() && app.restoreFocus?.isConnected && !app.restoreFocus.disabled) app.restoreFocus.focus({ preventScroll: true });
      app.restoreFocus = null;
      app.waveDirty = true;
    }
  }

  function cueIsVisible() {
    if (document.hidden || !document.hasFocus() || !app.recordingMode) return false;
    const cue = $("cueSymbol");
    const rect = cue.getBoundingClientRect();
    const style = window.getComputedStyle(cue);
    const viewportWidth = document.documentElement.clientWidth;
    const viewportHeight = document.documentElement.clientHeight;
    if (rect.width < 1 || rect.height < 1 || rect.left < 0 || rect.top < 0 || rect.right > viewportWidth || rect.bottom > viewportHeight || style.display === "none" || style.visibility !== "visible" || Number(style.opacity) === 0) return false;
    const front = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    return front === cue || cue.contains(front);
  }

  function confirmPresentedCue(recording, phase, onset) {
    const key = `${recording.id}:${phase.index}`;
    if (!cueIsVisible()) {
      app.presentedCue = null;
      void safetyStop("提示未完整显示在当前视口中，已停止录制并保留数据。");
      return false;
    }
    if (app.acked.has(key)) { app.lastVisibleElapsed = onset; return true; }
    if (app.presentedCue?.key === key) {
      // A long task can postpone the first actual paint. Do not backdate its
      // onset to the previous rAF, or accept an unbounded paint opportunity.
      if (onset - app.presentedCue.onset > .10) {
        void safetyStop("提示绘制延迟过大，已停止录制并保留数据。");
        return false;
      }
      app.lastVisibleElapsed = onset;
      void acknowledge(recording, phase, onset);
      app.presentedCue = null;
      return true;
    } else {
      app.presentedCue = { key, onset };
      return false;
    }
  }

  function clockText(seconds, decimals = false) {
    if (!finite(seconds)) return decimals ? "00:00.0" : "00:00";
    const value = Math.max(0, seconds);
    const mins = Math.floor(value / 60).toString().padStart(2, "0");
    const secs = decimals ? (value % 60).toFixed(1).padStart(4, "0") : Math.floor(value % 60).toString().padStart(2, "0");
    return `${mins}:${secs}`;
  }

  function notify(message, error = false, persist = false) {
    clearTimeout(app.noticeTimer);
    $("notice").textContent = message;
    $("notice").classList.toggle("error", error);
    $("notice").hidden = !message;
    if (message && !persist) app.noticeTimer = setTimeout(() => { $("notice").hidden = true; }, 9000);
  }

  function updateClock(serverTime, before, after, clockResolution) {
    if (!finite(serverTime) || !finite(clockResolution) || clockResolution <= 0 || isRecording()) return;
    const rtt = after - before;
    if (!finite(rtt) || rtt < 0) return;
    if (rtt <= app.bestRtt || app.offset === null) {
      app.offset = serverTime - (before + after) / 2;
      app.bestRtt = rtt;
      app.clockResolution = clockResolution;
      app.clockSetAt = after;
    }
  }

  async function synchronizeClock() {
    app.offset = null; app.bestRtt = Infinity; app.clockSetAt = 0;
    app.clockUncertainty = null; app.clockResolution = null;
    $("clockSyncStatus").textContent = "正在检查提示时钟…";
    for (let attempt = 0; attempt < 5; attempt++) {
      const before = performance.now() / 1000;
      try {
        const result = await request("/api/mi/clock", undefined, { timeout: 1200 });
        updateClock(result.server_time, before, performance.now() / 1000, result.clock_resolution_s);
      } catch (_) { /* Other lightweight samples may still provide a valid bound. */ }
    }
    // A fast request can span less than one server tick. RTT alone therefore
    // does not cover the difference between its timestamp and later receipts.
    const budget = app.bestRtt / 2 + app.clockResolution + BROWSER_CLOCK_QUANTIZATION_BUDGET_S;
    if (app.offset === null || !finite(budget) || budget > .05 || app.bestRtt > .10 || performance.now() / 1000 - app.clockSetAt >= 2) {
      app.offset = null;
      $("clockSyncStatus").textContent = "提示时钟检查未通过，请稍后重试。";
      throw new Error("提示时钟通信延迟过大或已过期，或计时分辨率使对齐预算超过 50 毫秒；尚未开始录制。请刷新页面后重试。");
    }
    app.clockUncertainty = budget;
    $("clockSyncStatus").textContent = `提示时钟估计误差 ±${Math.ceil(budget * 1000)} 毫秒（软件估计，未经硬件同步测量）`;
  }

  async function request(path, body, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeout || 10000);
    try {
      const response = await fetch(path, {
        method: body === undefined ? "GET" : "POST",
        headers: body === undefined ? {} : { "Content-Type": "application/json", "X-EEG-Token": app.token },
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store", signal: controller.signal, keepalive: Boolean(options.keepalive),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `请求失败（${response.status}）`);
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("请求超时，请检查本地服务是否仍在运行。");
      throw error;
    } finally { clearTimeout(timeout); }
  }

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  async function bootstrap() {
    const data = await request("/api/mi/bootstrap");
    if (data.csrf_token) app.token = data.csrf_token;
    window.MIRobotUI?.initialize({ token: app.token });
    if (Array.isArray(data.workbenches) && data.workbenches.length === 1 && data.workbenches[0] === "mi") {
      for (const link of document.querySelectorAll("[data-extra-workflow]")) link.hidden = true;
    }
    app.sessions = Array.isArray(data.sessions) ? data.sessions : [];
    app.models = Array.isArray(data.models) ? data.models : [];
    app.selected = new Set([...app.selected].filter((id) => app.sessions.some((s) => s.id === id)));
    renderLists();
  }

  function sessionText(session) {
    return `${session.subject || "未填编号"} · ${sourceText(session.source)} · ${fmt(session.duration_s, 0)} 秒 · ${session.id}`;
  }

  function updateSelect(node, records, title, describe, isDisabled = () => false) {
    const selected = node.value;
    node.replaceChildren(element("option", records.length ? title : `暂无${title}`));
    node.firstChild.value = "";
    for (const record of records) {
      const option = element("option", describe(record));
      option.value = record.id;
      option.disabled = isDisabled(record);
      node.append(option);
    }
    if (records.some((record) => record.id === selected && !isDisabled(record))) node.value = selected;
  }

  function renderLists() {
    updateSelect($("replaySession"), app.sessions, "选择录制", sessionText);
    updateSelect($("modelSelect"), app.models, "选择模型", (model) => `${model.compatible === false ? "[旧版 · 需重训] " : ""}${sourceText(model.source)} · ${model.feature || "—"} + ${String(model.classifier || "—").toUpperCase()} · ${model.id}`, (model) => model.compatible === false);
    const list = $("trainingSessions");
    list.replaceChildren();
    if (!app.sessions.length) list.append(element("p", "录制完成后，数据会出现在这里。", "empty-state"));
    for (const session of app.sessions) {
      const item = element("label", undefined, "session-item");
      const check = document.createElement("input");
      check.type = "checkbox";
      check.value = session.id;
      check.checked = app.selected.has(session.id);
      check.addEventListener("change", () => {
        if (check.checked) app.selected.add(session.id); else app.selected.delete(session.id);
        renderControls();
      });
      const info = element("span", undefined, "session-info");
      info.append(element("b", `${session.subject || "未填编号"} · ${sourceText(session.source)} · ${fmt(session.duration_s, 0)} 秒`));
      info.append(element("small", `${STATUS_LABELS[session.status] || session.status || "—"} · ${session.id}`));
      item.append(check, info);
      list.append(item);
    }
    renderSessionDetails();
    renderControls();
  }

  function renderSessionDetails() {
    const session = app.sessions.find((item) => item.id === $("replaySession").value);
    $("replayDetails").textContent = session ? `${sourceText(session.source)} · ${STATUS_LABELS[session.status] || session.status || "—"} · ${finite(session.sample_count) ? session.sample_count.toLocaleString() + " 个样本" : "标注随样本同步回放"}` : "先选择一次录制。";
  }

  function renderControls() {
    const recording = isRecording();
    const blocked = app.busy || !app.token || !app.online;
    const training = isTraining();
    const status = app.state?.status;
    const connecting = status === "connecting";
    const playback = app.state?.playback || {};
    $("sourceSelect").disabled = blocked || recording || training || status === "connecting" || isLive();
    $("deviceIp").disabled = blocked || recording || connecting || isLive();
    $("discoverButton").disabled = blocked || recording || connecting || isLive();
    for (const button of $("discoveryResults").querySelectorAll("button")) button.disabled = blocked || recording || connecting || isLive() || $("sourceSelect").value !== "device";
    $("connectButton").disabled = blocked || recording || training || status === "connecting" || isLive();
    $("connectButton").textContent = $("sourceSelect").value === "demo" ? "连接演示" : "连接设备";
    $("disconnectButton").disabled = blocked || !["running", "connecting", "error"].includes(status);
    $("recordButton").disabled = blocked || recording || training || !isLive();
    $("stopRecordButton").disabled = !app.token || !recording || app.safetyStopping;
    $("focusStopRecordButton").disabled = !app.token || !recording || app.safetyStopping;
    $("durationInput").disabled = blocked || recording;
    $("subjectInput").disabled = blocked || recording;
    $("loadReplayButton").disabled = blocked || recording || training || !$("replaySession").value;
    $("replaySession").disabled = blocked || recording;
    $("playReplayButton").disabled = blocked || recording || !playback.loaded || playback.playing;
    $("pauseReplayButton").disabled = blocked || !playback.loaded || !playback.playing;
    $("replaySeek").disabled = blocked || !playback.loaded || recording;
    $("replaySpeed").disabled = blocked || !playback.loaded || recording;
    const sampling = ["running", "connecting"].includes(status);
    $("trainButton").disabled = blocked || recording || training || sampling || !app.selected.size;
    $("stopForTrainingButton").hidden = !sampling;
    $("stopForTrainingButton").disabled = blocked || recording || training;
    $("stopForTrainingButton").textContent = app.state?.source === "replay" ? "先暂停回放" : "先停止采集";
    $("featureSelect").disabled = blocked || recording || training;
    $("classifierSelect").disabled = blocked || recording || training;
    const selectedModel = app.models.find((model) => model.id === $("modelSelect").value);
    $("loadModelButton").disabled = blocked || recording || training || !selectedModel || selectedModel.compatible === false;
    $("modelSelect").disabled = blocked || recording || training;
    for (const item of $("trainingSessions").querySelectorAll("input")) item.disabled = blocked || recording || training;
    $("selectedSessions").textContent = `已选 ${app.selected.size} 次录制`;
    for (const tab of document.querySelectorAll(".tab")) tab.disabled = recording && tab.dataset.tab !== "acquisition";
    setRecordingMode(recording);
  }

  function selectTab(name) {
    if (isRecording() && name !== "acquisition") return;
    for (const tab of document.querySelectorAll(".tab")) {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
      $(tab.getAttribute("aria-controls")).hidden = !active;
    }
    app.waveDirty = true;
  }

  async function action(callback) {
    if (app.busy) return;
    app.busy = true;
    renderControls();
    try { await callback(); await poll(); }
    catch (error) { notify(error.message, true, true); }
    finally { app.busy = false; renderControls(); }
  }

  // Keep the same explicit selection policy in the legacy workspace.
  function resolveDeviceAddress(entries, preferred) {
    const devices = [...new Set((Array.isArray(entries) ? entries : []).map((entry) => typeof entry === "string" ? entry : entry?.ip).filter((ip) => typeof ip === "string" && ip.trim()))];
    if (devices.length === 1) return { devices, ip: devices[0], kind: "unique" };
    if (devices.length > 1) return { devices, ip: devices.includes(preferred) ? preferred : "", kind: "multiple" };
    return { devices, ip: preferred, kind: preferred ? "manual" : "missing" };
  }

  async function discoverDevice(forConnection = false) {
    const preferred = $("deviceIp").value.trim();
    const results = $("discoveryResults");
    results.textContent = "正在确认设备地址…";
    let result;
    try { result = await request("/api/discover", { device_ip: preferred }); }
    catch (error) { results.textContent = `设备发现未完成：${error.message}`; throw error; }
    const selection = resolveDeviceAddress(result.devices, preferred);
    $("deviceIp").value = selection.ip;
    let message;
    if (selection.kind === "unique") {
      message = preferred && preferred !== selection.ip ? `设备地址已从 ${preferred} 更新为 ${selection.ip}（唯一身份回复）。` : `已确认并选用设备 ${selection.ip}。`;
    } else if (selection.kind === "multiple") {
      message = selection.ip ? `发现多个设备，已确认当前选择 ${selection.ip}。` : "发现多个设备，请点击下方地址选择后再连接；尚未选用任何设备。";
    } else if (selection.kind === "manual") {
      message = forConnection ? `未收到身份回复，正尝试手动地址 ${selection.ip}。` : `未收到身份回复，已保留手动地址 ${selection.ip}；点击连接可尝试直连。`;
    } else message = "未收到设备身份回复。请检查帽子与网络，或填写已知设备 IP 后再连接。";
    const messageNode = element("p", message, "field-note");
    results.replaceChildren(messageNode);
    for (const ip of selection.devices) {
      const button = element("button", ip, "secondary small"); button.type = "button";
      button.addEventListener("click", () => {
        if (app.busy || !app.online || !app.token || isRecording() || app.state?.status === "connecting" || isLive() || $("sourceSelect").value !== "device") return;
        $("deviceIp").value = ip;
        messageNode.textContent = `已选用设备 ${ip}，点击连接后会再次确认地址。`;
        notify(messageNode.textContent);
      });
      results.append(button);
    }
    renderControls();
    notify(message, !selection.ip, true);
    return selection.ip;
  }

  async function connectSource() {
    if (app.busy || !app.online || !app.token || isRecording() || isTraining() || app.state?.status === "connecting" || isLive()) return;
    await action(async () => {
      const source = $("sourceSelect").value;
      const ip = source === "device" ? await discoverDevice(true) : "";
      if (source === "device" && !ip) return;
      await request("/api/mi/start", { source, device_ip: ip });
      app.safetyMessage = "";
    });
  }

  async function startRecording() {
    if (!isLive() || isRecording() || app.busy || !app.online) return;
    const input = $("durationInput");
    if (!input.reportValidity()) return;
    if (document.hidden || !document.hasFocus()) { notify("请保持实验页面聚焦后开始录制。", true); return; }
    await action(async () => {
      await poll();
      app.recordRequested = true;
      app.focusLost = false;
      app.safetyMessage = "";
      try {
        await synchronizeClock();
        if (app.focusLost || document.hidden || !document.hasFocus()) throw new Error("请保持实验页面聚焦后重新开始录制。");
        if (performance.now() / 1000 - app.clockSetAt >= 2) throw new Error("提示时钟检查已过期，请重新开始。");
        app.recordingOffset = app.offset;
        app.lastFrameAt = null; app.lastVisibleElapsed = null;
        app.lastDisplayElapsed = -Infinity; app.lastDisplayPhase = undefined;
        const result = await request("/api/mi/record/start", {
          duration_s: Number(input.value), subject: $("subjectInput").value.trim(),
          display_timing_method: "raf-visible-heartbeat-v1", clock_uncertainty_s: app.clockUncertainty,
        });
        if (result.recording) acceptState({ ...app.state, server_time: result.server_time, recording: result.recording });
        else await poll();
        selectTab("acquisition");
        $("recordButton").blur();
        if (app.focusLost || document.hidden || !document.hasFocus()) void safetyStop("页面失去焦点，录制已停止并保留。");
      } finally { app.recordRequested = false; }
    });
  }

  async function safetyStop(message) {
    app.focusLost = true;
    // If starting is still in flight, its response performs the stop afterwards.
    if (app.recordRequested && !isRecording()) return;
    if (app.safetyStopping || (!isRecording() && !app.recordRequested) || !app.token) return;
    app.safetyStopping = true;
    app.safetyMessage = message;
    showCue("stopped", "■", "录制已中止", message, "请重新开始一次录制");
    renderControls();
    try {
      await request("/api/mi/record/stop", {
        id: app.state?.recording?.id,
        visibility_lost_s: finite(app.lastVisibleElapsed) ? Math.max(0, app.lastVisibleElapsed) : 0,
        reason: message,
      }, { keepalive: true, timeout: 6000 });
      await poll();
      notify(message, true, true);
    } catch (error) { notify(`录制停止请求未确认：${error.message} 请恢复服务并确认录制状态。`, true, true); }
    finally { app.safetyStopping = false; renderControls(); }
  }

  async function acknowledge(recording, phase, onset) {
    const key = `${recording.id}:${phase.index}`;
    if (app.acked.has(key)) return;
    app.acked.add(key);
    try {
      // onset_s is relative to recording.start_time, estimated at the displayed rAF.
      const result = await request("/api/mi/cue", { id: recording.id, phase_index: phase.index, onset_s: Math.max(0, onset) }, { timeout: 1800 });
      if (app.state?.recording?.id !== recording.id || !isRecording()) return;
      if (result.ignored === true) { await poll(); return; }
      if (result.accepted !== true) await safetyStop("提示起始标注未通过确认，已停止录制并保留数据。");
    } catch (error) {
      // Never invent an earlier acknowledgement after a late retry.
      if (app.state?.recording?.id === recording.id && isRecording()) await safetyStop(`提示标注未确认：${error.message} 已请求停止录制并保留数据。`);
    }
  }

  async function reportDisplay(recording, phaseIndex, observed) {
    if (app.displayInFlight || app.focusLost || app.safetyStopping) return;
    if (phaseIndex === app.lastDisplayPhase && observed - app.lastDisplayElapsed < .10) return;
    const flight = { id: recording.id };
    app.displayInFlight = flight;
    app.lastDisplayElapsed = observed; app.lastDisplayPhase = phaseIndex;
    try {
      const result = await request("/api/mi/display", {
        id: recording.id, phase_index: phaseIndex, observed_s: observed, visible: true,
        clock_uncertainty_s: app.clockUncertainty,
      }, { timeout: 500 });
      if (app.state?.recording?.id !== recording.id || !isRecording()) return;
      if (result.ignored === true) { await poll(); return; }
      if (result.accepted === false || result.ok !== true) await safetyStop("提示持续显示未通过确认，已停止录制并保留数据。");
    } catch (error) {
      if (app.state?.recording?.id === recording.id && isRecording()) await safetyStop(`提示持续显示确认中断：${error.message} 已请求停止录制并保留数据。`);
    } finally { if (app.displayInFlight === flight) app.displayInFlight = false; }
  }

  function showCue(key, symbol, title, instruction, trial, kind = "idle") {
    if (app.cueKey === key) return;
    app.cueKey = key;
    $("cueSymbol").textContent = symbol;
    $("cueTitle").textContent = title;
    $("cueInstruction").textContent = instruction;
    $("cueTrial").textContent = trial;
    $("cueStage").dataset.label = kind;
  }

  function frame() {
    const recording = app.state?.recording;
    const frameAt = performance.now();
    if (recording?.active && !app.focusLost && !app.safetyStopping) {
      if (app.lastFrameAt !== null && frameAt - app.lastFrameAt > 250) {
        void safetyStop("提示画面更新中断，已停止录制并保留数据。");
      } else if (performance.now() - app.lastStateAt > 3500) {
        void safetyStop("与本地服务的状态通信中断，已请求停止录制并保留数据。");
      } else if (finite(recording.start_time) && finite(app.recordingOffset)) {
        const elapsed = performance.now() / 1000 + app.recordingOffset - recording.start_time;
        const duration = recording.duration_s;
        $("recordElapsed").textContent = clockText(Math.min(duration, elapsed));
        $("recordRemaining").textContent = `剩余 ${clockText(Math.max(0, duration - elapsed))}`;
        $("recordProgress").style.width = `${clamp(elapsed / duration, 0, 1) * 100}%`;
        const phase = (recording.plan || []).find((item) => elapsed >= item.start_s && elapsed < item.end_s);
        if (phase) {
          const key = `${recording.id}:${phase.index}`;
          const task = phase.phase === "task";
          const kind = task ? phase.label : "interval";
          const symbol = kind === "left_mi" ? "←" : kind === "right_mi" ? "→" : kind === "rest" ? "○" : "+";
          const instruction = kind === "left_mi" ? "想象左手运动，保持身体静止" : kind === "right_mi" ? "想象右手运动，保持身体静止" : kind === "rest" ? "自然放松，不进行运动想象" : "注视十字，等待下一次提示";
          showCue(key, symbol, label(kind), instruction, task ? `第 ${phase.trial} 次任务 · 5 秒` : "间隔不属于静息训练标签", kind);
          $("cuePhase").textContent = task ? "任务进行中" : "十字间隔";
          if (confirmPresentedCue(recording, phase, elapsed)) void reportDisplay(recording, phase.index, elapsed);
        } else {
          // At the final boundary the preceding cue was still on screen up to
          // this frame. Send an explicit tail proof even with no next phase.
          if (elapsed >= duration && cueIsVisible()) {
            app.lastVisibleElapsed = elapsed;
            void reportDisplay(recording, null, elapsed);
          } else if (elapsed >= duration && !cueIsVisible()) {
            void safetyStop("提示末尾未保持可见，已停止录制并保留数据。");
          }
          showCue(`${recording.id}:end`, "+", elapsed < 0 ? "准备开始" : "录制结束", elapsed < 0 ? "请注视中央提示" : "正在保存录制与标注", "");
          $("cuePhase").textContent = "保存中";
        }
      }
      app.lastFrameAt = frameAt;
    } else if (!recording?.active) {
      $("recordElapsed").textContent = clockText(recording?.elapsed_s || 0);
      $("recordRemaining").textContent = recording?.id ? "录制已结束" : "尚未录制";
      $("recordProgress").style.width = `${finite(recording?.duration_s) && recording.duration_s > 0 ? clamp((recording.elapsed_s || 0) / recording.duration_s, 0, 1) * 100 : 0}%`;
      if (app.safetyMessage) {
        showCue("stopped", "■", "录制已中止", app.safetyMessage, "已采集数据可在回放区查看");
        $("cuePhase").textContent = "已中止";
      } else {
        showCue(recording?.id ? "complete" : "idle", "+", recording?.id ? "录制已保存" : "准备开始", recording?.id ? "可在数据回放中检查波形与提示标注" : "连接信号源后，按空格或点击“开始录制”", "保持舒适坐姿，注视中央提示");
        $("cuePhase").textContent = recording?.id ? "已结束" : "准备";
      }
    }
    if (app.waveDirty) { drawWaveform(); app.waveDirty = false; }
    requestAnimationFrame(frame);
  }

  function acceptState(state) {
    if (finite(state.server_time) && state.server_time < app.lastServerTime) return;
    if (finite(state.server_time)) app.lastServerTime = state.server_time;
    const wasRecording = isRecording();
    app.state = state;
    app.online = true;
    app.lastStateAt = performance.now();
    if (state.recording?.active && state.recording.id !== app.lastRecordId) {
      app.lastRecordId = state.recording.id;
      app.acked.clear();
      app.lastFrameAt = null; app.lastVisibleElapsed = null;
      app.lastDisplayElapsed = -Infinity; app.lastDisplayPhase = undefined;
      app.displayInFlight = false;
      selectTab("acquisition");
      if (!finite(app.recordingOffset) || !finite(app.clockUncertainty)) void safetyStop("页面没有本次录制的提示时钟，已请求停止；请重新开始录制。");
      if (app.focusLost || document.hidden || !document.hasFocus()) void safetyStop("页面失去焦点，录制已停止并保留。");
    }
    if (wasRecording && !isRecording()) {
      app.recordingOffset = null;
      void bootstrap().catch((error) => notify(error.message, true));
    }
    if (state.training?.status === "complete" && app.previousTraining !== "complete") void bootstrap().catch((error) => notify(error.message, true));
    app.previousTraining = state.training?.status || "idle";
    renderState();
  }

  function poll() {
    if (app.polling) return app.polling;
    app.polling = (async () => {
      try { acceptState(await request("/api/mi/state", undefined, { timeout: 3000 })); }
      catch (error) {
        app.online = false;
        $("connectionStatus").textContent = "服务不可达";
        $("connectionStatus").className = "status-pill error";
        $("sourceMessage").textContent = error.message;
        renderPrediction(null);
        window.MIRobotUI?.offline();
        renderControls();
      } finally { app.polling = null; }
    })();
    return app.polling;
  }

  function renderState() {
    const state = app.state;
    window.MIRobotUI?.render(state);
    const status = state.status;
    const statusNode = $("connectionStatus");
    statusNode.textContent = { idle: state.source === "replay" && state.playback?.loaded ? "回放暂停" : "未连接", connecting: "连接中", running: state.source === "replay" ? "正在回放" : "已连接", error: "采集异常" }[status] || status || "未知状态";
    statusNode.className = `status-pill${status === "running" ? " live" : status === "error" ? " error" : ""}`;
    $("sourceMessage").textContent = state.message || "等待数据";
    $("deviceTarget").hidden = state.source !== "device" || !state.device_ip;
    $("deviceTarget").textContent = state.device_ip ? `实际连接目标：${state.device_ip}` : "";
    $("sourceBadge").textContent = state.source === "device" ? "DEVICE" : state.source === "replay" ? "REPLAY" : "DEMO";
    const replaySession = app.sessions.find((item) => item.id === state.playback?.id);
    const simulated = state.source === "demo" || (state.source === "replay" && replaySession?.source === "demo");
    $("demoBanner").hidden = !simulated;
    const telemetry = state.telemetry || {};
    $("packetCount").textContent = finite(telemetry.packets) ? telemetry.packets.toLocaleString() : "—";
    $("lostCount").textContent = finite(telemetry.lost_packets) ? telemetry.lost_packets.toLocaleString() : "—";
    $("dataAge").textContent = state.source === "replay" ? "回放" : finite(telemetry.age_s) ? `${fmt(telemetry.age_s, 1)} s 前` : "—";
    $("recordBadge").textContent = state.recording?.active ? "录制中" : state.recording?.id ? "已保存" : "待开始";
    $("recordBadge").className = `tag${state.recording?.active ? " active" : ""}`;
    const playback = state.playback || {};
    if (!app.seeking) {
      $("replaySeek").max = Math.max(.02, playback.duration_s || .02);
      $("replaySeek").value = playback.position_s || 0;
      $("replayPosition").textContent = clockText(playback.position_s, true);
    }
    $("replayDuration").textContent = clockText(playback.duration_s, true);
    $("replayLabel").textContent = playback.loaded ? label(playback.label) : "—";
    $("replayTrial").textContent = playback.trial > 0 ? `试次 ${playback.trial}${playback.timing_valid === false ? " · 标注未通过校验" : ""}` : "";
    if (finite(playback.speed) && document.activeElement !== $("replaySpeed")) $("replaySpeed").value = String(playback.speed);
    $("waveSource").textContent = sourceText(state.source);
    $("waveformCaption").textContent = state.source === "replay" ? "回放时间轴 · 色带为录制标注，与预测独立" : "固定顺序：O2 · O1 · P8 · P7 · C4 · C3 · FP1 · FP2";
    renderPrediction(state.prediction);
    renderHistory(state.predictions || []);
    renderTraining(state.training || {}, state.model || {});
    renderRecordingValidation(state.recording?.validation);
    renderEvents(state.events || []);
    renderControls();
    app.waveDirty = true;
  }

  function renderRecordingValidation(validation) {
    const panel = $("recordingValidation");
    panel.hidden = isRecording() || !validation;
    if (panel.hidden) return;
    const details = $("recordingValidationDetails");
    details.replaceChildren();
    const status = validation.status;
    const diagnostics = validation.diagnostics || {};
    const audit = validation.display_audit || {};
    const retained = finite(diagnostics.accepted_trials) ? diagnostics.accepted_trials : 0;
    $("recordingValidationStatus").textContent = status === "checking" ? "正在检查" : status === "error" ? "检查失败" : retained > 0 ? "检查完成" : "无可训练试次";
    details.append(element("p", validation.message || (status === "checking" ? "正在读取原始记录并核对时序、提示和信号。" : status === "error" ? "检查未完成，请查看错误说明。" : "已完成录制数据检查。")));
    if (status !== "checked") return;
    details.append(element("p", `采集 ${finite(validation.sample_count) ? validation.sample_count.toLocaleString() : "—"} 点 · 保留 ${retained} 个试次、${diagnostics.accepted_windows || 0} 个训练窗口。`));
    const table = element("table", undefined, "diagnostic-table");
    const header = element("tr");
    for (const title of ["类别", "时序合格试次", "最终保留试次", "保留窗口"]) header.append(element("th", title));
    const head = element("thead"); head.append(header); table.append(head);
    const body = element("tbody");
    for (const kind of ["rest", "left_mi", "right_mi"]) {
      const row = element("tr");
      row.append(element("td", label(kind, true)), element("td", audit.per_class?.[kind]?.eligible_trials ?? "—"),
        element("td", diagnostics.per_class?.[kind]?.accepted_trials ?? 0), element("td", diagnostics.per_class?.[kind]?.accepted_windows ?? 0));
      body.append(row);
    }
    table.append(body); details.append(table);
    const displayReasons = {
      missing_or_late_ack: "缺少提示确认或起始延迟过大", invalid_display_order: "提示显示次序异常",
      heartbeat_missing: "缺少提示持续可见证明", heartbeat_gap: "提示持续可见确认中断",
      heartbeat_tail: "缺少提示末尾可见证明", heartbeat_invalid: "提示确认时间或时钟误差不可信",
      heartbeat_phase_mismatch: "任务中显示了其他提示", display_interrupted: "提示失焦、停顿或不可见",
      incomplete_task: "任务未完整采集", sample_discontinuity: "任务内采样断点或跨段",
      clock_future: "采样时间超前于收包", clock_unverified: "无法验证时钟",
    };
    for (const [title, values] of [["提示/时序排除", audit.rejection_reasons], ["信号/窗口排除", diagnostics.rejection_reasons]]) {
      const reasons = Object.entries(values || {}).filter(([, value]) => finite(value) && value > 0)
        .map(([reason, value]) => `${displayReasons[reason] || QUALITY_REASONS[reason] || reason}：${value}`).join("；");
      if (reasons) details.append(element("p", `${title}：${reasons}`, "diagnostic-warning"));
    }
    if (validation.timestamp_audit?.status === "suspect") details.append(element("p", "检测到采样时间与接收时间不一致；受影响的整试次已排除。", "diagnostic-warning"));
    if (diagnostics.warning_windows > 0) details.append(element("p", `${diagnostics.warning_windows} 个候选窗口有信号告警。告警不等于拒绝，最终可用数量见表格。`, "field-note"));
    if (!retained) details.append(element("p", "本次没有可训练试次；可回放检查原始波形，并根据上述原因重新录制。", "diagnostic-warning"));
  }

  function renderPrediction(prediction) {
    const invalid = !prediction || ["invalid", "unavailable", "no_model", "warming_up", "rejected", "artifact"].includes(prediction.status);
    const valid = !invalid && ["rest", "left_mi", "right_mi"].includes(prediction.label) && finite(prediction.confidence);
    $("predictionLabel").textContent = valid ? label(prediction.label) : prediction?.status === "no_model" ? "等待模型" : "暂无有效识别";
    $("predictionConfidence").textContent = valid ? percent(prediction.confidence) : "—";
    $("predictionBadge").textContent = !app.online ? "通信中断" : STATUS_LABELS[prediction?.status] || (valid ? "有效窗口" : "未就绪");
    $("predictionBadge").className = `tag${valid ? prediction.status === "uncertain" || prediction.accepted === false ? " caution" : " active" : ""}`;
    const directMapping = app.state?.robot?.decision_policy === "direct_prediction";
    const diagnosticNote = directMapping && valid && (prediction.status === "uncertain" || prediction.accepted === false)
      ? " 此标记仅供参考；机械臂映射直接采用当前类别，实际发送状态见上方。" : "";
    $("predictionReason").textContent = !app.online ? "无法获取最新状态，已停止显示旧的当前预测。" : (reasonText(prediction?.reason) || (valid ? `信号窗口 ${fmt(prediction.window_start, 2)}–${fmt(prediction.window_end, 2)} 秒。置信度不是分类准确率。` : "载入模型并积累有效 EEG 后开始识别。")) + diagnosticNote;
    const warnings = app.online && Array.isArray(prediction?.quality?.warnings) ? prediction.quality.warnings : [];
    const warningTexts = [...new Set(warnings.filter((item) => item && typeof item === "object").map((item) => `${item.channel || "未知通道"}：${QUALITY_REASONS[item.reason] || item.reason || "信号告警"}`))];
    $("predictionWarnings").hidden = !warningTexts.length;
    $("predictionWarnings").textContent = warningTexts.length ? `原始信号告警：${warningTexts.join("；")}。这些告警本身不剔除窗口；硬信号异常会停止机械臂发送。` : "";
    const order = prediction?.label_order || ["rest", "left_mi", "right_mi"];
    for (const [kind, id] of [["rest", "Rest"], ["left_mi", "Left"], ["right_mi", "Right"]]) {
      const value = valid ? prediction.probabilities?.[order.indexOf(kind)] : null;
      $(`prob${id}`).style.width = `${finite(value) ? clamp(value, 0, 1) * 100 : 0}%`;
      $(`prob${id}Value`).textContent = percent(value);
    }
  }

  function renderHistory(records) {
    const recent = records.slice(-60).reverse();
    const key = JSON.stringify(recent);
    if (key === app.historyKey) return;
    app.historyKey = key;
    $("predictionCount").textContent = `最近 ${recent.length} 条`;
    const rows = $("predictionRows");
    rows.replaceChildren();
    if (!recent.length) {
      const row = element("tr");
      const cell = element("td", "暂无预测窗口", "empty-state");
      cell.colSpan = 4; row.append(cell); rows.append(row);
    }
    for (const item of recent) {
      const row = element("tr");
      row.append(element("td", `${fmt(item.window_start, 2)}–${fmt(item.window_end, 2)}`), element("td", label(item.label, true)), element("td", percent(item.confidence)), element("td", STATUS_LABELS[item.status] || item.status || "—"));
      if (item.reason) row.title = reasonText(item.reason);
      rows.append(row);
    }
  }

  function renderTraining(training, model) {
    $("trainingMessage").textContent = training.message || "请选择录制与算法。每类至少需要 6 个完整有效试次。";
    $("trainingBadge").textContent = STATUS_LABELS[training.status] || "未训练";
    $("trainingBadge").className = `tag${training.status === "running" || training.status === "complete" ? " active" : training.status === "error" ? " error" : ""}`;
    $("activeModel").textContent = model.available ? `${model.feature || "—"} + ${String(model.classifier || "—").toUpperCase()}` : "未加载模型";
    $("modelDetails").textContent = model.available ? model.compatible === false ? `当前模型未启用：${model.reason || "模型与当前流程或数据来源不兼容"}` : `当前：${model.id}。真实设备需使用真实录制训练的模型。` : "采集与回放均可使用匹配数据来源的模型；旧版模型需重新训练后加载。";
    const hasAttempt = ["running", "complete", "error"].includes(training.status);
    const context = hasAttempt ? `本次训练 · ${STATUS_LABELS[training.status]}` : model.available ? `当前模型 · ${model.id}` : "尚未训练";
    $("trainingResultContext").textContent = context;
    // A new attempt owns this area even when it fails before metrics exist.
    // Never show a previous model's success as the new attempt's outcome.
    const metrics = hasAttempt ? training.status === "complete" ? training.metrics || {} : {} : model.metrics || {};
    renderDiagnostics(hasAttempt ? training.diagnostics : model.diagnostics, context, training.status);
    const key = JSON.stringify({ metrics, context, status: training.status, message: training.message });
    if (key === app.metricsKey) return;
    app.metricsKey = key;
    const panel = $("metricsPanel");
    panel.replaceChildren();
    if (!Object.keys(metrics).length) {
      const empty = training.status === "error" ? `本次训练未生成测试指标：${training.message || "请查看训练信息与下方数据诊断。"}` : training.status === "running" ? "本次训练进行中，完成后显示测试指标。" : "训练完成后显示留出测试指标与数据隔离方式。";
      panel.append(element("p", empty, "empty-state")); return;
    }
    const metricCards = element("div", undefined, "metric-cards");
    for (const [name, value] of [["测试试次平衡准确率", metrics.trial_balanced_accuracy], ["测试窗口准确率", metrics.accuracy ?? metrics.validation_accuracy], ["测试窗口平衡准确率", metrics.balanced_accuracy]]) {
      if (!finite(value)) continue;
      const metric = element("div", undefined, "metric"); metric.append(element("span", name), element("b", percent(value))); metricCards.append(metric);
    }
    panel.append(metricCards);
    const isolation = metrics.evaluation_isolation;
    if (isolation) panel.append(element("p", `${isolation.independent_sessions ? "整次录制独立留出" : "探索性试次留出（共享录制滤波历史）"}：${isolation.note || ""}`, isolation.independent_sessions ? "field-note" : "diagnostic-warning"));
    if (metrics.split_metrics) {
      const table = element("table", undefined, "diagnostic-table");
      const head = element("tr");
      for (const title of ["集合", "窗口准确率", "窗口平衡准确率", "试次平衡准确率", "预测窗数：静息 / 左 / 右", "召回率：静息 / 左 / 右"]) head.append(element("th", title));
      const thead = element("thead"); thead.append(head); table.append(thead);
      const tbody = element("tbody");
      for (const [key, name] of [["train", "训练"], ["calibration", "置信度校准"], ["test", "留出测试"]]) {
        const values = metrics.split_metrics[key];
        if (!values) continue;
        const row = element("tr");
        row.append(element("th", name), element("td", percent(values.accuracy)), element("td", percent(values.balanced_accuracy)), element("td", percent(values.trial_balanced_accuracy)), element("td", (values.prediction_counts || []).join(" / ")), element("td", (values.recall_per_class || []).map(percent).join(" / ")));
        tbody.append(row);
      }
      table.append(tbody);
      const wrapper = element("div", undefined, "diagnostic-table-wrap"); wrapper.append(table); panel.append(wrapper);
      panel.append(element("p", "校准集用于调整置信度；测试集不参与特征、分类器或校准拟合。若依据测试分数反复选择模型，请另留新录制做最终评估。", "field-note"));
    }
    const matrix = metrics.trial_confusion_matrix || metrics.confusion_matrix;
    if (Array.isArray(matrix) && matrix.length === 3 && matrix.every((row) => Array.isArray(row) && row.length === 3)) {
      panel.append(element("p", `留出测试${metrics.trial_confusion_matrix ? "试次" : "窗口"}混淆矩阵 · 行：真实标签，列：预测标签`, "matrix-caption"));
      const order = metrics.label_order || metrics.labels || ["rest", "left_mi", "right_mi"];
      const table = element("table", undefined, "confusion-table");
      const head = element("tr"); head.append(element("th", "真实 / 预测"));
      for (const kind of order) head.append(element("th", label(kind, true)));
      const thead = element("thead"); thead.append(head); table.append(thead);
      const tbody = element("tbody");
      matrix.forEach((values, index) => { const row = element("tr"); row.append(element("th", label(order[index], true))); values.forEach((value, column) => row.append(element("td", String(value), column === index ? "correct" : ""))); tbody.append(row); });
      table.append(tbody); panel.append(table);
    }
    const details = [];
    for (const [name, keyName] of [["训练窗口", "train_windows"], ["校准窗口", "calibration_windows"], ["测试窗口", "test_windows"]]) if (finite(metrics[keyName])) details.push(`${name} ${metrics[keyName]}`);
    if (details.length) panel.append(element("p", details.join(" · "), "field-note"));
    const testCounts = metrics.test_trials_per_class;
    if (testCounts && typeof testCounts === "object") {
      const entries = Array.isArray(testCounts) ? testCounts.map((value, index) => [["rest", "left_mi", "right_mi"][index], value]) : Object.entries(testCounts);
      panel.append(element("p", `测试试次：${entries.map(([kind, count]) => `${label(kind, true)} ${count}`).join(" · ")}`, "field-note"));
    }
    if (typeof metrics.usable === "boolean") panel.append(element("p", metrics.usable ? "模型通过验证门槛；仍需匹配录制来源及当前信号质量。" : "模型未通过验证门槛；有效窗口仍输出预测类别，请结合诊断判断效果。", "field-note"));
    if (typeof metrics.scope === "string") panel.append(element("p", `测试范围：${SCOPES[metrics.scope] || metrics.scope}`, "field-note"));
    if (typeof metrics.note === "string") panel.append(element("p", metrics.note, "field-note"));
    if (!metricCards.children.length && !Array.isArray(matrix)) panel.append(element("p", "训练已完成；指标详情见保存模型的元数据。", "field-note"));
  }

  function renderDiagnostics(diagnostics, context, status) {
    const key = JSON.stringify({ diagnostics, context, status });
    if (key === app.diagnosticsKey) return;
    app.diagnosticsKey = key;
    const panel = $("diagnosticsPanel");
    panel.replaceChildren();
    panel.append(element("h3", "训练数据保留情况"));
    if (!diagnostics || typeof diagnostics !== "object" || Object.keys(diagnostics).length === 0) {
      panel.append(element("p", status === "running" ? "正在统计本次训练数据。" : status === "error" ? "本次失败未返回数据诊断；请查看上方错误信息。" : "尚无数据诊断。训练后可查看候选窗口、保留试次与告警原因。", "field-note"));
      return;
    }
    const count = (value) => finite(value) ? value.toLocaleString("zh-CN") : "—";
    const totals = element("div", undefined, "diagnostic-totals");
    for (const [title, accepted, candidate] of [["保留 / 候选窗口", diagnostics.accepted_windows, diagnostics.candidate_windows], ["保留 / 候选试次", diagnostics.accepted_trials, diagnostics.candidate_trials]]) {
      const item = element("div", undefined, "diagnostic-total");
      item.append(element("span", title), element("b", `${count(accepted)} / ${count(candidate)}`)); totals.append(item);
    }
    panel.append(totals);
    const table = element("table", undefined, "diagnostic-table");
    const head = element("tr");
    for (const title of ["类别", "候选试次", "保留试次", "候选窗", "保留窗"]) head.append(element("th", title));
    const thead = element("thead"); thead.append(head); table.append(thead);
    const tbody = element("tbody");
    for (const kind of ["rest", "left_mi", "right_mi"]) {
      const values = diagnostics.per_class?.[kind] || {};
      const row = element("tr"); row.append(element("td", label(kind, true)));
      for (const field of ["candidate_trials", "accepted_trials", "candidate_windows", "accepted_windows"]) row.append(element("td", count(values[field])));
      tbody.append(row);
    }
    table.append(tbody);
    const wrapper = element("div", undefined, "diagnostic-table-wrap"); wrapper.append(table); panel.append(wrapper);
    panel.append(element("p", "候选窗口须完整处于已确认的有效任务内；至少保留一个窗口即计为保留试次。", "field-note"));
    panel.append(element("p", `完整有效任务范围外 ${count(diagnostics.outside_complete_task)} 窗 · 滤波预热拒绝 ${count(diagnostics.warmup_rejected)} 窗 · 硬质量拒绝 ${count(diagnostics.quality_rejected)} 窗`, "field-note"));
    if (diagnostics.clock_rejected_trials > 0) panel.append(element("p", `另有 ${count(diagnostics.clock_rejected_trials)} 个计划试次存在采样时间明显晚于实际收包时间的问题，整试次排除。原始文件和标签未改写；请展开下方录制明细。`, "diagnostic-warning"));
    if (diagnostics.clock_unverified_recordings > 0) panel.append(element("p", `${count(diagnostics.clock_unverified_recordings)} 次录制缺少时钟锚点，无法完成接收时间一致性检查。`, "diagnostic-warning"));
    const warningCount = finite(diagnostics.warning_windows) ? `候选中 ${count(diagnostics.warning_windows)} 个告警窗` : "告警仅作提示";
    const acceptedWarningCount = finite(diagnostics.accepted_warning_windows) ? `，其中 ${count(diagnostics.accepted_warning_windows)} 窗仍被保留` : "";
    panel.append(element("p", `${warningCount}${acceptedWarningCount}。原始直流偏置和大幅度告警本身不剔除窗口；同窗若有其他拒绝原因，仍会被剔除。`, "diagnostic-warning"));
    const describeCounts = (counts) => Object.entries(counts || {}).filter(([, value]) => finite(value) && value > 0).map(([reason, value]) => `${QUALITY_REASONS[reason] || reason} ${count(value)} 窗`).join("；");
    for (const [title, counts] of [["拒绝原因", diagnostics.rejection_reasons], ["告警原因", diagnostics.warning_reasons]]) {
      const line = element("p", undefined, "diagnostic-reasons");
      line.append(element("b", `${title}：`), element("span", describeCounts(counts) || "无")); panel.append(line);
    }
    panel.append(element("p", "同一窗口可有多个原因；下列原因与通道计数可能重叠。", "field-note"));
    const channelDetails = element("details", undefined, "diagnostic-details");
    channelDetails.append(element("summary", "按通道查看拒绝与告警"));
    const channelNames = [...new Set([...Object.keys(diagnostics.channel_rejections || {}), ...Object.keys(diagnostics.channel_warnings || {})])];
    for (const channel of channelNames) {
      const rejected = describeCounts(diagnostics.channel_rejections?.[channel]);
      const warned = describeCounts(diagnostics.channel_warnings?.[channel]);
      channelDetails.append(element("p", `${channel} · 拒绝：${rejected || "无"}；告警：${warned || "无"}`, "diagnostic-reasons"));
    }
    if (!channelNames.length) channelDetails.append(element("p", "无通道拒绝或告警。", "field-note"));
    panel.append(channelDetails);
    if (Array.isArray(diagnostics.per_recording) && diagnostics.per_recording.length) {
      const details = element("details", undefined, "diagnostic-details");
      details.append(element("summary", `按录制查看（${diagnostics.per_recording.length} 次）`));
      const recordings = element("div", undefined, "diagnostic-recordings");
      for (const session of diagnostics.per_recording) {
        const item = element("div", undefined, "diagnostic-recording");
        item.append(element("b", `${session.subject || "未填编号"} · ${sourceText(session.source)} · ${session.id || "未知录制"}`));
        item.append(element("p", `窗口保留 / 候选 ${count(session.accepted_windows)} / ${count(session.candidate_windows)}；试次保留 / 候选 ${count(session.accepted_trials)} / ${count(session.candidate_trials)}`));
        item.append(element("p", ["rest", "left_mi", "right_mi"].map((kind) => `${label(kind, true)}保留 ${count(session.per_class?.[kind]?.accepted_trials)} 试次`).join(" · ")));
        item.append(element("p", `告警 ${count(session.warning_windows)} 窗，其中保留 ${count(session.accepted_warning_windows)} 窗；拒绝原因：${describeCounts(session.rejection_reasons) || "无"}`));
        const audit = session.timestamp_audit;
        if (audit?.status === "suspect") item.append(element("p", `时间轴异常：最大超前 ${fmt(audit.max_future_lead_s, 3)} 秒，排除试次 ${(audit.rejected_trial_ids || []).join("、")}。超过 0.25 秒的整试次不用于训练。`, "diagnostic-warning"));
        else if (audit?.status === "unverified") item.append(element("p", audit.warning || "缺少时钟锚点，时间轴一致性未验证。", "diagnostic-warning"));
        recordings.append(item);
      }
      details.append(recordings); panel.append(details);
    }
  }

  function renderEvents(events) {
    const key = JSON.stringify(events);
    if (key === app.eventsKey) return;
    app.eventsKey = key;
    $("eventCount").textContent = String(events.length);
    $("eventList").replaceChildren();
    for (const event of events.slice().reverse()) {
      const stamp = typeof event.time === "string" ? event.time.replace("T", " ").slice(0, 19) : "";
      $("eventList").append(element("li", `${stamp} ${event.message || String(event)}`));
    }
    if (!events.length) $("eventList").append(element("li", "暂无事件"));
  }

  function drawWaveform() {
    const canvas = $("waveCanvas");
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const left = 44, right = 13, top = 4, bandHeight = 20, bottom = 26;
    const plotWidth = width - left - right;
    const rowHeight = (height - top - bottom - bandHeight - 8) / 8;
    const plotBottom = top + rowHeight * 8;
    const signal = app.state?.signal || {};
    const times = Array.isArray(signal.times) ? signal.times : [];
    const channels = signal.channel_names || CHANNELS;
    const hasData = times.length > 1 && finite(times[0]) && finite(times[times.length - 1]);
    // Use the true visible peak, not a percentile: automatic range must also
    // contain large transients. All channels retain one common amplitude scale.
    const peaks = CHANNELS.map((_, row) => {
      const trace = signal.traces?.[row] || [];
      let peak = 0;
      for (let index = 0; index < Math.min(times.length, trace.length); index++) {
        if (finite(times[index]) && finite(trace[index])) peak = Math.max(peak, Math.abs(trace[index]));
      }
      return peak;
    });
    const auto = $("waveScale").value === "auto";
    let amplitude = Number($("waveScale").value) || 50;
    if (auto) {
      const target = Math.max(25, Math.max(...peaks) * 1.05);
      const decade = 10 ** Math.floor(Math.log10(target));
      const desired = [1, 2, 2.5, 5, 10].find((step) => step * decade >= target) * decade;
      const now = performance.now();
      // Expand immediately; shrink only after a hold to avoid rapid pumping.
      if (!app.waveAutoAmplitude || desired > app.waveAutoAmplitude ||
          (desired <= app.waveAutoAmplitude / 2 && now - app.waveAutoUpdated >= 3000)) {
        app.waveAutoAmplitude = desired;
        app.waveAutoUpdated = now;
      }
      amplitude = app.waveAutoAmplitude;
    }
    const clipped = channels.filter((_, row) => peaks[row] > amplitude);
    $("waveScaleNote").textContent = `${auto ? "自动" : "手动"}统一量程：每行 ±${amplitude.toLocaleString("zh-CN")} µV（上下刻度）；八通道保持相同比例。` +
      (clipped.length ? ` ${clipped.join("、")} 超出量程，边缘部分被裁切；请选择自动或更大量程。` : "仅调整显示，不改变原始数据。");
    $("waveScaleNote").className = `wave-scale-note${clipped.length ? " clipped" : ""}`;
    const from = hasData ? times[0] : 0, to = hasData ? times[times.length - 1] : 5;
    const span = Math.max(to - from, .004);
    // Use the visible sample cadence, including backend decimation, so missing
    // samples make a break rather than an invented line across a data gap.
    const intervals = [];
    for (let index = 1; index < times.length; index++) {
      const delta = times[index] - times[index - 1];
      if (finite(delta) && delta > 0) intervals.push(delta);
    }
    intervals.sort((a, b) => a - b);
    const expectedStep = intervals.length >= 3 ? intervals[Math.floor(intervals.length / 2)] : 1 / 250;
    const gapThreshold = Math.max(1.5 / 250, expectedStep * 1.5) + 1e-9;
    const xFor = (time) => left + (time - from) / span * plotWidth;
    ctx.font = "10px Segoe UI, Microsoft YaHei, sans-serif";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 1;
    for (let row = 0; row < 8; row++) {
      const y = top + (row + .5) * rowHeight;
      if (row % 2 === 0) { ctx.fillStyle = "#fafcfc"; ctx.fillRect(left, top + row * rowHeight, plotWidth, rowHeight); }
      ctx.fillStyle = "#6e8792"; ctx.textAlign = "left"; ctx.fillText(channels[row] || CHANNELS[row], 1, y);
      ctx.strokeStyle = "#e9eff2"; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
      ctx.strokeStyle = "#f0f4f5"; ctx.beginPath();
      for (const edge of [-1, 1]) { const limit = y + edge * rowHeight * .43; ctx.moveTo(left, limit); ctx.lineTo(width - right, limit); }
      ctx.stroke();
    }
    const tickStep = span > 8 ? 2 : span > 3 ? 1 : .5;
    for (let tick = Math.ceil(from / tickStep) * tickStep; tick <= to + .001; tick += tickStep) {
      const x = xFor(tick);
      ctx.strokeStyle = "#edf1f3"; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, plotBottom); ctx.stroke();
      ctx.fillStyle = "#8c9ea7"; ctx.textAlign = "center"; ctx.fillText(`${tick.toFixed(tickStep < 1 ? 1 : 0)}s`, x, height - 8);
    }
    ctx.fillStyle = "#f0f3f4"; ctx.fillRect(left, plotBottom + 5, plotWidth, bandHeight);
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, plotWidth, height - top - bottom); ctx.clip();
    for (const segment of signal.annotation_segments || []) {
      if (!finite(segment.start) || !finite(segment.end)) continue;
      const start = Math.max(left, xFor(segment.start)), end = Math.min(width - right, xFor(segment.end));
      if (end <= start) continue;
      ctx.fillStyle = COLORS[segment.label] || COLORS.unknown; ctx.fillRect(start, plotBottom + 5, end - start, bandHeight);
      if (end - start > 50) { ctx.fillStyle = "#54747e"; ctx.textAlign = "center"; ctx.fillText(label(segment.label, true), (start + end) / 2, plotBottom + 5 + bandHeight / 2); }
    }
    for (let row = 0; row < 8; row++) {
      const trace = signal.traces?.[row] || [];
      const center = top + (row + .5) * rowHeight;
      ctx.save(); ctx.beginPath(); ctx.rect(left, center - rowHeight * .43, plotWidth, rowHeight * .86); ctx.clip();
      ctx.strokeStyle = row === 4 || row === 5 ? "#167e81" : "#577e92"; ctx.lineWidth = 1;
      ctx.beginPath(); let connected = false;
      for (let index = 0; index < Math.min(times.length, trace.length); index++) {
        if (!finite(trace[index]) || !finite(times[index])) { connected = false; continue; }
        if (index > 0 && times[index] - times[index - 1] > gapThreshold) connected = false;
        const x = xFor(times[index]), y = center - trace[index] / amplitude * rowHeight * .43;
        if (connected) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        connected = true;
      }
      ctx.stroke(); ctx.restore();
    }
    ctx.restore();
    $("waveEmpty").hidden = hasData;
  }

  // Acquisition actions never run automatically: initial mode is only a UI choice.
  for (const tab of document.querySelectorAll(".tab")) tab.addEventListener("click", () => selectTab(tab.dataset.tab));
  $("sourceSelect").addEventListener("change", () => { $("deviceFields").hidden = $("sourceSelect").value !== "device"; renderControls(); });
  $("connectButton").addEventListener("click", connectSource);
  $("disconnectButton").addEventListener("click", () => action(async () => { await request("/api/mi/stop", {}); }));
  $("recordButton").addEventListener("click", () => { void startRecording(); });
  for (const id of ["stopRecordButton", "focusStopRecordButton"]) $(id).addEventListener("click", () => { void safetyStop("录制已结束，已采集数据与标注保留。"); });
  $("discoverButton").addEventListener("click", () => {
    if (isRecording() || app.state?.status === "connecting" || isLive()) return;
    return action(() => discoverDevice());
  });
  for (const id of ["refreshSessionsButton", "refreshTrainingButton"]) $(id).addEventListener("click", () => action(bootstrap));
  $("replaySession").addEventListener("change", () => { renderSessionDetails(); renderControls(); });
  $("modelSelect").addEventListener("change", renderControls);
  $("loadReplayButton").addEventListener("click", () => action(async () => {
    await request("/api/mi/replay/load", { id: $("replaySession").value });
    notify("回放已载入。点击播放，或拖动时间轴查看信号与标注。");
  }));
  const replayAction = (name, extra = {}) => request("/api/mi/replay/control", { action: name, ...extra });
  $("playReplayButton").addEventListener("click", () => action(() => replayAction("play", { speed: Number($("replaySpeed").value) })));
  $("pauseReplayButton").addEventListener("click", () => action(() => replayAction("pause")));
  $("replaySpeed").addEventListener("change", () => action(() => replayAction(app.state?.playback?.playing ? "play" : "pause", { speed: Number($("replaySpeed").value) })));
  $("replaySeek").addEventListener("input", () => { app.seeking = true; $("replayPosition").textContent = clockText(Number($("replaySeek").value), true); });
  $("replaySeek").addEventListener("change", () => { const position = Number($("replaySeek").value); void action(async () => { try { await replayAction("seek", { position_s: position }); } finally { app.seeking = false; } }); });
  $("trainButton").addEventListener("click", () => action(async () => {
    await request("/api/mi/train", { ids: [...app.selected], feature: $("featureSelect").value, classifier: $("classifierSelect").value });
    notify("训练已提交，完成后将显示验证结果。");
  }));
  $("stopForTrainingButton").addEventListener("click", () => action(async () => {
    if (app.state?.source === "replay") await replayAction("pause"); else await request("/api/mi/stop", {});
  }));
  $("loadModelButton").addEventListener("click", () => action(async () => {
    const model = app.models.find((item) => item.id === $("modelSelect").value);
    if (!model || model.compatible === false) throw new Error("请选择兼容当前流程的模型；旧版模型需要重新训练。");
    await request("/api/mi/model/load", { id: model.id });
    notify("模型已加载，正在等待完整的连续信号窗口。");
  }));
  $("waveScale").addEventListener("change", () => { app.waveAutoAmplitude = null; app.waveDirty = true; });
  $("waveRowHeight").addEventListener("change", () => {
    const height = Number($("waveRowHeight").value);
    if ([64, 96, 144, 192].includes(height)) $("waveCanvas").parentElement.style.setProperty("--wave-row-height", `${height}px`);
    app.waveDirty = true;
  });
  window.addEventListener("resize", () => { app.waveDirty = true; });
  if (window.ResizeObserver) new ResizeObserver(() => { app.waveDirty = true; }).observe($("waveCanvas"));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Tab" && isRecording()) {
      event.preventDefault();
      $("focusStopRecordButton").focus({ preventScroll: true });
      return;
    }
    if (event.code !== "Space" || event.repeat || event.ctrlKey || event.altKey || event.metaKey || event.shiftKey) return;
    if (event.target.closest?.("input, textarea, select, button, a, summary, [contenteditable=true], [role=button]")) return;
    if (isLive() && !isRecording() && !app.busy) { event.preventDefault(); void startRecording(); }
  });
  const handleLostFocus = () => { if (isRecording() || app.recordRequested) void safetyStop("页面失去焦点，录制已停止并保留。"); };
  window.addEventListener("blur", handleLostFocus);
  document.addEventListener("visibilitychange", () => { if (document.hidden) handleLostFocus(); });
  window.addEventListener("pagehide", handleLostFocus);

  async function initialize() {
    renderControls();
    requestAnimationFrame(frame);
    try { await bootstrap(); await poll(); }
    catch (error) { notify(`初始化失败：${error.message} 请确认本地服务已启动并刷新页面。`, true, true); }
    setInterval(() => { if (app.token) void poll(); }, 250);
  }
  void initialize();
})();
