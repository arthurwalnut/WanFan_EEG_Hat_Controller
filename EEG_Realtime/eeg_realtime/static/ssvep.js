(function (global) {
  'use strict';
  const CLIENT_BUILD = '20260929-ssvep-1';
  const FRAME_TIMING_ACTION = 'warn_continue';
  // Advisory only. Severe display gaps are assessed separately at 250 ms.
  const FRAME_INTERVAL_POLICY = Object.freeze({id: 'ssvep-timestamp-advisory-v1', min_s: .003, max_s: .04, mean_tolerance_s: .003});
  const frameIntervalAllowed = ms => Number.isFinite(ms) && ms / 1000 >= FRAME_INTERVAL_POLICY.min_s - 1e-9 && ms / 1000 <= FRAME_INTERVAL_POLICY.max_s + 1e-9;
  function matchingFramePolicy(value) {
    return value?.id === FRAME_INTERVAL_POLICY.id && ['min_s', 'max_s', 'mean_tolerance_s'].every(key => typeof value[key] === 'number' && Number.isFinite(value[key]) && Math.abs(value[key] - FRAME_INTERVAL_POLICY[key]) <= 1e-12);
  }
  const median = values => { const v = [...values].sort((a, b) => a - b); return v[Math.floor(v.length / 2)]; };
  function assessRefresh(intervals) {
    if (intervals.length < 60 || intervals.some(x => !Number.isFinite(x) || x <= 0)) throw new Error('刷新检查样本不足');
    const period = median(intervals), hz = 1000 / period;
    if (hz < 50 || hz > 245) throw new Error(`当前约 ${hz.toFixed(1)} Hz；请使用 50–240 Hz 显示模式`);
    return {hz, period, maxInterval: Math.max(...intervals), irregularIntervals: intervals.filter(x => x > period * 1.8).length};
  }
  function chooseClock(samples, allowTimingWarnings = false) {
    const valid = samples.filter(x => x.domain === 'perf_counter' && Number.isFinite(x.server) && Number.isFinite(x.before) && Number.isFinite(x.after) && x.after >= x.before && Number.isFinite(x.resolution) && x.resolution > 0);
    if (valid.length < 5) throw new Error('高精度时钟同步样本不足');
    valid.sort((a, b) => (a.after - a.before) - (b.after - b.before));
    const best = valid[0], uncertainty = (best.after - best.before) / 2 + best.resolution + .0005;
    if (!allowTimingWarnings && uncertainty > .05) throw new Error(`时钟不确定度 ${(uncertainty * 1000).toFixed(2)} ms，超过 50 ms，不能开始刺激`);
    return {offset: best.server - (best.before + best.after) / 2, uncertainty};
  }
  const requestTimeout = path => /(?:^|\/)bootstrap$/.test(path) ? 30000 : 4000;
  const bootstrapRetryDelay = failures => Math.min(30000, 1000 * 2 ** Math.min(5, Math.max(0, failures - 1)));
  function clockAudit(frozen, measured, trialId, checkedAt, allowTimingWarnings = false) {
    const delta = measured.offset - frozen.offset, effective = Math.abs(delta) + measured.uncertainty;
    if (!Number.isFinite(effective) || !Number.isFinite(measured.uncertainty) || measured.uncertainty < 0 || !Number.isFinite(checkedAt) || checkedAt < 0 || (!allowTimingWarnings && effective > .05)) throw new Error('时钟漂移或复核误差超限，或时钟证据无效');
    return {trial_id: trialId, checked_at_s: checkedAt, offset_delta_s: delta,
      new_sync_uncertainty_s: measured.uncertainty, effective_sync_uncertainty_s: effective};
  }
  function verifyPostTrialClock(audit, callbackLag) {
    if (!Number.isFinite(callbackLag) || callbackLag < 0 || !Number.isFinite(audit.effective_sync_uncertainty_s) || audit.effective_sync_uncertainty_s < 0) throw new Error('事后时钟复核证据无效');
    return audit;
  }
  const timingCheckFresh = (checkedAt, now) => Number.isFinite(checkedAt) && now >= checkedAt && now - checkedAt <= 60000;
  function currentPrediction(prediction = {}, context = {}) {
    let p = prediction;
    if (context.phase === 'rest') p = {status: 'rest', reason: '休息中，等待下一试次。'};
    else if (context.timingInvalid) p = {status: 'timing_warning', reason: '当前试次时序不合格；原始 EEG 继续保存，等待后续正常试次。'};
    else if (context.trialId != null && p.trial_id !== context.trialId) p = {status: 'warming_up', reason: '新试次等待完整刺激与 EEG 窗口。'};
    else if (Number.isFinite(context.serverTime) && Number.isFinite(p.server_time) && context.serverTime - p.server_time > 1) p = {status: 'unavailable', reason: '预测已过期，等待新结果。'};
    if (!['accepted', 'predicted', 'uncertain'].includes(p.status)) return {...p, candidate_target: null, label: null, accepted: false, scores: [], correlations: [], margin: null};
    return p;
  }
  function metricsText(metrics) {
    if (!metrics || !Object.keys(metrics).length) return '';
    const pct = value => typeof value === 'number' && Number.isFinite(value) ? (100 * value).toFixed(1) + '%' : '—';
    return `留出测试：窗口准确率 ${pct(metrics.window_accuracy)} · 试次准确率 ${pct(metrics.trial_accuracy)} · 接受覆盖率 ${pct(metrics.coverage)} · 拒绝率 ${pct(metrics.rejection_rate)} · 已接受窗口准确率 ${pct(metrics.accepted_accuracy)}。${metrics.accepted_accuracy == null ? '没有已接受窗口时，准确率不定义。' : ''}重叠窗口并非独立样本；尚未验证不注视状态。${metrics.threshold_fallback ? ' 当前使用固定启发式阈值，校准可靠性尚未建立。' : ''}`;
  }
  function validationText(validation) {
    if (!validation) return '';
    if (validation.status === 'error') return `录制校验失败：${validation.error || '请查看事件'}`;
    const names = {missing_post_trial_clock_audit: '缺少试次后的时钟复核证据', invalid_post_trial_clock_audit: '试次后时钟复核不合格',
      missing_final_clock_audit: '缺少最后试次的结束时钟复核', invalid_final_clock_audit: '结束时钟复核不合格', post_clock_budget_exceeded: '事后总时序预算超限',
      incomplete_frames: '刺激帧不完整', missing_tail: '缺少结束帧', invalid_frame_evidence: '帧证据不合格', dropped_or_irregular_frame: '掉帧或帧间隔异常',
      invalid_clock_audit: '时钟复核证据不合格', display_interrupted: '显示中断或时钟回溯失效', raw_quality: '原始 EEG 硬质量检查不合格',
      timing_warning: '录制期间时序告警',
      incomplete_samples: 'EEG 覆盖不完整', sample_gap: 'EEG 缺样或跨段', sample_clock_future: 'EEG 时间估计异常', unverified_display_callback: '旧版缺少回调时序审计'};
    const reasons = Object.entries(validation.rejection_reasons || {}).map(([reason, count]) => `${names[reason] || reason} ${count} 试次`).join('；');
    return `最近录制：通过全部检查 ${validation.valid_trials ?? '—'} / ${validation.planned_trials ?? '—'} 试次；软件时序通过 ${validation.timing_valid_trials ?? '—'}，EEG 硬质量通过 ${validation.quality_valid_trials ?? '—'}。${reasons ? '排除原因：' + reasons + '。同一试次可有多个原因。' : ''}`;
  }
  function recordingTitle(item) {
    const date = new Date(item.created_utc);
    const started = Number.isFinite(date.getTime()) ? date.toLocaleString('zh-CN', {hour12: false}) : '开始时间未知';
    return `${item.recording_name || item.subject || item.id} · 受试者 ${item.subject || '未填写'} · ${started} · ${item.target_count ?? 4} 类`;
  }
  const FREQUENCIES = Object.freeze([8, 10, 12, 14]);
  const luminancesAt = seconds => FREQUENCIES.map(f => .5 + .5 * Math.sin(2 * Math.PI * f * seconds));
  class FrameRunner {
    constructor(options) {
      Object.assign(this, options);
      this.active = true; this.phase = 'rest'; this.trialIndex = 0; this.frameIndex = 0;
      this.codeFrame = 0; this.previous = null; this.pending = null; this.lastVisible = 0;
      this.warningKeys = new Set(); this.timingInvalidTrials = new Set();
      this.restUntil = this.startMs + this.restMs; this.taskStart = null;
      this.onRest(this.trials[0], 0);
    }
    abort(reason) {
      if (!this.active) return;
      this.active = false; this.pending = null; this.onBlank(); this.onAbort(reason, this.lastVisible);
    }
    warn(code, reason, severe = false) {
      const trial = this.trials[this.trialIndex], key = `${trial.trial_id}:${code}`;
      if (severe) this.timingInvalidTrials.add(trial.trial_id);
      if (this.warningKeys.has(key)) return;
      this.warningKeys.add(key); this.onWarning?.({trial_id: trial.trial_id, code, reason, severe});
    }
    tick(now) {
      if (!this.active) return;
      if (!Number.isFinite(now) || !this.isVisible()) return this.abort('页面失焦、隐藏或刺激区域不可见');
      if (this.phase === 'rest') {
        this.lastVisible = this.toRelative(now);
        if (now < this.restUntil || (this.readyForTask && !this.readyForTask(this.trials[this.trialIndex]))) return;
        this.phase = 'task'; this.codeFrame = 0; this.taskStart = now; this.previous = null;
        this.onTask(this.trials[this.trialIndex], this.trialIndex);
        if (!this.active) return;
      }
      const lag = ((this.readNow ? this.readNow() : now) - now) / 1000;
      if (!Number.isFinite(lag) || lag < -.001) return this.abort('浏览器回调时间无效');
      if (lag > .05) this.warn('callback_delay', '显示回调延迟，已记录实际时间；EEG 继续保存', lag > .25);
      if (this.previous !== null) {
        const dt = now - this.previous;
        if (dt <= 0) return this.abort('刺激时间未递增');
        if (dt > 40) this.warn('frame_interval', `帧间隔 ${dt.toFixed(1)} ms，已记录；EEG 继续保存`, dt > 250);
      }
      this.previous = now;
      if (this.pending) {
        const confirmed = this.pending; this.pending = null; this.lastVisible = confirmed.observed_s;
        this.onEvidence(confirmed);
        if (!this.active) return;
      }
      if (now - this.taskStart >= 6000 - 1e-7) {
        const trial = this.trials[this.trialIndex], tail = this.toRelative(now);
        this.onBlank(); this.lastVisible = tail;
        const finished = this.trialIndex + 1 >= this.trials.length;
        if (!finished) {
          this.phase = 'rest'; this.previous = null; this.restUntil = now + this.restMs;
          this.onRest(this.trials[this.trialIndex + 1], this.trialIndex + 1);
        }
        if (!this.active) return;
        const tailLag = ((this.readNow ? this.readNow() : now) - now) / 1000;
        if (!Number.isFinite(tailLag) || tailLag < -.001) return this.abort('清屏回调时间无效');
        if (tailLag > .05) this.warn('tail_callback_delay', '清屏回调迟到，已保留实际延迟', tailLag > .25);
        this.onTail(trial, tail, Math.max(0, tailLag));
        if (!this.active) return;
        this.trialIndex++;
        if (finished) { this.active = false; this.onComplete(); }
        return;
      }
      const trial = this.trials[this.trialIndex], luminances = luminancesAt((now - this.taskStart) / 1000);
      this.onFrame(luminances, this.codeFrame, trial);
      if (!this.active) return;
      const endLag = Math.max(0, ((this.readNow ? this.readNow() : now) - now) / 1000);
      this.pending = {frame_index: this.frameIndex++, code_frame: this.codeFrame++, observed_s: this.toRelative(now), trial_id: trial.trial_id, callback_end_lag_s: endLag, luminances};
    }
  }
  const core = {FREQUENCIES, luminancesAt, CLIENT_BUILD, FRAME_TIMING_ACTION, FRAME_INTERVAL_POLICY, frameIntervalAllowed, matchingFramePolicy, assessRefresh, chooseClock, FrameRunner, requestTimeout, bootstrapRetryDelay, currentPrediction, metricsText, validationText, recordingTitle, clockAudit, verifyPostTrialClock, timingCheckFresh};
  if (typeof module !== 'undefined' && module.exports) { module.exports = core; return; }
  global.SSVEPFrontend = core;
  if (typeof document === 'undefined') return;

  const $ = id => document.getElementById(id);
  const targetNodes = [...document.querySelectorAll('.target')];
  const channelNames = ['O2', 'O1', 'P8', 'P7', 'C4', 'C3', 'FP1', 'FP2'];
  let token = '', bootstrap = {}, state = {}, clock = null, refresh = null, checkedAt = null;
  let runner = null, raf = null, session = null, busy = false, pollBusy = false, ending = false;
  let evidence = [], reportQueue = null, listSignature = '', autoScale = 50;
  let bootstrapBusy = false, bootstrapFailures = 0, bootstrapRetry = null;
  let ownerToken = null, sessionClock = null, lastVisibleCutoff = 0, stoppingBody = null;
  let startGeneration = 0, starting = false;
  const timingAudits = new Map(), trialCallbackLags = new Map();
  const fmt = (value, digits = 1) => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
  const label = value => Number.isInteger(value) && value >= 0 && value < 4 ? `目标 ${value + 1} · ${FREQUENCIES[value]} Hz` : '未知意图';
  function selectedTargetCount() {
    const count = Number($('targetCount').value || 4);
    if (count !== 4) throw new Error('SSVEP 固定四分类');
    return count;
  }
  function targetLayout(count = selectedTargetCount()) {
    for (const n of [2, 4, 6]) $('targets').classList.toggle(`count-${n}`, n === count);
    targetNodes.forEach((node, i) => { node.hidden = i >= count; if (node.hidden) node.classList.remove('on', 'off', 'cued'); });
    const previous = Number($('demoTarget').value || 0);
    $('demoTarget').replaceChildren();
    for (let i = 0; i < count; i++) { const option = document.createElement('option'); option.value = String(i); option.textContent = label(i); $('demoTarget').append(option); }
    $('demoTarget').value = String(previous >= 0 && previous < count ? previous : 0);
    if (!runner) $('stageProgress').textContent = `${count} 个目标当前保持静态。`;
  }
  function targetSummary() {
    const count = selectedTargetCount(), trials = Number($('trials').value);
    $('targetSummary').textContent = Number.isInteger(trials) && trials >= 1 && trials <= 20 ? `${count} 类 × 每类 ${trials} 次 = ${count * trials} 试次，预计总时长约 ${(count * trials * 8 / 60).toFixed(1)} 分钟（含每次 2 秒休息）。FBCCA 使用录制数据校准阈值并做留出测试。` : '每个目标的试次数须为 1～20 的整数。';
    const modelCount = state.model?.target_count ?? 4;
    $('targetModelHint').textContent = state.model?.available && modelCount !== count ? `已加载 ${modelCount} 类 FBCCA 配置；在线识别需选择 ${modelCount} 类，或加载 ${count} 类 FBCCA 配置。` : '';
  }
  function notify(message, error = false) { $('notice').textContent = message; $('notice').classList.toggle('error', error); }
  function startFeedback(message, error = false) { $('startStatus').textContent = message; $('startStatus').classList.toggle('error', error); }
  async function api(path, body, timeoutMs = requestTimeout(path)) {
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const options = {signal: controller.signal, cache: 'no-store'};
      if (body !== undefined) Object.assign(options, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-EEG-Token': token}, body: JSON.stringify(body)});
      const response = await fetch(path.startsWith('/api/') ? path : '/api/ssvep/' + path, options);
      const result = await response.json();
      if (!response.ok || result.ok === false || result.error) {
        const ignored = response.ok && result.ignored === true;
        const endedSession = ignored && result.session?.id === body?.id && result.session?.active === false ? result.session : null;
        const message = ignored ? endedSession?.reason || '录制会话已结束或已切换，该请求未被接收。' : result.error || result.message || `请求失败 ${response.status}`;
        throw Object.assign(new Error(message), {ignored, endedSession});
      }
      return result;
    } catch (error) {
      if (controller.signal.aborted || error.name === 'AbortError') throw new Error(path === 'bootstrap' ? '加载基础信息超时；首次加载科学计算库可能较慢' : '请求超时，请检查本地服务');
      throw error;
    } finally { clearTimeout(timeout); }
  }
  async function action(fn) {
    if (busy) return;
    busy = true; controls();
    try { await fn(); } catch (error) { notify(error.message, true); }
    finally { busy = false; controls(); }
  }
  function controls() {
    const active = Boolean(runner || session?.active || state.session?.active || state.session?.finishing);
    const streaming = ['running', 'connecting'].includes(state.status);
    for (const id of ['source', 'deviceIp', 'discover', 'connect', 'loadReplay', 'train', 'models', 'loadModel']) $(id).disabled = !token || busy || active || streaming;
    $('disconnect').disabled = busy || active || !streaming;
    $('calibrate').disabled = busy || active || state.status !== 'running' || state.source === 'replay' || !$('consent').checked;
    $('calibrate').textContent = starting ? '正在检查并启动…' : '开始校准录制';
    $('startRequirements').textContent = active ? '实验正在进行，请先结束并保存。' : busy ? '正在处理操作，请等待结果。' : state.source === 'replay' ? '当前为回放，请先连接模拟或真实 EEG 数据源。' : state.status !== 'running' ? '连接信号源后可开始。' : !$('consent').checked ? '请先勾选闪烁刺激说明。' : '已就绪；开始前会检查显示与时钟。';
    $('targetCount').disabled = active || busy;
    $('subject').disabled = active || busy; $('recordingName').disabled = active || busy;
    $('online').disabled = $('calibrate').disabled || !state.model?.available || state.model?.compatible === false || (state.model?.target_count ?? 4) !== selectedTargetCount();
    $('demoTargetLabel').hidden = $('source').value !== 'demo'; $('demoTarget').disabled = active || busy;
    $('stopSession').disabled = !active; $('overlayStop').disabled = ending;
    $('stopSession').textContent = active && !ownerToken ? '紧急结束其他页面的录制' : '结束并保存';
    $('checkDisplay').disabled = !token || busy || active;
    $('retryLoad').disabled = bootstrapBusy;
    targetSummary();
    for (const id of ['play', 'pause', 'seek', 'speed']) $(id).disabled = busy || active || !state.playback?.loaded;
  }
  function blank() { targetNodes.forEach(node => {node.classList.remove('on', 'off'); node.style.removeProperty('background-color');}); }
  function setCue(trial, isTask) {
    const online = session?.kind === 'online';
    $('stageKind').textContent = `${state.source === 'demo' ? '模拟演示 · ' : ''}${online ? '在线自由注视' : '校准录制'} · ${isTask ? '刺激中' : '休息 / 准备'}`;
    $('stageCue').textContent = online ? (isTask ? '自由注视一个目标' : '休息，下一轮自由选择目标') : `${isTask ? '请注视' : '准备注视'}${label(trial.target)}`;
    targetNodes.forEach((node, i) => node.classList.toggle('cued', !online && i === trial.target));
  }
  function visible() {
    if (document.hidden || !document.hasFocus()) return false;
    const box = $('targets').getBoundingClientRect();
    return box.width > 120 && box.height > 100 && box.top >= 0 && box.bottom <= innerHeight + 1 && box.left >= 0 && box.right <= innerWidth + 1;
  }
  async function measureDisplay() {
    return new Promise((resolve, reject) => {
      let last = null, handle = null, values = [], settled = false;
      const end = (error) => { if (settled) return; settled = true; clearTimeout(timeout); cancelAnimationFrame(handle); error ? reject(error) : resolve(assessRefresh(values)); };
      const timeout = setTimeout(() => end(new Error('显示检查超时；请保持当前页面可见')), 6000);
      function sample(time) {
        if (document.hidden || !document.hasFocus()) return end(new Error('显示检查需要当前页面保持焦点'));
        if (!Number.isFinite(time) || performance.now() < time - 1) return end(new Error('浏览器时间值无效，请重新检查'));
        if (last !== null) values.push(time - last); last = time;
        if (values.length >= 120) { try { assessRefresh(values); end(); } catch (error) { end(error); } }
        else handle = requestAnimationFrame(sample);
      }
      handle = requestAnimationFrame(sample);
    });
  }
  async function syncClock(maxWaitMs = 2000, allowTimingWarnings = false) {
    const samples = [], deadline = performance.now() + maxWaitMs;
    for (let i = 0; i < 7; i++) {
      const remaining = deadline - performance.now();
      if (remaining <= 0) throw new Error('时钟复核超时');
      const before = performance.now() / 1000, result = await api('clock', undefined, Math.min(requestTimeout('clock'), remaining)), after = performance.now() / 1000;
      samples.push({before, after, server: result.server_time, domain: result.clock_domain, resolution: result.clock_resolution_s});
    }
    return chooseClock(samples, allowTimingWarnings);
  }
  async function checkTiming() {
    checkedAt = null; $('displayCheck').textContent = '静态检查中，请保持页面聚焦…';
    startFeedback('正在静态检查显示刷新与时钟，请保持当前页面可见…');
    try {
      refresh = await measureDisplay(); clock = await syncClock(); checkedAt = performance.now();
      $('displayCheck').textContent = `${fmt(refresh.hz, 2)} Hz · 时钟不确定度 ${fmt(clock.uncertainty * 1000, 2)} ms · 通过（软件估计）${refresh.irregularIntervals ? ` · ${refresh.irregularIntervals} 次帧间隔告警，允许开始` : ''}`;
      startFeedback('静态显示与时钟检查通过。' + (starting ? '正在请求创建录制…' : '可点击开始校准录制。'));
    } catch (error) { clock = null; refresh = null; $('displayCheck').textContent = error.message; startFeedback(`检查未通过：${error.message}`, true); throw error; }
  }
  function flushEvidence(trialId, tail, tailLag = 0) {
    const frames = evidence.splice(0), id = session?.id;
    if (!id || (!frames.length && tail === undefined)) return;
    if (frames.length > 16) { runner?.abort('帧报告积压'); return; }
    const audit = timingAudits.get(trialId);
    const queue = reportQueue;
    if (!audit || !ownerToken || !queue || queue.id !== id || queue.owner !== ownerToken) { runner?.abort('缺少当前试次时钟审计或页面所有权'); return; }
    const lag = Math.max(tailLag, ...frames.map(frame => frame.callback_end_lag_s));
    trialCallbackLags.set(trialId, Math.max(trialCallbackLags.get(trialId) ?? 0, lag));
    const body = {id, owner_token: ownerToken, trial_id: trialId, frames, visible: true,
      clock_audit: audit, clock_sync_uncertainty_s: audit.effective_sync_uncertainty_s,
      callback_end_lag_s: lag, clock_uncertainty_s: audit.effective_sync_uncertainty_s + lag};
    if (tail !== undefined) { body.tail_s = tail; body.tail_callback_end_lag_s = tailLag; }
    queue.pending++;
    // Keep a bounded ordered queue through short local-server stalls. Three
    // batches are only ~300ms; receipt lateness is assessed by the backend.
    if (queue.pending > 64) { queue.pending--; runner?.abort('帧报告持续积压超过缓存容量，已停止刺激并保留数据'); return; }
    if (queue.pending > 3 && !queue.backlogNotified) {
      queue.backlogNotified = true;
      $('timingWarning').textContent = '帧报告暂时排队；继续录制并保留实际帧时间，收到后按试次评估时序。';
    }
    queue.chain = queue.chain.then(async () => {
      const result = await api('frames', body);
      if (reportQueue === queue && session?.id === id && ownerToken === queue.owner && result.timing_valid === false) {
        runner?.timingInvalidTrials.add(trialId);
        const warning = result.timing_warnings?.[0];
        $('timingWarning').textContent = `试次 ${trialId} · ${warning?.reason || '时序不合格'}；原始 EEG 继续保存。`;
        if (runner?.active && runner.phase === 'task' && runner.trials[runner.trialIndex]?.trial_id === trialId) renderPrediction(currentPrediction({}, {timingInvalid: true}));
      }
    }).catch(error => {
      // Late responses from an earlier recording cannot stop or mutate a new one.
      if (reportQueue === queue && session?.id === id && ownerToken === queue.owner) {
        const reason = error.ignored ? error.message : `帧报告失败：${error.message}`;
        if (error.ignored) { queue.ignoredReason = reason; queue.endedSession = error.endedSession; }
        if (runner?.active) runner.abort(reason);
        else if (!ending) notify(reason, true);
      }
      throw error;
    }).finally(() => { queue.pending--; });
    // Attach a terminal handler immediately; completion still checks this queue.
    queue.chain.catch(() => {});
  }
  async function stopSession(reason = 'user_stop', cutoff = null) {
    const current = session || (state.session?.active ? state.session : null);
    if (ending || !current) return;
    ending = true; controls();
    const localSession = Boolean(session && ownerToken), queue = reportQueue;
    // Only reconcile a stop originating from a server-ignored frame report.
    // An in-flight frame response must not replace a local Esc/focus-loss reason.
    let reconcileEndedSession = Boolean(queue?.ignoredReason && queue.ignoredReason === reason);
    let endedSession = reconcileEndedSession ? queue.endedSession : null;
    const finalTrial = runner?.trials[runner.trials.length - 1]?.trial_id;
    lastVisibleCutoff = cutoff ?? runner?.lastVisible ?? lastVisibleCutoff;
    if (runner) runner.active = false;
    cancelAnimationFrame(raf); raf = null; blank();
    runner = null; evidence = [];
    const body = {id: current.id, reason};
    if (localSession) {
      body.owner_token = ownerToken;
      if (reason !== 'completed') body.invalid_since_s = Math.max(0, lastVisibleCutoff);
    } else { body.emergency = true; body.reason = '其他页面明确请求紧急停止'; }
    stoppingBody = body;
    startFeedback(reason === 'completed' ? '刺激已停止，正在复核末尾时钟并保存…' : `刺激已停止，正在保存。原因：${body.reason}`, reason !== 'completed' && reason !== 'user_stop');
    try {
      if (reason === 'completed') {
        try {
          await queue?.chain;
          $('stageKind').textContent = '刺激已停止 · 末尾时钟复核';
          const measured = await syncClock(2000, true);
          body.final_clock_audit = verifyPostTrialClock(clockAudit(sessionClock, measured, finalTrial, Math.max(0, performance.now() / 1000 + sessionClock.offset - current.start_time), true), trialCallbackLags.get(finalTrial) ?? 0);
        } catch (error) {
          // The last task has no following rest in which to audit its clock.
          // Without this final proof, conservatively invalidate that whole task.
          if (error.ignored) {
            reconcileEndedSession = true; endedSession = error.endedSession;
            body.reason = error.message; body.invalid_since_s = Math.max(0, lastVisibleCutoff);
          } else {
            body.final_clock_audit_error = String(error.message);
            notify(`末尾时钟复核失败：${error.message}；最后试次不会用于训练。`, true);
          }
        }
      }
      else if (localSession) {
        // Give already confirmed tail reports a short chance to finish before
        // the stop request; never wait indefinitely while the display is blank.
        await Promise.race([queue?.chain.catch(() => {}), new Promise(resolve => setTimeout(resolve, 150))]);
      }
      document.body.classList.remove('stimulating'); targetNodes.forEach(n => n.classList.remove('cued'));
      renderPrediction({status: 'unavailable', reason: '刺激已停止；历史结果保留在录制文件中。'});
      const result = await api('session/stop', body);
      if (reconcileEndedSession && result.session?.id === current.id && result.session?.active === false) endedSession = result.session;
      if (reconcileEndedSession && endedSession?.id === current.id && endedSession?.reason) body.reason = endedSession.reason;
      const backendStatus = reconcileEndedSession && endedSession?.id === current.id ? endedSession.status : null;
      const message = backendStatus === 'complete' ? `录制已完成并保存：${body.reason}` : backendStatus === 'aborted' ? `录制已中止并保留：${body.reason}` : backendStatus === 'error' ? `录制保存或校验异常：${body.reason}` : body.reason === 'completed' ? '刺激已完成，等待最后 EEG 数据并保存校验结果。' : body.reason === 'final_clock_audit_failed' ? '已结束并保留录制；末尾帧或时钟复核未通过，最后试次将被排除，请查看校验结果。' : `已停止刺激并保留录制：${body.reason}`;
      const failed = backendStatus ? backendStatus !== 'complete' : body.reason !== 'completed' && reason !== 'user_stop';
      notify(message, failed); startFeedback(message, failed);
    } catch (error) { const message = `结束请求失败：${error.message}。刺激已停止，请检查服务端状态。`; notify(message, true); startFeedback(message, true); }
    finally { current.active = false; session = null; ownerToken = null; stoppingBody = null; sessionClock = null; reportQueue = null; ending = false; checkedAt = null; $('stageKind').textContent = '刺激已停止'; $('stageCue').textContent = '重新检查后可开始下一次实验'; await poll(); await reloadLists(); controls(); }
  }
  async function auditTrialClock(trial, ownedSession, ownedToken) {
    try {
      const measured = await syncClock(2000, true);
      if (session?.id !== ownedSession.id || ownerToken !== ownedToken || ending) return;
      const audit = clockAudit(sessionClock, measured, trial.trial_id, Math.max(0, performance.now() / 1000 + sessionClock.offset - ownedSession.start_time), true);
      if (trial.trial_id > 1) verifyPostTrialClock(audit, trialCallbackLags.get(trial.trial_id - 1) ?? 0);
      timingAudits.set(trial.trial_id, audit);
      if (audit.effective_sync_uncertainty_s + (trialCallbackLags.get(trial.trial_id - 1) ?? 0) > .05) notify('时钟复核告警：继续保存原始 EEG，受影响试次按时序检查筛除。');
    } catch (error) {
      if (session?.id === ownedSession.id && ownerToken === ownedToken && runner?.active) {
        // A detected clock shift can affect the preceding trial; retaining it
        // as accurately aligned would be misleading.
        if (/时钟漂移|事后时钟复核/.test(error.message)) runner.lastVisible = timingAudits.get(trial.trial_id - 1)?.checked_at_s ?? 0;
        runner.abort(`试次前时钟复核失败：${error.message}`);
      }
    }
  }
  async function startSession(kind) {
    const attempt = ++startGeneration; starting = true;
    startFeedback('正在准备' + (kind === 'calibration' ? '校准录制' : '在线自由注视') + '…'); controls();
    const assertStillRequested = () => {
      if (attempt !== startGeneration || document.hidden || !document.hasFocus() || !$('consent').checked) throw new Error('启动已取消：页面焦点、显示条件或确认状态发生变化，请重新点击开始');
    };
    try {
    if (!$('consent').checked) throw new Error('请先阅读并确认闪烁刺激说明');
    if (document.hidden || !document.hasFocus()) throw new Error('请保持页面可见并获得焦点');
    if (bootstrap.client_build !== CLIENT_BUILD || !matchingFramePolicy(bootstrap.frame_interval_policy)) throw new Error('页面与后台版本不一致，请刷新页面；若仍不一致，请更新本地服务');
    if (!clock || !timingCheckFresh(checkedAt, performance.now())) await checkTiming();
    else { startFeedback('正在重新同步时钟…'); clock = await syncClock(); }
    assertStillRequested();
    const targetCount = selectedTargetCount();
    if (JSON.stringify(bootstrap.frequencies) !== JSON.stringify(FREQUENCIES)) throw new Error('后台频率配置不匹配');
    const count = Number($('trials').value);
    if (!Number.isInteger(count) || count < 1 || count > 20) throw new Error('每目标试次必须为 1～20 的整数');
    startFeedback('检查已通过，正在等待后台创建录制…');
    const result = await api('session/start', {kind, client_build: CLIENT_BUILD, frame_interval_policy: FRAME_INTERVAL_POLICY, target_count: targetCount, trials_per_target: count, trial_count: 12, subject: $('subject').value.trim(), recording_name: $('recordingName').value.trim(), demo_target: Number($('demoTarget').value), clock_uncertainty_s: .05, initial_clock_sync_uncertainty_s: clock.uncertainty, software_timing_budget_s: .05, display_timing_method: 'ssvep-timestamp-frames-v1'});
    session = result.session; ending = false; evidence = [];
    $('timingWarning').textContent = '';
    ownerToken = result.owner_token || null; sessionClock = {...clock}; timingAudits.clear(); trialCallbackLags.clear(); lastVisibleCutoff = 0;
    reportQueue = {id: session?.id, owner: ownerToken, pending: 0, chain: Promise.resolve()};
    try { assertStillRequested(); } catch (error) { await stopSession('start_cancelled', 0); throw error; }
    if (!session?.id || !Number.isFinite(session.start_time) || !Array.isArray(session.plan)) {
      if (session?.id) await stopSession('invalid_plan', 0);
      throw new Error('服务端未返回完整的刺激计划');
    }
    if (session.client_build !== CLIENT_BUILD || !matchingFramePolicy(session.frame_interval_policy)) {
      await stopSession('frame_policy_mismatch', 0); throw new Error('录制使用的版本或帧间隔策略与页面不一致，未开始闪烁');
    }
    const trials = session.plan.filter(phase => phase.phase === 'task');
    if (!trials.length) { await stopSession('invalid_plan', 0); throw new Error('刺激计划为空'); }
    const confirmedCount = session.target_count ?? session.contract?.targets?.length ?? 4;
    if (confirmedCount !== 4 || trials.some(trial => kind === 'calibration' ? !Number.isInteger(trial.target) || trial.target < 0 || trial.target >= 4 : trial.target !== null)) { await stopSession('target_contract_mismatch', 0); throw new Error('后台返回的目标数或标签不匹配'); }
    if (!ownerToken) { await stopSession('missing_owner'); throw new Error('缺少独立页面令牌，未启用刺激'); }
    targetLayout(targetCount);
    document.body.classList.add('stimulating');
    runner = new FrameRunner({trials, restMs: 2000,
      startMs: (session.start_time - sessionClock.offset) * 1000,
      toRelative: now => Math.max(0, now / 1000 + sessionClock.offset - session.start_time), isVisible: visible,
      readNow: () => performance.now(),
      getSyncUncertainty: trial => timingAudits.get(trial.trial_id)?.effective_sync_uncertainty_s ?? sessionClock.uncertainty,
      readyForTask: trial => timingAudits.has(trial.trial_id),
      onBlank: blank, onRest: (trial, i) => { setCue(trial, false); $('stageProgress').textContent = `第 ${i + 1} / ${trials.length} 试次 · 休息 2 秒 / 时钟复核`; auditTrialClock(trial, session, ownerToken); },
      onTask: (trial, i) => { setCue(trial, true); $('stageProgress').textContent = `第 ${i + 1} / ${trials.length} 试次 · 6 秒 · 8 / 10 / 12 / 14 Hz`; },
      onFrame: levels => targetNodes.forEach((node, i) => {const value = Math.round(255 * levels[i]); node.style.backgroundColor = `rgb(${value},${value},${value})`; }),
      onEvidence: frame => { evidence.push(frame); if (evidence.length >= 6) flushEvidence(frame.trial_id); },
      onTail: (trial, tail, lag) => flushEvidence(trial.trial_id, tail, lag),
      onWarning: warning => { $('timingWarning').textContent = `试次 ${warning.trial_id} · ${warning.reason}。原始 EEG 继续保存。`; if (warning.severe) renderPrediction(currentPrediction({}, {timingInvalid: true})); },
      onAbort: (reason, cutoff) => { stopSession(reason, cutoff); }, onComplete: () => { stopSession('completed'); }
    });
    function step(time) {
      const current = runner; if (!current?.active) return;
      try { current.tick(time); lastVisibleCutoff = current.lastVisible; }
      catch (error) { current.abort(`呈现处理失败：${error.message}`); }
      if (runner?.active) raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step); notify('刺激计划已开始；保持页面可见，按 Esc 可随时停止。'); startFeedback('录制已启动，正在静态休息与试次前时钟复核。'); controls();
    } catch (error) { startFeedback(`未能开始：${error.message}`, true); throw error; }
    finally { starting = false; }
  }
  async function discoverDevice() {
    $('devices').replaceChildren();
    const result = await api('/api/discover', {device_ip: $('deviceIp').value.trim()}), devices = result.devices || result.ips || [];
    for (const ip of devices) { const button = document.createElement('button'); button.textContent = ip; button.onclick = () => { $('deviceIp').value = ip; }; $('devices').append(button); }
    if (devices.length === 1) $('deviceIp').value = devices[0];
    notify(devices.length ? `发现 ${devices.length} 台身份已确认的设备；连接前请核对地址。` : '未发现设备；请确认电脑热点已开启、设备已加入且网络权限已配置。');
    return devices;
  }
  async function connectSource() {
    const source = $('source').value;
    if (source === 'device') {
      const preferred = $('deviceIp').value.trim(), devices = await discoverDevice();
      if (devices.length > 1 && !devices.includes(preferred)) throw new Error('发现多台设备，请先点击一个地址再连接');
      if (devices.includes(preferred)) $('deviceIp').value = preferred;
      if (!$('deviceIp').value.trim()) throw new Error('没有可连接地址');
    }
    await api('start', {source, device_ip: $('deviceIp').value.trim()}); await poll();
  }
  function populateSelect(id, items, title) {
    const node = $(id), previous = node.value; node.replaceChildren();
    const empty = document.createElement('option'); empty.value = ''; empty.textContent = title; node.append(empty);
    for (const item of items) { const option = document.createElement('option'), incompatible = item.compatible === false || item.status === 'incompatible'; option.value = item.id; option.textContent = `${item.kind ? recordingTitle(item) : `${item.subject || '未填写受试者'} · ${item.id} · ${item.target_count ?? 4} 类`} · ${item.source || ''}${item.kind ? ' · ' + item.kind : ''}${incompatible ? ' · 不兼容' : ''}`; option.disabled = incompatible; node.append(option); }
    if ([...node.options].some(option => option.value === previous && !option.disabled)) node.value = previous;
  }
  function renderLists(data) {
    const sessions = data.sessions || [], models = data.models || [];
    const signature = JSON.stringify([sessions, models]); if (signature === listSignature) return; listSignature = signature;
    const selected = new Set([...$('sessionList').querySelectorAll('input:checked')].map(n => n.value));
    $('sessionList').replaceChildren();
    if (!sessions.length) $('sessionList').textContent = '尚无录制';
    for (const item of sessions) {
      const row = document.createElement('label'), input = document.createElement('input'), text = document.createElement('span');
      input.type = 'checkbox'; input.value = item.id; input.checked = selected.has(item.id); input.disabled = item.kind === 'online';
      const savedStatus = {complete: '文件已保存，有效试次以校验为准', aborted: '已中止并保存', recording: '录制中', error: '保存或校验异常'};
      text.textContent = `${recordingTitle(item)} · ${item.source || '—'} · ${item.kind || 'calibration'} · ${fmt(item.duration_s)} 秒${item.status ? ' · ' + (savedStatus[item.status] || item.status) : ''}`;
      row.append(input, text); $('sessionList').append(row);
    }
    populateSelect('replaySessions', sessions, '选择录制'); populateSelect('models', models, '选择 FBCCA 配置');
  }
  async function reloadLists() {
    if (bootstrapBusy) return false;
    clearTimeout(bootstrapRetry); bootstrapRetry = null; bootstrapBusy = true; controls();
    try {
      const wasReady = Boolean(token), wasFailed = bootstrapFailures > 0, result = await api('bootstrap');
      if (typeof result.csrf_token !== 'string' || !result.csrf_token) throw new Error('服务未返回访问令牌，请重试加载');
      bootstrap = result; token = result.csrf_token; bootstrapFailures = 0;
      const matched = result.client_build === CLIENT_BUILD && matchingFramePolicy(result.frame_interval_policy) && JSON.stringify(result.frequencies) === JSON.stringify(FREQUENCIES);
      $('buildStatus').textContent = `页面版本 ${CLIENT_BUILD} · 后台版本 ${result.client_build || '旧版'} · ${matched ? '帧时序告警时继续录制' : '版本不一致，请刷新页面后再开始'}`;
      $('buildStatus').classList.toggle('error', !matched);
      renderLists(result); $('retryLoad').hidden = true;
      if (!wasReady) { notify('服务已就绪。连接信号后，先进行静态显示与时钟检查。'); await poll(); }
      else if (wasFailed) notify('服务信息已恢复加载。');
      return true;
    } catch (error) {
      const delay = bootstrapRetryDelay(++bootstrapFailures);
      notify(`${error.message}；${delay / 1000} 秒后自动重试，也可点击“重试加载服务”。`, true);
      $('retryLoad').hidden = false;
      bootstrapRetry = setTimeout(() => { bootstrapRetry = null; reloadLists(); }, delay);
      return false;
    } finally { bootstrapBusy = false; controls(); }
  }
  function renderPrediction(prediction = {}) {
    prediction = currentPrediction(prediction);
    const statusNames = {accepted: '已接受', predicted: '已预测', uncertain: '不确定', invalid: '无效窗口', timing_warning: '时序告警 · 继续录制', warming_up: '等待完整刺激', rest: '休息中', no_model: '尚无 FBCCA 校准', unavailable: '暂不可用'};
    $('predictionState').textContent = statusNames[prediction.status] || prediction.status || '尚无预测';
    const candidate = prediction.candidate_target ?? prediction.label;
    $('predictionTarget').textContent = Number.isInteger(candidate) ? label(candidate) : '—';
    $('predictionReason').textContent = `${prediction.accepted ? '模型接受此目标。' : Number.isInteger(candidate) ? '候选目标尚未通过接受门槛。' : ''} ${prediction.reason || ''}`;
    $('predictionDetail').textContent = `前两名 FBCCA 得分差 ${fmt(prediction.margin, 3)} · FBCCA 加权相关平方分数不是概率${state.source === 'demo' ? ' · 当前为模拟演示' : ''}`;
    $('correlations').replaceChildren();
    (prediction.scores || []).forEach((score, i) => { const node = document.createElement('div'), value = document.createElement('b'); node.className = 'correlation'; node.textContent = label(i); value.textContent = fmt(score, 3); node.append(value); $('correlations').append(node); });
    $('predictionWarnings').textContent = (prediction.quality?.warnings || []).map(x => `${x.channel || ''} ${x.reason || x}`).join('；');
  }
  function drawSignal(signal = {}) {
    const canvas = $('waveform'), rect = canvas.getBoundingClientRect(), width = Math.max(320, rect.width), height = 816, ratio = devicePixelRatio || 1;
    canvas.width = width * ratio; canvas.height = height * ratio; const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height);
    const times = signal.times || [], traces = signal.traces || [], selected = $('waveScale').value;
    let peak = 0;
    for (const trace of traces) for (const value of trace) if (Number.isFinite(value)) peak = Math.max(peak, Math.abs(value));
    if (selected === 'auto') autoScale = Math.max(25, Math.ceil(peak * 1.05 / 25) * 25);
    const range = selected === 'auto' ? autoScale : Number(selected), left = 60, right = width - 12, first = times[0], last = times[times.length - 1];
    const diffs = times.slice(1).map((value, i) => value - times[i]).filter(value => value > 0), dt = diffs.length ? median(diffs) : .004;
    let clipped = false;
    for (let ch = 0; ch < 8; ch++) {
      const top = ch * 98, center = top + 49;
      ctx.strokeStyle = '#e0e8e4'; ctx.beginPath(); ctx.moveTo(left, center); ctx.lineTo(right, center); ctx.stroke();
      ctx.fillStyle = '#47625a'; ctx.font = '12px system-ui'; ctx.fillText((signal.channel_names || channelNames)[ch], 4, center + 4);
      ctx.save(); ctx.beginPath(); ctx.rect(left, top + 4, right - left, 90); ctx.clip(); ctx.strokeStyle = '#177b71'; ctx.lineWidth = 1; ctx.beginPath(); let connected = false;
      const trace = traces[ch] || [];
      for (let i = 0; i < times.length; i++) {
        const value = trace[i]; if (!Number.isFinite(value)) { connected = false; continue; }
        if (Math.abs(value) > range) clipped = true;
        const x = left + (times[i] - first) / Math.max(.004, last - first) * (right - left), y = center - value / range * 45;
        if (!connected || (i && times[i] - times[i - 1] > dt * 1.8)) ctx.moveTo(x, y); else ctx.lineTo(x, y); connected = true;
      }
      ctx.stroke(); ctx.restore();
    }
    ctx.fillStyle = '#526a60'; ctx.font = '12px system-ui'; ctx.fillText(times.length ? `${fmt(first)} s` : '等待 EEG', left, 808); if (times.length) ctx.fillText(`${fmt(last)} s`, Math.max(left + 80, right - 70), 808);
    $('waveNote').textContent = `原始信号 · 所有通道 ±${fmt(range, 0)} μV · 每行 98 px · 缺样处断线${clipped ? ' · 存在超出当前量程的峰值，请增大量程' : ''}`;
    const segments = signal.annotation_segments || (signal.annotations || []).map(x => ({...x, start: x.onset_s, end: x.tail_s}));
    const shown = segments.filter(x => Number.isFinite(x.start) && Number.isFinite(x.end) && x.end > first && x.start < last);
    for (const segment of shown) {
      const x = left + (Math.max(first, segment.start) - first) / Math.max(.004, last - first) * (right - left);
      const end = left + (Math.min(last, segment.end) - first) / Math.max(.004, last - first) * (right - left);
      ctx.fillStyle = segment.valid === false ? '#cea59e' : ['#78a7b7', '#ac91bc', '#8aac86', '#bcac72', '#83a6a0', '#c49a78'][segment.target] || '#b4bfba';
      ctx.fillRect(x, 787, Math.max(1, end - x), 9);
    }
    $('annotation').textContent = '标注：' + (shown.length ? shown.map(x => `${fmt(x.start)}–${fmt(x.end)} s ${x.label ?? label(x.target)}${x.valid === false ? '（无效试次）' : ''}`).join('；') : '当前为休息或没有已确认的刺激标注');
  }
  function renderState() {
    const s = state, current = s.session || {};
    if (['running', 'connecting'].includes(s.status) && ['demo', 'device'].includes(s.source)) $('source').value = s.source;
    $('connection').textContent = `${s.status || 'idle'} · ${s.source === 'demo' ? '模拟演示' : s.source || '尚未选择'} · ${s.message || ''}`;
    const t = s.telemetry || {}; $('telemetry').textContent = `数据包 ${t.packets ?? t.stats?.packets ?? 0} · 丢包 ${t.lost_packets ?? t.stats?.lost_packets ?? 0} · CRC ${t.crc_errors ?? t.stats?.crc_errors ?? 0}`;
    $('sessionStatus').textContent = current.finishing ? '刺激已结束，正在检查时钟、接收末尾 EEG 并保存…' : current.active ? `当前 ${current.kind} · ${current.id}` : current.validation ? validationText(current.validation) : '校准目标标签与模型预测分开记录。在线自由注视不记录意图标签。';
    const training = s.training || {}; $('training').textContent = `${training.status || 'idle'} · ${training.message || ''}`;
    $('trainingMetrics').textContent = training.status === 'complete' ? metricsText(training.metrics) : '';
    const visibleMetrics = training.status === 'complete' ? training.metrics : null;
    global.TrainingReport.render($('trainingDetail'), training, s.model, {algorithm: 'SSVEP · FBCCA', labels: FREQUENCIES.map(f => `${f} Hz`), trainingFree: true});
    $('modelInfo').textContent = s.model?.available ? `当前 FBCCA 配置 ${s.model.id} · ${s.model.target_count ?? 4} 类 · ${s.model.source || ''}${s.model.compatible === false ? ' · ' + (s.model.reason || '与当前数据源不兼容') : ''} · ${metricsText(s.model.metrics)}` : '尚未加载个人 FBCCA 配置';
    const p = s.playback || {}; $('seek').max = p.duration_s || 1; if (document.activeElement !== $('seek')) $('seek').value = p.position_s || 0;
    const annotated = (s.signal?.annotations || []).find(x => Number.isFinite(x.onset_s) && Number.isFinite(x.tail_s) && p.position_s >= x.onset_s && p.position_s < x.tail_s);
    const annotationText = annotated ? `${label(annotated.target)} · 试次 ${annotated.trial_id}${annotated.valid === false ? '（无效）' : ''}` : '休息 / 未确认刺激';
    $('replayInfo').textContent = p.loaded ? `${p.playing ? '播放' : '暂停'} ${fmt(p.position_s)} / ${fmt(p.duration_s)} s · ${annotationText}` : '尚未载入回放';
    const visibleTrial = runner?.active && runner.phase === 'task' ? runner.trials[runner.trialIndex]?.trial_id : null;
    renderPrediction(currentPrediction(s.prediction, {phase: runner?.phase, trialId: visibleTrial, timingInvalid: visibleTrial != null && (runner.timingInvalidTrials.has(visibleTrial) || current.timing_invalid_trials?.includes(visibleTrial)), serverTime: s.server_time})); if (!runner) drawSignal(s.signal);
    if (current.active && current.timing_warnings?.length) {
      const last = current.timing_warnings[current.timing_warnings.length - 1];
      $('timingWarning').textContent = `原始 EEG 继续录制 · ${current.timing_invalid_trials?.length || 0} 个试次有时序告警。最近：试次 ${last.trial_id} · ${last.reason}`;
    }
    $('events').textContent = (s.events || []).slice(-20).map(x => typeof x === 'string' ? x : JSON.stringify(x)).join('\n');
    if (runner?.active && (s.status === 'error' || (current.id === session?.id && current.active === false && !current.finishing))) runner.abort(current.id === session?.id && current.reason ? current.reason : s.prediction?.reason || s.message || '服务端已终止录制');
    controls();
  }
  async function poll() {
    if (!token || pollBusy) return; pollBusy = true;
    try { state = await api('state'); renderState(); }
    catch (error) { renderPrediction({status: 'unavailable', reason: '服务状态不可用，旧预测已隐藏。'}); notify(error.message, true); if (runner?.active) runner.abort(`服务状态不可用：${error.message}`); }
    finally { pollBusy = false; }
  }
  $('checkDisplay').onclick = () => action(checkTiming);
  $('retryLoad').onclick = reloadLists;
  $('consent').onchange = controls;
  $('source').onchange = controls;
  $('targetCount').onchange = () => { if (runner || starting || state.session?.active) return; checkedAt = null; targetLayout(); controls(); };
  $('trials').oninput = targetSummary;
  $('discover').onclick = () => action(discoverDevice);
  $('connect').onclick = () => action(connectSource);
  $('disconnect').onclick = () => action(async () => { await api('stop', {}); await poll(); });
  $('calibrate').onclick = () => action(() => startSession('calibration'));
  $('online').onclick = () => action(() => startSession('online'));
  $('stopSession').onclick = $('overlayStop').onclick = () => stopSession();
  $('train').onclick = () => action(async () => { const ids = [...$('sessionList').querySelectorAll('input:checked')].map(n => n.value); if (!ids.length) throw new Error('请先选择校准录制'); await api('train', {ids}); await poll(); });
  $('loadModel').onclick = () => action(async () => { if (!$('models').value) throw new Error('请先选择FBCCA 配置'); await api('model/load', {id: $('models').value}); await poll(); });
  $('loadReplay').onclick = () => action(async () => { if (!$('replaySessions').value) throw new Error('请选择回放录制'); await api('replay/load', {id: $('replaySessions').value}); await poll(); });
  $('play').onclick = () => action(async () => { await api('replay/control', {action: 'play', speed: Number($('speed').value)}); await poll(); });
  $('pause').onclick = () => action(async () => { await api('replay/control', {action: 'pause'}); await poll(); });
  $('seek').onchange = () => action(async () => { await api('replay/control', {action: 'seek', position_s: Number($('seek').value), speed: Number($('speed').value)}); await poll(); });
  $('speed').onchange = () => action(() => api('replay/control', {action: state.playback?.playing ? 'play' : 'pause', speed: Number($('speed').value)}));
  $('waveScale').onchange = () => drawSignal(state.signal);
  function interrupt(reason) { checkedAt = null; startGeneration++; if (runner?.active) runner.abort(reason); else if (starting) { notify(`启动已取消：${reason}`, true); startFeedback(`启动已取消：${reason}`, true); } }
  window.addEventListener('blur', () => interrupt('窗口失焦'));
  document.addEventListener('visibilitychange', () => { if (document.hidden) interrupt('页面隐藏'); });
  window.addEventListener('resize', () => { interrupt('显示区域发生变化'); if (!runner) drawSignal(state.signal); });
  window.addEventListener('keydown', event => { if (event.key === 'Escape' && (runner?.active || starting)) { event.preventDefault(); interrupt('用户按 Esc 停止'); } });
  window.addEventListener('pagehide', () => {
    if (!session || !token || !ownerToken) return;
    const cutoff = runner?.lastVisible ?? lastVisibleCutoff;
    blank(); if (runner) runner.active = false;
    const body = stoppingBody && stoppingBody.reason !== 'completed' ? stoppingBody : {id: session.id, owner_token: ownerToken, reason: 'page_closed', invalid_since_s: cutoff};
    fetch('/api/ssvep/session/stop', {method: 'POST', keepalive: true, headers: {'Content-Type': 'application/json', 'X-EEG-Token': token}, body: JSON.stringify(body)}).catch(() => {});
  });
  targetLayout(); controls(); reloadLists();
  setInterval(poll, 500);
  setInterval(() => { if (!runner && !busy && !bootstrapRetry) reloadLists(); }, 5000);
})(typeof window === 'undefined' ? globalThis : window);
