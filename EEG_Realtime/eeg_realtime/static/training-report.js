(function (global) {
  'use strict';
  const escape = value => String(value ?? '—').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const pct = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
  const count = value => Number.isInteger(value) && value >= 0 ? String(value) : '—';
  const reasons = {incomplete_frames:'刺激帧不完整', missing_tail:'缺少刺激结束记录', invalid_frame_evidence:'刺激帧证据无效', dropped_or_irregular_frame:'掉帧或帧间隔异常', timing_warning:'时序告警', raw_quality:'脑电信号检查未通过', incomplete_samples:'脑电覆盖不完整', sample_gap:'脑电缺样或跨段', display_interrupted:'显示中断', display_gap:'显示停顿超过门槛', clock_uncertainty:'时钟误差超限', missing_post_trial_clock_audit:'缺少试次后时钟复核', invalid_post_trial_clock_audit:'试次后时钟复核失败', missing_final_clock_audit:'缺少结束时钟复核', post_clock_budget_exceeded:'事后时钟误差超限', task_duration:'刺激时长不完整', stimulus_luminance_mismatch:'亮度与目标频率不一致'};
  function table(headers, rows) {
    return `<div class="report-scroll"><table><thead><tr>${headers.map(x => `<th>${escape(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(x => `<td>${escape(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function confusion(matrix, labels, title) {
    if (!Array.isArray(matrix) || !matrix.length || matrix.some(row => !Array.isArray(row) || row.length !== matrix.length)) return '';
    const names = matrix.map((_, i) => labels[i] || `目标 ${i + 1}`);
    return `<h3>${escape(title)}</h3><p class="small">行：真实类别；列：预测类别；对角线：正确识别数。</p>${table(['真实 ↓ / 预测 →', ...names], matrix.map((row, i) => [names[i], ...row.map(count)]))}`;
  }
  function markup(training = {}, model = {}, options = {}) {
    const attempted = ['running', 'complete', 'error'].includes(training.status);
    const metrics = attempted ? (training.status === 'complete' ? training.metrics : null) : model.metrics;
    const diagnostics = attempted ? training.diagnostics : model.diagnostics;
    const reportSource = attempted ? (training.source || (training.model_id && training.model_id === model.id ? model.source : null)) : model.source;
    const name = options.algorithm || '个人模板相关';
    const labels = options.labels || [];
    let html = `<p class="report-context">${escape(attempted ? '本次训练 / 校准' : '当前加载模型')} · ${escape(name)}</p>`;
    if (metrics && Object.keys(metrics).length) {
      if (reportSource === 'demo') html += '<p class="report-warning">模拟数据 · 仅用于验证软件流程，下列结果不代表真实受试者识别效果。</p>';
      if (!reportSource) html += '<p class="small">该历史报告未提供数据来源字段，请核对录制来源。</p>';
      html += '<div class="report-kpis">' + [['测试试次准确率', metrics.trial_accuracy], ['测试窗口准确率', metrics.window_accuracy], ['接受覆盖率', metrics.coverage], ['已接受窗口准确率', metrics.accepted_accuracy]].map(([label, value]) => `<div><span>${label}</span><strong>${pct(value)}</strong></div>`).join('') + '</div>';
      const sets = options.trainingFree ? [['阈值校准', metrics.calibration], ['留出测试', metrics]] : [['训练', metrics.training], ['阈值校准', metrics.calibration], ['留出测试', metrics]];
      html += table(['数据分组', '试次', '窗口', '试次准确率', '窗口准确率', '接受覆盖率'], sets.filter(([,m]) => m).map(([label,m]) => [label, count(m.trial_count), count(m.window_count), pct(m.trial_accuracy), pct(m.window_accuracy), pct(m.coverage)]));
      html += '<p class="small">按完整试次分组，重叠窗口保留在同一组；同一份录制可能跨组。请结合试次指标评估结果。接受覆盖率是通过阈值的窗口比例；不注视状态的拒识效果尚未验证。</p>';
      if (options.trainingFree) html += '<p class="small">FBCCA 的频率参考、滤波器和权重固定，不拟合分类器；录制数据用于阈值校准与留出评估。</p>';
      if (metrics.threshold_fallback) html += '<p class="report-warning">校准证据不足，当前使用预设拒识阈值；接受覆盖率与正确率需继续验证。</p>';
      for (const warning of metrics.warnings || []) html += `<p class="report-warning">${escape(typeof warning === 'string' ? warning : JSON.stringify(warning))}</p>`;
      html += confusion(metrics.trial_confusion_matrix, labels, '留出测试 · 试次混淆矩阵');
      html += `<details><summary>查看窗口混淆矩阵</summary>${confusion(metrics.confusion_matrix, labels, '留出测试 · 窗口计数')}</details>`;
    } else {
      html += `<p>${escape(training.status === 'running' ? '正在处理数据，完成后显示本次结果。' : training.status === 'error' ? `本次未生成测试结果：${training.message || '请检查数据诊断。'}` : '完成训练 / 校准或加载模型后显示报告。')}</p>`;
    }
    const recordings = diagnostics?.per_recording;
    if (Array.isArray(recordings) && recordings.length) {
      html += '<h3>录制数据筛选</h3>' + table(['录制', '计划试次', '有效试次', '排除原因（可能重叠）'], recordings.map(row => [row.recording_name || row.id, count(row.planned_trials), count(row.valid_trials), Object.entries(row.rejection_reasons || {}).map(([key,value]) => `${reasons[key] || key}: ${value}`).join('；') || '无']));
    }
    const perTarget = diagnostics?.decoder?.per_target_valid_trials;
    if (perTarget) html += '<h3>各类可用试次</h3>' + table(['类别', '完整有效试次'], Object.entries(perTarget).map(([key, value]) => [labels[Number(key)] || `目标 ${Number(key) + 1}`, count(value)]));
    if (diagnostics?.subject?.warning) html += `<p class="report-warning">${escape(diagnostics.subject.warning)}</p>`;
    if ((metrics && Object.keys(metrics).length) || diagnostics) html += `<details class="raw-report"><summary>完整报告 · 原始 JSON</summary><pre>${escape(JSON.stringify({metrics, diagnostics}, null, 2))}</pre></details>`;
    return html;
  }
  function render(node, training, model, options) {
    const signature = JSON.stringify([training, model, options]);
    if (node.dataset.reportSignature === signature) return;
    node.dataset.reportSignature = signature;
    node.innerHTML = markup(training, model, options);
  }
  const api = {markup, render, escape, pct};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.TrainingReport = api;
})(typeof window === 'undefined' ? globalThis : window);
