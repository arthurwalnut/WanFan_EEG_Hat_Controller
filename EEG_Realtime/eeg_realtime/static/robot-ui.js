"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const clientId = crypto.randomUUID();
  const modes = { off: "未启用", preview: "仅预览 · 不发送动作", live: "机械臂发送已启用" };
  const commands = { CLENCH: "左手 → Elite", BLINK: "右手 → Piper", NEUTRAL: "静息 · 不触发动作", WAITING: "等待模型预测" };
  const labels = { rest: "静息", left_mi: "左手运动想象", right_mi: "右手运动想象" };
  const acknowledgements = {
    accepted: "接收端已接受，动作执行中", completed: "接收端报告动作脚本完成", failed: "接收端报告失败，请检查机器人端",
    busy: "接收端忙碌或冷却中，未排队", duplicate: "接收端忽略重复消息", ignored: "接收端未执行",
    test: "TEST 已收到，无动作", dry_run: "接收端为模拟运行，无实际动作",
  };
  let token = "", state = null, online = false, pending = false, heartbeatPending = false;
  let ownLive = false, generation = 0, attemptedLive = false, initialized = false;
  let localNoticeUntil = 0;
  const number = (v) => typeof v === "number" && Number.isFinite(v);

  async function post(action, body = {}, keepalive = false) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch(`/api/mi/robot/${action}`, {
        method: "POST", headers: { "Content-Type": "application/json", "X-EEG-Token": token },
        body: JSON.stringify(body), cache: "no-store", signal: controller.signal, keepalive,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `请求失败（${response.status}）`);
      return data;
    } finally { clearTimeout(timeout); }
  }

  function notice(message, error = false, fromState = false) {
    if (fromState && Date.now() < localNoticeUntil) return;
    if (!fromState) localNoticeUntil = Date.now() + 5000;
    $("robotNotice").textContent = message || "";
    $("robotNotice").className = `robot-notice${error ? " error" : ""}`;
  }

  function settings(mode) {
    return { host: $("robotHost").value.trim(), port: Number($("robotPort").value), mode, client_id: clientId };
  }

  function renderRobot(robot) {
    if (!robot) return;
    const mine = robot.mode === "live" && robot.owner_client_id === clientId;
    // A server snapshot never grants this newly loaded page a previous page's lease.
    ownLive = mine && attemptedLive && online;
    $("robotModeBadge").textContent = online ? modes[robot.mode] || "未启用" : "状态不可达 · 停发请求已发出";
    $("robotModeBadge").className = `tag${robot.armed && online ? " caution" : ""}`;
    if (!initialized) {
      $("robotHost").value = robot.host || "";
      $("robotPort").value = robot.port || 5005;
      initialized = true;
    }
    const armed = robot.mode === "live";
    if (armed) {
      $("robotHost").value = robot.host || "";
      $("robotPort").value = robot.port || 5005;
    }
    $("robotHost").disabled = pending || armed;
    $("robotPort").disabled = pending || armed;
    $("robotPreview").disabled = pending || !online || armed;
    $("robotTest").disabled = pending || !online || armed;
    $("robotEnable").disabled = pending || !online || armed || !robot.eligible;
    $("robotStop").disabled = !token;
    $("robotAvailability").textContent = robot.eligible
      ? "设备、模型和信号已就绪。核对接收端地址后，点击启用开始发送。"
      : robot.ineligible_reason || "发送需连接真实设备、加载匹配模型并持续收到数据；演示和回放仅可预览。";
    $("robotDecision").textContent = commands[robot.command] || "等待模型预测";
    const prediction = robot.last_prediction;
    const confidence = number(prediction?.confidence) ? ` · 置信度 ${(prediction.confidence * 100).toFixed(1)}%` : "";
    $("robotPrediction").textContent = labels[prediction?.label]
      ? `${labels[prediction.label]}${confidence} · 按当前类别直接映射；置信度仅展示。`
      : "等待模型给出类别；置信度仅作展示。";
    const lastStop = robot.last_stop;
    const stopTime = lastStop?.at ? new Date(lastStop.at) : null;
    const stopStamp = stopTime && Number.isFinite(stopTime.getTime()) ? `（${stopTime.toLocaleString()}）` : "";
    $("robotLastStop").textContent = lastStop?.reason ? `最近停发${stopStamp}：${lastStop.reason}` : "最近停发：暂无记录";
    $("robotLastStop").className = `robot-last-stop${lastStop?.reason ? " has-record" : ""}`;
    for (const [id, kind] of [["robotLeftCount", "clench"], ["robotRightCount", "blink"], ["robotRestCount", "neutral"], ["robotUnknownCount", "unknown"]]) {
      $(id).textContent = String(robot.counts?.[kind] || 0);
    }
    const span = number(robot.window_start_s) && number(robot.window_end_s)
      ? ` · ${robot.window_start_s.toFixed(2)}–${robot.window_end_s.toFixed(2)} s` : "";
    $("robotWindow").textContent = `最近 ${robot.window_count || 0} / ${robot.window_size || 15} 窗仅作统计${span}；每次有效类别直接映射。`;
    const last = robot.last_send;
    $("robotSent").textContent = last
      ? `${last.payload?.cmd || "UDP"} → ${last.target || `${robot.host}:${robot.port}`} · 已发出 ${robot.sent_count || 0} 个动作包；发送不等于执行确认`
      : "尚未发送动作指令";
    const ack = robot.last_action_ack || robot.last_ack;
    $("robotAck").textContent = ack
      ? `${ack.cmd || ""} · ${acknowledgements[ack.status] || ack.status || "收到回执"}${ack.reason ? ` · ${ack.reason}` : ""}${robot.last_action_ack && robot.last_ack?.status === "busy" ? "；后续指令处于冷却/忙碌，未排队" : ""}`
      : robot.sent_count || robot.test_count ? "尚无回执；旧版 receiver 不返回确认，请查看接收端终端" : "尚无回执";
    if (robot.error || robot.log_error) notice(robot.error || robot.log_error, true, true);
    else if (robot.reason && !pending) notice(robot.reason, false, true);
  }

  function update(robot) {
    if (!state) state = {};
    state.robot = robot;
    renderRobot(robot);
  }

  async function configure(mode) {
    const ticket = ++generation;
    pending = true;
    attemptedLive = mode === "live";
    renderRobot(state?.robot);
    try {
      const result = await post("configure", settings(mode));
      if (ticket !== generation || (mode === "live" && (document.hidden || !document.hasFocus()))) {
        attemptedLive = ownLive = false;
        await post("disarm", { client_id: clientId, reason: "启用响应到达前页面已失去焦点或操作已取消。" }, true);
        return;
      }
      online = true;
      update(result.robot || result);
      notice(mode === "live" ? "已启用，按有效预测直接输出；页面失焦、失联、断流或开始录制时停发。" : "仅预览类别与机械臂指令的映射，不发送动作包。");
    } catch (error) {
      attemptedLive = ownLive = false;
      // A timed-out configure can already have reached the server.
      if (mode === "live") void post("disarm", { client_id: clientId, reason: `启用请求失败：${error.message}` }, true).catch(() => {});
      notice(`设置失败：${error.message}`, true);
    } finally { pending = false; renderRobot(state?.robot); }
  }

  function stop(reason, force = false) {
    if (!force && !ownLive && !attemptedLive) return;
    ++generation;
    ownLive = attemptedLive = false;
    notice(reason);
    if (token) void post("disarm", { client_id: clientId, reason }, true).then((result) => update(result.robot || result)).catch(() => {
      notice("停发请求未获响应；服务端会在页面心跳超过 2 秒后关闭后续发送。", true);
    });
  }

  $("robotPreview").addEventListener("click", () => { void configure("preview"); });
  $("robotEnable").addEventListener("click", () => { void configure("live"); });
  $("robotStop").addEventListener("click", () => stop("已请求停止后续指令。已经开始的动作请在机器人端处理。", true));
  $("robotTest").addEventListener("click", async () => {
    pending = true; renderRobot(state?.robot);
    try {
      update(await post("configure", settings("off")));
      const result = await post("test");
      update(result.robot || result);
      notice("TEST 已发出，不触发动作。请查看接收端终端或等待回执。");
    } catch (error) { notice(`通信测试失败：${error.message}`, true); }
    finally { pending = false; renderRobot(state?.robot); }
  });
  setInterval(async () => {
    if (!ownLive || !online || heartbeatPending || document.hidden || !document.hasFocus()) return;
    heartbeatPending = true;
    try { const result = await post("heartbeat", { client_id: clientId }); update(result.robot || result); }
    catch (error) { stop(`机械臂心跳失败：${error.message}`); }
    finally { heartbeatPending = false; }
  }, 500);
  window.addEventListener("blur", () => stop("页面失去焦点，已请求停止机械臂发送。"));
  window.addEventListener("pagehide", () => stop("页面离开，已请求停止机械臂发送。"));
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop("页面不可见，已请求停止机械臂发送。"); });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") stop("Esc：已请求停止后续指令。", true); });
  window.MIRobotUI = {
    initialize(options) { token = options.token || token; },
    render(value) { state = value; online = true; renderRobot(value.robot); },
    offline() { online = false; stop("页面无法获取采集状态，已请求停止机械臂发送。"); renderRobot(state?.robot); },
  };
})();
