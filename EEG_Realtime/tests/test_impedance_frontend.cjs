const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
const assets = path.resolve(__dirname, '../eeg_realtime/static');
const source = fs.readFileSync(path.join(assets, 'mi.js'), 'utf8');
const html = fs.readFileSync(path.join(assets, 'mi.html'), 'utf8');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
assert.equal(new Set(ids).size, ids.length, 'HTML IDs are unique');
for (const match of source.matchAll(/\$\("(impedance[^"]+)"\)/g)) assert(ids.includes(match[1]), `${match[1]} exists`);
function node() {
  const item = {value: '', textContent: '', className: '', hidden: false, disabled: false,
    append() {}, querySelectorAll() { return []; }};
  item.lastElementChild = {textContent: '', append() {}};
  return item;
}
const nodes = new Map();
const $ = id => { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); };
let now = 1000, reads = 0, deferredGet, releaseCommand;
const app = {impedance: null, impedanceOnline: true, impedanceAt: 1000, impedanceRequestedAt: 950,
  impedancePolling: null, impedanceEpoch: 0, impedanceChanging: false, busy: false, token: 'test',
  online: true, state: {status: 'idle'}};
const state = () => ({active: true, status: 'running', fresh: true, device_ip: '192.168.1.8',
  telemetry: {age_s: .1, estimate_age_s: .1, packets: 50, crc_errors: 1, lost_packets: 2},
  stale_seconds: 1.2, window_seconds: 2, window_samples: 500, required_samples: 500,
  channels: ['O2', 'O1', 'P8', 'P7', 'C4', 'C3', 'FP1', 'FP2'].map(name => ({name, kohm: 9, abnormal: false, reason: 'ok'}))});
let requestImpl = async () => state();
const sandbox = {app, $, console, Promise, performance: {now: () => now},
  CHANNELS: ['O2', 'O1', 'P8', 'P7', 'C4', 'C3', 'FP1', 'FP2'],
  finite: x => typeof x === 'number' && Number.isFinite(x),
  fmt: (x, n = 1) => Number.isFinite(x) ? x.toFixed(n) : '—',
  isImpedanceActive: () => Boolean(app.impedance?.active) || ['running', 'connecting', 'stopping'].includes(app.impedance?.status),
  isRecording: () => false, isTraining: () => false, element: () => node(),
  request: (...args) => requestImpl(...args), renderControls: () => sandbox.renderImpedance(),
  action: async callback => { app.busy = true; try { await callback(); } finally { app.busy = false; } },
  selectTab: () => {}};
vm.createContext(sandbox);
const start = source.indexOf('  function renderImpedance()');
const end = source.indexOf('  async function discoverImpedance()', start);
vm.runInContext(source.slice(start, end), sandbox);
async function main() {
  app.impedance = state();
  sandbox.renderImpedance();
  assert.equal($('impedanceElectrode0').lastElementChild.textContent, '9.0', 'fresh measurements displayed');
  assert.equal($('impedanceRow0').className, 'grade-2', 'vendor color grade');
  now = 2200; sandbox.renderImpedance();
  assert.equal($('impedanceElectrode0').lastElementChild.textContent, '—', 'client-aged samples cleared');
  now = 1000; app.impedance.telemetry.estimate_age_s = 1.18; sandbox.renderImpedance();
  assert.equal($('impedanceElectrode0').lastElementChild.textContent, '—', 'HTTP elapsed time included in estimate age');
  app.impedance = state(); app.impedanceOnline = false; sandbox.renderImpedance();
  assert.equal($('impedanceElectrode0').lastElementChild.textContent, '—', 'offline values cleared');
  app.impedanceOnline = true; app.impedance.status = 'stopped'; app.impedance.active = true;
  sandbox.renderImpedance(); assert.equal($('impedanceStop').disabled, false, 'live worker keeps stop available');
  app.impedance = state();
  $('impedanceIp').value = '192.168.1.8';
  requestImpl = (path, body) => {
    if (body !== undefined) return new Promise(resolve => {releaseCommand = resolve;});
    reads++; return new Promise(resolve => {deferredGet = resolve;});
  };
  const oldRead = sandbox.pollImpedance();
  const command = sandbox.impedanceCommand('stop');
  assert.equal(app.impedanceChanging, true);
  assert.equal($('impedanceElectrode0').lastElementChild.textContent, '—', 'mode command clears values immediately');
  await sandbox.pollImpedance();
  assert.equal(reads, 1, 'periodic GET suppressed during command');
  deferredGet(state()); await oldRead;
  assert.equal(app.impedanceOnline, false, 'old in-flight GET cannot restore pre-command state');
  requestImpl = async () => ({...state(), status: 'stopped', active: false, fresh: false, channels: []});
  releaseCommand({ok: true}); await command;
  assert.equal(app.impedanceChanging, false);
  assert.equal(app.impedance.status, 'stopped', 'command completion reads authoritative state');
  assert.equal($('impedanceElectrode0').lastElementChild.textContent, '—');
  assert.equal($('impedanceEeg').disabled, true, 'command remains busy until action resolves');
  sandbox.renderImpedance();
  assert.equal($('impedanceEeg').disabled, false, 'stopped check can return to EEG');
  console.log('PASS: HTML IDs, live values, color grades, aged/offline values, pending worker Stop, command/GET race, and EEG return controls');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
