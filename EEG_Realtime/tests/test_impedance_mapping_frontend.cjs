'use strict';

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const assets = path.resolve(__dirname, '../eeg_realtime/static');
const source = fs.readFileSync(path.join(assets, 'mi.js'), 'utf8');
const html = fs.readFileSync(path.join(assets, 'mi.html'), 'utf8');
const expectedElectrodes = ['FP2', 'FP1', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2'];
const oldNames = ['O2', 'O1', 'P8', 'P7', 'C4', 'C3', 'FP1', 'FP2'];
const values = [2.1, 7.2, 12.3, 33.4, 4.5, 9.6, 16.7, 28.8];

function declaration(name) {
  const match = source.match(new RegExp(`\\bconst ${name}\\s*=\\s*\\[[^\\]]*\\];`));
  assert(match, `${name} array declaration exists`);
  return match[0];
}

function openingTag(id) {
  const match = html.match(new RegExp(`<[^>]*\\bid="${id}"[^>]*>`));
  assert(match, `${id} exists in HTML`);
  return match[0];
}

function hasClass(node, name) {
  return node.className.split(/\s+/).includes(name);
}

// A small DOM implementation keeps textContent replacement and child order real
// enough to catch accidental writes to labels, rows or reference electrodes.
const nodes = new Map();
class Element {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.style = {};
    this.className = '';
    this.title = '';
    this.hidden = false;
    this.disabled = false;
    this.value = '';
    this._text = '';
    this._id = '';
  }
  set id(value) {
    if (this._id) nodes.delete(this._id);
    assert(!nodes.has(value), `duplicate DOM ID ${value}`);
    this._id = value;
    nodes.set(value, this);
  }
  get id() { return this._id; }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  get textContent() {
    return this._text + this.children.map(child => child.textContent).join('');
  }
  get lastElementChild() { return this.children.at(-1) || null; }
  get firstElementChild() { return this.children[0] || null; }
  append(...children) {
    for (const child of children) {
      assert(child instanceof Element, 'DOM stub receives element children');
      this.children.push(child);
    }
  }
  querySelectorAll(tag) {
    const descendants = this.children.flatMap(child => [child, ...child.querySelectorAll('*')]);
    return tag === '*' ? descendants : descendants.filter(child => child.tagName === tag.toUpperCase());
  }
  setAttribute(name, value) { this[name === 'class' ? 'className' : name] = String(value); }
}

function element(tag, text, css) {
  const node = new Element(tag);
  if (text !== undefined) node.textContent = text;
  if (css) node.className = css;
  return node;
}

// Only static HTML IDs are pre-created: missing generated IDs must fail rather
// than being silently fabricated by the test harness.
for (const match of html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi)) {
  const node = element(match[1]);
  node.id = match[2];
}
const $ = id => {
  assert(nodes.has(id), `${id} was initialized`);
  return nodes.get(id);
};
const snapshot = node => JSON.stringify({
  tag: node.tagName, id: node.id, text: node.textContent,
  className: node.className, title: node.title, style: node.style,
  children: node.children.map(child => JSON.parse(snapshot(child))),
});

let now = 1000;
const app = {
  impedance: null, impedanceOnline: true, impedanceAt: 1000,
  impedanceRequestedAt: 950, impedanceChanging: false,
  busy: false, token: 'test', online: true, state: {status: 'idle'},
};
const sandbox = {
  app, $, element, console, performance: {now: () => now},
  finite: value => typeof value === 'number' && Number.isFinite(value),
  fmt: (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : '—',
  isImpedanceActive: () => Boolean(app.impedance?.active)
    || ['running', 'connecting', 'stopping'].includes(app.impedance?.status),
  isRecording: () => false, isTraining: () => false,
};
vm.createContext(sandbox);
const start = source.indexOf('  const IMPEDANCE_POSITIONS =');
const end = source.indexOf('  function pollImpedance()', start);
assert(start >= 0 && end > start, 'impedance initialization/render source found');
vm.runInContext(`${declaration('CHANNELS')}\n${declaration('IMPEDANCE_CHANNELS')}\n${source.slice(start, end)}`, sandbox);

assert.deepEqual(Array.from(vm.runInContext('CHANNELS', sandbox)), oldNames,
  'global EEG waveform channel order stays unchanged');
assert.deepEqual(Array.from(vm.runInContext('IMPEDANCE_CHANNELS', sandbox)), expectedElectrodes,
  'impedance display has its own physical channel mapping');
sandbox.initializeImpedance();

const cards = expectedElectrodes.map((_, index) => $(`impedanceElectrode${index}`));
const rows = expectedElectrodes.map((_, index) => $(`impedanceRow${index}`));
assert.equal($('impedanceElectrodes').children.length, 10, 'eight measured and two reference electrodes');
assert.equal($('impedanceRows').children.length, 10, 'eight measured and two reference rows');
expectedElectrodes.forEach((name, index) => {
  const cells = rows[index].children.map(cell => cell.textContent);
  assert.deepEqual(cells, [String(9 - index), `CH${index + 1}`, name, '—'],
    `CH${index + 1} row contains cap number ${9 - index} and ${name}`);
  assert.equal($('impedanceRows').children[index], rows[index], 'measurement row order follows CH1–CH8');
  assert(cards[index].children.some(child => child.tagName === 'B' && child.textContent === name),
    `CH${index + 1} card label is ${name}`);
  assert.equal(cards[index].lastElementChild.textContent, '—', `${name} starts without a reading`);
});

const referenceNodes = [];
for (const [offset, name, cap] of [[8, 'GND', '1'], [9, 'REF', '10']]) {
  const card = $(`impedance${name}`), row = $(`impedanceRow${name}`);
  assert(hasClass(card, 'impedance-reference'), `${name} is styled as a reference electrode`);
  assert.equal($('impedanceRows').children[offset], row, `${name} row follows measured channels`);
  assert.equal(row.children.length, 4, `${name} row has four columns`);
  assert.equal(row.children[0].textContent, cap, `${name} cap number`);
  assert.match(row.children[1].textContent, /^[-–—]$/, `${name} has no EEG channel`);
  assert.equal(row.children[2].textContent, name, `${name} table label`);
  assert.match(row.lastElementChild.textContent, /^[-–—]$/, `${name} has no measured impedance`);
  referenceNodes.push(card, row);
}
const referenceSnapshots = referenceNodes.map(snapshot);
function assertReferencesUnchanged(context) {
  assert.deepEqual(referenceNodes.map(snapshot), referenceSnapshots, `${context}: GND/REF stay static`);
  for (const node of referenceNodes) {
    assert.doesNotMatch(node.className, /grade-|\babnormal\b/, `${context}: references have no measurement color`);
    for (const value of [...values, 44.7]) {
      assert(!node.textContent.includes(value.toFixed(1)), `${context}: references contain no measured value`);
    }
  }
}

// The backend emits a fixed physical-channel order with historical name fields.
// Unique values deliberately disagree with the new display-name order.
app.impedance = {
  active: true, status: 'running', fresh: true, device_ip: '192.168.1.8',
  telemetry: {age_s: .1, estimate_age_s: .1, packets: 50, crc_errors: 0, lost_packets: 0},
  stale_seconds: 1.2, window_seconds: 2, window_samples: 500, required_samples: 500,
  channels: oldNames.map((name, index) => ({name, kohm: values[index], abnormal: false, reason: 'ok'})),
};
sandbox.renderImpedance();
expectedElectrodes.forEach((name, index) => {
  assert.equal(cards[index].lastElementChild.textContent, values[index].toFixed(1),
    `${name} reads physical CH${index + 1}, ignoring backend name ${oldNames[index]}`);
  assert.equal(rows[index].lastElementChild.textContent, values[index].toFixed(1), `${name} table value`);
  assert(cards[index].title.includes(name), `${name} tooltip uses the display mapping`);
});
assertReferencesUnchanged('fresh data');

const beforeCards = cards.map(snapshot), beforeRows = rows.map(snapshot);
app.impedance.channels[6].kohm = 44.7;
sandbox.renderImpedance();
for (let index = 0; index < 8; index++) {
  if (index === 6) {
    assert.equal(cards[index].lastElementChild.textContent, '44.7', 'CH7 changes the O1 card');
    assert.equal(rows[index].lastElementChild.textContent, '44.7', 'CH7 changes the seventh table row');
    assert(hasClass(cards[index], 'grade-4'), 'O1 color follows the new CH7 value');
  } else {
    assert.equal(snapshot(cards[index]), beforeCards[index], `CH7 change leaves ${expectedElectrodes[index]} card untouched`);
    assert.equal(snapshot(rows[index]), beforeRows[index], `CH7 change leaves table row ${index + 1} untouched`);
  }
}
assertReferencesUnchanged('CH7 update');

now = 2300;
sandbox.renderImpedance();
for (let index = 0; index < 8; index++) {
  assert.equal(cards[index].lastElementChild.textContent, '—', `stale CH${index + 1} card is cleared`);
  assert.equal(rows[index].lastElementChild.textContent, '—', `stale CH${index + 1} row is cleared`);
  assert.doesNotMatch(cards[index].className, /grade-|\babnormal\b/, 'stale card has no measurement color');
  assert.doesNotMatch(rows[index].className, /grade-|\babnormal\b/, 'stale row has no measurement color');
}
assertReferencesUnchanged('stale data');

assert.match(source, /activeTab:\s*['"]impedance['"]/, 'JS starts on the impedance tab');
assert.match(openingTag('impedanceTab'), /class="[^"]*\bactive\b/, 'impedance tab is active in HTML');
assert.match(openingTag('impedanceTab'), /aria-selected="true"/, 'impedance tab is selected in HTML');
assert.doesNotMatch(openingTag('impedancePane'), /\bhidden\b/, 'impedance pane is initially visible');
assert.doesNotMatch(openingTag('acquisitionTab'), /class="[^"]*\bactive\b/, 'acquisition tab is initially inactive');
assert.match(openingTag('acquisitionTab'), /aria-selected="false"/, 'acquisition tab is initially unselected');
assert.match(openingTag('acquisitionPane'), /\bhidden\b/, 'acquisition pane is initially hidden');
const initializeStart = source.indexOf('  async function initialize()');
const initializeEnd = source.indexOf('  void initialize();', initializeStart);
assert(initializeStart >= 0 && initializeEnd > initializeStart, 'startup initializer found');
assert.match(source.slice(initializeStart, initializeEnd), /selectTab\(app\.activeTab\)/,
  'startup applies the selected tab to shared page elements');

console.log('PASS: physical CH1–CH8 mapping, ten cap labels/rows, CH7-only O1 update, static GND/REF, stale clearing, unchanged EEG channel order, and default impedance page');
