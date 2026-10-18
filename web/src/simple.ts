// simple view: one model out of two, one prompt, and the guess for the next token animated through the layers
// loaded by simple-entry.ts, which also loads the styles and sends links for the advanced view there
import type { LogitLensResult } from '@wllama/wllama';
import { showPiece, layerLabel } from './heatmap';
import { webGpuProblem } from './gpu';
import { SIMPLE_MODELS, fmtBytes } from './hf';
import { newWllama, unloaded, recovered, cachedModels, esc, pct, setBar, showPreviewBadge } from './common';
import { modelParam, SIMPLE_PROMPT_MAX } from './views';

// each one has a clear answer that small models get right
const EXAMPLES = [
  { label: 'Eiffel Tower', prompt: 'The Eiffel Tower is located in the city of' },
  { label: 'Counting', prompt: 'One, two, three, four,' },
  { label: 'Opposites', prompt: 'The opposite of hot is' },
  { label: 'Weekdays', prompt: 'Monday, Tuesday, Wednesday,' },
  { label: 'Capital', prompt: 'The capital of Germany is' },
  { label: 'Who got the drink', prompt: 'Mary and John went to the store. John gave a drink to' },
  { label: 'Shakespeare', prompt: 'To be or not to be, that is the' },
];

// lens candidates per layer: 5 are shown, the rest let a word fade in smoothly as it climbs into the top 5
const K = 10;
const SHOWN = 5;

const urlParams = new URLSearchParams(location.search);
const modelUrl = modelParam(urlParams);

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const modelsEl = $<HTMLDivElement>('models');
const modelStatus = $<HTMLParagraphElement>('model-status');
const form = $<HTMLFormElement>('prompt-form');
const promptEl = $<HTMLInputElement>('prompt');
const btnRun = $<HTMLButtonElement>('btn-run');
const examplesEl = $<HTMLDivElement>('examples');
const advancedLink = $<HTMLAnchorElement>('advanced-link');
const stage = $<HTMLElement>('stage');
const svg = document.getElementById('lens-svg') as unknown as SVGSVGElement;
const intro = $<HTMLDivElement>('intro');
const progress = $<HTMLDivElement>('progress');
const progressPhase = $<HTMLParagraphElement>('progress-phase');
const progressBar = $<HTMLDivElement>('progress-bar');
const progressDetail = $<HTMLParagraphElement>('progress-detail');
const btnCancel = $<HTMLButtonElement>('btn-cancel');
const caption = $<HTMLParagraphElement>('caption');
const btnPlay = $<HTMLButtonElement>('btn-play');
const modelRow = $<HTMLDivElement>('model-row');
const btnModels = $<HTMLButtonElement>('btn-models');
const runStatus = $<HTMLParagraphElement>('run-status');
const scrub = $<HTMLInputElement>('scrub');

let wllama = newWllama();
let modelIdx = Math.max(0, SIMPLE_MODELS.findIndex((m) => m.url === modelUrl));
let loadedIdx = -1;
let busy = false;
let cached = new Set<string>();
let loadCtl: AbortController | null = null;

const clamp = (x: number, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const ease = (x: number) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);

/////////////////////////////////////////////
// model choice and loading

let gpuProblem: string | null = null;
const gpuChecked = webGpuProblem(wllama).then((p) => (gpuProblem = p));

async function refreshCache() {
  cached = new Set((await cachedModels(wllama)).map((m) => m.url));
}

// built once and updated in place, and never disabled (that blurs them), so the radio with keyboard focus keeps it
const modelInputs = SIMPLE_MODELS.map((m, i) => {
  const label = document.createElement('label');
  label.title = `${m.repo}, ${m.file}`;
  const input = Object.assign(document.createElement('input'), { type: 'radio', name: 'model', value: String(i) });
  // while busy, a change is undone instead of disabling the radios
  input.onchange = () => (busy ? renderModels() : selectModel(i));
  const size = Object.assign(document.createElement('span'), { className: 'size' });
  label.append(input, Object.assign(document.createElement('span'), { className: 'name', textContent: m.name }), size);
  modelsEl.append(label);
  return { input, size };
});

// arrow keys would move focus to a radio whose change is then undone
modelsEl.addEventListener('keydown', (ev) => {
  if (busy && ev.key.startsWith('Arrow')) ev.preventDefault();
});

function renderModels() {
  modelsEl.setAttribute('aria-busy', String(busy));
  modelInputs.forEach(({ input, size }, i) => {
    input.checked = i === modelIdx;
    size.textContent = cached.has(SIMPLE_MODELS[i].url) ? 'downloaded' : fmtBytes(SIMPLE_MODELS[i].size);
  });
}

function syncStatus() {
  const m = SIMPLE_MODELS[modelIdx];
  if (loadedIdx === modelIdx) modelStatus.textContent = `Ready, on the ${gpuProblem ? 'main processor' : 'graphics card'}.`;
  else if (cached.has(m.url)) modelStatus.textContent = `${m.note}. Loads from your browser on the first run.`;
  else modelStatus.textContent = `${m.note}. Press Run to download it (${fmtBytes(m.size)}, once).`;
}

function selectModel(i: number) {
  modelIdx = i;
  syncStatus();
  updateUrl();
  // after a first run, a model already in the browser shows its own answer right away; a download waits for Run
  if (res && i !== loadedIdx && cached.has(SIMPLE_MODELS[i].url)) run();
}

btnCancel.onclick = () => loadCtl?.abort();

async function load(i: number) {
  await gpuChecked;
  const m = SIMPLE_MODELS[i];
  const fromCache = cached.has(m.url);
  const ctl = (loadCtl = new AbortController());
  const setup = `Setting up the model${gpuProblem ? '' : ' and the graphics card'}; this takes a few seconds.`;
  progress.hidden = false;
  intro.hidden = true;
  btnCancel.hidden = fromCache;
  progressPhase.textContent = fromCache ? `Loading ${m.name} from your browser` : `Downloading ${m.name}`;
  progressDetail.textContent = fromCache ? setup : 'Starting...';
  setBar(progressBar, fromCache ? null : 0);
  const params = {
    n_ctx: 256,
    n_gpu_layers: gpuProblem ? 0 : 999,
    signal: ctl.signal,
    progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
      if (fromCache || !total) return;
      if (loaded >= total) {
        btnCancel.hidden = true;
        progressPhase.textContent = `Loading ${m.name}`;
        progressDetail.textContent = setup;
        return setBar(progressBar, null);
      }
      setBar(progressBar, loaded / total);
      progressDetail.textContent = `${fmtBytes(loaded)} of ${fmtBytes(total)}. Kept in your browser for the next visit.`;
    },
  };
  try {
    // download completely before unloading the current model, so a cancelled download keeps it
    const input = await wllama.modelManager.getModelOrDownload({ url: m.url }, params);
    btnCancel.hidden = true;
    loadedIdx = -1;
    wllama = await unloaded(wllama);
    await wllama.loadModel(input, params);
    loadedIdx = i;
  } catch (e) {
    wllama = await recovered(wllama);
    throw ctl.signal.aborted ? new Error('Download cancelled.') : e;
  } finally {
    loadCtl = null;
    progress.hidden = true;
    await refreshCache();
  }
}

// not disabled while busy (that drops keyboard focus): marked busy, and clicks are ignored
function setBusy(b: boolean) {
  busy = b;
  btnRun.setAttribute('aria-disabled', String(b));
  examplesEl.setAttribute('aria-busy', String(b));
  renderModels();
  syncStatus();
}

/** prompt and model for links */
function linkQuery() {
  const q = new URLSearchParams();
  // the loaded model, which the lens on screen comes from; the choice only once nothing is loaded
  q.set('model', SIMPLE_MODELS[loadedIdx >= 0 ? loadedIdx : modelIdx].url);
  if (promptEl.value) q.set('prompt', promptEl.value);
  return q;
}

/** the advanced view gets the same prompt and model; cheap, so it follows every keystroke */
const syncAdvancedLink = () => (advancedLink.href = `advanced.html?${linkQuery()}`);

/** prompt and model in the page URL, so the address bar is a shareable link; not per keystroke, browsers rate-limit replaceState */
function updateUrl() {
  history.replaceState(null, '', `${location.pathname}?${linkQuery()}`);
  syncAdvancedLink();
}

/** the caption changes every layer, so it is not a live region; errors also go to one that is */
function announce(msg: string) {
  caption.textContent = lastCaption = runStatus.textContent = msg;
}

/** another model's scene stays hidden, and cannot be played, until the new one is shown or cleared */
function hideScene(hide: boolean) {
  svg.style.visibility = hide ? 'hidden' : '';
  scrub.disabled = btnPlay.disabled = hide || !lens;
  // bars still easing would keep redrawing the old caption
  if (hide && frameId) {
    cancelAnimationFrame(frameId);
    frameId = 0;
  }
}

async function run() {
  if (busy) return;
  if (!promptEl.value.trim()) {
    announce('Type a prompt first, or pick one of the examples.');
    return promptEl.focus();
  }
  setBusy(true);
  pause();
  runStatus.textContent = '';
  try {
    if (loadedIdx !== modelIdx) {
      hideScene(true);
      caption.textContent = lastCaption = '';
      await load(modelIdx);
    }
    caption.textContent = 'Running the model...';
    const r = await wllama.logitLens({ prompt: promptEl.value, k: K, mtp: false, mtpDraft: 0 });
    show(r);
    updateUrl();
  } catch (e) {
    // a cancelled download keeps the previous model: go back to it; the scene stays only if it is that model's
    if (loadedIdx >= 0) modelIdx = loadedIdx;
    if (loadedIdx < 0 || resIdx !== loadedIdx) clearScene();
    announce(`Something went wrong: ${(e as Error).message}`);
  } finally {
    hideScene(false);
    setBusy(false);
  }
}

form.onsubmit = (ev) => {
  ev.preventDefault();
  run();
};

/////////////////////////////////////////////
// the lens at the last position

interface Lens {
  r: LogitLensResult;
  nL: number; // rows: embedding, then one per layer
  probs: Map<number, number>[]; // per row: top-K token -> probability
  final: number; // the model's prediction
  finalP: Float32Array; // its probability per row, also outside the top K
  finalRank: Int32Array;
  kl: Float32Array; // KL(final || row), nats
  top: number[]; // top-1 per row
  settle: number; // first row from which every row's top-1 is the final prediction
  finalEog: boolean; // asked of the model the lens came from, not whichever is loaded later
  input: number; // the last prompt token
}

let res: LogitLensResult | null = null;
let resIdx = -1; // the model res came from
let lens: Lens | null = null;

function makeLens(r: LogitLensResult): Lens {
  const nL = r.nLayers;
  const t = r.nTokens - 1;
  const cell = (l: number) => l * r.nTokens + t;
  const probs: Map<number, number>[] = [];
  const top: number[] = [];
  const finalP = new Float32Array(nL);
  const finalRank = new Int32Array(nL);
  const kl = new Float32Array(nL);
  for (let l = 0; l < nL; l++) {
    const c = cell(l);
    const m = new Map<number, number>();
    for (let j = 0; j < r.k; j++) m.set(r.topIds[c * r.k + j], r.topProbs[c * r.k + j]);
    probs.push(m);
    top.push(r.topIds[c * r.k]);
    // the default target is the final row's top-1
    finalP[l] = r.targetProbs[c];
    finalRank[l] = r.targetRanks[c];
    kl[l] = r.kl[c];
  }
  const final = top[nL - 1];
  let settle = nL - 1;
  while (settle > 0 && top[settle - 1] === final) settle--;
  return { r, nL, probs, final, finalP, finalRank, kl, top, settle, finalEog: wllama.isTokenEOG(final), input: r.tokens[t] };
}

const piece = (id: number) => showPiece(res?.pieces.get(id));
/** a token for running text: quoted, the leading space dropped */
const word = (id: number) => `“<span class="tok">${esc(piece(id).replace(/^␣(?=.)/, '').trim() || piece(id))}</span>”`;
/** a token for SVG text: quoted, so leading spaces stay visible */
const quoted = (id: number) => `\u201c${piece(id)}\u201d`;
// row l > 0 is the output of block l - 1: layers count from 0, as in the advanced view (layerLabel)
const rowName = (l: number) => (l === 0 ? 'the embedding' : `layer ${layerLabel(l)}`);

/** probability of a token at a fractional row; 0 where it is outside the top K */
function probAt(L: Lens, id: number, lc: number): number {
  const i = Math.floor(lc);
  const j = Math.min(L.nL - 1, i + 1);
  const at = (l: number) => (id === L.final ? L.finalP[l] : (L.probs[l].get(id) ?? 0));
  return at(i) + (at(j) - at(i)) * (lc - i);
}

function captionFor(L: Lens, lc: number): string {
  const l = Math.round(lc);
  const last = L.nL - 1;
  const fp = pct(L.finalP[l]);
  if (L.finalEog && l === last) return `<b>Last layer.</b> The model predicts the end of the text here.`;
  if (l === last) return `<b>Last layer.</b> Here the lens is exact: this is the model's own prediction, ${word(L.final)} at ${fp}.`;
  if (l === 0) {
    const echo = L.top[0] === L.input ? `mostly points back at the input token ${word(L.input)} itself` : `gives a shallow guess, ${word(L.top[0])}`;
    return `<b>Embedding.</b> No layer has run yet. Read through the lens, the input ${echo}.`;
  }
  if (l >= L.settle) return `<b>Layer ${layerLabel(l)}.</b> ${word(L.final)} has been the top guess since ${rowName(L.settle)}. The remaining layers mostly make the model more sure (${fp} now).`;
  if (L.top[l] === L.final) return `<b>Layer ${layerLabel(l)}.</b> ${word(L.final)} is on top here, but later layers still waver.`;
  const rank = L.finalRank[l] + 1;
  const where = rank <= 1000 ? `at rank ${rank}` : 'far down the list';
  return `<b>Layer ${layerLabel(l)}.</b> The top guess is ${word(L.top[l])}. The final answer ${word(L.final)} is still ${where} (${pct(L.finalP[l])}).`;
}

/////////////////////////////////////////////
// drawing: a stack of layers on the left, the lens readout on the right

const NS = 'http://www.w3.org/2000/svg';
function el<K extends keyof SVGElementTagNameMap>(name: K, attrs: Record<string, string | number>, parent: Element = svg, text?: string, style?: Partial<CSSStyleDeclaration>): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, name);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  if (text !== undefined) e.textContent = text;
  if (style) Object.assign(e.style, style);
  parent.appendChild(e);
  return e;
}

interface Bar {
  g: SVGGElement;
  fill: SVGRectElement;
  label: SVGTextElement;
  value: SVGTextElement;
  note: SVGTextElement;
  y: number;
  o: number;
}

interface Scene {
  blocks: SVGRectElement[];
  active: SVGLineElement;
  lens: SVGGElement;
  lensLabel: SVGTextElement;
  outBox: SVGRectElement;
  outText: SVGTextElement;
  title: SVGTextElement;
  bars: Map<number, Bar>;
  barLayer: SVGGElement;
  other: SVGTextElement;
  klPath: SVGPathElement | null;
  klDot: SVGCircleElement | null;
  klValue: SVGTextElement | null;
  yh: (l: number) => number;
  sx: number;
  row: { x: number; labelW: number; barX: number; barW: number; y0: number; h: number; fs: number; narrow: boolean };
  kx: (l: number) => number;
  ky: (v: number) => number;
}

let scene: Scene | null = null;

function build() {
  svg.replaceChildren();
  scene = null;
  if (!lens) return;
  const L = lens;
  const { width: W, height: H } = stage.getBoundingClientRect();
  if (W < 50 || H < 50) return;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const narrow = W < 640;
  const fs = narrow ? 12 : 13;

  // stack: output head on top, embedding at the bottom, one node per row in between
  const DW = narrow ? Math.max(120, W * 0.36) : clamp(W * 0.34, 230, 440);
  const SX = DW * 0.6;
  const boxH = Math.min(30, H * 0.07);
  const yTop = 2 + boxH + 12;
  const yBot = H - 2 - boxH - 12;
  const s = (yBot - yTop) / (L.nL - 1);
  const yh = (l: number) => yBot - l * s;
  const boxW = Math.min(DW * 0.76 - 8, 230);
  const boxX = clamp(SX - boxW / 2, 2, DW - boxW - 2);
  const box = (y: number, text: string) => {
    const r = el('rect', { x: boxX, y, width: boxW, height: boxH, rx: 6 }, svg, undefined, { fill: 'var(--sunk)', stroke: 'var(--rule)', strokeWidth: '1.5' });
    const t = el('text', { x: boxX + boxW / 2, y: y + boxH / 2 + 4.5, 'text-anchor': 'middle', class: 'sv-svg-text' }, svg, text, { fontSize: `${fs}px`, fill: 'var(--ink)' });
    return [r, t] as const;
  };
  box(H - 2 - boxH, `embed ${quoted(L.input)}`);
  const [outBox, outText] = box(2, 'output');
  el('line', { x1: SX, y1: yBot + 12, x2: SX, y2: yTop - 12 }, svg, undefined, { stroke: 'var(--rule)', strokeWidth: '4', strokeLinecap: 'round' });
  const active = el('line', { x1: SX, y1: yBot + 12, x2: SX, y2: yBot }, svg, undefined, { stroke: 'var(--data)', strokeWidth: '4', strokeLinecap: 'round' });
  const blocks: SVGRectElement[] = [];
  const nodeR = clamp(s * 0.3, 2.5, 8);
  const blockR = SX - 18;
  // layer numbers in an axis column left of the blocks, not inside them
  const lfs = clamp(s * 0.85, 10, 12);
  const axisW = Math.ceil(lfs * 0.62 * 2) + 6;
  const blockL = 2 + axisW;
  const bh = Math.min(s * 0.72, 26);
  for (let l = 1; l < L.nL; l++) {
    const mid = (yh(l) + yh(l - 1)) / 2;
    el('path', { d: `M${SX} ${yh(l - 1) - nodeR} L${blockR} ${mid}`, fill: 'none' }, svg, undefined, { stroke: 'var(--ink-3)', strokeWidth: '1.2', opacity: '0.6' });
    el('path', { d: `M${blockR} ${mid} L${SX - nodeR} ${yh(l)}`, fill: 'none' }, svg, undefined, { stroke: 'var(--ink-3)', strokeWidth: '1.5' });
    blocks.push(el('rect', { x: blockL, y: mid - bh / 2, width: blockR - blockL, height: bh, rx: Math.min(5, bh / 3) }, svg, undefined, { fill: 'var(--panel)', stroke: 'var(--ink-3)', strokeWidth: '1.2' }));
    // every layer where the rows have room, else every fifth and the last
    if (s >= 14 || (l - 1) % 5 === 0 || l === L.nL - 1) el('text', { x: axisW, y: mid + lfs * 0.36, 'text-anchor': 'end', class: 'sv-svg-mono' }, svg, layerLabel(l), { fontSize: `${lfs}px`, fill: 'var(--ink-3)' });
    el('circle', { cx: SX, cy: yh(l), r: nodeR }, svg, undefined, { fill: 'var(--panel)', stroke: 'var(--ink-3)', strokeWidth: '1.2' });
  }

  // the lens: a ring on the stream, a dotted line to the right, a magnifier
  const lensG = el('g', {});
  const ringR = nodeR + 5;
  const mx = DW - 16;
  el('line', { x1: SX + ringR, y1: 0, x2: mx - 14, y2: 0, 'stroke-dasharray': '2 6' }, lensG, undefined, { stroke: 'var(--trace)', strokeWidth: '3', strokeLinecap: 'round' });
  el('circle', { cx: SX, cy: 0, r: ringR, fill: 'none' }, lensG, undefined, { stroke: 'var(--trace)', strokeWidth: '3' });
  el('circle', { cx: mx, cy: 0, r: 13 }, lensG, undefined, { fill: 'color-mix(in srgb, var(--trace) 22%, var(--panel))', stroke: 'var(--trace)', strokeWidth: '3.5' });
  el('line', { x1: mx + 9, y1: 9, x2: mx + 15, y2: 15 }, lensG, undefined, { stroke: 'var(--trace)', strokeWidth: '4', strokeLinecap: 'round' });
  const lensLabel = el('text', { x: SX + ringR + 4, y: -7, class: 'sv-svg-mono' }, lensG, '', { fontSize: '11px', fill: 'var(--ink)', fontWeight: '600' });

  // readout
  const RX = DW + (narrow ? 14 : 40);
  const RW = W - RX;
  const title = el('text', { x: RX, y: 20, class: 'sv-svg-text' }, svg, '', { fontSize: narrow ? '15px' : '18px', fontWeight: '700', fill: 'var(--ink)' });
  el('text', { x: RX, y: 40, class: 'sv-svg-text' }, svg, `Its top guesses for the token after ${quoted(L.input)}`, { fontSize: `${fs}px`, fill: 'var(--ink-2)' });
  const KH = H >= 340 ? 92 : 0;
  const y0 = 56;
  const h = clamp((H - y0 - KH - 28) / (SHOWN + 1), 20, 40);
  const labelW = clamp(RW * 0.3, 60, 170);
  const barX = RX + labelW + 8;
  const barW = Math.max(30, RW - labelW - 8 - 52);
  const barLayer = el('g', {});
  const other = el('text', { x: RX, y: y0 + (SHOWN + 1) * h + 14, class: 'sv-svg-text' }, svg, '', { fontSize: `${fs}px`, fill: 'var(--ink-2)' });

  // how far each row is from the final answer
  let klPath: SVGPathElement | null = null;
  let klDot: SVGCircleElement | null = null;
  let klValue: SVGTextElement | null = null;
  const kmax = Math.max(1e-6, ...L.kl);
  const kx = (l: number) => RX + (l / (L.nL - 1)) * Math.max(40, RW - 90);
  const ky = (v: number) => H - 8 - (v / kmax) * (KH - 40);
  if (KH) {
    el('text', { x: RX, y: H - KH + 14, class: 'sv-svg-text' }, svg, narrow ? 'Distance from the final answer (KL)' : 'Distance from the final answer (KL divergence)', { fontSize: `${fs}px`, fill: 'var(--ink-2)' });
    el('line', { x1: kx(0), y1: H - 8, x2: kx(L.nL - 1), y2: H - 8 }, svg, undefined, { stroke: 'var(--rule)', strokeWidth: '1.5' });
    klPath = el('path', { fill: 'none' }, svg, undefined, { stroke: 'var(--data)', strokeWidth: '2.5', strokeLinejoin: 'round' });
    klDot = el('circle', { r: 5 }, svg, undefined, { fill: 'var(--trace)' });
    klValue = el('text', { x: kx(L.nL - 1) + 10, y: H - 10, class: 'sv-svg-mono' }, svg, '', { fontSize: '12px', fill: 'var(--ink)' });
  }
  svg.append(lensG);
  builtW = W;
  builtH = H;
  scene = { blocks, active, lens: lensG, lensLabel, outBox, outText, title, bars: new Map(), barLayer, other, klPath, klDot, klValue, yh, sx: SX, row: { x: RX, labelW, barX, barW, y0, h, fs, narrow }, kx, ky };
}

function makeBar(sc: Scene, id: number, y: number): Bar {
  const { labelW, barX, barW, h } = sc.row;
  const g = el('g', {}, sc.barLayer);
  // smaller on phones, where the label column is narrow
  const fs = clamp(h * 0.42, 11, sc.row.narrow ? 13 : 16);
  const bh = h * 0.66;
  // condensed mono: about 0.56 em per character
  const chars = Math.max(3, Math.floor(labelW / (fs * 0.56)));
  // quoted, so a leading space shows: " cold" and "cold" are different tokens
  const text = quoted(id);
  const label = el('text', { x: barX - 8, y: bh / 2 + fs * 0.36, 'text-anchor': 'end', class: 'sv-svg-mono' }, g, text.length > chars ? `${text.slice(0, chars - 1)}…` : text, { fontSize: `${fs}px`, fill: 'var(--ink)' });
  if (text.length > chars) el('title', {}, label, text);
  el('rect', { x: barX, y: 0, width: barW, height: bh, rx: 4 }, g, undefined, { fill: 'var(--sunk)' });
  const fill = el('rect', { x: barX, y: 0, width: 0, height: bh, rx: 4 }, g, undefined, { fill: id === lens!.final ? 'var(--trace)' : 'var(--data)' });
  const value = el('text', { x: barX + barW + 8, y: bh / 2 + 4.5, class: 'sv-svg-text' }, g, '', { fontSize: '13px', fill: 'var(--ink)', fontWeight: '600' });
  const note = el('text', { x: barX + 8, y: bh / 2 + 4, class: 'sv-svg-text' }, g, '', { fontSize: '11px', fill: 'var(--ink-2)' });
  return { g, fill, label, value, note, y, o: 0 };
}

/////////////////////////////////////////////
// animation: lc is the row the lens reads, fractional between rows

let lc = 0;
let playing = false;
let playStart = 0; // time at which lc would have been 0
let lastFrame = 0;
let frameId = 0;
let lastCaption = '';
const HOLD = 0.8; // seconds on the embedding before the lens moves
const stepSec = () => clamp(10 / Math.max(1, (lens?.nL ?? 2) - 1), 0.3, 1.1);

/** lc after t seconds of playing: each step eases in, then holds */
function lcAt(t: number): number {
  const k = Math.max(0, t - HOLD) / stepSec();
  const i = Math.floor(k);
  return Math.min(lens!.nL - 1, i + ease(clamp((k - i) / 0.5)));
}

function play() {
  if (!lens) return;
  if (lc >= lens.nL - 1 - 1e-3) lc = 0;
  // continue from the current row: the inverse of lcAt, so the first frame does not jump
  const i = Math.floor(lc);
  const f = lc - i;
  const g = f < 0.5 ? Math.sqrt(f / 2) : 1 - Math.sqrt(2 * (1 - f)) / 2;
  playStart = performance.now() / 1000 - (lc > 0 ? HOLD + (i + g * 0.5) * stepSec() : 0);
  playing = true;
  btnPlay.textContent = 'Pause';
  kick();
}

function pause() {
  playing = false;
  // at rest the lens sits on a whole layer, so the bars, the title and the caption all read the same row
  if (lens && lc !== Math.round(lc)) {
    lc = Math.round(lc);
    kick();
  }
  btnPlay.textContent = lens && lc >= lens.nL - 1 - 1e-3 ? 'Replay' : 'Play';
}

btnPlay.onclick = () => (playing ? pause() : play());
scrub.oninput = () => {
  pause();
  lc = Number(scrub.value);
  btnPlay.textContent = lens && lc >= lens.nL - 1 - 1e-3 ? 'Replay' : 'Play';
  kick();
};

function kick() {
  if (!frameId) frameId = requestAnimationFrame(frame);
}

function frame(nowMs: number) {
  frameId = 0;
  const now = nowMs / 1000;
  const dt = Math.min(0.1, now - (lastFrame || now));
  lastFrame = now;
  if (!lens || !scene) return;
  if (playing) {
    lc = lcAt(now - playStart);
    if (lc >= lens.nL - 1) pause();
  }
  scrub.value = String(lc);
  const settling = draw(dt);
  if (playing || settling) kick();
  else lastFrame = 0;
}

/** one frame; true while bars are still moving */
function draw(dt: number): boolean {
  const L = lens!;
  const sc = scene!;
  const last = L.nL - 1;
  const atEnd = lc >= last - 1e-3;
  const li = Math.round(lc);
  sc.blocks.forEach((b, i) => (b.style.fill = i + 1 <= lc + 0.01 ? 'color-mix(in srgb, var(--data) 28%, var(--panel))' : 'var(--panel)'));
  const y = sc.yh(lc);
  sc.active.setAttribute('y2', String(y));
  sc.lens.setAttribute('transform', `translate(0 ${y})`);
  sc.lensLabel.textContent = li === 0 ? 'embedding' : `layer ${layerLabel(li)}`;
  // below the line near the top, where the output box would cover it
  sc.lensLabel.setAttribute('y', y - sc.yh(L.nL - 1) < 20 ? '17' : '-7');
  sc.outBox.style.stroke = atEnd ? 'var(--trace)' : 'var(--rule)';
  // rebuilt only when it changes, every frame would be wasteful
  if (sc.outText.childElementCount !== (atEnd ? 1 : 0)) {
    sc.outText.textContent = 'output';
    if (atEnd) el('tspan', {}, sc.outText, ` → ${piece(L.final)}`, { fill: 'var(--trace)', fontWeight: '700' });
  }
  sc.title.textContent = li === 0 ? 'Lens at the embedding' : `Lens at layer ${layerLabel(li)} (of 0\u2013${layerLabel(last)})`;

  // top 5 at this row, the final answer pinned below them if it is not among them
  const ids = new Set<number>([...L.probs[Math.floor(lc)].keys(), ...L.probs[Math.min(last, Math.floor(lc) + 1)].keys(), L.final]);
  const ranked = [...ids].map((id) => ({ id, p: probAt(L, id, lc) })).sort((a, b) => b.p - a.p);
  const shown = ranked.slice(0, SHOWN);
  if (!shown.some((x) => x.id === L.final)) shown.push(ranked.find((x) => x.id === L.final)!);
  const k = 1 - Math.exp(-dt * 10);
  let moving = false;
  const { y0, h } = sc.row;
  const want = new Map(shown.map((x, i) => [x.id, { p: x.p, y: y0 + i * h }]));
  for (const [id, w] of want) if (!sc.bars.has(id)) sc.bars.set(id, makeBar(sc, id, w.y));
  for (const [id, b] of sc.bars) {
    const w = want.get(id);
    const ty = w?.y ?? b.y;
    const to = w ? 1 : 0;
    b.y += (ty - b.y) * k;
    b.o += (to - b.o) * k;
    if (Math.abs(ty - b.y) > 0.3 || Math.abs(to - b.o) > 0.01) moving = true;
    if (!w && b.o < 0.02) {
      b.g.remove();
      sc.bars.delete(id);
      continue;
    }
    b.g.setAttribute('transform', `translate(0 ${b.y})`);
    b.g.style.opacity = String(b.o);
    if (!w) continue;
    b.fill.setAttribute('width', String(sc.row.barW * w.p));
    b.value.textContent = pct(w.p);
    // the pinned final answer shows how far down it is
    const pinned = id === L.final && w.y >= y0 + SHOWN * h - 0.5;
    // short form where the bar is narrow
    const noteX = Math.max(8, sc.row.barW * w.p + 6);
    const rank = L.finalRank[li] + 1;
    b.note.textContent = !pinned ? '' : sc.row.barW - noteX > 130 ? `final answer, rank ${rank}` : `rank ${rank}`;
    b.note.setAttribute('x', String(sc.row.barX + noteX));
  }
  const rest = 1 - shown.reduce((a, x) => a + x.p, 0);
  sc.other.setAttribute('y', String(y0 + shown.length * h + 14));
  sc.other.textContent = `All other tokens: ${pct(Math.max(0, rest))}`;

  if (sc.klPath) {
    let d = '';
    for (let l = 0; l <= lc + 1e-6; l += 0.25) d += `${d ? 'L' : 'M'}${sc.kx(l).toFixed(1)} ${sc.ky(klAt(L, l)).toFixed(1)} `;
    d += `L${sc.kx(lc).toFixed(1)} ${sc.ky(klAt(L, lc)).toFixed(1)}`;
    sc.klPath.setAttribute('d', d);
    sc.klDot!.setAttribute('cx', String(sc.kx(lc)));
    sc.klDot!.setAttribute('cy', String(sc.ky(klAt(L, lc))));
    sc.klValue!.textContent = `${klAt(L, lc).toFixed(2)} nats`;
  }

  const c = captionFor(L, lc);
  if (c !== lastCaption) caption.innerHTML = lastCaption = c;
  return moving;
}

function klAt(L: Lens, l: number): number {
  const i = Math.floor(l);
  const j = Math.min(L.nL - 1, i + 1);
  return L.kl[i] + (L.kl[j] - L.kl[i]) * (l - i);
}

/** after a run only the chosen model shows, with a button to pick another */
function collapseModels(collapse: boolean) {
  modelRow.classList.toggle('collapsed', collapse);
  btnModels.hidden = !collapse;
}
btnModels.onclick = () => {
  collapseModels(false);
  modelInputs[modelIdx].input.focus();
};

function show(r: LogitLensResult) {
  collapseModels(true);
  res = r;
  resIdx = loadedIdx;
  lens = makeLens(r);
  intro.hidden = true;
  lastCaption = '';
  build();
  scrub.max = String(lens.nL - 1);
  scrub.disabled = false;
  btnPlay.disabled = false;
  // always from the embedding up: the animation is the point of this view, even with reduced motion set
  lc = 0;
  play();
}

function clearScene() {
  pause();
  res = null;
  lens = null;
  scene = null;
  svg.replaceChildren();
  intro.hidden = false;
  scrub.disabled = btnPlay.disabled = true;
}

// rebuilt when the width or the height changes a lot; a caption that wraps to one more line only scales the
// drawing a little (the viewBox keeps its aspect), instead of a jump and a flicker of the bars
let builtW = 0;
let builtH = 0;
new ResizeObserver(() => {
  if (!lens) return;
  const { width, height } = stage.getBoundingClientRect();
  if (scene && Math.abs(width - builtW) < 1 && Math.abs(height - builtH) < 60) return;
  const first = !scene;
  build();
  if (!scene) return;
  lastFrame = 0;
  // the clock ran while there was no scene to draw: start the climb now
  if (first && playing) return play();
  // place everything at once after a resize
  draw(1);
  kick();
}).observe(stage);

/////////////////////////////////////////////
// setup

examplesEl.replaceChildren(
  ...EXAMPLES.map((ex) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = ex.label;
    b.title = ex.prompt;
    b.setAttribute('aria-pressed', String(promptEl.value === ex.prompt));
    b.onclick = () => {
      if (busy) return;
      promptEl.value = ex.prompt;
      // keeps the pressed state, the advanced link and the URL in step even if the run fails or is cancelled
      promptEl.dispatchEvent(new Event('input'));
      updateUrl();
      run();
    };
    return b;
  })
);
promptEl.addEventListener('input', () => {
  examplesEl.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(promptEl.value === x.title)));
  syncAdvancedLink();
});
// once editing ends (blur or Enter); a run writes it too
promptEl.addEventListener('change', updateUrl);

showPreviewBadge($('preview-badge'));

promptEl.maxLength = SIMPLE_PROMPT_MAX;
const promptParam = urlParams.get('prompt');
if (promptParam !== null) promptEl.value = promptParam;
promptEl.dispatchEvent(new Event('input'));
renderModels();
syncStatus();
refreshCache().then(() => {
  renderModels();
  syncStatus();
});
gpuChecked.then(syncStatus);
