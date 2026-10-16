import '@fontsource-variable/martian-mono/wdth.css';
import '@fontsource-variable/atkinson-hyperlegible-next/wght.css';
import './style.css';
import { Wllama, type LogitLensResult } from '@wllama/wllama';
import { Heatmap, rampGradientCss, divergingGradientCss, showPiece, layerLabel, CELL_W, CELL_H, LEFT, TOP, type ColorMode } from './heatmap';
import { drawTrace, type TracePoint } from './trace';
import { ChatFormat } from './chat';
import { info, initGlossary } from './glossary';
import { SUPPORTED_ARCHS, MAX_FILE_BYTES, tooLargeForBrowser, onlySmallFilesFit, searchRepos, getRepo, listGgufFiles, isSupported, pickDefaultFile, quantLabel, fmtBytes, fmtParams, fmtCount, DEFAULT_REPO, PRESET_FILES, type RepoInfo } from './hf';

// shown when the search box is empty
// hint: download size of the file preselected by pickDefaultFile
const SUGGESTED: RepoInfo[] = [
  { id: DEFAULT_REPO, arch: 'llama', params: 0.135e9, downloads: 0, gated: false, hint: 'default \u00b7 145 MB' },
  { id: 'unsloth/Qwen3.5-0.8B-GGUF', arch: 'qwen35', params: 0.75e9, downloads: 0, gated: false, hint: '811 MB' },
  { id: 'Qwen/Qwen3-0.6B-GGUF', arch: 'qwen3', params: 0.6e9, downloads: 0, gated: false, hint: '639 MB' },
  { id: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF', arch: 'qwen2', params: 0.5e9, downloads: 0, gated: false, hint: '676 MB' },
  { id: 'unsloth/Llama-3.2-1B-Instruct-GGUF', arch: 'llama', params: 1.2e9, downloads: 0, gated: false, hint: '807 MB' },
];

// ?model=<url> adds a direct-URL entry, e.g. a dev-served /local-models/... file
const urlParams = new URLSearchParams(location.search);
const directParam = urlParams.get('model');
const directUrl = directParam ? new URL(directParam, location.href).href : null;
// a shared Hugging Face link is shown as repo + quantization, parsed from the URL alone
const directHf = directUrl ? /^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\/main\/(.+)$/.exec(directUrl) : null;
const DIRECT_ID = '(direct URL)';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const examplesEl = $<HTMLDivElement>('examples');
const traceActions = $<HTMLDivElement>('trace-actions');
const wordRadio = document.querySelector<HTMLInputElement>('input[name=color][value=word]')!;
const contrastRadio = document.querySelector<HTMLInputElement>('input[name=color][value=contrast]')!;
const exampleNote = $<HTMLParagraphElement>('example-note');
const summaryEl = $<HTMLParagraphElement>('summary');

// each one shows something at a glance in small models; the note says what to look for
const EXAMPLES: { label: string; prompt: string; prompt2?: string; note: string }[] = [
  { label: 'A fact', prompt: 'The Eiffel Tower is located in the city of', note: 'Look at the last column: in which layer does "Paris" first show up, and where does the model settle on it?' },
  { label: 'Counting', prompt: 'One, two, three, four,', note: 'An easy pattern: the model is sure of "five" long before the last layer.' },
  { label: 'Who did what', prompt: 'Mary and John went to the store. John gave a drink to', note: 'The model has to keep track of who is who. Watch how late in the layers "Mary" appears.' },
  { label: 'Opposites', prompt: 'The opposite of hot is', note: '"cold" shows up in the middle layers, but the model keeps hesitating before it commits.' },
  {
    label: 'Two prompts',
    prompt: 'The capital of France is',
    prompt2: 'The capital of Germany is',
    note: 'Two prompts one word apart, on the same scale. Compare the last columns: in which layer do the two guesses split?',
  },
];
const modelCurrent = $<HTMLParagraphElement>('model-current');
const modelIndicator = $<HTMLParagraphElement>('model-indicator');
const railLoad = $<HTMLDivElement>('rail-load');
const modelOptions = $<HTMLDetailsElement>('model-options');
const btnContinue = $<HTMLButtonElement>('btn-continue');
const btnContinue5 = $<HTMLButtonElement>('btn-continue5');
const btnCopyLink = $<HTMLButtonElement>('btn-copy-link');
const btnExportJson = $<HTMLButtonElement>('btn-export-json');
const btnExportPng = $<HTMLButtonElement>('btn-export-png');
const searchInput = $<HTMLInputElement>('model-search');
const repoMeta = $<HTMLDivElement>('repo-meta');
const repoList = $<HTMLDivElement>('repo-list');
const quantBlock = $<HTMLDivElement>('quant-block');
const quantList = $<HTMLDivElement>('quant-list');
const quantNote = $<HTMLParagraphElement>('quant-note');
const quantRepo = $<HTMLSpanElement>('quant-repo');
const emptyEl = $<HTMLElement>('empty');
const timingsEl = $<HTMLElement>('timings');
const timingsBox = $<HTMLDetailsElement>('timings-box');
const timingsSummary = $<HTMLElement>('timings-summary');
// keyboard shortcut hint only where there is likely a keyboard
const idleHint = window.matchMedia('(hover: hover)').matches ? 'Ctrl+Enter runs the lens.' : '';
const btnLoad = $<HTMLButtonElement>('btn-load');
const btnLoadEmpty = $<HTMLButtonElement>('btn-load-empty');
const emptyTitle = $<HTMLParagraphElement>('empty-title');
const fileInput = $<HTMLInputElement>('file-input');
const useGpu = $<HTMLInputElement>('use-gpu');
const modelStatus = $<HTMLSpanElement>('model-status');
const loadPanel = $<HTMLElement>('load-panel');
const loadPhase = $<HTMLParagraphElement>('load-phase');
const loadBar = $<HTMLDivElement>('load-bar');
const loadDetail = $<HTMLParagraphElement>('load-detail');
const btnCancel = $<HTMLButtonElement>('btn-cancel');
const cacheRow = $<HTMLDivElement>('cache-row');
const cacheInfo = $<HTMLSpanElement>('cache-info');
const btnClearCache = $<HTMLButtonElement>('btn-clear-cache');
const downloadNotes = document.querySelectorAll<HTMLElement>('[data-download-note]');
const promptEl = $<HTMLTextAreaElement>('prompt');
const useChat = $<HTMLInputElement>('use-chat');
const chatLabel = $<HTMLLabelElement>('chat-label');
const chatNote = $<HTMLSpanElement>('chat-note');
const btnRun = $<HTMLButtonElement>('btn-run');
const runStatus = $<HTMLSpanElement>('run-status');
const resultEl = $<HTMLElement>('result');
const legend = $<HTMLDivElement>('legend');
const heatmapWrap = $<HTMLDivElement>('heatmap-wrap');
const tableWrap = $<HTMLDivElement>('table-wrap');
const tooltip = $<HTMLDivElement>('tooltip');
const traceEl = $<HTMLElement>('trace');
const traceDesc = $<HTMLParagraphElement>('trace-desc');
const traceChips = $<HTMLDivElement>('trace-chips');
const tracePlot = $<HTMLDivElement>('trace-plot');

const promptB = $<HTMLTextAreaElement>('prompt-b');
const fieldB = $<HTMLDivElement>('field-b');
const promptsEl = $<HTMLDivElement>('prompts');
const useCompare = $<HTMLInputElement>('use-compare');
const capA = $<HTMLParagraphElement>('cap-a');
const capB = $<HTMLParagraphElement>('cap-b');
const rowB = $<HTMLDivElement>('row-b');

const WASM = { default: new URL('wasm/wllama.wasm', document.baseURI).href };
let wllama = new Wllama(WASM, { suppressNativeLog: true });

// the word the grid can be shaded by (#3), and the second word it can be contrasted with (#2)
interface WordShade {
  id: number;
  probs: Float32Array;
  ranks: Int32Array;
}

/**
 * One prompt and its grid. Compare mode (#11) shows a second one, B.
 * The lens keeps the hidden states of only its last forward pass, so every call that reuses them
 * goes through ensureCached(side), which runs that side's prompt again when the other one is cached.
 */
interface Side {
  name: 'A' | 'B';
  canvas: HTMLCanvasElement;
  heatmap: Heatmap;
  res: LogitLensResult | null;
  prompt: string; // the prompt as typed, for the caption
  input: string; // the text the lens read
  owner: Wllama | null; // reuse is only valid in the instance that ran it
  word: WordShade | null;
  contrast: WordShade | null;
}
const side = (name: 'A' | 'B', id: string): Side => {
  const canvas = $<HTMLCanvasElement>(id);
  return { name, canvas, heatmap: new Heatmap(canvas, $<HTMLCanvasElement>(`${id}-axis`)), res: null, prompt: '', input: '', owner: null, word: null, contrast: null };
};
const A = side('A', 'heatmap');
const B = side('B', 'heatmap-b');
/** the sides on screen: B only in compare mode, after a run that included it */
const shown = () => (useCompare.checked && B.res ? [A, B] : [A]);
const comparing = () => shown().length === 2;
// the lens input whose hidden states the lens currently holds, null when unknown
let lensCached: string | null = null;

let selected: { side: Side; layer: number; pos: number } | null = null;
let traced: { side: Side; pos: number } | null = null;
let shadeId: number | null = null;
let contrastId: number | null = null;
// token pieces of the results on screen, see syncPieces
let pieces = new Map<number, string>();
const syncPieces = () => (pieces = new Map([...(A.res?.pieces ?? []), ...(B.res?.pieces ?? [])]));
let busy = false;

// chat format (#9): the prompt becomes the user message, and Continue grows the model's reply
let chatFormat: ChatFormat | null = null;
let reply = '';
let lastInput = '';
const chatOn = () => useChat.checked && !!chatFormat;
/** the text the lens reads for one prompt; the reply only grows with Continue, which compare mode does not offer */
const lensInput = (text: string) => (chatOn() ? chatFormat!.render(text) + (useCompare.checked ? '' : reply) : text);

const colorMode = (): ColorMode => (document.querySelector<HTMLInputElement>('input[name=color]:checked')!.value as ColorMode);
const viewMode = () => document.querySelector<HTMLInputElement>('input[name=view]:checked')!.value;
const pct = (p: number) => (p >= 0.995 ? '100%' : p < 0.001 ? '<0.1%' : `${(p * 100).toFixed(1)}%`);
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
const piece = (id: number) => showPiece(pieces.get(id));
/** a word in quotes for sentences, with the leading space trimmed */
const quote = (id: number) => `\u201c<span class="tok">${esc(showPiece(pieces.get(id)).trim() || piece(id))}</span>\u201d`;

/////////////////////////////////////////////
// model search

let cachedUrls = new Set<string>();
let cachedBytes = 0;
let shownRepos: RepoInfo[] = [];
let selectedRepo: string | null = null;
let searchCtl: AbortController | null = null;
let filesCtl: AbortController | null = null;
let selectedUrl = '';
let selectedLabel = '';
let selectedSize = NaN;
let confirmUrl: string | null = null; // set after the first click on a large download

const CONFIRM_BYTES = 1e9;
const needsConfirm = () =>
  !cachedUrls.has(selectedUrl) &&
  (selectedSize >= CONFIRM_BYTES || (Number.isNaN(selectedSize) && new URL(selectedUrl).host !== location.host));
const confirmText = () => `Confirm download${Number.isNaN(selectedSize) ? '' : ` \u00b7 ${fmtBytes(selectedSize)}`}`;

// both load buttons follow the selected file; the empty-state one is the first-visit shortcut
function syncLoad() {
  const loaded = wllama.isModelLoaded();
  btnLoad.disabled = busy || !selectedUrl;
  btnLoadEmpty.disabled = busy || !selectedUrl;
  btnLoadEmpty.hidden = loaded || !selectedUrl;
  const confirming = confirmUrl === selectedUrl && needsConfirm();
  btnLoad.textContent = confirming ? confirmText() : 'Load model';
  btnLoadEmpty.textContent = confirming ? confirmText() : `Load ${selectedLabel}`;
  let note = '';
  if (selectedUrl && cachedUrls.has(selectedUrl)) {
    note = "Loads from your browser's cache, no download.";
  } else if (selectedUrl) {
    const size = Number.isNaN(selectedSize) ? '' : `${fmtBytes(selectedSize)} `;
    note = `Downloads ${size}from ${new URL(selectedUrl).host} over your connection. The file is then kept in your browser for later visits.`;
    if (!size) note += ' Its size is shown once the download starts.';
    if (confirming) note += ' Click again to start the download, or pick a smaller file under Model and settings.';
    else if (needsConfirm()) note += size ? ' Files this large ask for a second click.' : ' Downloads of unknown size ask for a second click.';
  }
  downloadNotes.forEach((el) => (el.textContent = note));
  modelCurrent.textContent = selectedLabel;
  // the empty state has the same button and note; show the rail copy only once that is gone
  railLoad.hidden = !emptyEl.hidden && !btnLoadEmpty.hidden;
  emptyTitle.textContent = loaded ? 'Model loaded. Run the lens to see the readout.' : 'Load a model to start.';
}

async function refreshCache() {
  try {
    const models = await wllama.modelManager.getModels();
    cachedUrls = new Set(models.map((m) => m.url));
    cachedBytes = models.reduce((a, m) => a + m.size, 0);
  } catch {
    // OPFS unavailable, no cache status
    cachedUrls = new Set();
    cachedBytes = 0;
  }
  cacheRow.hidden = cachedUrls.size === 0;
  cacheInfo.textContent = `${cachedUrls.size} ${cachedUrls.size === 1 ? 'model' : 'models'} cached in this browser, ${fmtBytes(cachedBytes)}.`;
}

const repoHasCache = (id: string) => [...cachedUrls].some((u) => u.startsWith(`https://huggingface.co/${id}/`));

function renderRepos() {
  repoList.replaceChildren(
    ...shownRepos.map((r) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('aria-pressed', String(r.id === selectedRepo));
      const supported = r.id === DIRECT_ID || isSupported(r);
      const [owner, name] = r.id === DIRECT_ID ? (directHf ? directHf[1].split('/') : ['', directUrl!.split('/').pop()!]) : r.id.split('/');
      const meta: string[] = [];
      if (r.id === DIRECT_ID) meta.push(directHf ? 'shared link' : 'direct URL');
      if (r.params && r.arch !== 'clip') meta.push(`${fmtParams(r.params)} params`);
      if (r.id !== DIRECT_ID && tooLargeForBrowser(r.params)) meta.push('too large for the browser');
      else if (r.id !== DIRECT_ID && onlySmallFilesFit(r.params)) meta.push('large: only small files fit');
      if (r.downloads) meta.push(`${fmtCount(r.downloads)} downloads`);
      if (r.hint) meta.push(r.hint);
      if (repoHasCache(r.id) || (r.id === DIRECT_ID && cachedUrls.has(directUrl!))) meta.push('cached');
      const arch = r.arch ? `${r.arch}${supported ? '' : '?'}` : r.id === DIRECT_ID ? '' : 'unknown';
      b.innerHTML =
        `<span class="name">${owner ? `<span class="owner">${esc(owner)}/</span>` : ''}${esc(decodeURIComponent(name))}</span>` +
        `<span class="arch${supported ? '' : ' warn'}" title="${supported ? 'Architecture' : 'Architecture may be unsupported'}">${esc(arch)}</span>` +
        (meta.length ? `<span class="meta">${meta.join(' \u00b7 ')}</span>` : '');
      b.onclick = () => selectRepo(r);
      return b;
    })
  );
}

async function search(query: string) {
  searchCtl?.abort();
  const q = query.trim();
  if (!q) {
    shownRepos = directUrl ? [{ id: DIRECT_ID, arch: null, params: null, downloads: 0, gated: false }, ...SUGGESTED] : SUGGESTED;
    repoMeta.textContent = 'Suggested small models. Type to search Hugging Face.';
    renderRepos();
    return;
  }
  const ctl = (searchCtl = new AbortController());
  repoMeta.textContent = 'Searching...';
  try {
    const exactId = /^[\w.-]+\/[\w.-]+$/.test(q);
    const [found, exact] = await Promise.all([searchRepos(q, ctl.signal), exactId ? getRepo(q, ctl.signal) : Promise.resolve(null)]);
    const usable = found.filter((r) => isSupported(r) && !r.gated && !tooLargeForBrowser(r.params));
    // an exact repo id is always listed, the architecture tag can be wrong for repos with projector files
    if (exact && !exact.gated && !usable.some((r) => r.id === exact.id)) usable.unshift(exact);
    shownRepos = usable;
    const hidden = found.length - usable.filter((r) => found.includes(r)).length;
    repoMeta.textContent = usable.length
      ? `${usable.length} matching ${usable.length === 1 ? 'repo' : 'repos'}${hidden ? `, ${hidden} hidden (unsupported architecture, gated or too large)` : ''}. Supported: ${SUPPORTED_ARCHS.join(', ')}.`
      : `No supported repos found${hidden ? ` (${hidden} hidden: unsupported architecture, gated or too large)` : ''}. Supported: ${SUPPORTED_ARCHS.join(', ')}.`;
    renderRepos();
  } catch (e) {
    if (ctl.signal.aborted) return;
    repoMeta.textContent = `Search failed: ${(e as Error).message}`;
  }
}

function setQuantStatus(text: string) {
  quantList.replaceChildren(Object.assign(document.createElement('p'), { className: 'hint', textContent: text }));
}

async function selectRepo(r: RepoInfo) {
  selectedRepo = r.id;
  selectedUrl = '';
  quantNote.hidden = true;
  renderRepos();
  filesCtl?.abort();
  const ctl = (filesCtl = new AbortController());
  syncLoad();
  quantBlock.hidden = false;
  quantRepo.textContent = r.id === DIRECT_ID ? (directHf ? `in ${directHf[1]}` : '') : `in ${r.id}`;
  const files =
    r.id === DIRECT_ID && directUrl
      ? [{ path: decodeURIComponent(directUrl.split('/').pop()!), size: NaN, shards: 1, url: directUrl }]
      : (PRESET_FILES[r.id] ?? null);
  if (!files) setQuantStatus('Listing files...');
  try {
    const list = files ?? (await listGgufFiles(r.id, ctl.signal));
    const repoName = r.id.split('/').pop()!.replace(/-GGUF$/i, '');
    if (!list.length) return setQuantStatus('This repo has no model .gguf files.');
    const def = r.id === DIRECT_ID ? list[0] : pickDefaultFile(list, r.params);
    quantList.replaceChildren(
      ...list.map((f) => {
        const tooBig = f.size > MAX_FILE_BYTES;
        const label = document.createElement('label');
        label.title = `${f.path}${tooBig ? ' - too large to load in the browser' : ''}`;
        const input = Object.assign(document.createElement('input'), { type: 'radio', name: 'quant', value: f.url, checked: f === def, disabled: tooBig });
        input.onchange = () => {
          selectedUrl = f.url;
          selectedLabel = label.dataset.label!;
          selectedSize = f.size;
          confirmUrl = null;
          syncLoad();
        };
        const size = Number.isNaN(f.size) ? 'local' : `${fmtBytes(f.size)}${f.shards > 1 ? ` \u00b7 ${f.shards} parts` : ''}${tooBig ? ' \u00b7 too large' : ''}`;
        label.dataset.label =
          r.id === DIRECT_ID
            ? directHf
              ? `${directHf[1].split('/').pop()!.replace(/-GGUF$/i, '')} \u00b7 ${quantLabel(f.path)}`
              : f.path.replace(/\.gguf$/i, '')
            : `${repoName} \u00b7 ${quantLabel(f.path)} \u00b7 ${fmtBytes(f.size)}`;
        label.dataset.size = String(f.size);
        label.append(input);
        label.insertAdjacentHTML('beforeend', `<span class="q">${esc(quantLabel(f.path))}</span><span class="s${cachedUrls.has(f.url) ? ' cached' : ''}">${size}</span>`);
        return label;
      })
    );
    // outside the radiogroup, so screen readers do not read it as part of the group
    if (!def && r.id !== DIRECT_ID) {
      quantNote.textContent = `No file in this repo fits in the browser: the smallest is ${fmtBytes(Math.min(...list.map((f) => f.size)))}, the limit is ${fmtBytes(MAX_FILE_BYTES)}. Pick a smaller model.`;
      quantNote.hidden = false;
    }
    selectedUrl = def?.url ?? '';
    selectedLabel = quantList.querySelector<HTMLLabelElement>('label:has(input:checked)')?.dataset.label ?? '';
    selectedSize = def?.size ?? NaN;
    confirmUrl = null;
    syncLoad();
  } catch (e) {
    if (!ctl.signal.aborted) setQuantStatus(`Could not list files: ${(e as Error).message}`);
  }
}

let searchTimer = 0;
searchInput.addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = window.setTimeout(() => search(searchInput.value), 300);
});

function setBusy(b: boolean) {
  busy = b;
  btnClearCache.disabled = b;
  syncLoad();
  fileInput.disabled = b;
  // stays clickable without a model, so a click can explain what to do
  btnRun.disabled = b;
  // Continue grows one prompt; with two, each would append a different word, so it is off in compare mode
  btnContinue.disabled = btnContinue5.disabled = b || !A.res || !wllama.isModelLoaded() || useCompare.checked;
  btnContinue.title = useCompare.checked ? 'Continue works on one prompt. Turn off the comparison to use it.' : 'Append the final prediction to the prompt and run again';
  btnContinue5.title = useCompare.checked ? btnContinue.title : 'Append five predictions, one at a time';
  useCompare.disabled = b;
  syncChat();
}

function syncChat() {
  useChat.disabled = busy || !chatFormat;
  chatLabel.title = !wllama.isModelLoaded()
    ? 'Load a model first'
    : chatFormat
      ? "Wrap the prompt in the model's chat format: your prompt becomes the user message, and the last column predicts the first word of the model's reply"
      : 'This model has no chat template. Base models usually have none.';
  if (!chatOn()) return chatNote.replaceChildren();
  if (useCompare.checked) return (chatNote.textContent = 'Both prompts are user messages; the last columns predict the start of each reply.');
  if (!reply) return (chatNote.textContent = 'Your prompt is the user message; the last column predicts the start of the reply.');
  const clear = Object.assign(document.createElement('button'), { type: 'button', className: 'link-btn', textContent: 'Clear reply' });
  clear.onclick = () => {
    reply = '';
    syncChat();
    updateUrl();
  };
  chatNote.replaceChildren('Reply so far: \u201c', Object.assign(document.createElement('span'), { className: 'tok', textContent: reply }), '\u201d ', clear);
}

/////////////////////////////////////////////
// timing

const fmtMs = (ms: number) => (ms < 1000 ? `${ms.toFixed(ms < 10 ? 1 : 0)} ms` : `${(ms / 1000).toFixed(2)} s`);
let loadMs: number | null = null;

/** live elapsed time in a status line until stop() is called */
function ticker(el: HTMLElement, text: () => string) {
  const t0 = performance.now();
  const tick = () => (el.textContent = `${text()} ${fmtMs(performance.now() - t0)}`);
  tick();
  const id = window.setInterval(tick, 100);
  return () => {
    clearInterval(id);
    return performance.now() - t0;
  };
}

/** rs: the results of one run, two in compare mode */
function renderTimings(rs: LogitLensResult[], totalMs: number) {
  const r = rs[0];
  const decode = rs.reduce((a, x) => a + x.tDecodeMs, 0);
  const project = rs.reduce((a, x) => a + x.tProjectMs, 0);
  const other = Math.max(0, totalMs - decode - project);
  const items: [string, string, string?][] = [];
  if (loadMs !== null) items.push(['Model load', fmtMs(loadMs)]);
  items.push(['Forward pass', decode > 0 ? fmtMs(decode) : 'reused']);
  items.push(['Projection', fmtMs(project)]);
  items.push(['Other', fmtMs(other)]);
  items.push(['Lens total', fmtMs(totalMs), 'total']);
  const tokens = rs.map((x) => x.nTokens).join(' + ');
  timingsSummary.textContent = `Lens ${fmtMs(totalMs)} \u00b7 ${tokens} tokens \u00d7 ${r.nLayers} layers`;
  timingsEl.innerHTML =
    items.map(([k, v, cls]) => `<div${cls ? ` class="${cls}"` : ''}><dt>${k}</dt><dd>${v}</dd></div>`).join('') +
    `<div class="note">${tokens} tokens \u00d7 ${r.nLayers} layers${rs.length > 1 ? ', two prompts,' : ''} on ${esc(r.backend.replace(/^(\w+): \1$/, '$1'))}</div>`;
  timingsBox.hidden = false;
}

/** keep the repo list and chips in step with the cache, without changing the selection */
async function refreshSelection() {
  await refreshCache();
  renderRepos();
  const keep = selectedUrl;
  const repo = shownRepos.find((r) => r.id === selectedRepo);
  if (repo) await selectRepo(repo);
  const input = quantList.querySelector<HTMLInputElement>(`input[value="${CSS.escape(keep)}"]`);
  if (input) {
    input.checked = true;
    const label = input.closest('label')!;
    selectedUrl = keep;
    selectedLabel = label.dataset.label!;
    selectedSize = Number(label.dataset.size);
  }
  syncLoad();
}

function setBar(fraction: number | null) {
  loadBar.classList.toggle('indeterminate', fraction === null);
  const pct = fraction === null ? 0 : Math.round(fraction * 1000) / 10;
  (loadBar.firstElementChild as HTMLElement).style.width = fraction === null ? '' : `${pct}%`;
  if (fraction === null) loadBar.removeAttribute('aria-valuenow');
  else loadBar.setAttribute('aria-valuenow', String(pct));
}

const fmtDuration = (s: number) => (s < 60 ? `${Math.ceil(s)} s` : `${Math.floor(s / 60)} min ${Math.ceil(s % 60)} s`);

let loadCtl: AbortController | null = null;
let loadedUrl: string | null = null; // null for local files, which cannot be shared
btnCancel.onclick = () => loadCtl?.abort();

async function load(source: { url: string } | { files: File[] }) {
  setBusy(true);
  const ctl = (loadCtl = new AbortController());
  const fromCache = 'url' in source && cachedUrls.has(source.url);
  const host = 'url' in source ? new URL(source.url).host : '';
  const loadingPhase = 'Loading the model into memory';
  loadPanel.hidden = false;
  emptyEl.hidden = true;
  btnCancel.hidden = !('url' in source) || fromCache;
  loadPhase.textContent = 'url' in source ? (fromCache ? "Reading the model from your browser's cache" : `Downloading from ${host}`) : 'Opening the local file';
  loadDetail.textContent = 'Starting...';
  setBar('url' in source && !fromCache ? 0 : null);
  try {
    let phase = 'Loading';
    const stop = ticker(modelStatus, () => phase);
    // transfer rate as a moving average, so the time estimate does not jump around
    let lastT = performance.now();
    let lastBytes = 0;
    let rate = 0;
    const params = {
      n_ctx: 256,
      n_gpu_layers: useGpu.checked ? 999 : 0,
      signal: ctl.signal,
      progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
        if (fromCache || !total) return;
        if (loaded >= total) {
          phase = 'Loading';
          btnCancel.hidden = true;
          loadPhase.textContent = loadingPhase;
          loadDetail.textContent = `Downloaded ${fmtBytes(total)}. Setting up the model${useGpu.checked ? ' and the graphics card' : ''}; this can take a few seconds.`;
          setBar(null);
          return;
        }
        const now = performance.now();
        const dt = (now - lastT) / 1000;
        if (dt > 0.2) {
          const r = (loaded - lastBytes) / dt;
          rate = rate ? 0.7 * rate + 0.3 * r : r;
          lastT = now;
          lastBytes = loaded;
        }
        phase = `Downloading ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB,`;
        setBar(loaded / total);
        const eta = rate > 0 ? ` \u00b7 about ${fmtDuration((total - loaded) / rate)} left` : '';
        loadDetail.textContent = `${fmtBytes(loaded)} of ${fmtBytes(total)} \u00b7 ${Math.round((loaded / total) * 100)}%${rate > 0 ? ` \u00b7 ${(rate / 1e6).toFixed(1)} MB/s` : ''}${eta}`;
      },
    };
    if (fromCache || !('url' in source)) {
      loadDetail.textContent = `Setting up the model${useGpu.checked ? ' and the graphics card' : ''}; this can take a few seconds.`;
    }
    try {
      // download completely before unloading the current model, so a cancelled download keeps it
      const input = 'url' in source ? await wllama.modelManager.getModelOrDownload({ url: source.url }, params) : source.files;
      btnCancel.hidden = true;
      if (wllama.isModelLoaded()) {
        await wllama.exit();
        wllama = new Wllama(WASM, { suppressNativeLog: true });
      }
      await wllama.loadModel(input, params);
    } finally {
      loadMs = stop();
    }
    loadedUrl = 'url' in source ? source.url : null;
    lensCached = null;
    chatFormat = ChatFormat.from(wllama.getChatTemplate());
    if (!chatFormat) useChat.checked = false;
    updateUrl();
    const meta = wllama.getModelMetadata();
    const info = wllama.getLoadedContextInfo();
    const arch = meta.meta['general.architecture'] ?? '?';
    const name = meta.meta['general.name'] ?? ('url' in source ? source.url.split('/').pop() : source.files[0].name);
    modelIndicator.querySelector('a')!.textContent = `${name} \u00b7 loaded`;
    modelIndicator.hidden = false;
    modelStatus.textContent = `${name} is ready. It has ${info.n_layer} layers. Loaded in ${fmtMs(loadMs!)} (${arch}, ${wllama.isMultithread() ? `${wllama.getNumThreads()} threads` : 'single thread'}).`;
  } catch (e) {
    const kept = wllama.isModelLoaded() ? ' The previous model is still loaded.' : '';
    modelStatus.textContent = ctl.signal.aborted ? `Download cancelled.${kept}` : `Could not load the model: ${(e as Error).message}`;
  } finally {
    loadCtl = null;
    loadPanel.hidden = true;
    emptyEl.hidden = !resultEl.hidden;
    setBusy(false);
    await refreshSelection();
  }
}

/** model and prompt in the page URL, so the address bar is always a shareable link */
function updateUrl() {
  const q = new URLSearchParams();
  if (loadedUrl) q.set('model', loadedUrl.startsWith(location.origin) ? loadedUrl.slice(location.origin.length) : loadedUrl);
  if (promptEl.value) q.set('prompt', promptEl.value);
  if (useChat.checked) q.set('chat', '1');
  if (useChat.checked && reply && !useCompare.checked) q.set('reply', reply);
  if (useCompare.checked) q.set('prompt2', promptB.value);
  history.replaceState(null, '', `${location.pathname}?${q}`);
}

/** scroll heatmap and table to the last column; waits for the redraw that widens the canvas */
function scrollToNewest(smooth = true) {
  const behavior: ScrollBehavior = smooth && !window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'smooth' : 'auto';
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      for (const el of [heatmapWrap, tableWrap]) if (!el.hidden) el.scrollTo({ left: el.scrollWidth, behavior });
    })
  );
}

/** explain that a model is needed and point at the load button that is on screen */
function needModel() {
  runStatus.textContent = 'No model is loaded yet. Load one first with the highlighted button, then run the lens.';
  const target = !btnLoadEmpty.hidden && !emptyEl.hidden ? btnLoadEmpty : btnLoad;
  const smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  target.scrollIntoView({ block: 'center', behavior: smooth ? 'smooth' : 'auto' });
  target.classList.remove('attention');
  void target.offsetWidth; // restart the animation
  target.classList.add('attention');
  target.focus({ preventScroll: true });
}

/** a forward pass; it replaces the hidden states the lens holds */
async function lensForward(input: string, incremental: boolean): Promise<LogitLensResult> {
  lensCached = null;
  const r = await wllama.logitLens({ prompt: input, k: 5, incremental });
  lensCached = input;
  return r;
}

/** make the lens hold this side's hidden states, so a reuse call reads the right prompt */
async function ensureCached(s: Side) {
  if (!s.res || s.owner !== wllama) throw new Error('The model changed since this run. Run the lens again.');
  // incremental: only the positions after the prefix the two prompts share run again
  if (lensCached !== s.input) await lensForward(s.input, true);
}

/** probability and rank of one word in every cell of one side, by re-projecting its layer outputs */
async function projectWord(s: Side, id: number): Promise<WordShade> {
  await ensureCached(s);
  const r = await wllama.logitLens({ prompt: '', reuse: true, k: 1, targetIds: [id] });
  return { id, probs: r.targetProbs, ranks: r.targetRanks };
}

/** the shown sides, the one the lens holds first, so at most one forward pass runs again */
const cachedFirst = () => shown().sort((x, y) => Number(y.input === lensCached) - Number(x.input === lensCached));

async function run(continuing = false): Promise<boolean> {
  if (busy) return false;
  if (!wllama.isModelLoaded()) {
    needModel();
    return false;
  }
  const compare = useCompare.checked;
  if (compare && !promptB.value.trim()) {
    runStatus.textContent = 'Type a second prompt to compare with, or turn the comparison off.';
    promptB.focus();
    return false;
  }
  const prevTokens = A.res?.tokens ?? [];
  setBusy(true);
  const stop = ticker(runStatus, () => (compare ? 'Running both prompts' : 'Running'));
  try {
    let ms: number;
    // computed first and shown together, so a failure halfway never pairs a new grid with an old one
    const out: { s: Side; prompt: string; input: string; res: LogitLensResult; word: WordShade | null; contrast: WordShade | null }[] = [];
    try {
      for (const s of compare ? [A, B] : [A]) {
        const prompt = s === A ? promptEl.value : promptB.value;
        const input = lensInput(prompt);
        // continuing: the lens runs only the new tokens if the old ones are unchanged; B runs only what it does not share with A
        const res = await lensForward(input, continuing || s === B);
        const reuse = (id: number | null) => (id === null ? null : wllama.logitLens({ prompt: '', reuse: true, k: 1, targetIds: [id] }).then((r) => ({ id, probs: r.targetProbs, ranks: r.targetRanks })));
        // the chosen words, while the lens still holds this prompt
        out.push({ s, prompt, input, res, word: await reuse(shadeId), contrast: await reuse(contrastId) });
      }
    } finally {
      ms = stop();
    }
    if (!compare) Object.assign(B, { res: null, word: null, contrast: null });
    for (const o of out) Object.assign(o.s, { prompt: o.prompt, input: o.input, res: o.res, owner: wllama, word: o.word, contrast: o.contrast });
    lastInput = A.input;
    syncResults();
    selected = null;
    traced = null;
    traceEl.hidden = true;
    resultEl.hidden = false;
    emptyEl.hidden = true;
    runStatus.textContent = idleHint && 'Ctrl+Enter runs the lens again.';
    renderTimings(out.map((o) => o.res), ms);
    updateUrl();
    await document.fonts.load('500 condensed 12px "Martian Mono Variable"').catch(() => {});
    // re-tokenizing the extended prompt can change the last old token too, so animate from the first difference
    const res = A.res!;
    let same = 0;
    while (continuing && same < prevTokens.length && same < res.nTokens && prevTokens[same] === res.tokens[same]) same++;
    render(true, Math.min(same, res.nTokens - 1));
    // the last column holds the final prediction, so start there; the layer axis stays pinned on the left
    scrollToNewest(continuing);
    return true;
  } catch (e) {
    if (/loadModel\(\) is not yet called/.test((e as Error).message)) needModel();
    else {
      const msg = (e as Error).message;
      // the token limit counts the tokens the chat format adds
      runStatus.textContent = `Something went wrong: ${msg}${chatOn() && /limit/.test(msg) ? ' The chat format adds tokens of its own.' : ''}`;
    }
    return false;
  } finally {
    setBusy(false);
  }
}

/** after the results or the mode changed: pieces, shared scales, column alignment, word shades */
function syncResults() {
  syncPieces();
  const sides = shown();
  for (const s of [A, B]) if (s.res) s.heatmap.setResult(s.res);
  // one color scale for both grids
  const max = (f: (r: LogitLensResult) => Float32Array) => Math.max(...sides.map((s) => f(s.res!).reduce((a, x) => Math.max(a, x), 0))) || 1;
  const maxEntropy = max((r) => r.entropy);
  const maxKl = max((r) => r.kl);
  // columns line up by position; the marker starts at the first token that differs
  const a = A.res!.tokens;
  const b = B.res?.tokens ?? [];
  let d = 0;
  while (d < a.length && d < b.length && a[d] === b[d]) d++;
  const cmp = sides.length === 2;
  for (const s of sides) {
    s.heatmap.maxEntropy = maxEntropy;
    s.heatmap.maxKl = maxKl;
    s.heatmap.cols = cmp ? Math.max(a.length, b.length) : 0;
    s.heatmap.diffFrom = cmp && (d < a.length || d < b.length) ? d : null;
  }
  applyWordShades();
}

/** append the final layer's top prediction at the last position and run again, n times */
async function continueRun(n: number) {
  for (let i = 0; i < n && A.res && !useCompare.checked; i++) {
    const res = A.res;
    const id = res.topIds[((res.nLayers - 1) * res.nTokens + res.nTokens - 1) * res.k];
    if (wllama.isTokenEOG(id)) {
      runStatus.textContent = chatOn() ? 'The model predicts the end of its reply here.' : 'The model predicts the end of the text here.';
      return;
    }
    // in chat format the prediction continues the reply, so the prompt is wrapped only once
    if (chatOn()) reply += res.pieces.get(id) ?? '';
    else promptEl.value += res.pieces.get(id) ?? '';
    if (!(await run(true))) return;
  }
}

/////////////////////////////////////////////
// share and export

function download(blob: Blob, name: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

const fileStem = () => `logitlens-${promptEl.value.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 40).replace(/^-|-$/g, '') || 'result'}`;

function flash(btn: HTMLButtonElement, text: string) {
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = old), 1500);
}

btnCopyLink.onclick = async () => {
  updateUrl();
  try {
    await navigator.clipboard.writeText(location.href);
    flash(btnCopyLink, loadedUrl ? 'Link copied' : 'Copied (prompt only)');
  } catch {
    runStatus.textContent = `Copy this link: ${location.href}`;
  }
};

/** cells of one result for the JSON export */
function exportLayers(r: LogitLensResult) {
  const p = (id: number) => pieces.get(id) ?? '';
  const layers = [];
  for (let l = 0; l < r.nLayers; l++) {
    const cells = [];
    for (let t = 0; t < r.nTokens; t++) {
      const c = l * r.nTokens + t;
      const top = [];
      for (let j = 0; j < r.k; j++) top.push({ token: p(r.topIds[c * r.k + j]), id: r.topIds[c * r.k + j], prob: r.topProbs[c * r.k + j] });
      cells.push({ top, entropy: r.entropy[c], kl: r.kl[c], target: { token: p(r.targetIds[c]), id: r.targetIds[c], prob: r.targetProbs[c], rank: r.targetRanks[c] } });
    }
    layers.push({ layer: layerLabel(l), cells });
  }
  return { tokens: r.tokens.map((id) => ({ token: p(id), id })), layers };
}

btnExportJson.onclick = () => {
  if (!A.res) return;
  const chat = chatOn() && { chat: true, ...(!comparing() && { reply }), input: A.input };
  const data = {
    model: loadedUrl ?? 'local file',
    prompt: A.prompt,
    ...chat,
    ...exportLayers(A.res),
    // compare mode: the second prompt, same layout
    ...(comparing() && { second: { prompt: B.prompt, ...(chatOn() && { input: B.input }), ...exportLayers(B.res!) } }),
    note: 'layers[0] is the token embedding, layers[i] the output of block i-1; target is the token traced per position (default: final layer top-1); entropy and kl in nats, kl = KL(final layer || this layer) at the same position',
  };
  download(new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' }), `${fileStem()}.json`);
};

btnExportPng.onclick = () => {
  if (!A.res) return;
  // draw without the selection outline, then restore the on-screen state
  const sides = shown();
  for (const s of sides) s.heatmap.draw({ mode: colorMode(), selected: null });
  const parts = sides.map((s) => s.heatmap.composite());
  // compare mode: the two grids stacked, with a gap
  const gap = parts.length > 1 ? Math.round(16 * (parts[0].height / A.canvas.offsetHeight || 1)) : 0;
  const out = document.createElement('canvas');
  out.width = Math.max(...parts.map((c) => c.width));
  out.height = parts.reduce((a, c) => a + c.height, 0) + gap * (parts.length - 1);
  const ctx = out.getContext('2d')!;
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--panel').trim();
  ctx.fillRect(0, 0, out.width, out.height);
  let y = 0;
  for (const c of parts) {
    ctx.drawImage(c, 0, y);
    y += c.height + gap;
  }
  out.toBlob((blob) => {
    if (blob) download(blob, `${fileStem()}.png`);
    render();
  }, 'image/png');
};

const fmtTimes = (x: number) => (x >= 1e4 ? `${Math.round(x / 1000)}k` : x >= 100 ? String(Math.round(x)) : x.toFixed(x >= 10 ? 0 : 1));

function applyWordShades() {
  const sides = shown();
  for (const s of sides) {
    s.heatmap.wordProbs = s.word?.probs ?? null;
    if (s.word && s.contrast) {
      const a = s.word.probs;
      const b = s.contrast.probs;
      const d = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) d[i] = Math.log(Math.max(a[i], 1e-30)) - Math.log(Math.max(b[i], 1e-30));
      s.heatmap.setContrast(d);
    } else s.heatmap.setContrast(null);
  }
  // one scale for both grids
  const diffMax = Math.max(...sides.map((s) => s.heatmap.diffMax));
  for (const s of sides) s.heatmap.diffMax = diffMax;
  wordRadio.disabled = !sides.every((s) => s.word);
  contrastRadio.disabled = !sides.every((s) => s.word && s.contrast);
  if ((colorMode() === 'word' && wordRadio.disabled) || (colorMode() === 'contrast' && contrastRadio.disabled)) {
    document.querySelector<HTMLInputElement>('input[name=color][value=prob]')!.checked = true;
  }
}

/** shade every grid on screen by a word; in compare mode it is projected for each prompt */
async function chooseWord(kind: 'word' | 'contrast', id: number) {
  if (busy || !A.res) return;
  setBusy(true);
  try {
    const got = new Map<Side, WordShade>();
    for (const s of cachedFirst()) got.set(s, await projectWord(s, id));
    for (const [s, w] of got) {
      if (kind === 'word') {
        s.word = w;
        if (contrastId === id) s.contrast = null;
      } else s.contrast = w;
    }
    if (kind === 'word') {
      shadeId = id;
      if (contrastId === id) contrastId = null;
    } else contrastId = id;
    applyWordShades();
    (kind === 'word' ? wordRadio : contrastRadio).checked = true;
    render();
    if (traced) trace(traced.side, traced.pos);
  } catch (e) {
    runStatus.textContent = `Something went wrong: ${(e as Error).message}`;
  } finally {
    setBusy(false);
  }
}

/** one plain-language sentence about the last position: final guess, confidence, where it settles */
function renderSummary() {
  if (comparing()) return renderPairSummary();
  const r = A.res!;
  const t = r.nTokens - 1;
  const top = (l: number) => r.topIds[(l * r.nTokens + t) * r.k];
  const fin = top(r.nLayers - 1);
  const prob = r.topProbs[((r.nLayers - 1) * r.nTokens + t) * r.k];
  const after = `After ${quote(r.tokens[t])}`;
  if (wllama.isTokenEOG(fin)) {
    summaryEl.innerHTML = `${after}, the model would most likely end the text here (${pct(prob)}).`;
    return;
  }
  let text = `${after}, the model's final guess is ${quote(fin)} (${pct(prob)}).`;
  // from the top predictions, not the emergence line, which follows the traced token
  let settle = r.nLayers - 1;
  while (settle > 0 && top(settle - 1) === fin) settle--;
  let first = 0;
  while (first < r.nLayers && top(first) !== fin) first++;
  if (settle === r.nLayers - 1)
    text += first < settle ? ` It first shows up at ${layerName(first)}, but only the last layer commits to it.` : ' Only the last layer arrives at it; the layers below predict other words.';
  else if (settle === 0) text += ' Even the input row already points to it.';
  else text += ` It settles on this answer from ${layerName(settle)} upward${first < settle ? `, after first showing up at ${layerName(first)}` : ''}.`;
  if (prob < 0.2) text += ' The model is unsure, though: other words are almost as likely.';
  summaryEl.innerHTML = text;
}

const layerName = (l: number) => (l === 0 ? 'the input row' : `layer ${layerLabel(l)}`);

/** compare mode: the two final guesses, and in which layers the last columns differ */
function renderPairSummary() {
  const [a, b] = [A.res!, B.res!];
  const top = (r: LogitLensResult, l: number) => r.topIds[(l * r.nTokens + r.nTokens - 1) * r.k];
  const prob = (r: LogitLensResult) => r.topProbs[((r.nLayers - 1) * r.nTokens + r.nTokens - 1) * r.k];
  const guess = (r: LogitLensResult) => (wllama.isTokenEOG(top(r, r.nLayers - 1)) ? 'the end of the text' : quote(top(r, r.nLayers - 1)));
  const n = a.nLayers;
  const fa = top(a, n - 1);
  const fb = top(b, n - 1);
  let text =
    fa === fb
      ? `Both prompts end on the same final guess, ${guess(a)} (${pct(prob(a))} after ${quote(a.tokens[a.nTokens - 1])} in A, ${pct(prob(b))} after ${quote(b.tokens[b.nTokens - 1])} in B).`
      : `After ${quote(a.tokens[a.nTokens - 1])} in A, the final guess is ${guess(a)} (${pct(prob(a))}); after ${quote(b.tokens[b.nTokens - 1])} in B, it is ${guess(b)} (${pct(prob(b))}).`;
  // top words of the two last columns, layer by layer
  const differ = Array.from({ length: n }, (_, l) => top(a, l) !== top(b, l));
  const count = differ.filter(Boolean).length;
  const first = differ.indexOf(true);
  let tail = n - 1; // lowest layer from which all layers above agree (fa === fb) or differ (fa !== fb)
  while (tail > 0 && differ[tail - 1] === differ[n - 1]) tail--;
  if (count === 0) text += ' Their last columns predict the same word in every layer.';
  else if (fa !== fb) text += tail === first ? ` Their last columns differ from ${layerName(tail)} upward.` : ` Their last columns first differ at ${layerName(first)}, and for good from ${layerName(tail)} upward.`;
  else text += ` Their last columns differ in ${count} of ${n} layers, first at ${layerName(first)}, and agree from ${layerName(tail)} upward.`;
  summaryEl.innerHTML = text;
}

function render(animate = false, fromCol = 0) {
  if (!A.res) return;
  const sides = shown();
  const cmp = sides.length === 2;
  renderSummary();
  const mode = colorMode();
  const q = (id: number | null) => (id === null ? '' : quote(id));
  if (mode === 'word' && shadeId !== null) {
    legend.innerHTML = `<span class="item"><span>Probability of ${q(shadeId)}</span><span>0%</span><span class="bar" style="background:${rampGradientCss()}"></span><span>100%</span></span><span class="item"><span class="line"></span><span>From here up, the layer agrees with the final prediction</span></span>`;
  } else if (mode === 'contrast' && shadeId !== null && contrastId !== null) {
    const n = fmtTimes(Math.exp(A.heatmap.diffMax));
    legend.innerHTML = `<span class="item ends"><span>${q(contrastId)} ${n}× more likely</span><span class="bar" style="background:${divergingGradientCss()}"></span><span>${q(shadeId)} ${n}× more likely</span></span><span class="item"><span>gray: equally likely</span></span>`;
  }
  const scale =
    mode === 'prob'
      ? `<span>How sure the top word is (probability) ${info('probability')}</span><span>0%</span><span class="bar" style="background:${rampGradientCss()}"></span><span>100%</span>`
      : mode === 'entropy'
        ? `<span>How spread out the guesses are (entropy) ${info('entropy')}</span><span>0</span><span class="bar" style="background:${rampGradientCss()}"></span><span>${A.heatmap.maxEntropy.toFixed(1)} nats</span>`
        : `<span>Difference from the final layer (KL) ${info('kl')}</span><span>0</span><span class="bar" style="background:${rampGradientCss(true)}"></span><span>${A.heatmap.maxKl.toFixed(1)} nats</span>`;
  if (mode === 'prob' || mode === 'entropy' || mode === 'kl') legend.innerHTML = `<span class="item">${scale}</span><span class="item"><span class="line"></span><span>From here up, the layer agrees with the final prediction</span></span>`;
  if (cmp && A.heatmap.diffFrom !== null) legend.innerHTML += `<span class="item"><span class="diffmark"></span><span>From here on, the prompts differ</span></span>`;
  // captions name the prompt each grid belongs to, as it was run
  const cap = (s: Side) => `<b>${s.name}</b><span class="tok">${esc(s.prompt)}</span>`;
  capA.hidden = capB.hidden = rowB.hidden = !cmp;
  if (cmp) {
    capA.innerHTML = cap(A);
    capB.innerHTML = cap(B);
  }
  const table = viewMode() === 'table';
  heatmapWrap.hidden = table;
  tableWrap.hidden = !table;
  if (table) {
    tableWrap.innerHTML = sides.map((s) => (cmp ? `<p class="hm-cap">${cap(s)}</p>` : '') + tableHtml(s.res!)).join('');
    return;
  }
  for (const s of sides) {
    const opts = { mode, selected: selected?.side === s ? selected : null };
    if (animate) s.heatmap.reveal(opts, fromCol);
    else s.heatmap.draw(opts);
  }
}

function tableHtml(r: LogitLensResult): string {
  let html = '<table class="lens-table"><thead><tr><th scope="col">Layer</th>';
  for (let t = 0; t < r.nTokens; t++) html += `<th scope="col" class="tok">${esc(piece(r.tokens[t]))}</th>`;
  html += '</tr></thead><tbody>';
  for (let l = r.nLayers - 1; l >= 0; l--) {
    html += `<tr><th scope="row">${layerLabel(l)}</th>`;
    for (let t = 0; t < r.nTokens; t++) {
      const c = (l * r.nTokens + t) * r.k;
      html += `<td><span class="tok">${esc(piece(r.topIds[c]))}</span><span class="p">${pct(r.topProbs[c])}</span></td>`;
    }
    html += '</tr>';
  }
  return html + '</tbody></table>';
}

function wordLines(s: Side, cell: number): string {
  let out = '';
  const w = (x: WordShade) => `<span class="tok">${esc(piece(x.id))}</span>`;
  if (s.word) out += `<br>${w(s.word)}: rank ${s.word.ranks[cell] + 1}, ${pct(s.word.probs[cell])}`;
  if (s.word && s.contrast) {
    const d = Math.log(Math.max(s.word.probs[cell], 1e-30)) - Math.log(Math.max(s.contrast.probs[cell], 1e-30));
    const [hi, lo] = d >= 0 ? [s.word, s.contrast] : [s.contrast, s.word];
    out += `<br>${w(hi)} is ${fmtTimes(Math.exp(Math.abs(d)))}× more likely than ${w(lo)}`;
  }
  return out;
}

function showTooltip(s: Side, layer: number, pos: number, x: number, y: number) {
  const r = s.res!;
  const cell = layer * r.nTokens + pos;
  let rows = '';
  for (let j = 0; j < r.k; j++) {
    const i = cell * r.k + j;
    rows += `<tr><td class="tok">${esc(piece(r.topIds[i]))}</td><td class="num">${pct(r.topProbs[i])}</td></tr>`;
  }
  // compare mode: the other prompt's top word in the same cell
  const o = comparing() ? (s === A ? B : A) : null;
  const other = o && pos < o.res!.nTokens ? o.res! : null;
  const oc = other ? (layer * other.nTokens + pos) * other.k : 0;
  tooltip.innerHTML =
    `<div class="tt-head">${o ? `${s.name} · ` : ''}Layer ${layerLabel(layer)}, after <span class="tok">${esc(piece(r.tokens[pos]))}</span></div>` +
    `<table>${rows}</table>` +
    `<div class="tt-foot">Entropy ${r.entropy[cell].toFixed(2)} nats, KL to final layer ${r.kl[cell].toFixed(2)} nats<br>${r.targetIds[cell] === r.topIds[((r.nLayers - 1) * r.nTokens + pos) * r.k] ? 'Final prediction' : 'Traced word'} <span class="tok">${esc(piece(r.targetIds[cell]))}</span>: rank ${r.targetRanks[cell] + 1}, ${pct(r.targetProbs[cell])}${wordLines(s, cell)}` +
    `${other ? `<br>Prompt ${o!.name} here, after <span class="tok">${esc(piece(other.tokens[pos]))}</span>: <span class="tok">${esc(piece(other.topIds[oc]))}</span> ${pct(other.topProbs[oc])}` : ''}</div>`;
  tooltip.hidden = false;
  const tw = tooltip.offsetWidth;
  const th = tooltip.offsetHeight;
  tooltip.style.left = `${Math.max(8, Math.min(window.innerWidth - tw - 8, x + 14))}px`;
  tooltip.style.top = `${y + th + 24 > window.innerHeight ? y - th - 14 : y + 14}px`;
}

function trace(s: Side, pos: number) {
  const r = s.res!;
  traced = { side: s, pos };
  traceEl.hidden = false;
  const pts: TracePoint[] = [];
  for (let l = 0; l < r.nLayers; l++) {
    const cell = l * r.nTokens + pos;
    pts.push({ layer: l, rank: r.targetRanks[cell], prob: r.targetProbs[cell] });
  }
  const target = r.targetIds[pos];
  const where = comparing() ? ` in prompt ${s.name}` : '';
  traceDesc.innerHTML = `Where <span class="tok">${esc(piece(target))}</span> ranks among each layer's guesses after <span class="tok">${esc(piece(r.tokens[pos]))}</span> (position ${pos}${where}); rank 1 is the top guess. Pick another word to follow, from the final layer or the selected cell:`;
  const last = (r.nLayers - 1) * r.nTokens + pos;
  const cands = Array.from(r.topIds.slice(last * r.k, last * r.k + r.k));
  if (selected?.side === s) {
    const top = r.topIds[(selected.layer * r.nTokens + pos) * r.k];
    if (!cands.includes(top)) cands.push(top);
  }
  traceChips.replaceChildren(
    ...cands.map((id) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'tok';
      b.textContent = piece(id);
      b.setAttribute('aria-pressed', String(id === target));
      b.onclick = () => retarget(s, pos, id);
      return b;
    })
  );
  drawTrace(tracePlot, pts, tooltip, piece(target));

  const quoted = (id: number) => `“${showPiece(pieces.get(id)).trim() || piece(id)}”`;
  const actions: HTMLButtonElement[] = [];
  const action = (text: string, fn: () => void) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'btn btn-quiet';
    b.textContent = text;
    b.onclick = fn;
    actions.push(b);
  };
  const grid = comparing() ? 'both grids' : 'the whole grid';
  if (shadeId !== target) action(`Shade ${grid} by ${quoted(target)}`, () => chooseWord('word', target));
  if (shadeId !== null && shadeId !== target && contrastId !== target) action(`Compare ${quoted(shadeId)} with ${quoted(target)}`, () => chooseWord('contrast', target));
  traceActions.replaceChildren(...actions);
}

async function retarget(s: Side, pos: number, id: number) {
  if (busy || !s.res) return;
  setBusy(true);
  try {
    const old = s.res;
    const targets = Array.from(old.targetIds.slice(0, old.nTokens));
    targets[pos] = id;
    const t0 = performance.now();
    await ensureCached(s);
    const res = await wllama.logitLens({ prompt: '', reuse: true, k: old.k, targetIds: targets });
    res.kl = old.kl; // reuse with targets skips KL, it does not depend on them
    renderTimings([res], performance.now() - t0);
    for (const [k, v] of old.pieces) if (!res.pieces.has(k)) res.pieces.set(k, v);
    s.res = res;
    syncResults();
    render();
    trace(s, pos);
  } catch (e) {
    runStatus.textContent = `Something went wrong: ${(e as Error).message}`;
  } finally {
    setBusy(false);
  }
}

// a tapped tooltip goes away on a real scroll or on a tap anywhere else; opening the trace panel can nudge the scroll position by a few pixels, which must not count
let tapScroll = [0, 0, 0];
// touch has no hover, so a tap also shows the cell's top 5
let lastPointer = 'mouse';
for (const s of [A, B]) {
  const { canvas, heatmap } = s;
  canvas.addEventListener('pointermove', (ev) => {
    const h = s.res && heatmap.hit(ev.clientX, ev.clientY);
    if (h) showTooltip(s, h.layer, h.pos, ev.clientX, ev.clientY);
    else tooltip.hidden = true;
  });
  canvas.addEventListener('pointerleave', () => (tooltip.hidden = true));
  canvas.addEventListener('pointerdown', (ev) => (lastPointer = ev.pointerType));
  canvas.addEventListener('click', (ev) => {
    const h = s.res && heatmap.hit(ev.clientX, ev.clientY);
    if (!h) return;
    selected = { side: s, ...h };
    render();
    trace(s, h.pos);
    if (lastPointer !== 'mouse') {
      showTooltip(s, h.layer, h.pos, ev.clientX, ev.clientY);
      tapScroll = [scrollX, scrollY, heatmapWrap.scrollLeft];
    }
  });
  canvas.addEventListener('keydown', (ev) => {
    const res = s.res;
    if (!res) return;
    const cur = selected?.side === s ? selected : { layer: res.nLayers - 1, pos: 0 };
    const moves: Record<string, [number, number]> = { ArrowUp: [1, 0], ArrowDown: [-1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    if (ev.key in moves) {
      ev.preventDefault();
      const [dl, dp] = moves[ev.key];
      selected = {
        side: s,
        layer: Math.max(0, Math.min(res.nLayers - 1, cur.layer + dl)),
        pos: Math.max(0, Math.min(res.nTokens - 1, cur.pos + dp)),
      };
      render();
      const rect = canvas.getBoundingClientRect();
      showTooltip(s, selected.layer, selected.pos, rect.left + LEFT + (selected.pos + 0.5) * CELL_W, rect.top + TOP + (res.nLayers - selected.layer) * CELL_H);
    } else if (ev.key === 'Enter' && selected?.side === s) {
      trace(s, selected.pos);
    } else if (ev.key === 'Escape') {
      tooltip.hidden = true;
    }
  });
  canvas.addEventListener('blur', () => (tooltip.hidden = true));
}
const hideOnScroll = () => {
  const [x, y, l] = tapScroll;
  if (Math.abs(scrollX - x) + Math.abs(scrollY - y) + Math.abs(heatmapWrap.scrollLeft - l) > 40) tooltip.hidden = true;
};
const hideTooltip = () => (tooltip.hidden = true);
window.addEventListener('scroll', hideOnScroll, { passive: true });
heatmapWrap.addEventListener('scroll', hideOnScroll, { passive: true });
document.addEventListener('pointerdown', (ev) => {
  if (ev.target !== A.canvas && ev.target !== B.canvas) hideTooltip();
});

btnLoad.onclick = btnLoadEmpty.onclick = () => {
  if (!selectedUrl) return;
  if (needsConfirm() && confirmUrl !== selectedUrl) {
    confirmUrl = selectedUrl;
    return syncLoad();
  }
  confirmUrl = null;
  load({ url: selectedUrl });
};

// two clicks: the first arms the button, the second deletes; it disarms itself after a few seconds
let clearTimer = 0;
btnClearCache.onclick = async () => {
  if (!btnClearCache.classList.contains('armed')) {
    btnClearCache.classList.add('armed');
    btnClearCache.textContent = `Delete ${fmtBytes(cachedBytes)}? Click again`;
    clearTimer = window.setTimeout(() => {
      btnClearCache.classList.remove('armed');
      btnClearCache.textContent = 'Delete cached models';
    }, 4000);
    return;
  }
  clearTimeout(clearTimer);
  btnClearCache.classList.remove('armed');
  btnClearCache.textContent = 'Delete cached models';
  btnClearCache.disabled = true;
  try {
    await wllama.modelManager.clear();
    modelStatus.textContent = wllama.isModelLoaded() ? 'Cached models deleted. The loaded model stays usable until you reload the page.' : 'Cached models deleted.';
  } catch (e) {
    modelStatus.textContent = `Could not delete cached models: ${(e as Error).message}`;
  } finally {
    btnClearCache.disabled = false;
    await refreshSelection();
  }
};
fileInput.onchange = () => {
  if (fileInput.files?.length) load({ files: Array.from(fileInput.files) });
};
btnRun.onclick = () => run();
btnContinue.onclick = () => continueRun(1);
btnContinue5.onclick = () => continueRun(5);
useChat.onchange = () => {
  reply = '';
  syncChat();
  updateUrl();
  if (A.res && wllama.isModelLoaded()) run();
};

function syncCompare() {
  fieldB.hidden = !useCompare.checked;
  promptsEl.classList.toggle('comparing', useCompare.checked);
}
useCompare.onchange = () => setCompare(true);
/** after the compare box changed; byHand: focus the second prompt to edit it */
function setCompare(byHand: boolean) {
  // the second grid is shown again only after a run with both prompts
  Object.assign(B, { res: null, word: null, contrast: null });
  if (traced?.side === B) {
    traced = null;
    traceEl.hidden = true;
  }
  if (selected?.side === B) selected = null;
  if (useCompare.checked) {
    // a minimal pair starts from a copy of the first prompt; Continue and its reply are for one prompt
    if (!promptB.value) promptB.value = promptEl.value;
    reply = '';
    if (byHand) {
      promptB.focus();
      promptB.setSelectionRange(promptB.value.length, promptB.value.length);
      runStatus.textContent = 'Edit the second prompt, then run the lens on both.';
    }
  } else if (byHand) runStatus.textContent = idleHint;
  syncCompare();
  setBusy(busy);
  updateUrl();
  if (A.res) {
    syncResults();
    render();
  }
  if (traced) trace(traced.side, traced.pos);
}
promptB.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) run();
});
promptEl.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) run();
});
document.querySelectorAll('input[name=color], input[name=view]').forEach((e) => e.addEventListener('change', () => render()));
// canvas and svg colors are read at draw time, so redraw on any theme change
function redrawForTheme() {
  render();
  if (traced && !traceEl.hidden) trace(traced.side, traced.pos);
}
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawForTheme);

function applyTheme(theme: string) {
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  try {
    if (theme === 'auto') localStorage.removeItem('theme');
    else localStorage.setItem('theme', theme);
  } catch {
    // storage unavailable, the choice lasts for this page only
  }
  redrawForTheme();
}
const themeInputs = document.querySelectorAll<HTMLInputElement>('input[name=theme]');
const savedTheme = document.documentElement.dataset.theme ?? 'auto';
themeInputs.forEach((input) => {
  input.checked = input.value === savedTheme;
  input.addEventListener('change', () => applyTheme(input.value));
});

if (!wllama.isSupportWebGPU()) {
  useGpu.checked = false;
  useGpu.disabled = true;
  modelStatus.textContent = 'This browser cannot use the graphics card (no WebGPU), so the model runs on the main processor. That is slower, but works.';
}
// the options disclosure remembers its state per browser
try {
  modelOptions.open = localStorage.getItem('modelOptions') === 'open';
} catch {
  // storage unavailable, start closed
}
modelOptions.addEventListener('toggle', () => {
  try {
    localStorage.setItem('modelOptions', modelOptions.open ? 'open' : 'closed');
  } catch {
    // storage unavailable, the state lasts for this page only
  }
});

examplesEl.replaceChildren(
  ...EXAMPLES.map((ex) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = ex.label;
    b.title = ex.prompt;
    b.setAttribute('aria-pressed', 'false');
    b.onclick = () => {
      promptEl.value = ex.prompt;
      reply = '';
      // a pair turns comparison on, a single prompt turns it off
      if (useCompare.checked !== !!ex.prompt2) {
        useCompare.checked = !!ex.prompt2;
        setCompare(false);
      }
      if (ex.prompt2) promptB.value = ex.prompt2;
      syncChat();
      examplesEl.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      exampleNote.textContent = ex.note;
      updateUrl();
      if (wllama.isModelLoaded()) run();
      else exampleNote.textContent += ' Load a model, then run the lens.';
    };
    return b;
  })
);
// editing a prompt by hand leaves the example
promptB.addEventListener('input', () => {
  examplesEl.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
  exampleNote.textContent = '';
});
promptEl.addEventListener('input', () => {
  // a new user message starts a new reply
  if (reply) {
    reply = '';
    syncChat();
    updateUrl();
  }
  examplesEl.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
  exampleNote.textContent = '';
});

// set by the deploy workflow for non-main branches
const previewBranch = import.meta.env.VITE_BRANCH as string | undefined;
if (previewBranch) {
  const badge = $<HTMLParagraphElement>('preview-badge');
  badge.innerHTML = `Preview of branch ${esc(previewBranch)} &middot; <a href="https://logitlens.gdkrmr.com/">main site</a>`;
  badge.hidden = false;
  document.title = `[${previewBranch}] ${document.title}`;
}

runStatus.textContent = idleHint;
initGlossary();
// desktop shows the full timing strip; phones start with the one-line summary
timingsBox.open = window.matchMedia('(min-width: 861px)').matches;

const promptParam = urlParams.get('prompt');
if (promptParam !== null) promptEl.value = promptParam;
// checked but disabled until a model with a chat template is loaded
useChat.checked = urlParams.get('chat') === '1';
const prompt2Param = urlParams.get('prompt2');
useCompare.checked = prompt2Param !== null;
promptB.value = prompt2Param ?? '';
syncCompare();
reply = (useChat.checked && !useCompare.checked && urlParams.get('reply')) || '';

refreshCache().then(async () => {
  await search('');
  // preselect so a first visit is one click; the default's file list ships with the app
  await selectRepo(shownRepos.find((r) => r.id === (directUrl ? DIRECT_ID : DEFAULT_REPO))!);
});
