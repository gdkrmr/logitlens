import '@fontsource-variable/martian-mono/wdth.css';
import '@fontsource-variable/atkinson-hyperlegible-next/wght.css';
import './style.css';
import { Wllama, type LogitLensResult } from '@wllama/wllama';
import { Heatmap, rampGradientCss, showPiece, layerLabel, CELL_W, CELL_H, LEFT, TOP, type ColorMode } from './heatmap';
import { drawTrace, type TracePoint } from './trace';
import { SUPPORTED_ARCHS, MAX_FILE_BYTES, searchRepos, getRepo, listGgufFiles, isSupported, pickDefaultFile, quantLabel, fmtBytes, fmtParams, fmtCount, DEFAULT_REPO, PRESET_FILES, type RepoInfo } from './hf';

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
const directParam = new URLSearchParams(location.search).get('model');
const directUrl = directParam ? new URL(directParam, location.href).href : null;
const DIRECT_ID = '(direct URL)';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const searchInput = $<HTMLInputElement>('model-search');
const repoMeta = $<HTMLDivElement>('repo-meta');
const repoList = $<HTMLDivElement>('repo-list');
const quantBlock = $<HTMLDivElement>('quant-block');
const quantList = $<HTMLDivElement>('quant-list');
const quantRepo = $<HTMLSpanElement>('quant-repo');
const emptyEl = $<HTMLElement>('empty');
const timingsEl = $<HTMLElement>('timings');
const btnLoad = $<HTMLButtonElement>('btn-load');
const btnLoadEmpty = $<HTMLButtonElement>('btn-load-empty');
const emptyTitle = $<HTMLParagraphElement>('empty-title');
const fileInput = $<HTMLInputElement>('file-input');
const useGpu = $<HTMLInputElement>('use-gpu');
const modelStatus = $<HTMLSpanElement>('model-status');
const progress = $<HTMLProgressElement>('progress');
const promptEl = $<HTMLTextAreaElement>('prompt');
const btnRun = $<HTMLButtonElement>('btn-run');
const runStatus = $<HTMLSpanElement>('run-status');
const resultEl = $<HTMLElement>('result');
const legend = $<HTMLDivElement>('legend');
const canvas = $<HTMLCanvasElement>('heatmap');
const heatmapWrap = $<HTMLDivElement>('heatmap-wrap');
const tableWrap = $<HTMLDivElement>('table-wrap');
const tooltip = $<HTMLDivElement>('tooltip');
const traceEl = $<HTMLElement>('trace');
const traceDesc = $<HTMLParagraphElement>('trace-desc');
const traceChips = $<HTMLDivElement>('trace-chips');
const tracePlot = $<HTMLDivElement>('trace-plot');

const WASM = { default: new URL('wasm/wllama.wasm', document.baseURI).href };
let wllama = new Wllama(WASM, { suppressNativeLog: true });
const heatmap = new Heatmap(canvas);
let res: LogitLensResult | null = null;
let selected: { layer: number; pos: number } | null = null;
let tracedPos: number | null = null;
let busy = false;

const colorMode = (): ColorMode => (document.querySelector<HTMLInputElement>('input[name=color]:checked')!.value as ColorMode);
const viewMode = () => document.querySelector<HTMLInputElement>('input[name=view]:checked')!.value;
const pct = (p: number) => (p >= 0.995 ? '100%' : p < 0.001 ? '<0.1%' : `${(p * 100).toFixed(1)}%`);
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
const piece = (id: number) => showPiece(res?.pieces.get(id));

/////////////////////////////////////////////
// model search

let cachedUrls = new Set<string>();
let shownRepos: RepoInfo[] = [];
let selectedRepo: string | null = null;
let searchCtl: AbortController | null = null;
let filesCtl: AbortController | null = null;
let selectedUrl = '';
let selectedLabel = '';

// both load buttons follow the selected file; the empty-state one is the first-visit shortcut
function syncLoad() {
  const loaded = wllama.isModelLoaded();
  btnLoad.disabled = busy || !selectedUrl;
  btnLoadEmpty.disabled = busy || !selectedUrl;
  btnLoadEmpty.hidden = loaded || !selectedUrl;
  btnLoadEmpty.textContent = `Load ${selectedLabel}`;
  emptyTitle.textContent = loaded ? 'Model loaded. Run the lens to see the readout.' : 'Load a model to start.';
}

async function refreshCache() {
  try {
    cachedUrls = new Set((await wllama.modelManager.getModels()).map((m) => m.url));
  } catch {
    // OPFS unavailable, no cache status
    cachedUrls = new Set();
  }
}

const repoHasCache = (id: string) => [...cachedUrls].some((u) => u.startsWith(`https://huggingface.co/${id}/`));

function renderRepos() {
  repoList.replaceChildren(
    ...shownRepos.map((r) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('aria-pressed', String(r.id === selectedRepo));
      const supported = r.id === DIRECT_ID || isSupported(r);
      const [owner, name] = r.id === DIRECT_ID ? ['', directUrl!.split('/').pop()!] : r.id.split('/');
      const meta: string[] = [];
      if (r.id === DIRECT_ID) meta.push('direct URL');
      if (r.params && r.arch !== 'clip') meta.push(`${fmtParams(r.params)} params`);
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
    const usable = found.filter((r) => isSupported(r) && !r.gated);
    // an exact repo id is always listed, the architecture tag can be wrong for repos with projector files
    if (exact && !exact.gated && !usable.some((r) => r.id === exact.id)) usable.unshift(exact);
    shownRepos = usable;
    const hidden = found.length - usable.filter((r) => found.includes(r)).length;
    repoMeta.textContent = usable.length
      ? `${usable.length} matching ${usable.length === 1 ? 'repo' : 'repos'}${hidden ? `, ${hidden} hidden (unsupported architecture or gated)` : ''}. Supported: ${SUPPORTED_ARCHS.join(', ')}.`
      : `No supported repos found${hidden ? ` (${hidden} hidden: unsupported architecture or gated)` : ''}. Supported: ${SUPPORTED_ARCHS.join(', ')}.`;
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
  renderRepos();
  filesCtl?.abort();
  const ctl = (filesCtl = new AbortController());
  syncLoad();
  quantBlock.hidden = false;
  quantRepo.textContent = r.id === DIRECT_ID ? '' : `in ${r.id}`;
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
          syncLoad();
        };
        const size = Number.isNaN(f.size) ? 'local' : `${fmtBytes(f.size)}${f.shards > 1 ? ` \u00b7 ${f.shards} parts` : ''}${tooBig ? ' \u00b7 too large' : ''}`;
        label.dataset.label = r.id === DIRECT_ID ? f.path.replace(/\.gguf$/i, '') : `${repoName} \u00b7 ${quantLabel(f.path)} \u00b7 ${fmtBytes(f.size)}`;
        label.append(input);
        label.insertAdjacentHTML('beforeend', `<span class="q">${esc(quantLabel(f.path))}</span><span class="s${cachedUrls.has(f.url) ? ' cached' : ''}">${size}</span>`);
        return label;
      })
    );
    selectedUrl = def?.url ?? '';
    selectedLabel = quantList.querySelector<HTMLLabelElement>('label:has(input:checked)')?.dataset.label ?? '';
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
  syncLoad();
  fileInput.disabled = b;
  btnRun.disabled = b || !wllama.isModelLoaded();
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

function renderTimings(r: LogitLensResult, totalMs: number) {
  const other = Math.max(0, totalMs - r.tDecodeMs - r.tProjectMs);
  const items: [string, string, string?][] = [];
  if (loadMs !== null) items.push(['Model load', fmtMs(loadMs)]);
  items.push(['Forward pass', r.tDecodeMs > 0 ? fmtMs(r.tDecodeMs) : 'reused']);
  items.push(['Projection', fmtMs(r.tProjectMs)]);
  items.push(['Other', fmtMs(other)]);
  items.push(['Lens total', fmtMs(totalMs), 'total']);
  timingsEl.innerHTML =
    items.map(([k, v, cls]) => `<div${cls ? ` class="${cls}"` : ''}><dt>${k}</dt><dd>${v}</dd></div>`).join('') +
    `<div class="note">${r.nTokens} tokens \u00d7 ${r.nLayers} layers on ${esc(r.backend.replace(/^(\w+): \1$/, '$1'))}</div>`;
  timingsEl.hidden = false;
}

async function load(source: { url: string } | { files: File[] }) {
  setBusy(true);
  progress.hidden = false;
  progress.value = 0;
  try {
    if (wllama.isModelLoaded()) {
      await wllama.exit();
      wllama = new Wllama(WASM, { suppressNativeLog: true });
    }
    let phase = 'Loading';
    const stop = ticker(modelStatus, () => phase);
    const params = {
      n_ctx: 256,
      n_gpu_layers: useGpu.checked ? 999 : 0,
      progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
        progress.value = total ? loaded / total : 0;
        phase = loaded < total ? `Downloading ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB,` : 'Loading';
      },
    };
    try {
      if ('url' in source) {
        await wllama.loadModelFromUrl(source.url, params);
      } else {
        progress.hidden = true;
        await wllama.loadModel(source.files, params);
      }
    } finally {
      loadMs = stop();
    }
    const meta = wllama.getModelMetadata();
    const info = wllama.getLoadedContextInfo();
    const arch = meta.meta['general.architecture'] ?? '?';
    const name = meta.meta['general.name'] ?? ('url' in source ? source.url.split('/').pop() : source.files[0].name);
    modelStatus.textContent = `${name} (${arch}, ${info.n_layer} layers) loaded in ${fmtMs(loadMs!)}, ${wllama.isMultithread() ? `${wllama.getNumThreads()} threads` : 'single thread'}.`;
  } catch (e) {
    modelStatus.textContent = `Load failed: ${(e as Error).message}`;
  } finally {
    progress.hidden = true;
    setBusy(false);
    await refreshCache();
    renderRepos();
    const keep = selectedUrl;
    const repo = shownRepos.find((r) => r.id === selectedRepo);
    if (repo) await selectRepo(repo);
    const input = quantList.querySelector<HTMLInputElement>(`input[value="${CSS.escape(keep)}"]`);
    if (input) {
      input.checked = true;
      selectedUrl = keep;
      selectedLabel = input.closest('label')!.dataset.label!;
    }
    syncLoad();
  }
}

async function run() {
  if (busy) return;
  setBusy(true);
  const stop = ticker(runStatus, () => 'Running');
  try {
    let ms: number;
    try {
      res = await wllama.logitLens({ prompt: promptEl.value, k: 5 });
    } finally {
      ms = stop();
    }
    heatmap.setResult(res);
    selected = null;
    traceEl.hidden = true;
    resultEl.hidden = false;
    emptyEl.hidden = true;
    runStatus.textContent = 'Ctrl+Enter runs the lens again.';
    renderTimings(res, ms);
    await document.fonts.load('500 condensed 12px "Martian Mono Variable"').catch(() => {});
    render(true);
  } catch (e) {
    runStatus.textContent = `Error: ${(e as Error).message}`;
  } finally {
    setBusy(false);
  }
}

function render(animate = false) {
  if (!res) return;
  const mode = colorMode();
  const scale =
    mode === 'prob'
      ? `<span>Top-1 probability</span><span>0%</span><span class="bar" style="background:${rampGradientCss()}"></span><span>100%</span>`
      : `<span>Entropy</span><span>0</span><span class="bar" style="background:${rampGradientCss()}"></span><span>${heatmap.maxEntropy.toFixed(1)} nats</span>`;
  legend.innerHTML = `<span class="item">${scale}</span><span class="item"><span class="line"></span><span>From here up, the layer agrees with the final prediction</span></span>`;
  const table = viewMode() === 'table';
  heatmapWrap.hidden = table;
  tableWrap.hidden = !table;
  if (table) renderTable();
  else if (animate) heatmap.reveal({ mode, selected });
  else heatmap.draw({ mode, selected });
}

function renderTable() {
  const r = res!;
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
  tableWrap.innerHTML = html + '</tbody></table>';
}

function showTooltip(layer: number, pos: number, x: number, y: number) {
  const r = res!;
  const cell = layer * r.nTokens + pos;
  let rows = '';
  for (let j = 0; j < r.k; j++) {
    const i = cell * r.k + j;
    rows += `<tr><td class="tok">${esc(piece(r.topIds[i]))}</td><td class="num">${pct(r.topProbs[i])}</td></tr>`;
  }
  tooltip.innerHTML =
    `<div class="tt-head">Layer ${layerLabel(layer)}, after <span class="tok">${esc(piece(r.tokens[pos]))}</span></div>` +
    `<table>${rows}</table>` +
    `<div class="tt-foot">Entropy ${r.entropy[cell].toFixed(2)} nats<br>Final prediction <span class="tok">${esc(piece(r.targetIds[cell]))}</span>: rank ${r.targetRanks[cell] + 1}, ${pct(r.targetProbs[cell])}</div>`;
  tooltip.hidden = false;
  const tw = tooltip.offsetWidth;
  const th = tooltip.offsetHeight;
  tooltip.style.left = `${Math.max(8, Math.min(window.innerWidth - tw - 8, x + 14))}px`;
  tooltip.style.top = `${y + th + 24 > window.innerHeight ? y - th - 14 : y + 14}px`;
}

function trace(pos: number) {
  const r = res!;
  tracedPos = pos;
  traceEl.hidden = false;
  const pts: TracePoint[] = [];
  for (let l = 0; l < r.nLayers; l++) {
    const cell = l * r.nTokens + pos;
    pts.push({ layer: l, rank: r.targetRanks[cell], prob: r.targetProbs[cell] });
  }
  const target = r.targetIds[pos];
  traceDesc.innerHTML = `Rank of <span class="tok">${esc(piece(target))}</span> at position ${pos} (after <span class="tok">${esc(piece(r.tokens[pos]))}</span>) across layers. Pick another candidate from the final layer or the selected cell:`;
  const last = (r.nLayers - 1) * r.nTokens + pos;
  const cands = Array.from(r.topIds.slice(last * r.k, last * r.k + r.k));
  if (selected) {
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
      b.onclick = () => retarget(pos, id);
      return b;
    })
  );
  drawTrace(tracePlot, pts, tooltip, piece(target));
}

async function retarget(pos: number, id: number) {
  if (busy || !res) return;
  setBusy(true);
  try {
    const targets = Array.from(res.targetIds.slice(0, res.nTokens));
    targets[pos] = id;
    const pieces = res.pieces;
    const t0 = performance.now();
    res = await wllama.logitLens({ prompt: '', reuse: true, k: res.k, targetIds: targets });
    renderTimings(res, performance.now() - t0);
    for (const [k, v] of pieces) if (!res.pieces.has(k)) res.pieces.set(k, v);
    heatmap.setResult(res);
    render();
    trace(pos);
  } catch (e) {
    runStatus.textContent = `Error: ${(e as Error).message}`;
  } finally {
    setBusy(false);
  }
}

canvas.addEventListener('pointermove', (ev) => {
  const h = heatmap.hit(ev.clientX, ev.clientY);
  if (h) showTooltip(h.layer, h.pos, ev.clientX, ev.clientY);
  else tooltip.hidden = true;
});
canvas.addEventListener('pointerleave', () => (tooltip.hidden = true));
canvas.addEventListener('click', (ev) => {
  const h = heatmap.hit(ev.clientX, ev.clientY);
  if (!h) return;
  selected = h;
  render();
  trace(h.pos);
});
canvas.addEventListener('keydown', (ev) => {
  if (!res) return;
  const s = selected ?? { layer: res.nLayers - 1, pos: 0 };
  const moves: Record<string, [number, number]> = { ArrowUp: [1, 0], ArrowDown: [-1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  if (ev.key in moves) {
    ev.preventDefault();
    const [dl, dp] = moves[ev.key];
    selected = {
      layer: Math.max(0, Math.min(res.nLayers - 1, s.layer + dl)),
      pos: Math.max(0, Math.min(res.nTokens - 1, s.pos + dp)),
    };
    render();
    const rect = canvas.getBoundingClientRect();
    showTooltip(selected.layer, selected.pos, rect.left + LEFT + (selected.pos + 0.5) * CELL_W, rect.top + TOP + (res.nLayers - selected.layer) * CELL_H);
  } else if (ev.key === 'Enter' && selected) {
    trace(selected.pos);
  } else if (ev.key === 'Escape') {
    tooltip.hidden = true;
  }
});
canvas.addEventListener('blur', () => (tooltip.hidden = true));

btnLoad.onclick = btnLoadEmpty.onclick = () => selectedUrl && load({ url: selectedUrl });
fileInput.onchange = () => {
  if (fileInput.files?.length) load({ files: Array.from(fileInput.files) });
};
btnRun.onclick = run;
promptEl.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) run();
});
document.querySelectorAll('input[name=color], input[name=view]').forEach((e) => e.addEventListener('change', () => render()));
// canvas and svg colors are read at draw time, so redraw on any theme change
function redrawForTheme() {
  render();
  if (res && tracedPos !== null && !traceEl.hidden) trace(tracedPos);
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
  modelStatus.textContent = 'WebGPU is not available in this browser, the lens will run on the CPU.';
}
refreshCache().then(async () => {
  await search('');
  // preselect so a first visit is one click; the default's file list ships with the app
  await selectRepo(shownRepos.find((r) => r.id === (directUrl ? DIRECT_ID : DEFAULT_REPO))!);
});
