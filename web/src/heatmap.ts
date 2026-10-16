import type { LogitLensResult } from '@wllama/wllama';


export type ColorMode = 'prob' | 'entropy' | 'kl' | 'word' | 'contrast';

// sequential blue, light -> dark
const BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];

const hexToRgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

export function isDark(): boolean {
  const t = document.documentElement.dataset.theme;
  if (t === 'dark') return true;
  if (t === 'light') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

// diverging red (second word) <- gray -> blue (first word); validated poles, equal steps per arm
const DIVERGING_LIGHT = ['#962221', '#d83a39', '#ec7f7c', '#f4b8b5', '#f0efec', '#b7d3f6', '#6da7ec', '#2a78d6', '#184f95'];
const DIVERGING_DARK = ['#ee8a86', '#bb625f', '#8a4644', '#5c3a39', '#383835', '#344a66', '#3d6597', '#5585c9', '#6da7ec'];

/** v in [-1,1] -> color; negative favors the second word, positive the first */
export function divergingColor(v: number): { fill: string; ink: string } {
  return interpolate(isDark() ? DIVERGING_DARK : DIVERGING_LIGHT, (Math.max(-1, Math.min(1, v)) + 1) / 2);
}

export function divergingGradientCss(): string {
  return `linear-gradient(to right, ${(isDark() ? DIVERGING_DARK : DIVERGING_LIGHT).join(',')})`;
}

/** t in [0,1] -> color; low values recede toward the surface in both themes */
export function rampColor(t: number): { fill: string; ink: string } {
  return interpolate(isDark() ? [...BLUE].reverse() : BLUE, t);
}

function interpolate(stops: string[], t: number): { fill: string; ink: string } {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = hexToRgb(stops[i]);
  const b = hexToRgb(stops[i + 1]);
  const c = a.map((v, j) => Math.round(v + (b[j] - v) * f));
  // pick ink by luminance of the fill
  const lum = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
  return { fill: `rgb(${c[0]},${c[1]},${c[2]})`, ink: lum > 0.55 ? '#0b0b0b' : '#ffffff' };
}

/** reversed: strong color on the left, for scales where low values matter (KL) */
export function rampGradientCss(reversed = false): string {
  const stops = isDark() !== reversed ? [...BLUE].reverse() : BLUE;
  return `linear-gradient(to right, ${stops.join(',')})`;
}

/** printable form of a token piece */
export function showPiece(s: string | undefined): string {
  if (s === undefined) return '?';
  if (s.length > 0 && s.trim() === '' && !/[\n\t\r]/.test(s)) return '\u2423'.repeat(s.length);
  return s.replace(/\n/g, '\\n').replace(/\t/g, '\\t').replace(/\r/g, '\\r');
}

export const layerLabel = (l: number) => (l === 0 ? 'emb' : String(l - 1));

/** MTP stages, bottom to top: short axis labels and full names */
export const MTP_STAGE_LABELS = ['proj', 'attn', 'MTP'];
export const MTP_STAGE_NAMES = ['after eh_proj', 'after attention', 'MTP output'];

export interface HeatmapOpts {
  mode: ColorMode;
  selected: { layer: number; pos: number } | null;
}

export const CELL_W = 84;
export const CELL_H = 22;
// the layer axis is a separate sticky canvas, AXIS_W wide; LEFT is the cell canvas's own margin
export const AXIS_W = 40;
export const LEFT = 2;
export const TOP = 8;
const BOTTOM = 30;
// space between the MTP rows and the final layer
const MTP_GAP = 9;
const FONT = '"Martian Mono Variable", ui-monospace, monospace';

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export class Heatmap {
  private ctx: CanvasRenderingContext2D;
  private axisCtx: CanvasRenderingContext2D;
  res: LogitLensResult | null = null;
  maxEntropy = 1;
  maxKl = 1;
  // per position: lowest layer from which the top token equals the target all the way up, or null
  emergence: (number | null)[] = [];
  private anim = 0;

  constructor(
    private canvas: HTMLCanvasElement,
    private axis: HTMLCanvasElement
  ) {
    this.ctx = canvas.getContext('2d')!;
    this.axisCtx = axis.getContext('2d')!;
  }

  /**
   * MTP rows above the final layer, one per stage of the NextN block (0 without it).
   * Row layer nLayers + s is stage s: 0 after eh_proj, 1 after attention, 2 the block output.
   */
  get mtpRows(): number {
    return this.res && this.res.nMtp > 0 ? this.res.nMtp : 0;
  }

  /** the block output stage, used by the summary */
  get mtpOut(): number {
    return this.mtpRows - 1;
  }

  /**
   * MTP cell drawn in column t: stage s made at position t - 1, which guesses the token after t,
   * the same token as every layer in column t; -1 for the first column
   */
  mtpCell(t: number, s = this.mtpOut): number {
    const r = this.res!;
    return t < 1 ? -1 : s * r.nTokens + t - 1;
  }

  /** draft columns after the prompt: the MTP block chains its own guesses, only the MTP rows have cells there */
  get draftCols(): number {
    return this.mtpRows && this.res!.nDraft > 0 ? this.res!.nDraft : 0;
  }

  /** columns drawn: prompt plus draft */
  get nCols(): number {
    return this.res ? this.res.nTokens + this.draftCols : 0;
  }

  /** MTP cell of stage s in column t, from the prompt run or, past the prompt, from the draft; null for an empty cell */
  mtpAt(t: number, s = this.mtpOut): { ids: Int32Array; probs: Float32Array; entropy: Float32Array; c: number } | null {
    const r = this.res!;
    if (t >= r.nTokens) return t < this.nCols ? { ids: r.draftTopIds, probs: r.draftTopProbs, entropy: r.draftEntropy, c: s * r.nDraft + t - r.nTokens } : null;
    const c = this.mtpCell(t, s);
    return c < 0 || r.mtpTopIds[c * r.k] < 0 ? null : { ids: r.mtpTopIds, probs: r.mtpTopProbs, entropy: r.mtpEntropy, c };
  }

  /** input token of column t: the prompt token, or in a draft column the token the draft step reads */
  colToken(t: number): number {
    const r = this.res!;
    return t < r.nTokens ? r.tokens[t] : r.draftTokens[t - r.nTokens];
  }

  /** top of a row; layer nLayers + s is MTP stage s, the output stage on top */
  rowY(layer: number): number {
    const r = this.res!;
    if (layer >= r.nLayers) return TOP + (r.nLayers + this.mtpRows - 1 - layer) * CELL_H;
    return TOP + (this.mtpRows ? this.mtpRows * CELL_H + MTP_GAP : 0) + (r.nLayers - 1 - layer) * CELL_H;
  }

  setResult(res: LogitLensResult) {
    this.res = res;
    let m = 0;
    for (const h of res.entropy) m = Math.max(m, h);
    // one entropy scale for the layers and all MTP stages; the last prompt column has no MTP cell
    for (let c = 0; c < res.nMtp * res.nTokens; c++) if (res.mtpTopIds[c * res.k] >= 0) m = Math.max(m, res.mtpEntropy[c]);
    for (let c = 0; res.nMtp > 0 && c < res.nMtp * res.nDraft; c++) m = Math.max(m, res.draftEntropy[c]);
    this.maxEntropy = m || 1;
    m = 0;
    for (const d of res.kl) m = Math.max(m, d);
    this.maxKl = m || 1;
    this.emergence = [];
    for (let t = 0; t < res.nTokens; t++) {
      const target = res.targetIds[(res.nLayers - 1) * res.nTokens + t];
      let from: number | null = null;
      for (let l = res.nLayers - 1; l >= 0; l--) {
        if (res.topIds[(l * res.nTokens + t) * res.k] !== target) break;
        from = l;
      }
      this.emergence.push(from);
    }
  }

  // per-cell values for the word and contrast modes, computed by re-projection in main.ts
  wordProbs: Float32Array | null = null;
  diff: Float32Array | null = null; // ln(p_a / p_b)
  diffMax = 1;

  setContrast(diff: Float32Array | null) {
    this.diff = diff;
    let m = 0;
    if (diff) for (const d of diff) m = Math.max(m, Math.abs(d));
    // cap at e^10 (about 22000x) so one extreme cell does not wash out the rest
    this.diffMax = Math.min(m, 10) || 1;
  }

  value(cell: number, mode: ColorMode): number {
    const r = this.res!;
    if (mode === 'word') return this.wordProbs?.[cell] ?? 0;
    if (mode === 'contrast') return (this.diff?.[cell] ?? 0) / this.diffMax;
    // log scale, since the embedding row is often far above the rest; KL 0 (same as the output) is strongest
    if (mode === 'kl') return 1 - Math.log1p(r.kl[cell]) / Math.log1p(this.maxKl);
    return mode === 'prob' ? r.topProbs[cell * r.k] : r.entropy[cell] / this.maxEntropy;
  }

  color(cell: number, mode: ColorMode) {
    return mode === 'contrast' ? divergingColor(this.value(cell, mode)) : rampColor(this.value(cell, mode));
  }

  /** MTP cells have no KL and no word probabilities, so those modes draw them neutral (null) */
  mtpColor(t: number, s: number, mode: ColorMode): { fill: string; ink: string } | null {
    const m = this.mtpAt(t, s);
    if (!m) return null;
    if (mode === 'prob') return rampColor(m.probs[m.c * this.res!.k]);
    if (mode === 'entropy') return rampColor(m.entropy[m.c] / this.maxEntropy);
    return null;
  }

  /** map pointer position to (layer, pos) */
  hit(clientX: number, clientY: number): { layer: number; pos: number } | null {
    const r = this.res;
    if (!r) return null;
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left - LEFT;
    const y = clientY - rect.top - TOP;
    const pos = Math.floor(x / CELL_W);
    if (x < 0 || pos >= this.nCols) return null;
    const layer = this.rowAt(y);
    // only the MTP rows reach into the draft columns
    if (layer === null || (layer < r.nLayers && pos >= r.nTokens)) return null;
    return { layer, pos };
  }

  /** row under y (relative to TOP), null in the gap or outside */
  private rowAt(y: number): number | null {
    const r = this.res!;
    if (y < 0) return null;
    const top = this.mtpRows ? this.mtpRows * CELL_H + MTP_GAP : 0;
    if (y < this.mtpRows * CELL_H) return r.nLayers + this.mtpRows - 1 - Math.floor(y / CELL_H);
    if (y < top) return null;
    const layer = r.nLayers - 1 - Math.floor((y - top) / CELL_H);
    return layer < 0 ? null : layer;
  }

  /** full name for an axis label under the pointer, for the axis title; the MTP labels are short */
  axisTitle(clientY: number): string {
    const r = this.res;
    if (!r) return '';
    const l = this.rowAt(clientY - this.axis.getBoundingClientRect().top - TOP);
    if (l === null) return '';
    if (l >= r.nLayers) return `MTP block, ${MTP_STAGE_NAMES[l - r.nLayers] ?? l - r.nLayers}`;
    return l === 0 ? 'Embeddings' : `Layer ${layerLabel(l)}`;
  }

  /**
   * draw with a bottom-up reveal: embeddings first, final layer last.
   * Columns before fromCol are drawn complete; continuing passes the new column, since appending a token
   * leaves the earlier positions unchanged (causal model). A running reveal is replaced, not replayed.
   */
  reveal(opts: HeatmapOpts, fromCol = 0) {
    cancelAnimationFrame(this.anim);
    const r = this.res;
    if (!r || reducedMotion()) return this.draw(opts);
    const n = r.nLayers + this.mtpRows;
    const dur = 140 + n * 14;
    const t0 = performance.now();
    const step = (now: number) => {
      const p = Math.min(1, (now - t0) / dur);
      this.draw(opts, p * n, fromCol);
      if (p < 1) this.anim = requestAnimationFrame(step);
    };
    this.anim = requestAnimationFrame(step);
  }

  draw(opts: HeatmapOpts, rows = Infinity, fromCol = 0) {
    const r = this.res;
    if (!r) return;
    if (rows === Infinity) cancelAnimationFrame(this.anim);
    const w = LEFT + this.nCols * CELL_W + 4;
    const h = this.rowY(0) + CELL_H + BOTTOM;
    // canvases wider than 32767 device px stay blank; long prompts trade sharpness for that
    const dpr = Math.min(window.devicePixelRatio || 1, 30000 / (w + AXIS_W));
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.canvas.style.width = `${w}px`;
      this.canvas.style.height = `${h}px`;
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const v = (name: string) => css.getPropertyValue(name).trim();
    const panel = v('--panel');
    this.drawAxis(dpr, h, panel, v('--ink-3'));
    ctx.fillStyle = panel;
    ctx.fillRect(0, 0, w, h);
    ctx.textBaseline = 'middle';

    for (let l = 0; l < r.nLayers; l++) {
      const y = this.rowY(l);
      const alpha = Math.max(0, Math.min(1, rows - l));
      for (let t = 0; t < r.nTokens; t++) {
        // columns before fromCol are already known and shown complete
        ctx.globalAlpha = t < fromCol ? 1 : alpha;
        if (ctx.globalAlpha <= 0) continue;
        const cell = l * r.nTokens + t;
        const x = LEFT + t * CELL_W;
        const { fill, ink } = this.color(cell, opts.mode);
        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.roundRect(x + 1, y + 1, CELL_W - 2, CELL_H - 2, 3);
        ctx.fill();
        ctx.fillStyle = ink;
        ctx.font = `500 condensed 12px ${FONT}`;
        ctx.textAlign = 'left';
        ctx.save();
        ctx.beginPath();
        ctx.rect(x + 4, y, CELL_W - 8, CELL_H);
        ctx.clip();
        ctx.fillText(showPiece(r.pieces.get(r.topIds[cell * r.k])), x + 5, y + CELL_H / 2 + 0.5);
        ctx.restore();
      }
      ctx.globalAlpha = 1;
    }
    if (this.draftCols) this.drawDraftBand(v);
    if (this.mtpRows) this.drawMtp(opts.mode, rows, fromCol, v);

    // input tokens along the bottom
    ctx.font = `400 condensed 12px ${FONT}`;
    ctx.textAlign = 'left';
    for (let t = 0; t < r.nTokens; t++) {
      const x = LEFT + t * CELL_W;
      const y = this.rowY(0) + CELL_H + 15;
      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 2, y - 10, CELL_W - 4, 20);
      ctx.clip();
      ctx.fillStyle = v('--ink-2');
      ctx.fillText(showPiece(r.pieces.get(r.tokens[t])), x + 5, y);
      ctx.restore();
    }
    // draft inputs: x1 is the final prediction, the rest are the MTP block's own guesses
    ctx.font = `italic 400 condensed 12px ${FONT}`;
    for (let t = r.nTokens; t < this.nCols; t++) {
      const x = LEFT + t * CELL_W;
      const y = this.rowY(0) + CELL_H + 15;
      ctx.beginPath();
      ctx.roundRect(x + 1.5, y - 9.5, CELL_W - 3, 19, 3);
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = v('--ink-3');
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 2, y - 10, CELL_W - 4, 20);
      ctx.clip();
      ctx.fillStyle = v('--ink-3');
      ctx.fillText(showPiece(r.pieces.get(this.colToken(t))), x + 5, y);
      ctx.restore();
    }

    // during a reveal the line covers only the complete columns, so it never disappears and comes back
    this.drawEmergence(v('--trace'), panel, rows >= r.nLayers ? r.nTokens : fromCol);

    if (opts.selected) {
      const { layer, pos } = opts.selected;
      ctx.strokeStyle = v('--ink');
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(LEFT + pos * CELL_W + 1, this.rowY(layer) + 1, CELL_W - 2, CELL_H - 2, 3);
      ctx.stroke();
    }
  }

  /** draft columns: a sunk band over the layer rows, with a label, since only the MTP rows have cells there */
  private drawDraftBand(v: (name: string) => string) {
    const r = this.res!;
    const ctx = this.ctx;
    const x0 = LEFT + r.nTokens * CELL_W;
    const y0 = this.rowY(r.nLayers - 1);
    const y1 = this.rowY(0) + CELL_H;
    ctx.fillStyle = v('--sunk');
    ctx.beginPath();
    ctx.roundRect(x0 + 1, y0 + 1, this.draftCols * CELL_W - 2, y1 - y0 - 2, 3);
    ctx.fill();
    ctx.fillStyle = v('--ink-3');
    ctx.font = `500 condensed 11px ${FONT}`;
    ctx.textAlign = 'left';
    for (let j = 0; j < this.draftCols; j++) ctx.fillText(`draft ${j + 1}`, x0 + j * CELL_W + 6, y0 + CELL_H / 2 + 0.5);
  }

  /**
   * the MTP stage rows above a thin rule, block output on top; the reveal goes on upward through the stages.
   * The first cell of each stage stays empty, there is no earlier position to guess from.
   */
  private drawMtp(mode: ColorMode, rows: number, fromCol: number, v: (name: string) => string) {
    const r = this.res!;
    const ctx = this.ctx;
    const ruleY = TOP + this.mtpRows * CELL_H + MTP_GAP / 2;
    ctx.fillStyle = v('--rule');
    ctx.fillRect(LEFT + 1, ruleY - 0.5, this.nCols * CELL_W - 2, 1);
    for (let s = 0; s < this.mtpRows; s++) {
      const y = this.rowY(r.nLayers + s);
      const alpha = Math.max(0, Math.min(1, rows - r.nLayers - s));
      for (let t = 0; t < this.nCols; t++) {
        ctx.globalAlpha = t < fromCol ? 1 : alpha;
        if (ctx.globalAlpha <= 0) continue;
        const x = LEFT + t * CELL_W;
        const m = this.mtpAt(t, s);
        const id = m ? m.ids[m.c * r.k] : -1;
        ctx.beginPath();
        ctx.roundRect(x + 1.5, y + 1.5, CELL_W - 3, CELL_H - 3, 3);
        if (id < 0) {
          ctx.setLineDash([3, 3]);
          ctx.strokeStyle = v('--rule');
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.setLineDash([]);
          continue;
        }
        const col = this.mtpColor(t, s, mode);
        ctx.fillStyle = col ? col.fill : v('--sunk');
        ctx.fill();
        ctx.fillStyle = col ? col.ink : v('--ink-2');
        ctx.font = `500 condensed 12px ${FONT}`;
        ctx.textAlign = 'left';
        ctx.save();
        ctx.beginPath();
        ctx.rect(x + 4, y, CELL_W - 8, CELL_H);
        ctx.clip();
        ctx.fillText(showPiece(r.pieces.get(id)), x + 5, y + CELL_H / 2 + 0.5);
        ctx.restore();
      }
    }
    // each draft column gets one dashed outline around its stages, they come from the block's own earlier guess
    for (let t = r.nTokens; t < this.nCols; t++) {
      ctx.globalAlpha = t < fromCol ? 1 : Math.max(0, Math.min(1, rows - r.nLayers - this.mtpRows + 1));
      if (ctx.globalAlpha <= 0) continue;
      ctx.beginPath();
      ctx.roundRect(LEFT + t * CELL_W + 0.5, TOP + 0.5, CELL_W - 1, this.mtpRows * CELL_H - 1, 3.5);
      ctx.setLineDash([3, 2]);
      ctx.strokeStyle = v('--ink');
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.globalAlpha = 1;
  }

  private drawAxis(dpr: number, h: number, panel: string, ink: string) {
    const r = this.res!;
    const a = this.axis;
    if (a.width !== Math.round(AXIS_W * dpr) || a.height !== Math.round(h * dpr)) {
      a.width = Math.round(AXIS_W * dpr);
      a.height = Math.round(h * dpr);
      a.style.width = `${AXIS_W}px`;
      a.style.height = `${h}px`;
    }
    const ctx = this.axisCtx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = panel;
    ctx.fillRect(0, 0, AXIS_W, h);
    ctx.fillStyle = ink;
    ctx.font = `400 condensed 11px ${FONT}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let l = 0; l < r.nLayers; l++) {
      // only label every other layer when the grid is tall
      if (r.nLayers > 16 && l % 2 !== 0 && l !== r.nLayers - 1) continue;
      ctx.fillText(layerLabel(l), AXIS_W - 5, this.rowY(l) + CELL_H / 2);
    }
    // MTP stages: the block output in bold, the earlier stages lighter
    for (let s = 0; s < this.mtpRows; s++) {
      const out = s === this.mtpOut;
      ctx.font = `${out ? 600 : 400} condensed 11px ${FONT}`;
      ctx.globalAlpha = out ? 1 : 0.8;
      ctx.fillText(this.mtpRows > 1 ? (MTP_STAGE_LABELS[s] ?? String(s)) : 'MTP', AXIS_W - 5, this.rowY(r.nLayers + s) + CELL_H / 2);
    }
    ctx.globalAlpha = 1;
  }

  /** axis and cells as one canvas, for export */
  composite(): HTMLCanvasElement {
    const out = document.createElement('canvas');
    out.width = this.axis.width + this.canvas.width;
    out.height = this.canvas.height;
    const ctx = out.getContext('2d')!;
    ctx.drawImage(this.axis, 0, 0);
    ctx.drawImage(this.canvas, this.axis.width, 0);
    return out;
  }

  // step line along the lower edge of each column's emergence layer
  private drawEmergence(color: string, ring: string, cols: number) {
    const r = this.res!;
    const ctx = this.ctx;
    const yOf = (l: number) => this.rowY(l) + CELL_H;
    const path = new Path2D();
    let open = false;
    for (let t = 0; t < cols; t++) {
      const l = this.emergence[t];
      if (l === null) {
        open = false;
        continue;
      }
      const x0 = LEFT + t * CELL_W + 1;
      const x1 = LEFT + (t + 1) * CELL_W - 1;
      if (open) path.lineTo(x0, yOf(l));
      else path.moveTo(x0, yOf(l));
      path.lineTo(x1, yOf(l));
      open = true;
    }
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = ring;
    ctx.lineWidth = 6;
    ctx.stroke(path);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke(path);
  }
}
