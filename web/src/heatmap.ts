import type { LogitLensResult } from '@wllama/wllama';

export type ColorMode = 'prob' | 'entropy';

// sequential blue, light -> dark
const BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];

const hexToRgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

export function isDark(): boolean {
  const t = document.documentElement.dataset.theme;
  if (t === 'dark') return true;
  if (t === 'light') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

/** t in [0,1] -> color; low values recede toward the surface in both themes */
export function rampColor(t: number): { fill: string; ink: string } {
  const stops = isDark() ? [...BLUE].reverse() : BLUE;
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

export function rampGradientCss(): string {
  const stops = isDark() ? [...BLUE].reverse() : BLUE;
  return `linear-gradient(to right, ${stops.join(',')})`;
}

/** printable form of a token piece */
export function showPiece(s: string | undefined): string {
  if (s === undefined) return '?';
  if (s.length > 0 && s.trim() === '' && !/[\n\t\r]/.test(s)) return '\u2423'.repeat(s.length);
  return s.replace(/\n/g, '\\n').replace(/\t/g, '\\t').replace(/\r/g, '\\r');
}

export const layerLabel = (l: number) => (l === 0 ? 'emb' : String(l - 1));

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
const FONT = '"Martian Mono Variable", ui-monospace, monospace';

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export class Heatmap {
  private ctx: CanvasRenderingContext2D;
  private axisCtx: CanvasRenderingContext2D;
  res: LogitLensResult | null = null;
  maxEntropy = 1;
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

  setResult(res: LogitLensResult) {
    this.res = res;
    let m = 0;
    for (const h of res.entropy) m = Math.max(m, h);
    this.maxEntropy = m || 1;
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

  value(cell: number, mode: ColorMode): number {
    const r = this.res!;
    return mode === 'prob' ? r.topProbs[cell * r.k] : r.entropy[cell] / this.maxEntropy;
  }

  /** map pointer position to (layer, pos) */
  hit(clientX: number, clientY: number): { layer: number; pos: number } | null {
    const r = this.res;
    if (!r) return null;
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left - LEFT;
    const y = clientY - rect.top - TOP;
    const pos = Math.floor(x / CELL_W);
    const layer = r.nLayers - 1 - Math.floor(y / CELL_H);
    if (x < 0 || y < 0 || pos >= r.nTokens || layer < 0 || layer >= r.nLayers) return null;
    return { layer, pos };
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
    const dur = 140 + r.nLayers * 14;
    const t0 = performance.now();
    const step = (now: number) => {
      const p = Math.min(1, (now - t0) / dur);
      this.draw(opts, p * r.nLayers, fromCol);
      if (p < 1) this.anim = requestAnimationFrame(step);
    };
    this.anim = requestAnimationFrame(step);
  }

  draw(opts: HeatmapOpts, rows = Infinity, fromCol = 0) {
    const r = this.res;
    if (!r) return;
    if (rows === Infinity) cancelAnimationFrame(this.anim);
    const dpr = window.devicePixelRatio || 1;
    const w = LEFT + r.nTokens * CELL_W + 4;
    const h = TOP + r.nLayers * CELL_H + BOTTOM;
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
      const y = TOP + (r.nLayers - 1 - l) * CELL_H;
      const alpha = Math.max(0, Math.min(1, rows - l));
      for (let t = 0; t < r.nTokens; t++) {
        // columns before fromCol are already known and shown complete
        ctx.globalAlpha = t < fromCol ? 1 : alpha;
        if (ctx.globalAlpha <= 0) continue;
        const cell = l * r.nTokens + t;
        const x = LEFT + t * CELL_W;
        const { fill, ink } = rampColor(this.value(cell, opts.mode));
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

    // input tokens along the bottom
    ctx.font = `400 condensed 12px ${FONT}`;
    ctx.textAlign = 'left';
    for (let t = 0; t < r.nTokens; t++) {
      const x = LEFT + t * CELL_W;
      const y = TOP + r.nLayers * CELL_H + 15;
      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 2, y - 10, CELL_W - 4, 20);
      ctx.clip();
      ctx.fillStyle = v('--ink-2');
      ctx.fillText(showPiece(r.pieces.get(r.tokens[t])), x + 5, y);
      ctx.restore();
    }

    // during a reveal the line covers only the complete columns, so it never disappears and comes back
    this.drawEmergence(v('--trace'), panel, rows >= r.nLayers ? r.nTokens : fromCol);

    if (opts.selected) {
      const { layer, pos } = opts.selected;
      ctx.strokeStyle = v('--ink');
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(LEFT + pos * CELL_W + 1, TOP + (r.nLayers - 1 - layer) * CELL_H + 1, CELL_W - 2, CELL_H - 2, 3);
      ctx.stroke();
    }
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
      ctx.fillText(layerLabel(l), AXIS_W - 5, TOP + (r.nLayers - 1 - l) * CELL_H + CELL_H / 2);
    }
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
    const yOf = (l: number) => TOP + (r.nLayers - l) * CELL_H;
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
