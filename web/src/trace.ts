import { layerLabel } from './heatmap';

export interface TracePoint {
  layer: number;
  rank: number; // 0 = top token
  prob: number;
}

const NS = 'http://www.w3.org/2000/svg';
const el = (tag: string, attrs: Record<string, string | number> = {}) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
};

/** rank of one token across layers; y is log-scaled with rank 1 at the top */
export function drawTrace(host: HTMLElement, pts: TracePoint[], tooltip: HTMLElement, label: string) {
  host.replaceChildren();
  const W = 720;
  const H = 220;
  const m = { l: 52, r: 16, t: 12, b: 30 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const maxRank = Math.max(10, ...pts.map((p) => p.rank + 1));
  const decades = Math.ceil(Math.log10(maxRank));
  const ymax = Math.log10(10 ** decades);
  const x = (l: number) => m.l + (pts.length <= 1 ? 0 : (l / (pts.length - 1)) * iw);
  const y = (rank: number) => m.t + (Math.log10(rank + 1) / ymax) * ih;

  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `Rank of ${label} across layers` });
  const grid = el('g', { class: 'grid' });
  const axis = el('g', { class: 'axis' });
  for (let d = 0; d <= decades; d++) {
    const r = 10 ** d - 1;
    grid.append(el('line', { x1: m.l, x2: W - m.r, y1: y(r), y2: y(r), 'stroke-width': 1 }));
    const t = el('text', { x: m.l - 8, y: y(r) + 4, 'text-anchor': 'end' });
    t.textContent = d === 0 ? 'rank 1' : `${10 ** d}`;
    axis.append(t);
  }
  const step = Math.max(1, Math.ceil(pts.length / 13));
  pts.forEach((p, i) => {
    if (i % step !== 0 && i !== pts.length - 1) return;
    const t = el('text', { x: x(i), y: H - 8, 'text-anchor': 'middle' });
    t.textContent = layerLabel(p.layer);
    axis.append(t);
  });
  svg.append(grid, axis);

  const accent = getComputedStyle(document.documentElement).getPropertyValue('--data').trim();
  const surface = getComputedStyle(document.documentElement).getPropertyValue('--panel').trim();
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.rank).toFixed(1)}`).join('');
  svg.append(el('path', { d, fill: 'none', stroke: accent, 'stroke-width': 2, 'stroke-linejoin': 'round' }));

  const cross = el('line', { y1: m.t, y2: m.t + ih, stroke: accent, 'stroke-width': 1, opacity: 0 });
  const dot = el('circle', { r: 5, fill: accent, stroke: surface, 'stroke-width': 2, opacity: 0 });
  svg.append(cross, dot);

  // hit layer: whole plot area, nearest layer wins
  const hit = el('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent' });
  svg.append(hit);
  const show = (ev: PointerEvent) => {
    const rect = svg.getBoundingClientRect();
    const sx = ((ev.clientX - rect.left) / rect.width) * W;
    const i = Math.round(((sx - m.l) / iw) * (pts.length - 1));
    const p = pts[Math.max(0, Math.min(pts.length - 1, i))];
    const idx = pts.indexOf(p);
    cross.setAttribute('x1', String(x(idx)));
    cross.setAttribute('x2', String(x(idx)));
    cross.setAttribute('opacity', '0.4');
    dot.setAttribute('cx', String(x(idx)));
    dot.setAttribute('cy', String(y(p.rank)));
    dot.setAttribute('opacity', '1');
    tooltip.innerHTML = `<div class="tt-head">Layer ${layerLabel(p.layer)}</div><table><tr><td>Rank</td><td class="num">${p.rank + 1}</td></tr><tr><td>Probability</td><td class="num">${(p.prob * 100).toFixed(2)}%</td></tr></table>`;
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(window.innerWidth - 220, ev.clientX + 14)}px`;
    tooltip.style.top = `${ev.clientY + 14}px`;
  };
  hit.addEventListener('pointermove', show as EventListener);
  hit.addEventListener('pointerleave', () => {
    tooltip.hidden = true;
    cross.setAttribute('opacity', '0');
    dot.setAttribute('opacity', '0');
  });
  host.append(svg);
}
