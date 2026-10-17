// helpers shared by the simple and the advanced view
import { Wllama } from '@wllama/wllama';

const WASM = { default: new URL('wasm/wllama.wasm', document.baseURI).href };
export const newWllama = () => new Wllama(WASM, { suppressNativeLog: true });

/** an instance without a model: loadModel works once per instance, so a loaded one is replaced */
export async function unloaded(w: Wllama): Promise<Wllama> {
  if (!w.isModelLoaded()) return w;
  await w.exit();
  return newWllama();
}

/** after a failed load the worker is started without a model; without a fresh instance the next load fails with "already initialized" */
export async function recovered(w: Wllama): Promise<Wllama> {
  if (w.isModelLoaded()) return w;
  await w.exit().catch(() => {});
  return newWllama();
}

/** models in the browser's cache; none if OPFS is unavailable */
export async function cachedModels(w: Wllama): Promise<{ url: string; size: number }[]> {
  try {
    return await w.modelManager.getModels();
  } catch {
    return [];
  }
}

export const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
export const pct = (p: number) => (p >= 0.995 ? '100%' : p < 0.001 ? '<0.1%' : `${(p * 100).toFixed(1)}%`);

/** progress bar with a .bar-fill child; null: indeterminate */
export function setBar(bar: HTMLElement, fraction: number | null) {
  bar.classList.toggle('indeterminate', fraction === null);
  const p = fraction === null ? 0 : Math.round(fraction * 1000) / 10;
  (bar.firstElementChild as HTMLElement).style.width = fraction === null ? '' : `${p}%`;
  if (fraction === null) bar.removeAttribute('aria-valuenow');
  else bar.setAttribute('aria-valuenow', String(p));
}

/** set by the deploy workflow for non-main branches */
export function showPreviewBadge(badge: HTMLElement) {
  const branch = import.meta.env.VITE_BRANCH as string | undefined;
  if (!branch) return;
  badge.innerHTML = `Preview of branch ${esc(branch)} &middot; <a href="https://logitlens.gdkrmr.com/">main site</a>`;
  badge.hidden = false;
  document.title = `[${branch}] ${document.title}`;
}
