// Hugging Face Hub search for GGUF repos the lens supports

export const SUPPORTED_ARCHS = ['llama', 'qwen2', 'qwen3', 'qwen35', 'gemma', 'gemma2', 'gemma3', 'gemma4', 'phi2', 'phi3'];
// the wasm heap is capped at 4 GB and also holds the KV cache and compute buffers
export const MAX_FILE_BYTES = 3.5e9;
// smallest GGUFs seen on the Hub take about 0.26 bytes per parameter (2-bit quants), Q4 files about 0.56
export const tooLargeForBrowser = (params: number | null) => params !== null && params * 0.25 > MAX_FILE_BYTES;
export const onlySmallFilesFit = (params: number | null) => params !== null && params * 0.56 > MAX_FILE_BYTES;

const API = 'https://huggingface.co/api';

export interface RepoInfo {
  id: string;
  arch: string | null;
  params: number | null;
  downloads: number;
  gated: boolean;
  hint?: string; // shown in the list, e.g. the download size of the preselected file
}

export interface GgufFile {
  path: string; // first shard for split files
  size: number; // all shards
  shards: number;
  url: string;
}

interface ApiModel {
  id: string;
  downloads?: number;
  gated?: boolean | string;
  gguf?: { architecture?: string; total?: number };
}

const toRepo = (m: ApiModel): RepoInfo => ({
  id: m.id,
  arch: m.gguf?.architecture ?? null,
  params: m.gguf?.total ?? null,
  downloads: m.downloads ?? 0,
  gated: !!m.gated,
});

const EXPAND = ['gguf', 'downloads', 'gated'].map((e) => `expand[]=${e}`).join('&');

/** search GGUF repos by text; repos are sorted by downloads */
export async function searchRepos(query: string, signal?: AbortSignal): Promise<RepoInfo[]> {
  const q = encodeURIComponent(query.trim());
  const res = await fetch(`${API}/models?search=${q}&filter=gguf&sort=downloads&direction=-1&limit=50&${EXPAND}`, { signal });
  if (!res.ok) throw new Error(`Hugging Face search failed (${res.status})`);
  return ((await res.json()) as ApiModel[]).map(toRepo);
}

/** fetch one repo by exact id, null if it does not exist */
export async function getRepo(id: string, signal?: AbortSignal): Promise<RepoInfo | null> {
  const res = await fetch(`${API}/models/${id}?${EXPAND}`, { signal });
  if (res.status === 404 || res.status === 401) return null;
  if (!res.ok) throw new Error(`Hugging Face request failed (${res.status})`);
  return toRepo((await res.json()) as ApiModel);
}

/** list model GGUF files in a repo; split files are merged into one entry; projector, imatrix and MTP draft files are skipped */
export async function listGgufFiles(repo: string, signal?: AbortSignal): Promise<GgufFile[]> {
  const res = await fetch(`${API}/models/${repo}/tree/main?recursive=1`, { signal });
  if (!res.ok) throw new Error(`Could not list files of ${repo} (${res.status})`);
  const entries = (await res.json()) as { type: string; path: string; size: number; lfs?: { size: number } }[];
  const groups = new Map<string, GgufFile>();
  for (const e of entries) {
    if (e.type !== 'file' || !e.path.toLowerCase().endsWith('.gguf')) continue;
    const parts = e.path.toLowerCase().split('/');
    const name = parts[parts.length - 1];
    if (name.includes('mmproj') || name.startsWith('imatrix') || name.startsWith('mtp-') || parts.slice(0, -1).includes('mtp')) continue;
    const size = e.lfs?.size ?? e.size;
    const m = /^(.+)-(\d{5})-of-(\d{5})\.gguf$/i.exec(e.path);
    const key = m ? `${m[1]}-of-${m[3]}` : e.path;
    const first = m ? `${m[1]}-00001-of-${m[3]}.gguf` : e.path;
    const g = groups.get(key) ?? { path: first, size: 0, shards: 0, url: fileUrl(repo, first) };
    g.size += size;
    g.shards += 1;
    groups.set(key, g);
  }
  return [...groups.values()].sort((a, b) => a.size - b.size);
}

export const DEFAULT_REPO = 'unsloth/SmolLM2-135M-Instruct-GGUF';

// file lists shipped with the app, so preselecting the default model needs no request to Hugging Face
export const PRESET_FILES: Record<string, GgufFile[]> = {
  [DEFAULT_REPO]: (
    [
      ['SmolLM2-135M-Instruct-Q2_K.gguf', 88201792],
      ['SmolLM2-135M-Instruct-Q3_K_M.gguf', 93510208],
      ['SmolLM2-135M-Instruct-Q4_K_M.gguf', 105454144],
      ['SmolLM2-135M-Instruct-Q5_K_M.gguf', 112103488],
      ['SmolLM2-135M-Instruct-Q6_K.gguf', 138382912],
      ['SmolLM2-135M-Instruct-Q8_0.gguf', 144811072],
      ['SmolLM2-135M-Instruct-F16.gguf', 270885952],
    ] as const
  ).map(([path, size]) => ({ path, size, shards: 1, url: fileUrl(DEFAULT_REPO, path) })),
};

export function fileUrl(repo: string, path: string) {
  return `https://huggingface.co/${repo}/resolve/main/${path.split('/').map(encodeURIComponent).join('/')}`;
}


export const isSupported = (r: RepoInfo) => !!r.arch && SUPPORTED_ARCHS.includes(r.arch);

/** default file: Q8_0 for small models, else Q4_K_M, else the smallest file that fits */
export function pickDefaultFile(files: GgufFile[], params: number | null): GgufFile | undefined {
  const fits = files.filter((f) => f.size <= MAX_FILE_BYTES);
  const find = (q: string) => fits.find((f) => f.path.toUpperCase().includes(q));
  const small = params !== null && params < 1.2e9;
  return (small ? find('Q8_0') : undefined) ?? find('Q4_K_M') ?? find('Q8_0') ?? fits[0];
}

export const fmtBytes = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.round(n / 1e6)} MB`);
export const fmtParams = (n: number | null) => (n === null ? '' : n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : `${Math.round(n / 1e6)}M`);
export const fmtCount = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(n));

/** short quantization name from a file name, e.g. "Q4_K_M", "UD-Q4_K_XL", "BF16" */
export function quantLabel(path: string): string {
  const name = path.split('/').pop()!.replace(/-\d{5}-of-\d{5}\.gguf$/i, '').replace(/\.gguf$/i, '');
  const m = /(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4|NVFP4))$/i.exec(name);
  return m ? m[1].toUpperCase() : name;
}

// the two models of the simple view; Q8_0 keeps the early layers clean on small models
export const SIMPLE_MODELS = [
  { name: 'SmolLM2 135M', repo: DEFAULT_REPO, file: 'SmolLM2-135M-Instruct-Q8_0.gguf', size: 144811072, note: 'Small, quick to download' },
  { name: 'Qwen3.5 0.8B', repo: 'unsloth/Qwen3.5-0.8B-GGUF', file: 'Qwen3.5-0.8B-Q8_0.gguf', size: 811843840, note: 'Larger, clearer middle layers' },
].map((m) => ({ ...m, url: fileUrl(m.repo, m.file) }));
