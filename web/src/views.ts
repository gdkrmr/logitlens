// which view a link belongs to; kept free of wllama, so the simple view can redirect before anything starts
import { SIMPLE_MODELS } from './hf';

/** a ?model= value as an absolute URL; null if missing or unparsable (e.g. a truncated link) */
export function modelParam(params: URLSearchParams): string | null {
  const v = params.get('model');
  if (!v) return null;
  try {
    return new URL(v, location.href).href;
  } catch {
    return null;
  }
}

/** URL settings only the advanced view has; its updateUrl can write no others */
export const ADVANCED_PARAMS = ['chat', 'reply', 'mtp', 'draft'] as const;
export type AdvancedParam = (typeof ADVANCED_PARAMS)[number];

/** the simple view's prompt is one line of at most this many characters */
export const SIMPLE_PROMPT_MAX = 600;

/** the simple view takes only a prompt it can hold and one of its models; links with anything else are for the advanced view */
export function advancedOnly(params: URLSearchParams): boolean {
  const model = modelParam(params);
  const prompt = params.get('prompt') ?? '';
  return (
    ADVANCED_PARAMS.some((k) => params.has(k)) ||
    (!!model && !SIMPLE_MODELS.some((m) => m.url === model)) ||
    /[\r\n]/.test(prompt) ||
    prompt.length > SIMPLE_PROMPT_MAX
  );
}
