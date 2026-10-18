import type { Wllama } from '@wllama/wllama';

/**
 * why WebGPU is not usable, or null. navigator.gpu alone is not enough: Chromium on Linux often has it but gives no
 * adapter, only the SwiftShader software one, or one that never answers; llama.cpp also needs shader-f16
 */
export async function webGpuProblem(wllama: Wllama): Promise<string | null> {
  if (!wllama.isSupportWebGPU()) return 'no WebGPU';
  const gpu = (navigator as any).gpu;
  const TIMEOUT = Symbol('timeout');
  const within = <T>(p: Promise<T>): Promise<T | typeof TIMEOUT> => {
    let timer = 0;
    const late = new Promise<typeof TIMEOUT>((r) => (timer = window.setTimeout(() => r(TIMEOUT), 5000)));
    return Promise.race([p, late]).finally(() => clearTimeout(timer));
  };
  try {
    const adapter: any = await within(gpu.requestAdapter());
    if (adapter === TIMEOUT) return 'the graphics driver did not answer';
    if (!adapter) return 'no graphics card available to WebGPU';
    if (adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter) return 'WebGPU offers only a software renderer';
    if (!adapter.features.has('shader-f16')) return 'the graphics card lacks 16-bit float support';
    const pending = adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
    const device: any = await within(pending);
    if (device === TIMEOUT) {
      // free the device if it still arrives
      pending.then((d: any) => d?.destroy?.(), () => {});
      return 'the graphics driver did not answer';
    }
    device.destroy();
    return null;
  } catch {
    return 'the graphics card did not start';
  }
}
