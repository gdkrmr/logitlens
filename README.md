# Logit lens

A browser-based logit lens for GGUF models, built on a fork of wllama (llama.cpp in wasm with WebGPU). Live at https://logitlens.gdkrmr.com, source at https://github.com/gdkrmr/logitlens. Feature ideas and open items are tracked as [GitHub issues](https://github.com/gdkrmr/logitlens/issues).

## Layout

- `third_party/wllama/` - wllama fork. Lens code: `cpp/wllama-lens.h` (capture via `cb_eval` plus the projection graph), the `logit_lens` action in `cpp/wllama-context.h`, `wllama.logitLens()` in `src/wllama.ts`.
- `native/` - `lens-cli`, a native build of the same lens code (CPU or Vulkan).
- `web/` - the visualizer (Vite, TypeScript, no framework).

## How it works

1. **Capture.** The lens runs the prompt in its own `llama_context`, separate from the one wllama uses for generation, with an eval callback (`cb_eval`). The callback copies each layer's residual-stream output (`l_out-<il>`) back from the GPU while the forward pass runs. Synchronous WebGPU readback inside the callback works in Chrome with the JSPI build. In current llama.cpp, `inp_embd` names an unused placeholder; the real token embedding is the first `GET_ROWS` on `token_embd.weight`, so the callback matches that op.
2. **Project.** The captured rows go through the model's `output_norm` and output matrix (`tok_embd` when embeddings are tied) in a second ggml graph on the GPU, chunked to 32 MB of logits at a time. Softmax, top-k, entropy and the rank of a target token are computed there.
3. **Return.** Only those small per-cell results go back to JavaScript, as packed typed arrays, and the web app draws the layer x position heatmap.

## Run the web app

```sh
cd web
npm install
LOCAL_MODELS=~/.lmstudio/models npm run dev
```

Open http://localhost:5173. Search Hugging Face for a GGUF repo (or pick a suggested one), choose a file, and load it; or open a local `.gguf`. With `LOCAL_MODELS` set, the dev server also serves that folder at `/local-models/...`, so a local file can be preselected with `?model=/local-models/unsloth/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf`.

`npm run build` writes a static site to `web/dist/`, including the `_headers` file with the cross-origin isolation headers (needed for multi-threading).

## Rebuild the wasm

After changing anything in `third_party/wllama/cpp/`:

```sh
cd third_party/wllama
SKIP_COMPAT=1 ./scripts/build_wasm.sh    # Docker, about 10-20 min from scratch
node cpp/generate_glue_prototype.js      # after changing glue.hpp
./scripts/build_worker.sh
```

`npm run dev` in `web/` copies the new `wllama.wasm` into place.

## Native tool

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF
cmake --build native/build --target lens-cli -j
native/build/lens-cli -m model.gguf -p "The capital of France is" [-ngl 0] [--json out.json]
```

It prints the top-1 token per layer and position. As a sanity check it also reports how many positions disagree with the model's own prediction on the last layer; this should be 0.

## Limits

- Architectures: llama, qwen2, qwen3, qwen35.
- Lens prompts are capped at 64 tokens by default, since llama.cpp computes full logits for every position.
- The whole model must fit in browser memory, even with GPU offload, and the wasm heap is capped at 4 GB. The model picker disables files over 3.5 GB.
- Browsers: Chrome and Edge on desktop. The Firefox/Safari compat build is skipped (`SKIP_COMPAT=1`).

## Deploy

`web/dist/` is a static site with no backend. The host must provide:

- **HTTPS (or localhost).** WebGPU needs a secure context, and `file://` won't load the wasm or the worker.
- **Cross-origin isolation headers**, for the multi-threaded build (`SharedArrayBuffer`):

  ```
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  ```

  Without them the app still works but falls back to a single thread. If they block cross-origin resources, `Cross-Origin-Embedder-Policy: credentialless` is the usual fix.
- **`.wasm` served as `application/wasm`**, which common static hosts do by default.

The live site, https://logitlens.gdkrmr.com, is GitHub Pages, deployed by `.github/workflows/deploy.yml` on every push to `main`. The workflow builds the wasm with Docker only when the wllama fork commit changes (cached otherwise), then builds and publishes `web/dist/`. GitHub Pages can't send the isolation headers, so the page loads `coi-serviceworker`, which adds them in the browser; on the very first visit the page reloads once while it installs. Where the server does send the headers (the Vite dev and preview servers, or Cloudflare Pages and Netlify via the `_headers` file in `dist/`), the service worker does nothing.

## License

MIT, see `LICENSE`. Third-party components and their licenses are listed in `THIRD_PARTY_NOTICES.txt`, which the web app also serves next to the page.
