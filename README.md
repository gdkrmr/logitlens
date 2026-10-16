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

- Architectures: llama, qwen2, qwen3, qwen35, gemma, gemma2, gemma3, gemma4, phi2, phi3.
- Lens prompts are capped at 256 tokens by default. The prompt is decoded in batches of 64 tokens and the output head is skipped, so the hidden states (layers x tokens x n_embd floats) and the projection time grow with the prompt, not the logits buffers.
- The whole model must fit in browser memory, even with GPU offload, and the wasm heap is capped at 4 GB. The model picker disables files over 3.5 GB.
- Browsers: Chrome and Edge on desktop. The Firefox/Safari compat build is skipped (`SKIP_COMPAT=1`).

## Performance

Browser inference on WebGPU vs the CPU build (16 threads), measured in Chromium 153 on an AMD Ryzen AI Max (Radeon 8060S iGPU, Vulkan), October 2026. 67-token prompt, 64 generated tokens, temperature 0, prompt cache off; second of two runs.

| Model (Q4_K_XL) | Prompt, WebGPU | Prompt, CPU | Generation, WebGPU | Generation, CPU |
| --- | --- | --- | --- | --- |
| Qwen3.5 0.8B | 206 tok/s | 92 tok/s | 31 tok/s | 39 tok/s |
| Qwen3.5 2B | 142 tok/s | 38 tok/s | 31 tok/s | 20 tok/s |

WebGPU processes prompts 2-4x faster, and the lens is a single prompt-processing pass plus the projection, so it benefits most. For token-by-token generation the per-token GPU overhead dominates on small models: the CPU build is faster at 0.8B, WebGPU at 2B.

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

The live site, https://logitlens.gdkrmr.com, is GitHub Pages served from the `gh-pages` branch, which `.github/workflows/deploy.yml` writes:

- `main` is published at the site root.
- Every other branch of this repo is published at `https://logitlens.gdkrmr.com/preview/<branch>/` (lowercased, other characters replaced by `-`), with a "Preview of branch" badge, and removed when the branch is deleted. `robots.txt` keeps previews out of search engines.
- The workflow runs only on pushes to this repo, never on pull requests from forks, so only people with write access can publish. The domain is verified for the GitHub account, so no other account can serve Pages under it.

The wasm is built with Docker only when the wllama fork commit changes (cached otherwise). GitHub Pages can't send the isolation headers, so the page loads `coi-serviceworker`, which adds them in the browser; on the very first visit the page reloads once while it installs. Where the server does send the headers (the Vite dev and preview servers, or Cloudflare Pages and Netlify via the `_headers` file in `dist/`), the service worker does nothing.

## License

MIT, see `LICENSE`. Third-party components and their licenses are listed in `THIRD_PARTY_NOTICES.txt`, which the web app also serves next to the page.
