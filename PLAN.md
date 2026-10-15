# Logit lens visualizer: implementation plan

A browser-based logit lens for GGUF models. It runs entirely client-side on a fork of
[wllama](https://github.com/ngxson/wllama) with the llama.cpp WebGPU backend, and is
deployed as a static website.

## Approach

The forked wllama already handles model loading (local files and Hugging Face), caching,
and running inference in a Web Worker. This project adds one capability on top:

1. Capture each layer's residual-stream output during the forward pass with llama.cpp's
   eval callback (`cb_eval`), using the graph tensor names `inp_embd` and `l_out-<il>`.
2. Project the captured hidden states through the model's final norm and output matrix
   in a second ggml graph on the GPU, and reduce to top-k tokens, probabilities, and
   entropy per (layer, position) cell.
3. Return only those small results to JavaScript and draw a layer × position heatmap.

The phases are ordered so the riskiest unknown, synchronous readback of WebGPU tensors
inside the eval callback, is resolved before any UI work.

## Assumptions

- Solo developer, so tasks form one ordered sequence.
- Target browsers are Chrome and Edge on desktop; Firefox and Safari come later.
- The test model is Qwen3.5-0.8B (`qwen35`, Q4_K_XL) from the local LM Studio folder; no downloads.
- Only Llama-, Qwen2/3- and Qwen3.5-family architectures are supported at first.

## Phase 0: Feasibility spike

Prove that intermediate tensors can be read back from WebGPU in the browser.

- [x] Build wllama from source, unmodified (Docker emsdk 4.0.20, `SKIP_COMPAT=1`).
- [x] emdawnwebgpu and `GGML_WEBGPU=ON` are already part of upstream wllama's build.
- [ ] Load the test model and confirm generation is clearly faster than the CPU build (not benchmarked).
- [x] Add a `cb_eval` that captures `l_out-*` (superseded by the full capture in `cpp/wllama-lens.h`).
- [x] Decide the capture strategy. **Decision:** synchronous `ggml_backend_tensor_get`
      inside `cb_eval` works on WebGPU in Chrome (JSPI build), so no fallback is needed.
      The lens runs in its own `llama_context`, so the server slots are untouched.

**Done when:** one hidden-state tensor is read back from WebGPU inside `cb_eval` in Chrome.

## Phase 1: Native lens tool

Build the logit lens natively first, for fast iteration. No separate validation against
Hugging Face: the only check is that the lens's top token on the last layer matches the
model's own next-token prediction.

- [x] Build llama.cpp natively and write a `lens-cli` tool (`native/`) that captures the token
      embedding and every `l_out-*` via `cb_eval`. Note: in current llama.cpp `inp_embd` names an
      unused placeholder; the real embedding is the first `GET_ROWS` on `token_embd.weight`.
- [x] Apply `output_norm` and `output` (or `tok_embd` when embeddings are tied) and print
      top-k per layer and position.

```cpp
static bool lens_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * st = (LensState *) ud;
    bool want = strncmp(t->name, "l_out-", 6) == 0
             || strcmp(t->name, "inp_embd") == 0;
    if (ask) return want;          // "do I want this tensor?"
    if (want) st->capture(t);      // after compute: copy it out
    return true;                   // continue graph evaluation
}

// cparams.cb_eval = lens_cb;
// cparams.cb_eval_user_data = &state;
```

**Done when:** `lens-cli` prints a plausible layer × position table for the test model.

## Phase 2: GPU lens graph

Move the vocabulary projection onto the GPU and read back only small results. Doing it
on the CPU in WASM is too slow: for Qwen2.5-0.5B it's about 3 billion multiply-adds per
token across all layers.

- [x] Build the projection graph: stack hidden states into `[n_embd, n_layers × n_tokens]`,
      apply `ggml_rms_norm`, multiply by `output_norm`, then `ggml_mul_mat` with the output
      matrix. Reach the weight tensors through the internal `llama-model.h` header.
- [x] Add softmax and `ggml_top_k`, plus per-cell entropy and the probability of the final
      layer's top token. Read back only these.
- [x] Chunk the work by rows so the logits buffer stays under 32 MB per chunk. (Full
      logits for 24 layers × 32 tokens × 152k vocab would be about 470 MB in float32.)
- [x] Allow only `llama`, `qwen2`, `qwen3` and `qwen35` architectures and show a clear message for others.

**Done when:** the native tool runs the projection on the Vulkan backend.

## Phase 3: WASM integration

- [x] Add a `logit_lens` action to the wllama C++ glue, taking the prompt, k, and the layer range.
- [x] Return token IDs, probabilities, and entropies as packed typed arrays, not JSON.
- [x] Expose `wllama.logitLens()` in TypeScript, running inside the existing worker.

**Done when:** `wllama.logitLens()` returns results in Chrome. Done: 10 tokens x 25 rows in
~0.7 s on WebGPU with the test model.

Known limit: llama.cpp always computes full logits for every requested output row, so lens
prompts are capped at 64 tokens by default (`maxTokens`), about 64 MB of logits for a 248k vocab.

## Phase 4: Visualizer UI

- [x] Ship a plain HTML table of top-1 tokens per layer and position (minimum viable version).
- [x] Add a model picker: a curated list of ungated Hugging Face GGUFs plus a local file
      picker, with download progress and cache status.
- [x] Search the Hugging Face Hub for GGUF repos (filtered to supported architectures, ungated),
      with a per-repo file picker; the curated list is shown when the search is empty.
- [x] Draw a canvas heatmap with layers bottom to top (embeddings at the bottom), tokens
      across, and a toggle to color by probability or entropy.
- [x] Add hover detail (top-5 tokens) and click-to-plot a final-row token's rank across layers.

**Done when:** a colleague can load a model and interpret a heatmap without help.

## Phase 5: Extensions (optional)

- [ ] Export results as JSON and the heatmap as PNG.
- [ ] Compare two prompts side by side.
- [ ] Next-token mode: append the predicted token and rerun the lens.
- [ ] Tuned lens: load per-layer `n_embd × n_embd` translators from a separate file and
      apply them before the norm.
- [ ] More architectures: Gemma (embedding scaling, final logit soft-capping) and Phi
      (LayerNorm with bias).

## Deployment

The app is a static site with no backend. Hosting requirements:

- **HTTPS or localhost.** WebGPU requires a secure context, and `file://` won't load the
  WASM or workers.
- **Cross-origin isolation** for wllama's multi-threaded build (`SharedArrayBuffer`):

  ```
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  ```

  Without these, wllama falls back to single-threaded mode. If cross-origin resources
  get blocked, `Cross-Origin-Embedder-Policy: credentialless` is the usual fix.
- **`.wasm` served as `application/wasm`**, which common static hosts do by default.

Plan: develop locally with Vite (headers set in `vite.config`), deploy to Cloudflare
Pages or Netlify with a `_headers` file. GitHub Pages can't set headers; use
`coi-serviceworker` there if needed. Any university web server where these two headers
can be added works too.

## Risks

| Risk | Mitigation |
| --- | --- |
| Synchronous WebGPU readback inside `cb_eval` fails in the browser | Resolved in Phase 0; fallbacks are a post-pass buffer copy or the WASM CPU backend |
| Emscripten/Dawn toolchain friction | Start from wllama's existing build, change one thing at a time |
| Quantization noise in early-layer decodings | Recommend Q8_0 or F16 for small models in the UI |
| The full model must fit in main memory, even with GPU offload | Curate small models (0.5–1.5B) in the picker |
| Architecture-specific quirks | Restrict to llama/qwen2/qwen3/qwen35 until Phase 5 |
