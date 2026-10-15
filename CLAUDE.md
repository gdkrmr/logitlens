# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Browser logit lens for GGUF models: each layer's residual stream is decoded through the model's own final norm and output head, and shown as a layer x position heatmap. Runs client-side on a fork of wllama (llama.cpp compiled to wasm, WebGPU backend). Live at https://logitlens.gdkrmr.com; open work is tracked as GitHub issues on gdkrmr/logitlens.

## Commands

```sh
# web app (Vite dev server, sends COOP/COEP headers)
cd web && npm install
LOCAL_MODELS=~/.lmstudio/models npm run dev        # http://localhost:5173
# LOCAL_MODELS serves that folder at /local-models/...; preselect a file with
#   ?model=/local-models/unsloth/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf
npm run build                                      # assets + tsc --noEmit + vite build -> web/dist/
npx tsc --noEmit -p .                              # typecheck only (also checks the wllama TS sources)

# wasm, after changing third_party/wllama/cpp/ (Docker emsdk, 10-20 min cold, ~3 min incremental)
cd third_party/wllama
SKIP_COMPAT=1 ./scripts/build_wasm.sh              # writes src/wasm/wllama.{wasm,js}
node cpp/generate_glue_prototype.js                # after changing cpp/glue.hpp -> src/glue/messages.ts
./scripts/build_worker.sh                          # re-embeds wllama.js into src/workers-code/generated.ts
# web's `npm run assets` (run by dev/build) copies the new wllama.wasm into web/public/wasm/

# native tool: same lens code, CPU or Vulkan, much faster to iterate on than the wasm build
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF
cmake --build native/build --target lens-cli -j
native/build/lens-cli -m model.gguf -p "The capital of France is" [-k 5] [-ngl 0] [--json out.json]

# compile-check the wllama glue natively (catches C++ errors before a Docker build)
cd third_party/wllama
cmake -S dev/native-test -B dev/native-test/build -DCMAKE_BUILD_TYPE=Release
cmake --build dev/native-test/build --target native-test -j
```

There is no linter and no test suite for `web/` or `native/`. `lens-cli` prints "last-layer mismatches=N", which must be 0: the lens's top token on the last layer has to equal the model's own prediction. That is the only correctness check the project uses (by design, see below).

Test model: `~/.lmstudio/models/unsloth/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (arch `qwen35`). Use local GGUFs; ask before downloading models.

## Architecture

Three parts, one lens implementation:

- `third_party/wllama/` - git submodule, fork https://github.com/gdkrmr/wllama, branch `logitlens`. All lens C++ lives in `cpp/wllama-lens.h` (header-only, shared by wasm and native).
- `native/lens-cli.cpp` - native driver; links llama.cpp from the submodule and includes `wllama-lens.h`.
- `web/` - Vite + TypeScript app, no framework. Imports the fork's TS sources directly via the `@wllama/wllama` alias (`vite.config.ts`, `tsconfig.json` paths), not a published package.

Data flow for one lens run:

1. `wllama.logitLens()` (`src/wllama.ts`) sends a GLUE message to the worker. GLUE is wllama's binary protocol: structs in `cpp/glue.hpp` are the source of truth, `generate_glue_prototype.js` generates `src/glue/messages.ts`. Large numeric arrays travel as `raw` fields (packed little-endian typed arrays), decoded in `logitLens()`.
2. `action_logit_lens` (`cpp/wllama-context.h`, dispatched from `cpp/wllama.cpp` via `WLLAMA_ACTION`) runs in a **separate `llama_context`** created lazily with `cb_eval = lens_cb_eval`, so it never touches the server slots used for generation.
3. `lens_cb_eval` copies hidden states back with synchronous `ggml_backend_tensor_get` inside the callback (works on WebGPU with the JSPI build). Slot 0 is the token embedding, matched as the first `GET_ROWS` on `token_embd.weight`, because in current llama.cpp `inp_embd` names an unused placeholder. Slot il+1 is `l_out-<il>`. All batch tokens request logits so the last layer is not cropped to one row. The prompt is decoded in batches of `lens_batch` (64) tokens, which bounds llama.cpp's logits buffers; after the last `l_out` the callback returns false, which breaks the rest of that scheduler split, so the output head is not computed (`lens-cli` runs a separate plain pass for its mismatch check).
4. `lens_projector` builds its own ggml graph on the model's output device (internal `llama-model.h` gives `output_norm`, `output`, `output_s`): rms_norm, mul by norm weight, mul_mat with the output matrix, final-logit soft-cap (Gemma 2/3/4), softmax, `top_k`, entropy, and target-token prob/rank. Chunked to 32 MB of logits per graph. Only the small per-cell results are read back. Gemma scales its embeddings by sqrt(n_embd) after the captured lookup, so `lens_run` scales slot 0 itself.
5. Results are **layer-major**: cell = layer * nTokens + position, layer 0 = embeddings, layer i = block i-1 output. Target token defaults to the final layer's top-1 per position. `reuse: true` re-projects cached hidden states with new targets (no forward pass), used by the trace view.

Web app (`web/src/`): `main.ts` (UI state, model loading, run, tooltips, trace, timings, theme), `hf.ts` (Hugging Face search/file listing, arch filter, quant labels), `heatmap.ts` (canvas renderer, color ramp, emergence line), `trace.ts` (SVG rank plot).

Limits that shape the code: lens prompts capped at 256 tokens by default (`max_tokens`; hidden states are layers x tokens x n_embd floats, and projection time grows with layers x tokens); architectures allowlisted in both `lens_arch_supported` (C++) and `SUPPORTED_ARCHS` (`hf.ts`): llama, qwen2, qwen3, qwen35, gemma, gemma2, gemma3, gemma4; wasm heap max 4 GB, so the picker disables files over 3.5 GB (`MAX_FILE_BYTES`).

## Conventions and gotchas

- **Submodule workflow:** commit C++/TS changes inside `third_party/wllama` on branch `logitlens`, push to the `fork` remote (`origin` there is upstream ngxson/wllama), then commit the bumped submodule pointer in this repo. The deploy cache is keyed on the fork commit, so an unpushed or unbumped fork change never reaches the site.
- **Inside `third_party/wllama/`, follow its `AGENTS.md`:** ASCII only, terse comments in simple English, smallest diff, never edit `llama.cpp/` (fix things in `cpp/` or `src/`), and commit trailers use `Assisted-by: <model>` instead of `Co-authored-by:`. Upstream llama.cpp sync procedure is in that file too.
- `src/wasm/*.wasm` is gitignored in the fork; `wllama.js`, `source-map.ts` and `workers-code/generated.ts` are tracked and must match the built wasm.
- Never add `llama.cpp/src` to the include path: its `unicode.h` shadows `common/unicode.h`. `wllama-lens.h` includes `../llama.cpp/src/llama-model.h` by relative path instead.
- Canvas: setting `ctx.font` resets `ctx.fontStretch`, so put the stretch keyword in the font shorthand (`500 condensed 12px ...`). Await `document.fonts.load(...)` before drawing.
- Colors are CSS custom properties in `style.css`; dark values are defined twice (media query and `[data-theme='dark']`). Canvas and SVG read them at draw time, so theme changes must call a redraw (`redrawForTheme`).
- Cross-origin isolation: the Vite servers send COOP/COEP; GitHub Pages cannot, so `index.html` loads `coi-serviceworker.min.js` first (copied from node_modules by `npm run assets`). Vite `base` is `'./'`.
- Deploy: `.github/workflows/deploy.yml` runs on every push to this repo (not fork PRs), rebuilds the wasm only when the fork commit changes, and writes the `gh-pages` branch: `main` at the root, other branches at `preview/<branch>/` (removed on branch delete). Pages serves `gh-pages`. Push the fork before the parent repo, or the checkout of the submodule commit fails.
- Third-party code bundled into the site must get an entry in `THIRD_PARTY_NOTICES.txt` (served with the app).
- Screenshots through the Chrome extension time out once WebGPU is active; for visual checks drive the system Chromium with Playwright from `third_party/wllama` (it has `playwright` installed): `chromium.launch({ executablePath: '/usr/bin/chromium', args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan'] })`.
- Scope: the goal is visualization. Do not add validation suites or reference implementations (e.g. a Python/transformers comparison) unless asked.
