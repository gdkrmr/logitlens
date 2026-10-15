# Logit lens

A browser-based logit lens for GGUF models, built on a fork of wllama (llama.cpp in wasm with WebGPU). See `PLAN.md` for the design and status.

## Layout

- `third_party/wllama/` - wllama fork. Lens code: `cpp/wllama-lens.h` (capture via `cb_eval` plus the projection graph), the `logit_lens` action in `cpp/wllama-context.h`, `wllama.logitLens()` in `src/wllama.ts`.
- `native/` - `lens-cli`, a native build of the same lens code (CPU or Vulkan).
- `web/` - the visualizer (Vite, TypeScript, no framework).

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

## License

MIT, see `LICENSE`. Third-party components and their licenses are listed in `THIRD_PARTY_NOTICES.txt`, which the web app also serves next to the page.
