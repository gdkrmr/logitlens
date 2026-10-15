# TODO

Ideas for the logit lens visualizer, roughly ordered by value for effort within each group.

## Reading the lens

- [ ] **Shade by KL divergence to the final layer.** Entropy only says how uncertain a layer is,
      not whether it is heading toward the final prediction. KL(final || layer) per cell shows
      that directly: dark means "already thinks like the output". Compute it on the GPU next to
      the existing projection (needs the final layer's distribution per position) and add it as
      a third shading option beside Probability and Entropy.
- [ ] **Contrast two answers (logit difference).** Pick two tokens, e.g. " Paris" vs " London",
      and shade each cell by `logit(a) - logit(b)` with a diverging palette. This is the standard
      way to see where a model commits to one answer. Reuses the target-token machinery
      (`targetIds`, `get_rows` on the logits).
- [ ] **Track one token everywhere.** Type a token and shade the whole grid by its probability or
      rank at every (layer, position). Same mechanism as the trace plot, applied to all columns.
- [ ] **Show the model's continuation.** Append the predicted token and rerun, a few steps at a
      time, so the lens covers text the model generates itself.

## Visualizing MTP (multi-token prediction)

Qwen3.5 GGUFs from the `-MTP-` repos carry an extra NextN block (`blk.24.nextn.*` in the test
model). It takes the final hidden state at position t together with the embedding of token t+1,
runs one more attention + FFN block, and predicts token **t+2** through a shared head. llama.cpp
uses it as a built-in draft model for speculative decoding.

- [ ] **MTP row in the heatmap.** Add a row above the final layer labelled "MTP (t+2)" with the
      NextN block's top token per position. Inside the prompt the true next token is known, so
      the block can be fed teacher-forced: hidden state at t plus the embedding of prompt token
      t+1.
- [ ] **Agreement with the main model.** Mark MTP cells where the t+2 guess matches what the main
      model predicts one step later (the final-layer top token at position t+1), and where it
      matches the actual prompt token t+2. The share of matches is the draft acceptance rate,
      which is what makes MTP speed up generation, so it is worth showing as a number too.
- [ ] **Lens inside the NextN block.** Project the block's intermediate states (after `eh_proj`,
      after its attention, after its FFN) through `shared_head_norm` + head, to see how the t+2
      guess forms inside the extra block, like a mini logit lens.
- [ ] **Implementation notes.** llama.cpp builds the MTP graph in `src/models/qwen35.cpp`
      (`h_nextn`, `cparams.embeddings_nextn`). The simplest route is probably to run the
      context with NextN outputs enabled and capture its tensors through the same `cb_eval`. The
      alternative is rebuilding the block in our own ggml graph, which needs its attention over
      all positions, so reusing llama.cpp's path is preferable. Only applies to models with
      `nextn_predict_layers > 0`; hide the row otherwise.

## Practical

- [ ] **Chat template toggle.** Instruct models behave differently inside their chat format. One
      checkbox that wraps the prompt in the model's template (wllama already parses it).
- [ ] **Shareable link and export.** Encode model URL, quantization and prompt in the page URL.
      Export results as JSON and the heatmap as PNG.
- [ ] **Side-by-side prompts.** Two aligned heatmaps for minimal pairs, e.g. "The doctor said
      she" vs "The doctor said he".

## Model coverage and limits

- [ ] **Gemma support.** Five Gemma 4 GGUFs are in the local LM Studio folder, and the lens
      currently refuses them. Needs embedding scaling (sqrt(n_embd)) and final-logit soft-capping
      in the projection graph in `cpp/wllama-lens.h`, then add the arch to the allowlist (C++ and
      `web/src/hf.ts`).
- [ ] **Phi support.** Phi models use LayerNorm with bias as the final norm, so the projection
      graph needs `ggml_norm` + weight + bias instead of `ggml_rms_norm` for them.
- [ ] **Longer prompts.** The 64-token cap exists because llama.cpp computes full logits for every
      position (~1 MB per token for a 248k vocab). Stopping the graph right after the last
      `l_out-*` in `cb_eval` (return false) would skip the LM head compute; also avoid the
      oversized logits buffer, then raise `maxTokens`.
- [ ] **Tuned lens.** Load per-layer `n_embd x n_embd` translators from a separate file and apply
      them before the norm.

## Housekeeping

- [x] **Publish on GitHub.** Pushed to https://github.com/gdkrmr/logitlens.
- [ ] **Link the GitHub repo** from the app (rail footer next to the license links) and from
      README.md.
- [x] **Push the wllama fork.** The `logitlens` branch is on https://github.com/gdkrmr/wllama
      and the submodule points there.
- [ ] **Benchmark WebGPU generation against the CPU build.** Never measured.
