// native logit lens: prints the top-1 token per layer and position
// usage: lens-cli -m model.gguf -p "prompt" [-k 5] [-ngl 99] [--json out.json] [--continue N]
// --continue N: append N greedy tokens one by one with incremental runs, then compare with a full run
// -ub N: tokens per decode call; llama.cpp results depend a bit on the batch split, use -ub 1 for an exact compare
// --mtp: also run the NextN (MTP) block, print its t+2 guesses and check them against llama.cpp's own MTP logits

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "wllama-lens.h"
// staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith
#include "../llama.cpp/src/llama-ext.h"

static std::string piece(const llama_vocab *vocab, llama_token t)
{
  char buf[256];
  int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
  std::string s = n > 0 ? std::string(buf, n) : std::string();
  std::string out;
  for (char c : s)
  {
    if (c == '\n')
      out += "\\n";
    else if (c == '"' || c == '\\')
      out += std::string("\\") + c;
    else
      out += c;
  }
  return out;
}

int main(int argc, char **argv)
{
  std::string model_path, prompt = "The capital of France is", json_path;
  int k = 5, ngl = 99, n_cont = 0, n_ub = lens_batch;
  bool use_mtp = false;
  for (int i = 1; i < argc; i++)
  {
    std::string a = argv[i];
    if (a == "-m" && i + 1 < argc)
      model_path = argv[++i];
    else if (a == "-p" && i + 1 < argc)
      prompt = argv[++i];
    else if (a == "-k" && i + 1 < argc)
      k = atoi(argv[++i]);
    else if (a == "-ngl" && i + 1 < argc)
      ngl = atoi(argv[++i]);
    else if (a == "--json" && i + 1 < argc)
      json_path = argv[++i];
    else if (a == "-ub" && i + 1 < argc)
      n_ub = atoi(argv[++i]);
    else if (a == "--continue" && i + 1 < argc)
      n_cont = atoi(argv[++i]);
    else if (a == "--mtp")
      use_mtp = true;
    else
    {
      fprintf(stderr, "usage: %s -m model.gguf -p prompt [-k 5] [-ngl 99] [--json out.json] [--continue N] [-ub 64] [--mtp]\n", argv[0]);
      return 1;
    }
  }

  llama_log_set([](ggml_log_level level, const char *text, void *)
                { if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr); },
                nullptr);
  llama_backend_init();

  auto mparams = llama_model_default_params();
  mparams.n_gpu_layers = ngl;
  mparams.load_mtp = use_mtp;
  llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
  if (!model)
    return 1;
  std::string arch;
  if (!lens_arch_supported(model, arch))
  {
    fprintf(stderr, "architecture '%s' is not supported\n", arch.c_str());
    return 1;
  }
  const llama_vocab *vocab = llama_model_get_vocab(model);
  if (use_mtp && !lens_has_mtp(model))
  {
    fprintf(stderr, "model has no MTP block, --mtp ignored\n");
    use_mtp = false;
  }

  std::vector<llama_token> tokens(prompt.size() + 8);
  int n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), tokens.data(), (int)tokens.size(), true, true);
  tokens.resize(n);

  lens_capture cap;
  auto cparams = llama_context_default_params();
  cparams.n_ctx = std::max(512, n + n_cont);
  cparams.n_batch = n_ub;
  cparams.n_ubatch = n_ub;
  cparams.n_seq_max = 1;
  cparams.cb_eval = lens_cb_eval;
  cparams.cb_eval_user_data = &cap;
  llama_context *lctx = llama_init_from_model(model, cparams);

  lens_mtp mtp;
  lens_mtp *pmtp = nullptr;
  auto mparams_ctx = cparams;
  mparams_ctx.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
  if (use_mtp)
  {
    mparams_ctx.cb_eval_user_data = &mtp.cap;
    mtp.ctx = llama_init_from_model(model, mparams_ctx);
    if (!mtp.ctx)
      return 1;
    pmtp = &mtp;
  }

  lens_projector proj;
  proj.init(model);

  lens_run_output out;
  const int64_t t0 = ggml_time_us();
  lens_run(lctx, cap, proj, tokens, k, 0, 0, {}, out, false, false, pmtp);
  const int64_t t1 = ggml_time_us();

  if (n_cont > 0)
  {
    for (int i = 0; i < n_cont; i++)
    {
      tokens.push_back(out.res.top_ids[((out.n_layers - 1) * out.n_tokens + out.n_tokens - 1) * k]);
      lens_run(lctx, cap, proj, tokens, k, 0, 0, {}, out, false, true, pmtp);
      printf("continue tokens=%d forward %.1f ms, projection %.1f ms\n", out.n_tokens, out.t_decode_ms, out.t_project_ms);
    }
    // hidden states too: RMSNorm hides a wrong scale of a whole row from the readout
    auto inc_hidden = cap.hidden;
    if (pmtp)
    {
      inc_hidden.push_back(cap.h_nextn);
      inc_hidden.insert(inc_hidden.end(), mtp.cap.hidden.begin(), mtp.cap.hidden.end());
    }
    // also append all new tokens in one incremental run
    lens_run_output jump;
    lens_run(lctx, cap, proj, std::vector<llama_token>(tokens.begin(), tokens.begin() + n), k, 0, 0, {}, jump, false, false, pmtp);
    lens_run(lctx, cap, proj, tokens, k, 0, 0, {}, jump, false, true, pmtp);
    printf("one step +%d tokens: forward %.1f ms, projection %.1f ms\n", n_cont, jump.t_decode_ms, jump.t_project_ms);
    lens_run_output full;
    lens_run(lctx, cap, proj, tokens, k, 0, 0, {}, full, false, false, pmtp);
    auto compare = [&](const char *name, const lens_result &a, const lens_result &b)
    {
      int top1_diff = 0, id_diff = 0;
      float max_diff = 0;
      for (int r = 0; r < a.n_rows && a.n_rows == b.n_rows; r++)
      {
        top1_diff += a.top_ids[r * k] != b.top_ids[r * k];
        for (int j = 0; j < k; j++)
        {
          id_diff += a.top_ids[r * k + j] != b.top_ids[r * k + j];
          max_diff = std::max(max_diff, std::abs(a.top_probs[r * k + j] - b.top_probs[r * k + j]));
        }
        id_diff += a.target_ids[r] != b.target_ids[r];
        id_diff += a.target_rank[r] != b.target_rank[r];
        for (float d : {a.entropy[r] - b.entropy[r], a.target_prob[r] - b.target_prob[r], a.kl[r] - b.kl[r]})
          max_diff = std::max(max_diff, std::abs(d));
      }
      printf("%s vs full: cells=%d/%d top-1 diffs=%d id/rank diffs=%d max abs diff=%g\n", name, a.n_rows, b.n_rows, top1_diff, id_diff, max_diff);
    };
    compare("one by one", out.res, full.res);
    compare("one step", jump.res, full.res);
    if (pmtp)
    {
      compare("mtp one by one", out.mtp, full.mtp);
      compare("mtp one step", jump.mtp, full.mtp);
    }
    float max_h = 0;
    auto full_hidden = cap.hidden;
    if (pmtp)
    {
      full_hidden.push_back(cap.h_nextn);
      full_hidden.insert(full_hidden.end(), mtp.cap.hidden.begin(), mtp.cap.hidden.end());
    }
    size_t n_h = 0;
    for (size_t l = 0; l < full_hidden.size(); l++)
    {
      n_h += full_hidden[l].size() == inc_hidden[l].size();
      for (size_t i = 0; i < full_hidden[l].size() && i < inc_hidden[l].size(); i++)
        max_h = std::max(max_h, std::abs(full_hidden[l][i] - inc_hidden[l][i]));
    }
    printf("one by one vs full: hidden states max abs diff=%g (%zu/%zu slots same size)\n", max_h, n_h, full_hidden.size());
    printf("full run: forward %.1f ms, projection %.1f ms\n", full.t_decode_ms, full.t_project_ms);
  }

  // sanity check: last lens layer must agree with the model's own prediction
  // the lens pass skips the output head, so run a plain pass for the logits
  const int n_vocab = llama_vocab_n_tokens(vocab);
  int mismatches = 0;
  const int64_t n_embd = llama_model_n_embd(model);
  std::vector<float> h_ref; // llama.cpp's own MTP input
  if (pmtp)
    llama_set_embeddings_nextn(lctx, true, false);
  llama_memory_clear(llama_get_memory(lctx), true);
  for (int i0 = 0; i0 < out.n_tokens; i0 += n_ub)
  {
    const int nb = std::min(n_ub, out.n_tokens - i0);
    if (lens_decode(lctx, tokens, i0, nb) != 0)
      return 1;
    for (int t = 0; pmtp && t < nb; t++)
    {
      const float *h = llama_get_embeddings_nextn_ith(lctx, t);
      h_ref.insert(h_ref.end(), h, h + n_embd);
    }
    for (int t = 0; t < nb; t++)
    {
      const float *logits = llama_get_logits_ith(lctx, t);
      int best = 0;
      for (int v = 1; v < n_vocab; v++)
        if (logits[v] > logits[best])
          best = v;
      const int row = (out.n_layers - 1) * out.n_tokens + i0 + t;
      if (out.res.top_ids[row * k] != best)
        mismatches++;
    }
  }
  float last_kl = 0;
  for (int t = 0; t < out.n_tokens; t++)
    last_kl = std::max(last_kl, out.res.kl[(out.n_layers - 1) * out.n_tokens + t]);

  // MTP check: llama.cpp's own MTP context (no capture) on its own h_nextn, as in its MTP draft
  // column t: NextN reads h at t and token t + 1, so its guess is for token t + 2
  int mtp_mismatches = 0, acc_prompt = 0, acc_main = 0;
  float max_dh = 0;
  if (pmtp)
  {
    for (size_t i = 0; i < h_ref.size() && i < cap.h_nextn.size(); i++)
      max_dh = std::max(max_dh, std::abs(h_ref[i] - cap.h_nextn[i]));
    mparams_ctx.cb_eval = nullptr;
    mparams_ctx.cb_eval_user_data = nullptr;
    llama_context *mref = llama_init_from_model(model, mparams_ctx);
    if (!mref)
      return 1;
    const int s_last = out.n_mtp - 1;
    for (int i0 = 0; i0 < out.n_tokens; i0 += n_ub)
    {
      const int nb = std::min(n_ub, out.n_tokens - i0);
      if (lens_mtp_decode(mref, tokens, h_ref, n_embd, i0, nb, true) != 0)
        return 1;
      for (int i = 0; i < nb; i++)
      {
        const int t = i0 + i - 1;
        if (t < 0)
          continue;
        const float *logits = llama_get_logits_ith(mref, i);
        int best = 0;
        for (int v = 1; v < n_vocab; v++)
          if (logits[v] > logits[best])
            best = v;
        mtp_mismatches += out.mtp.top_ids[(s_last * out.n_tokens + t) * k] != best;
      }
    }
    llama_free(mref);
    for (int t = 0; t + 1 < out.n_tokens; t++)
    {
      const int guess = out.mtp.top_ids[(s_last * out.n_tokens + t) * k];
      acc_main += guess == out.res.top_ids[((out.n_layers - 1) * out.n_tokens + t + 1) * k];
      acc_prompt += t + 2 < out.n_tokens && guess == tokens[t + 2];
    }
  }

  printf("arch=%s backend=%s tokens=%d layers=%d time=%.1f ms (forward %.1f ms, projection %.1f ms) last-layer mismatches=%d kl=%g\n",
         arch.c_str(), proj.backend_name(), out.n_tokens, out.n_layers, (t1 - t0) / 1000.0, out.t_decode_ms, out.t_project_ms, mismatches, last_kl);
  if (pmtp)
    printf("mtp: mismatches vs llama.cpp MTP=%d of %d, h_nextn max abs diff=%g, t+2 acceptance: prompt %d/%d, main model %d/%d\n",
           mtp_mismatches, out.n_tokens - 1, max_dh, acc_prompt, std::max(0, out.n_tokens - 2), acc_main, out.n_tokens - 1);
  printf("\n");

  // table: layers top to bottom (last layer first), positions across
  printf("%6s", "");
  for (int t = 0; t < out.n_tokens; t++)
    printf(" | %-14.14s", piece(vocab, tokens[t]).c_str());
  printf("\n");
  // MTP rows: column t shows the guess for token t + 2
  for (int s = out.n_mtp - 1; s >= 0; s--)
  {
    static const char *labels[] = {"mtp.e", "mtp.a", "mtp"};
    printf("%6s", labels[s]);
    for (int t = 0; t < out.n_tokens; t++)
    {
      const int row = s * out.n_tokens + t;
      char cell[64] = "";
      if (out.mtp.top_ids[row * k] >= 0)
        snprintf(cell, sizeof(cell), "%s %.2f", piece(vocab, out.mtp.top_ids[row * k]).c_str(), out.mtp.top_probs[row * k]);
      printf(" | %-14.14s", cell);
    }
    printf("\n");
  }
  for (int l = out.n_layers - 1; l >= 0; l--)
  {
    printf("%6s", l == 0 ? "embd" : std::to_string(l - 1).c_str());
    for (int t = 0; t < out.n_tokens; t++)
    {
      const int row = l * out.n_tokens + t;
      char cell[64];
      snprintf(cell, sizeof(cell), "%s %.2f", piece(vocab, out.res.top_ids[row * k]).c_str(), out.res.top_probs[row * k]);
      printf(" | %-14.14s", cell);
    }
    printf("\n");
  }

  if (!json_path.empty())
  {
    std::ofstream f(json_path);
    f << "{\"tokens\":[";
    for (int t = 0; t < out.n_tokens; t++)
      f << (t ? "," : "") << "\"" << piece(vocab, tokens[t]) << "\"";
    f << "],\"n_layers\":" << out.n_layers << ",\"k\":" << k << ",\"cells\":[";
    for (int r = 0; r < out.res.n_rows; r++)
    {
      f << (r ? "," : "") << "{\"top\":[";
      for (int j = 0; j < k; j++)
        f << (j ? "," : "") << "[\"" << piece(vocab, out.res.top_ids[r * k + j]) << "\"," << out.res.top_probs[r * k + j] << "]";
      f << "],\"entropy\":" << out.res.entropy[r] << ",\"target_prob\":" << out.res.target_prob[r] << ",\"target_rank\":" << out.res.target_rank[r] << ",\"kl\":" << out.res.kl[r] << "}";
    }
    f << "]";
    if (out.n_mtp > 0)
    {
      // [stage][t], guess for token t + 2, null for the last position
      f << ",\"mtp_stages\":" << out.n_mtp << ",\"mtp\":[";
      for (int r = 0; r < out.mtp.n_rows; r++)
      {
        f << (r ? "," : "");
        if (out.mtp.top_ids[r * k] < 0)
        {
          f << "null";
          continue;
        }
        f << "{\"top\":[";
        for (int j = 0; j < k; j++)
          f << (j ? "," : "") << "[\"" << piece(vocab, out.mtp.top_ids[r * k + j]) << "\"," << out.mtp.top_probs[r * k + j] << "]";
        f << "],\"entropy\":" << out.mtp.entropy[r] << "}";
      }
      f << "]";
    }
    f << "}\n";
  }

  if (mtp.ctx)
    llama_free(mtp.ctx);
  llama_free(lctx);
  llama_model_free(model);
  return 0;
}
