// native logit lens: prints the top-1 token per layer and position
// usage: lens-cli -m model.gguf -p "prompt" [-k 5] [-ngl 99] [--json out.json]

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "wllama-lens.h"

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
  int k = 5, ngl = 99;
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
    else
    {
      fprintf(stderr, "usage: %s -m model.gguf -p prompt [-k 5] [-ngl 99] [--json out.json]\n", argv[0]);
      return 1;
    }
  }

  llama_log_set([](ggml_log_level level, const char *text, void *)
                { if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr); },
                nullptr);
  llama_backend_init();

  auto mparams = llama_model_default_params();
  mparams.n_gpu_layers = ngl;
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

  std::vector<llama_token> tokens(prompt.size() + 8);
  int n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), tokens.data(), (int)tokens.size(), true, true);
  tokens.resize(n);

  lens_capture cap;
  auto cparams = llama_context_default_params();
  cparams.n_ctx = std::max(512, n);
  cparams.n_batch = lens_batch;
  cparams.n_ubatch = lens_batch;
  cparams.n_seq_max = 1;
  cparams.cb_eval = lens_cb_eval;
  cparams.cb_eval_user_data = &cap;
  llama_context *lctx = llama_init_from_model(model, cparams);

  lens_projector proj;
  proj.init(model);

  lens_run_output out;
  const int64_t t0 = ggml_time_us();
  lens_run(lctx, cap, proj, tokens, k, 0, 0, {}, out);
  const int64_t t1 = ggml_time_us();

  // sanity check: last lens layer must agree with the model's own prediction
  // the lens pass skips the output head, so run a plain pass for the logits
  const int n_vocab = llama_vocab_n_tokens(vocab);
  int mismatches = 0;
  llama_memory_clear(llama_get_memory(lctx), true);
  for (int i0 = 0; i0 < out.n_tokens; i0 += lens_batch)
  {
    const int nb = std::min(lens_batch, out.n_tokens - i0);
    if (lens_decode(lctx, tokens, i0, nb) != 0)
      return 1;
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

  printf("arch=%s backend=%s tokens=%d layers=%d time=%.1f ms (forward %.1f ms, projection %.1f ms) last-layer mismatches=%d kl=%g\n\n",
         arch.c_str(), proj.backend_name(), out.n_tokens, out.n_layers, (t1 - t0) / 1000.0, out.t_decode_ms, out.t_project_ms, mismatches, last_kl);

  // table: layers top to bottom (last layer first), positions across
  printf("%6s", "");
  for (int t = 0; t < out.n_tokens; t++)
    printf(" | %-14.14s", piece(vocab, tokens[t]).c_str());
  printf("\n");
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
    f << "]}\n";
  }

  llama_free(lctx);
  llama_model_free(model);
  return 0;
}
