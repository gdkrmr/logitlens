import { Template } from '@huggingface/jinja';

/**
 * Renders the model's chat template around one user message, ending with the start of the
 * assistant turn, so the last position predicts the first token of the reply.
 */
export class ChatFormat {
  private tpl: Template;

  constructor(source: string) {
    this.tpl = new Template(source);
  }

  /** null if the template is missing, does not parse, or fails a trial render */
  static from(source: string | null): ChatFormat | null {
    if (!source) return null;
    try {
      const f = new ChatFormat(source);
      f.render('');
      return f;
    } catch {
      return null;
    }
  }

  render(user: string): string {
    return this.tpl.render({
      messages: [{ role: 'user', content: user }],
      add_generation_prompt: true,
      // empty: the tokenizer adds BOS itself where the model wants it, so a BOS written by the template would be doubled
      bos_token: '',
      eos_token: '',
      // raise_exception and strftime_now (today's date, as in llama.cpp) are built into the renderer
    });
  }
}
