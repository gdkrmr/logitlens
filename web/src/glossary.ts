// glossary markers (#23): a small "i" button next to a term opens a short note.
// Click, tap, Enter or Space toggles it; a mouse also opens it on hover. One shared
// live region shows the note, so screen readers read it when it opens.

const TERMS: Record<string, { title: string; ask: string; text: string }> = {
  token: {
    title: 'Token',
    ask: 'What is a token?',
    text: 'A piece of text the model handles as one unit: a whole word, part of a longer word, or a punctuation mark. Most tokens include the space before them.',
  },
  layer: {
    title: 'Layer',
    ask: 'What is a layer?',
    text: 'One processing step inside the model. The text passes through all layers in order, and each one refines the guess for the next word. Small models have about 20 to 30.',
  },
  embedding: {
    title: 'Embedding',
    ask: 'What is an embedding?',
    text: 'The starting point: each token turned into a long list of numbers, before any layer has worked on it. The bottom row reads a guess straight from these numbers.',
  },
  probability: {
    title: 'Probability',
    ask: 'What does probability mean here?',
    text: 'How sure a layer is about its top word, from 0% to 100%. A stronger blue means more confident.',
  },
  entropy: {
    title: 'Entropy',
    ask: 'What is entropy?',
    text: 'How spread out the guesses are. Low: the layer favors one word. High: many words seem about equally likely. Measured in nats.',
  },
  kl: {
    title: 'KL divergence',
    ask: 'What is KL divergence?',
    text: "How different a layer's whole list of guesses is from the final layer's. 0 means the same. A stronger blue means closer to the final answer; the scale is logarithmic.",
  },
  mtp: {
    title: 'MTP (multi-token prediction)',
    ask: 'What is MTP?',
    text: 'An extra small block some models have. It takes the last layer and the next word, and guesses the word after that, two tokens ahead. Its guesses are used to speed up writing text. The MTP rows put each guess in the column of the word it guesses, so they line up with the layers below; the first column is empty.',
  },
  mtpstages: {
    title: 'MTP steps (proj, attn, MTP)',
    ask: 'What are the three MTP rows?',
    text: 'The MTP block works in three steps, read from the bottom up. proj: it mixes the last layer with the next word. attn: it looks back over the text. MTP: it refines that into its output. Each row reads a guess out at that step, like the layer rows do, so you can watch the guess form.',
  },
  draft: {
    title: 'Draft tokens',
    ask: 'What are draft tokens?',
    text: "Tokens the MTP block guesses past the end of your prompt. It starts from the model's final prediction, then feeds each of its own guesses back in, like llama.cpp does for speculative decoding. The full model would then check them; here they are not checked.",
  },
  parameters: {
    title: 'Parameters',
    ask: 'What are parameters?',
    text: 'The numbers a model learned in training. More parameters usually means a more capable, but larger and slower, model. 135M means 135 million.',
  },
  quantization: {
    title: 'Quantization',
    ask: 'What is quantization?',
    text: "Storing the model's numbers with fewer digits to make the file smaller. Q4 files are small and slightly less exact; Q8_0 and F16 are larger and closer to the original.",
  },
  chat: {
    title: 'Chat format',
    ask: 'What is the chat format?',
    text: 'Chat models are trained on conversations with special markers around each message. With this on, your prompt is wrapped as a user message, and you see how the model starts its reply.',
  },
  webgpu: {
    title: 'WebGPU',
    ask: 'What is WebGPU?',
    text: 'Lets the browser use your graphics card, which runs the model much faster. Without it, the model runs on the main processor (CPU).',
  },
};

/** markup for a marker, for HTML built in code */
export function info(term: string): string {
  const t = TERMS[term];
  return `<button type="button" class="info" data-term="${term}" aria-label="${t.ask}" aria-expanded="false" aria-controls="gloss">i</button>`;
}

let open: HTMLButtonElement | null = null;
let pinned = false;
let hideTimer = 0;

export function initGlossary() {
  const pop = document.getElementById('gloss')!;
  // markers written in index.html get the same attributes as info()
  document.querySelectorAll<HTMLButtonElement>('button.info[data-term]').forEach((b) => {
    const t = TERMS[b.dataset.term!];
    // an unknown term must not break the page
    if (!t) return;
    b.type = 'button';
    b.textContent = 'i';
    b.setAttribute('aria-label', t.ask);
    b.setAttribute('aria-expanded', 'false');
    b.setAttribute('aria-controls', 'gloss');
  });

  const close = () => {
    clearTimeout(hideTimer);
    open?.setAttribute('aria-expanded', 'false');
    open = null;
    pinned = false;
    pop.replaceChildren();
  };

  const show = (b: HTMLButtonElement) => {
    const t = TERMS[b.dataset.term!];
    if (!t) return;
    if (open && open !== b) close();
    open = b;
    b.setAttribute('aria-expanded', 'true');
    pop.replaceChildren(Object.assign(document.createElement('strong'), { textContent: t.title }), ' ', t.text);
    // below the marker, or above when there is no room; kept inside the window
    const r = b.getBoundingClientRect();
    const w = pop.offsetWidth;
    const h = pop.offsetHeight;
    pop.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, r.left + r.width / 2 - w / 2))}px`;
    pop.style.top = `${r.bottom + 8 + h > window.innerHeight - 8 && r.top - 8 - h > 8 ? r.top - 8 - h : r.bottom + 8}px`;
  };

  const marker = (ev: Event) => (ev.target as Element | null)?.closest?.<HTMLButtonElement>('button.info[data-term]') ?? null;

  document.addEventListener('click', (ev) => {
    const b = marker(ev);
    if (!b) return;
    // a marker inside a summary must not toggle the details
    ev.preventDefault();
    if (open === b && pinned) return close();
    show(b);
    pinned = true;
  });
  document.addEventListener('pointerover', (ev) => {
    const b = marker(ev);
    if (b && (ev as PointerEvent).pointerType === 'mouse' && !pinned) {
      clearTimeout(hideTimer);
      show(b);
    }
  });
  // a hover-opened note closes after a short delay, so the pointer can move onto it and stay there
  const closeSoon = () => {
    clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => !pinned && close(), 300);
  };
  document.addEventListener('pointerout', (ev) => {
    if (marker(ev) && marker(ev) === open && !pinned) closeSoon();
  });
  pop.addEventListener('pointerenter', () => clearTimeout(hideTimer));
  pop.addEventListener('pointerleave', () => open && !pinned && closeSoon());
  // tap or click anywhere else closes it
  document.addEventListener('pointerdown', (ev) => {
    if (open && !marker(ev) && !pop.contains(ev.target as Node)) close();
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && open) {
      const b = open;
      close();
      b.focus();
    }
  });
  document.addEventListener('focusout', (ev) => {
    if (open && ev.target === open && pinned) close();
  });
  window.addEventListener('scroll', () => open && close(), { passive: true, capture: true });
  window.addEventListener('resize', () => open && close());
}
