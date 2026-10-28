export const metadata = { title: "Inside the model" };

const STEPS = [
  {
    title: "Tokenizer",
    text: "Text is cut into tokens with a byte-level BPE tokenizer trained on the novels: frequent pairs of bytes are merged into new tokens 3,839 times, so common words like “ the” become one token and rare words are spelled from pieces. Any text can be encoded, even emoji.",
  },
  {
    title: "Embeddings",
    text: "Each token becomes a vector of 384 numbers, plus a second vector for its position in the passage (up to 256 tokens). Both are learned during training.",
  },
  {
    title: "6 transformer blocks",
    text: "Each block does two things. Attention: 6 heads each decide which earlier tokens matter for the current one and gather information from them. Then a small feed-forward network (384 → 1,536 → 384, GELU) transforms what was gathered. Layer normalisation comes before each step and the result is added back (residual connection), which keeps training stable.",
  },
  {
    title: "Causal mask",
    text: "A token can only attend to itself and the tokens before it, never to the future. That is what makes the model able to write: at every position it has learned to predict the next token without seeing it.",
  },
  {
    title: "Prediction",
    text: "After the last block, the vector is compared with every token's embedding (the same matrix as the input, called weight tying) to give a score for each of the 4,096 tokens. Softmax turns the scores into probabilities.",
  },
  {
    title: "Sampling",
    text: "Temperature sharpens or flattens the probabilities; top-k and top-p cut the unlikely tail; then one token is drawn at random and fed back in. Low temperature is safe and repetitive, high temperature is creative and less coherent.",
  },
];

const ENGINE = [
  ["Training", "PyTorch, AdamW with weight decay, cosine learning-rate schedule with warm-up, mixed precision, gradient clipping, on one free Colab GPU."],
  ["Compression", "The 12 M weights are stored as 8-bit integers with one scale per row: 12.5 MB instead of 49 MB, with almost no loss of quality."],
  ["In your browser", "A transformer engine written from scratch in TypeScript runs in a Web Worker: matrix products, LayerNorm, attention and GELU, with a key/value cache so each new token only costs one pass. About 75 tokens per second on a laptop."],
  ["Verified", "The browser engine is tested against an independent NumPy implementation of the same model: their outputs agree to within 0.00001."],
];

export default function Inside() {
  return (
    <main className="wrap" style={{ paddingTop: 36, paddingBottom: 60, maxWidth: 880 }}>
      <h1>Inside the model</h1>
      <p className="muted" style={{ marginTop: 14, fontSize: "1.1rem", maxWidth: "62ch" }}>
        Glance is a GPT-style transformer, the same family as the models behind ChatGPT, more than ten thousand times smaller than GPT-3. Every part was written by
        hand.
      </p>
      <ol style={{ listStyle: "none", padding: 0, margin: "36px 0 0", display: "grid", gap: 14 }}>
        {STEPS.map((s, i) => (
          <li key={s.title} className="panel" style={{ display: "grid", gridTemplateColumns: "2.2em 1fr", gap: 8 }}>
            <span style={{ fontFamily: "var(--serif)", fontSize: "1.4rem", color: "var(--heat)" }}>{i + 1}</span>
            <div>
              <h3 style={{ fontSize: "1.3rem" }}>{s.title}</h3>
              <p className="muted" style={{ marginTop: 8, lineHeight: 1.65 }}>
                {s.text}
              </p>
            </div>
          </li>
        ))}
      </ol>
      <h2 style={{ marginTop: 48 }}>From training to your screen</h2>
      <dl style={{ marginTop: 20, display: "grid", gap: 18 }}>
        {ENGINE.map(([k, v]) => (
          <div key={k}>
            <dt style={{ fontWeight: 600 }}>{k}</dt>
            <dd className="muted" style={{ margin: "4px 0 0", lineHeight: 1.65 }}>
              {v}
            </dd>
          </div>
        ))}
      </dl>
    </main>
  );
}
