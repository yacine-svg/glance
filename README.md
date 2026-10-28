# Glance: a language model you can watch think

Glance is a **12-million-parameter GPT-style transformer trained from scratch** on 25 classic English novels (Austen, Dickens, the Brontës, Shelley, Eliot…). It runs **entirely in the browser** with an inference engine written from scratch in TypeScript, and it shows, for every word it writes, **which earlier words each attention head looked at** and **what else it considered**.

Built by **Sekhri Mohammed Yassine** (MSc Artificial Intelligence & Data Science).

## What's built from scratch

| Part | Where | What it does |
| --- | --- | --- |
| Byte-level BPE tokenizer | `train/tokenizer.py`, `lib/engine/tokenizer.ts` | GPT-2-style pre-tokenisation, 3,839 learned merges (vocabulary 4,096). Training updates pair counts incrementally instead of recounting. The Python and TypeScript versions are tested to give identical token ids. |
| Transformer | `train/model.py` | Pre-LayerNorm decoder blocks, multi-head causal self-attention, GELU MLP, learned positional embeddings, weight tying, GPT-2 residual scaling at init |
| Training loop | `train/train.py` | AdamW (decay on matrices only), warm-up + cosine LR, mixed precision (bf16 / fp16 + loss scaling), gradient clipping, held-out evaluation, checkpoint samples with a fixed seed |
| Compression | `train/export.py` | Per-row absmax **int8 quantisation**: 49 MB → 12.5 MB, one self-describing binary file |
| Browser engine | `lib/engine/model.ts` | Matrix-vector products, LayerNorm, attention with a **KV cache**, tanh-GELU, temperature / top-k / top-p sampling with a seeded RNG, in a Web Worker (~75 tokens/s on a laptop CPU) |
| Reference model | `train/reference.py` | Independent NumPy forward pass. The browser engine must match it: max logit error 2.6 × 10⁻⁶ |

```
text ─▶ BPE tokens ─▶ token + position embeddings
        ─▶ 6 × [ LayerNorm → 6-head causal attention → +residual
                 LayerNorm → MLP 384→1536→384 (GELU) → +residual ]
        ─▶ LayerNorm ─▶ · embeddingᵀ (tied) ─▶ softmax ─▶ sample ─▶ next token
```

Default size: d = 384, 6 layers, 6 heads, context 256 tokens, vocabulary 4,096 ≈ 12.2 M parameters.

## Train it (free, ~20 minutes)

1. Open `train/Glance.ipynb` in Google Colab (File → Open notebook → GitHub → `yacine-svg/glance`).
2. Runtime → Change runtime type → **T4 GPU**, then **Run all**.
3. Upload the three downloaded files (`glance.bin`, `tokenizer.json`, `training_log.json`) to `public/model/`. Vercel redeploys and the model is live.

Locally with a GPU: `pip install torch numpy` then `python -m train.train` (try `--steps 300` for a smoke test).

## Run the site

```bash
npm install
npm run dev        # http://localhost:3000
npm test           # cross-language tests (bun + python3 with numpy)
```

Pages: **Write with it** (playground + attention inspector + attention map), **How it learned** (loss curve, samples at every checkpoint), **Inside the model** (architecture walk-through).

## Tests

`tests/make_fixtures.py` builds a random model and a tokenizer with the Python code; `tests/glance.test.ts` checks that the TypeScript engine reproduces the NumPy logits and every attention weight, that the tokenizers agree token for token, and that sampling behaves (greedy at low temperature, top-k respected, seeded reproducibility).
