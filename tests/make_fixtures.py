"""Builds test fixtures with the Python side (NumPy only) for tests/glance.test.ts:
a small random model exported to glance.bin with its reference outputs, and a
tokenizer with reference encodings. Usage: python3 tests/make_fixtures.py <dir>
"""

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.export import export, load  # noqa: E402
from train.reference import forward, random_state  # noqa: E402
from train.tokenizer import Tokenizer  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

# model
cfg = {"vocab_size": 300, "ctx": 32, "d": 48, "n_layer": 3, "n_head": 4, "dropout": 0.0}
W = random_state(cfg, 1)
export(W, cfg, os.path.join(out, "tiny.bin"), {"params": 1})
_, Wq = load(os.path.join(out, "tiny.bin"))
ids = [5, 17, 299, 0, 42, 42, 7, 120, 3, 250, 11, 64]
logits, attn = forward(Wq, cfg, ids)
logits_f32, _ = forward(W, cfg, ids)
json.dump(
    {
        "ids": ids,
        "logits": logits.tolist(),
        "attn": [a.tolist() for a in attn],  # [layer][head][t][t]
        "quant_rel_error": float(np.abs(logits - logits_f32).max() / np.abs(logits_f32).max()),
    },
    open(os.path.join(out, "tiny.json"), "w"),
)

# tokenizer
random.seed(0)
words = "the of and to a in that it was he her his she you not with as had for my on at but be him is all by have so which this from said".split()
text = "\n".join(" ".join(random.choice(words) for _ in range(random.randint(5, 20))).capitalize() + "." for _ in range(8000))
tok = Tokenizer.train(text, vocab_size=600, verbose=False)
tok.save(os.path.join(out, "tokenizer.json"))
cases = [
    "“Well,” said Mr. Darcy, “I can’t—it’s 1813, café!”",
    "She had not been there   since the 12th of May.\nIt was all.",
    "ünïcode ☃ and 你好, emoji 🙂",
]
json.dump([{"text": c, "ids": tok.encode(c), "decoded": tok.decode(tok.encode(c))} for c in cases], open(os.path.join(out, "tokens.json"), "w"))
print("fixtures written to", out)
