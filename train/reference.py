"""NumPy forward pass of Glance, line for line the same maths as train/model.py.

Used to check that the browser engine (lib/engine/model.ts) computes exactly
what PyTorch computes, without needing PyTorch in the test environment.
"""

from __future__ import annotations

import numpy as np


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def forward(W: dict[str, np.ndarray], cfg: dict, ids: list[int]):
    """Returns logits (T, V) and attention weights [layer] (H, T, T)."""
    T, d, H = len(ids), cfg["d"], cfg["n_head"]
    hd = d // H
    x = W["wte.weight"][ids] + W["wpe.weight"][:T]
    mask = np.tril(np.ones((T, T), dtype=bool))
    attns = []
    for i in range(cfg["n_layer"]):
        p = f"blocks.{i}."
        h = layer_norm(x, W[p + "ln1.weight"], W[p + "ln1.bias"])
        qkv = h @ W[p + "attn.qkv.weight"].T + W[p + "attn.qkv.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(T, H, hd).transpose(1, 0, 2)
        k = k.reshape(T, H, hd).transpose(1, 0, 2)
        v = v.reshape(T, H, hd).transpose(1, 0, 2)
        att = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
        att = np.where(mask, att, -np.inf)
        att = softmax(att)
        attns.append(att)
        y = (att @ v).transpose(1, 0, 2).reshape(T, d)
        x = x + y @ W[p + "attn.proj.weight"].T + W[p + "attn.proj.bias"]
        h = layer_norm(x, W[p + "ln2.weight"], W[p + "ln2.bias"])
        h = gelu_tanh(h @ W[p + "mlp.fc.weight"].T + W[p + "mlp.fc.bias"])
        x = x + h @ W[p + "mlp.proj.weight"].T + W[p + "mlp.proj.bias"]
    x = layer_norm(x, W["ln_f.weight"], W["ln_f.bias"])
    return x @ W["wte.weight"].T, attns


def random_state(cfg: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """Random weights with the exact names and shapes of the PyTorch model."""
    r = np.random.default_rng(seed)
    d, V = cfg["d"], cfg["vocab_size"]
    n = lambda *s: (r.standard_normal(s) * 0.2).astype(np.float32)  # noqa: E731
    W = {"wte.weight": n(V, d), "wpe.weight": n(cfg["ctx"], d), "ln_f.weight": 1 + n(d), "ln_f.bias": n(d)}
    for i in range(cfg["n_layer"]):
        p = f"blocks.{i}."
        W |= {
            p + "ln1.weight": 1 + n(d), p + "ln1.bias": n(d),
            p + "attn.qkv.weight": n(3 * d, d), p + "attn.qkv.bias": n(3 * d),
            p + "attn.proj.weight": n(d, d), p + "attn.proj.bias": n(d),
            p + "ln2.weight": 1 + n(d), p + "ln2.bias": n(d),
            p + "mlp.fc.weight": n(4 * d, d), p + "mlp.fc.bias": n(4 * d),
            p + "mlp.proj.weight": n(d, 4 * d), p + "mlp.proj.bias": n(d),
        }
    return W
