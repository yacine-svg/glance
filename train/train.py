"""Train Glance from scratch.

    python -m train.train                 # defaults: ~12M parameters, 5,000 steps (~15 min on a Colab T4)
    python -m train.train --steps 300     # quick smoke test

Outputs in --out (default out/):
    tokenizer.json       the BPE vocabulary
    glance.bin           int8 weights for the browser
    training_log.json    loss curve, evaluations and text samples over training
    best.pt              PyTorch checkpoint (best validation loss)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .data import download
from .export import export
from .model import Config, Glance
from .tokenizer import Tokenizer

PROMPTS = ['"My dear', "It was a dark and", "Elizabeth looked at him and", "The house stood"]


def build_data(out: str, vocab_size: int):
    print("1/4 Corpus")
    books = download(os.path.join(out, "books"))
    if not books:
        raise SystemExit("No book could be downloaded. Check the internet connection.")

    tok_path = os.path.join(out, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tokenizer.load(tok_path)
    else:
        print("2/4 Tokenizer (byte-level BPE)")
        # train the merges on a sample of every book (the vocabulary converges well before all 25 MB)
        sample = "".join(text[: 400_000] for _, text in books)
        t0 = time.time()
        tok = Tokenizer.train(sample, vocab_size=vocab_size)
        tok.save(tok_path)
        print(f"  {tok.vocab_size} tokens in {time.time() - t0:.0f}s")

    print("3/4 Encoding")
    train_ids, val_ids = [], []
    for title, text in books:
        ids = tok.encode(text) + [tok.eot]
        cut = int(len(ids) * 0.97)  # the last 3% of every book is held out
        train_ids += ids[:cut]
        val_ids += ids[cut:]
    train = np.array(train_ids, dtype=np.uint16)
    val = np.array(val_ids, dtype=np.uint16)
    chars = sum(len(t) for _, t in books)
    print(f"  {len(train) / 1e6:.2f} M training tokens, {len(val) / 1e3:.0f} k validation tokens ({chars / len(train_ids + val_ids):.2f} characters per token)")
    return tok, train, val, [t for t, _ in books]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: no GPU found. In Colab: Runtime → Change runtime type → T4 GPU.")

    tok, train, val, titles = build_data(args.out, args.vocab)

    cfg = Config(vocab_size=tok.vocab_size, ctx=args.ctx, d=args.d, n_layer=args.layers, n_head=args.heads, dropout=args.dropout)
    model = Glance(cfg).to(device)
    print(f"4/4 Training a {model.n_params() / 1e6:.1f} M-parameter model on {device}")

    # mixed precision: bfloat16 where supported, otherwise float16 with loss scaling (T4)
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8  # Ampere+; a T4 uses float16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))
    opt = model.optimizer(args.lr)

    def batch(split: np.ndarray):
        ix = np.random.randint(0, len(split) - cfg.ctx - 1, args.batch)
        x = torch.from_numpy(np.stack([split[i : i + cfg.ctx] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([split[i + 1 : i + 1 + cfg.ctx] for i in ix]).astype(np.int64))
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def lr_at(step: int) -> float:
        # linear warm-up, then cosine decay to min-lr
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(1.0, p)))

    @torch.no_grad()
    def evaluate(n: int = 40) -> dict:
        model.eval()
        out = {}
        for name, split in (("train", train), ("val", val)):
            losses = []
            for _ in range(n):
                x, y = batch(split)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
                    _, loss, _ = model(x, y)
                losses.append(loss.item())
            out[name] = float(np.mean(losses))
        model.train()
        return out

    @torch.no_grad()
    def samples() -> list[dict]:
        model.eval()
        g = torch.Generator(device=device).manual_seed(7)  # same randomness at every checkpoint, so samples are comparable
        res = []
        for p in PROMPTS:
            idx = torch.tensor([tok.encode(p)], device=device)
            for _ in range(60):
                logits, _, _ = model(idx[:, -cfg.ctx :])
                probs = (logits[:, -1, :].float() / 0.8).softmax(-1)
                v, i = probs.topk(40)
                nxt = i.gather(1, torch.multinomial(v / v.sum(), 1, generator=g))
                idx = torch.cat([idx, nxt], 1)
            res.append({"prompt": p, "text": tok.decode(idx[0].tolist())})
        model.train()
        return res

    log = {"config": cfg.dict(), "args": vars(args), "books": titles, "train_tokens": int(len(train)), "loss": [], "evals": [], "samples": []}
    best = float("inf")
    t0 = time.time()
    tokens_per_step = args.batch * cfg.ctx
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate()
            log["evals"].append({"step": step, **ev})
            print(f"  step {step:5d}  train {ev['train']:.3f}  val {ev['val']:.3f}  ({time.time() - t0:.0f}s)")
            if ev["val"] < best:
                best = ev["val"]
                torch.save({"model": model.state_dict(), "config": cfg.dict(), "step": step, "val": best}, os.path.join(args.out, "best.pt"))
        if step % args.sample_every == 0 or step == args.steps:
            s = samples()
            log["samples"].append({"step": step, "samples": s})
            print(f"    “{s[0]['text'][:110]}…”")
        if step == args.steps:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch(train)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
            _, loss, _ = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            log["loss"].append({"step": step, "loss": round(loss.item(), 4), "lr": lr_at(step)})
        if step % 100 == 0 and step:
            el = time.time() - t0
            print(f"    step {step}: loss {loss.item():.3f}, {step * tokens_per_step / el / 1e3:.0f}k tokens/s")

    # ---------- export the best checkpoint ----------
    ck = torch.load(os.path.join(args.out, "best.pt"), map_location="cpu")
    state = {k: v.float().numpy() for k, v in ck["model"].items()}
    meta = {
        "params": model.n_params(),
        "best_step": ck["step"],
        "val_loss": round(ck["val"], 4),
        "perplexity": round(math.exp(ck["val"]), 2),
        "tokens_seen": (ck["step"] or 1) * tokens_per_step,
        "books": titles,
        "train_minutes": round((time.time() - t0) / 60, 1),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    info = export(state, cfg.dict(), os.path.join(args.out, "glance.bin"), meta)
    log["meta"] = meta
    with open(os.path.join(args.out, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f)
    print(f"Done. Best validation loss {ck['val']:.3f} (perplexity {math.exp(ck['val']):.1f}) at step {ck['step']}.")
    print(f"glance.bin: {info['bytes'] / 1e6:.1f} MB. Upload glance.bin, tokenizer.json and training_log.json to public/model/.")


if __name__ == "__main__":
    main()
