/*
 * Runs the model off the main thread so the page stays smooth while it thinks.
 * Protocol (main ⇄ worker):
 *   → { type: "load", base }                       ← progress…, ready | error
 *   → { type: "generate", prompt, max, … }          ← token… (one per prompt or generated token), done | error
 *   → { type: "stop" }
 */

import { GlanceModel, rng, sample, softmax, topCandidates, type SampleOptions } from "./model";
import { Tokenizer, type TokenizerFile } from "./tokenizer";
import type { FromWorker, ToWorker } from "./protocol";

// typed by hand so this file compiles with the DOM lib only
const ctx = self as unknown as {
  postMessage: (m: FromWorker, transfer: Transferable[]) => void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};
let model: GlanceModel | null = null;
let tok: Tokenizer | null = null;
let stopRequested = false;

const post = (m: FromWorker, transfer: Transferable[] = []) => ctx.postMessage(m, transfer);

async function fetchWithProgress(url: string, onProgress: (loaded: number, total: number) => void): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(res.status === 404 ? "not-trained" : `Could not load ${url} (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out.buffer;
}

async function load(base: string) {
  const [tokFile, weights] = await Promise.all([
    fetch(`${base}tokenizer.json`).then((r) => {
      if (!r.ok) throw new Error("not-trained");
      return r.json() as Promise<TokenizerFile>;
    }),
    fetchWithProgress(`${base}glance.bin`, (loaded, total) => post({ type: "progress", loaded, total })),
  ]);
  tok = new Tokenizer(tokFile);
  model = new GlanceModel(weights);
  post({ type: "ready", config: model.cfg, meta: model.meta });
}

async function generate(m: Extract<ToWorker, { type: "generate" }>) {
  if (!model || !tok) throw new Error("The model is not loaded yet.");
  stopRequested = false;
  const opts: SampleOptions = { temperature: m.temperature, topK: m.topK, topP: m.topP };
  const random = rng(m.seed);
  let ids = tok.encode(m.prompt);
  if (ids.length === 0) ids = [tok.eot];
  ids = ids.slice(-Math.floor(model.cfg.ctx * 0.75)); // leave room to generate
  const room = model.cfg.ctx - ids.length;
  const max = Math.max(0, Math.min(m.max, room));

  model.reset();
  let logits: Float32Array | null = null;
  let index = 0;

  const emit = (id: number, isPrompt: boolean, attn: Float32Array[], prev: Float32Array | null) => {
    // the probability the model gave to THIS token one step earlier
    const p = prev ? softmax(prev)[id] : null;
    const flat = new Float32Array(attn.reduce((n, a) => n + a.length, 0));
    let o = 0;
    for (const a of attn) {
      flat.set(a, o);
      o += a.length;
    }
    post({ type: "token", index, id, piece: tok!.piece(id), isPrompt, prob: p, attn: flat }, [flat.buffer]);
    index++;
  };

  const candidates = (lg: Float32Array) => topCandidates(softmax(lg), 8).map((c) => ({ ...c, piece: tok!.piece(c.id) }));

  for (const id of ids) {
    const prev = logits;
    const out = model.step(id);
    emit(id, true, out.attn, prev);
    logits = out.logits;
  }
  post({ type: "candidates", index: index - 1, top: candidates(logits!) });

  const generated: number[] = [];
  for (let i = 0; i < max; i++) {
    if (stopRequested) break;
    const id = sample(logits!, opts, random);
    if (id === tok.eot) break;
    const prev = logits;
    const out = model.step(id);
    emit(id, false, out.attn, prev);
    generated.push(id);
    logits = out.logits;
    post({ type: "candidates", index: index - 1, top: candidates(logits) });
    if (i % 2 === 1) await new Promise((r) => setTimeout(r, 0)); // let "stop" messages in
  }
  post({ type: "done", reason: stopRequested ? "stopped" : generated.length >= max ? (max === room ? "context" : "length") : "end" });
}

ctx.onmessage = async (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  try {
    if (m.type === "load") await load(m.base);
    else if (m.type === "generate") await generate(m);
    else if (m.type === "stop") stopRequested = true;
  } catch (err) {
    post({ type: "error", message: (err as Error).message });
  }
};
