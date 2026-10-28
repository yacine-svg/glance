"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { GlanceConfig, GlanceMeta } from "@/lib/engine/model";
import type { Candidate, FromWorker, ToWorker } from "@/lib/engine/protocol";

export type Tok = { index: number; id: number; piece: string; isPrompt: boolean; prob: number | null; attn: Float32Array };
export type Status =
  | { kind: "loading"; loaded: number; total: number }
  | { kind: "missing" }
  | { kind: "ready" }
  | { kind: "running" }
  | { kind: "error"; message: string };

export type GenerateOptions = { prompt: string; max: number; temperature: number; topK: number; topP: number; seed: number };

/** Owns the model's Web Worker and the tokens it streams back. */
export function useGlance() {
  const worker = useRef<Worker | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "loading", loaded: 0, total: 0 });
  const [config, setConfig] = useState<GlanceConfig | null>(null);
  const [meta, setMeta] = useState<GlanceMeta | null>(null);
  const [tokens, setTokens] = useState<Tok[]>([]);
  const [candidates, setCandidates] = useState<Record<number, Candidate[]>>({});
  const [doneReason, setDoneReason] = useState<string | null>(null);
  const buffer = useRef<Tok[]>([]);
  const frame = useRef<number | null>(null);

  // batch token updates per animation frame so long generations stay smooth
  const flush = useCallback(() => {
    frame.current = null;
    const add = buffer.current;
    buffer.current = [];
    if (add.length) setTokens((t) => [...t, ...add]);
  }, []);

  useEffect(() => {
    const w = new Worker(new URL("./engine/worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      switch (m.type) {
        case "progress":
          setStatus({ kind: "loading", loaded: m.loaded, total: m.total });
          break;
        case "ready":
          setConfig(m.config);
          setMeta(m.meta);
          setStatus({ kind: "ready" });
          break;
        case "token":
          buffer.current.push({ index: m.index, id: m.id, piece: m.piece, isPrompt: m.isPrompt, prob: m.prob, attn: m.attn });
          if (frame.current === null) frame.current = requestAnimationFrame(flush);
          break;
        case "candidates":
          setCandidates((c) => ({ ...c, [m.index]: m.top }));
          break;
        case "done":
          flush();
          setDoneReason(m.reason);
          setStatus({ kind: "ready" });
          break;
        case "error":
          setStatus(m.message === "not-trained" ? { kind: "missing" } : { kind: "error", message: m.message });
          break;
      }
    };
    const msg: ToWorker = { type: "load", base: "/model/" };
    w.postMessage(msg);
    return () => w.terminate();
  }, [flush]);

  const generate = useCallback((o: GenerateOptions) => {
    if (!worker.current) return;
    buffer.current = [];
    setTokens([]);
    setCandidates({});
    setDoneReason(null);
    setStatus({ kind: "running" });
    const msg: ToWorker = { type: "generate", ...o };
    worker.current.postMessage(msg);
  }, []);

  const stop = useCallback(() => {
    const msg: ToWorker = { type: "stop" };
    worker.current?.postMessage(msg);
  }, []);

  return { status, config, meta, tokens, candidates, doneReason, generate, stop };
}

/** Attention of token `tok` over earlier tokens for one layer and one head (or the mean of all heads). */
export function attentionOf(tok: Tok, cfg: GlanceConfig, layer: number, head: number | "mean"): Float32Array {
  const T = tok.index + 1;
  const H = cfg.n_head;
  const base = layer * H * T;
  const out = new Float32Array(T);
  if (head === "mean") {
    for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) out[t] += tok.attn[base + h * T + t] / H;
  } else {
    out.set(tok.attn.subarray(base + head * T, base + head * T + T));
  }
  return out;
}
