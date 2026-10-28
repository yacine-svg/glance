"use client";

import { useEffect, useMemo, useState } from "react";

type Log = {
  config: { d: number; n_layer: number; n_head: number; ctx: number; vocab_size: number };
  books: string[];
  train_tokens: number;
  loss: { step: number; loss: number; lr: number }[];
  evals: { step: number; train: number; val: number }[];
  samples: { step: number; samples: { prompt: string; text: string }[] }[];
  meta?: { params: number; best_step: number; val_loss: number; perplexity: number; tokens_seen: number; train_minutes: number; device: string };
};

function LossChart({ log }: { log: Log }) {
  const W = 760;
  const H = 300;
  const P = { l: 44, r: 16, t: 16, b: 36 };
  const maxStep = Math.max(...log.loss.map((p) => p.step), ...log.evals.map((e) => e.step), 1);
  const all = [...log.loss.map((p) => p.loss), ...log.evals.map((e) => e.val)];
  const yMax = Math.ceil(Math.max(...all));
  const yMin = Math.floor(Math.min(...all));
  const x = (s: number) => P.l + (s / maxStep) * (W - P.l - P.r);
  const y = (v: number) => P.t + ((yMax - v) / Math.max(1e-6, yMax - yMin)) * (H - P.t - P.b);
  // light smoothing (EMA) of the noisy per-batch loss
  let ema = log.loss[0]?.loss ?? 0;
  const train = log.loss.map((p) => {
    ema = 0.85 * ema + 0.15 * p.loss;
    return `${x(p.step).toFixed(1)},${y(ema).toFixed(1)}`;
  });
  const ticks = Array.from({ length: yMax - yMin + 1 }, (_, i) => yMin + i);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training loss and validation loss falling over training steps">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={P.l} x2={W - P.r} y1={y(t)} y2={y(t)} stroke="rgba(190,205,255,0.12)" />
          <text x={P.l - 8} y={y(t) + 4} textAnchor="end">
            {t}
          </text>
        </g>
      ))}
      <text x={W / 2} y={H - 6} textAnchor="middle">
        training step
      </text>
      <polyline fill="none" stroke="#8fb4ff" strokeWidth="2" points={train.join(" ")} />
      <polyline fill="none" stroke="#f6b45e" strokeWidth="2.5" points={log.evals.map((e) => `${x(e.step)},${y(e.val)}`).join(" ")} />
      {log.evals.map((e) => (
        <circle key={e.step} cx={x(e.step)} cy={y(e.val)} r="3.5" fill="#f6b45e" />
      ))}
      <g transform={`translate(${W - P.r - 210}, ${P.t + 8})`}>
        <line x1="0" x2="18" y1="6" y2="6" stroke="#8fb4ff" strokeWidth="2" />
        <text x="24" y="10">
          training loss
        </text>
        <line x1="110" x2="128" y1="6" y2="6" stroke="#f6b45e" strokeWidth="2.5" />
        <text x="134" y="10">
          held-out loss
        </text>
      </g>
    </svg>
  );
}

export default function Training() {
  const [log, setLog] = useState<Log | null>(null);
  const [missing, setMissing] = useState(false);
  const [at, setAt] = useState(0);

  useEffect(() => {
    fetch("/model/training_log.json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((l: Log) => {
        setLog(l);
        setAt(l.samples.length - 1);
      })
      .catch(() => setMissing(true));
  }, []);

  const snap = useMemo(() => log?.samples[at], [log, at]);
  const evalAt = useMemo(() => (snap && log ? log.evals.find((e) => e.step === snap.step) : undefined), [snap, log]);

  if (missing)
    return (
      <div className="panel">
        <p className="muted">No training log yet: train the model and upload training_log.json to public/model/.</p>
      </div>
    );
  if (!log) return <p className="muted">Loading the training log…</p>;

  const m = log.meta;
  return (
    <div style={{ display: "grid", gap: 20 }}>
      {m && (
        <div className="panel stats">
          <div>
            <b>{(m.params / 1e6).toFixed(1)} M</b>
            <span className="muted small">parameters</span>
          </div>
          <div>
            <b>{(log.train_tokens / 1e6).toFixed(1)} M</b>
            <span className="muted small">training tokens from {log.books.length} novels</span>
          </div>
          <div>
            <b>{m.perplexity}</b>
            <span className="muted small">perplexity on held-out text (lower is better)</span>
          </div>
          <div>
            <b>{m.train_minutes} min</b>
            <span className="muted small">on one {m.device}</span>
          </div>
        </div>
      )}

      <div className="panel">
        <h2 style={{ fontSize: "1.5rem" }}>The loss curve</h2>
        <p className="muted" style={{ marginTop: 8, marginBottom: 16, maxWidth: "70ch" }}>
          Loss measures how surprised the model is by the next word, on average. It starts near ln({log.config.vocab_size}) ≈{" "}
          {Math.log(log.config.vocab_size).toFixed(1)}, which is random guessing among {log.config.vocab_size} tokens. The held-out loss is measured on the
          last pages of every book, which the model never trained on: if it rose while training loss kept falling, the model would be memorising.
        </p>
        <LossChart log={log} />
      </div>

      <div className="panel">
        <h2 style={{ fontSize: "1.5rem" }}>Watch it learn to write</h2>
        <p className="muted" style={{ marginTop: 8, maxWidth: "70ch" }}>
          The same prompts and the same random seed at every checkpoint, so the only thing changing is what the model knows.
        </p>
        <label htmlFor="step" style={{ marginTop: 18 }}>
          Step {snap?.step.toLocaleString()} of {log.samples[log.samples.length - 1].step.toLocaleString()}
          {evalAt ? `, held-out loss ${evalAt.val.toFixed(2)}` : ""}
        </label>
        <input id="step" type="range" min={0} max={log.samples.length - 1} value={at} onChange={(e) => setAt(+e.target.value)} />
        <div style={{ display: "grid", gap: 18, marginTop: 18 }}>
          {snap?.samples.map((s) => (
            <p key={s.prompt} className="sample">
              <span className="p">{s.prompt}</span>
              {s.text.slice(s.prompt.length)}
            </p>
          ))}
        </div>
      </div>

      <div className="panel">
        <h2 style={{ fontSize: "1.5rem" }}>What it read</h2>
        <p className="muted" style={{ marginTop: 10, lineHeight: 1.8 }}>
          {log.books.join(", ")}. All public domain, from Project Gutenberg.
        </p>
      </div>
    </div>
  );
}
