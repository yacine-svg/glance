"use client";

import { useMemo, useState } from "react";
import AttentionMatrix from "@/components/AttentionMatrix";
import { attentionOf, useGlance } from "@/lib/useGlance";

const PROMPTS = [
  "It was a dark and stormy night, and",
  '"My dear," said Mrs. Bennet,',
  "The letter arrived in the morning. She opened it and",
  "He had never seen the sea before.",
];

const pct = (p: number) => (p >= 0.995 ? "99%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`);
const show = (piece: string) => piece.replace(/\n/g, "↵").replace(/^ /, "␣");

export default function Playground() {
  const g = useGlance();
  const [prompt, setPrompt] = useState(PROMPTS[0]);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.95);
  const [max, setMax] = useState(160);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState<number | "mean">("mean");
  const [pinned, setPinned] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  const running = g.status.kind === "running";
  const ready = g.status.kind === "ready" || running;
  const sel = hover ?? pinned ?? (g.tokens.length ? g.tokens.length - 1 : null);
  const selTok = sel !== null ? g.tokens[sel] : undefined;
  const weights = useMemo(() => (selTok && g.config ? attentionOf(selTok, g.config, Math.min(layer, g.config.n_layer - 1), head) : null), [selTok, g.config, layer, head]);
  const maxW = weights ? Math.max(...weights) : 1;
  const strongest = useMemo(() => {
    if (!weights) return [];
    return Array.from(weights.keys())
      .sort((a, b) => weights[b] - weights[a])
      .slice(0, 5);
  }, [weights]);

  const start = () => {
    setPinned(null);
    g.generate({ prompt, max, temperature, topK, topP, seed: (Math.random() * 2 ** 31) | 0 });
  };

  if (g.status.kind === "missing") {
    return (
      <div className="wrap play">
        <div className="panel">
          <h2>The model hasn't been trained yet</h2>
          <p className="muted" style={{ marginTop: 12, maxWidth: "60ch" }}>
            Run the training notebook in <code>train/Glance.ipynb</code> on Google Colab, then upload <code>glance.bin</code>,{" "}
            <code>tokenizer.json</code> and <code>training_log.json</code> to <code>public/model/</code>.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="wrap play">
      <section>
        <div className="panel">
          <label htmlFor="prompt">Start a sentence, Glance continues it</label>
          <textarea id="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} maxLength={600} />
          <div className="row" style={{ marginTop: 10 }}>
            {PROMPTS.map((p) => (
              <button key={p} type="button" className="small" style={{ padding: "4px 12px", fontWeight: 400 }} onClick={() => setPrompt(p)}>
                {p.length > 30 ? `${p.slice(0, 30)}…` : p}
              </button>
            ))}
          </div>
          <div className="sliders">
            <div>
              <label htmlFor="temp">Temperature {temperature.toFixed(2)}</label>
              <input id="temp" type="range" min={0.1} max={1.5} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} />
            </div>
            <div>
              <label htmlFor="topk">Top-k {topK === 0 ? "off" : topK}</label>
              <input id="topk" type="range" min={0} max={200} step={5} value={topK} onChange={(e) => setTopK(+e.target.value)} />
            </div>
            <div>
              <label htmlFor="topp">Top-p {topP.toFixed(2)}</label>
              <input id="topp" type="range" min={0.5} max={1} step={0.01} value={topP} onChange={(e) => setTopP(+e.target.value)} />
            </div>
            <div>
              <label htmlFor="len">Length {max} tokens</label>
              <input id="len" type="range" min={20} max={220} step={10} value={max} onChange={(e) => setMax(+e.target.value)} />
            </div>
          </div>
          <div className="row" style={{ marginTop: 18 }}>
            {running ? (
              <button type="button" onClick={g.stop}>
                Stop
              </button>
            ) : (
              <button type="button" className="primary" onClick={start} disabled={!ready || !prompt.trim()}>
                Write
              </button>
            )}
            {g.status.kind === "loading" && (
              <span className="muted small">
                Loading the model{g.status.total ? ` (${(g.status.loaded / 1e6).toFixed(1)} of ${(g.status.total / 1e6).toFixed(1)} MB)` : "…"}
              </span>
            )}
            {g.status.kind === "error" && <span style={{ color: "#ff8a7a" }}>{g.status.message}</span>}
            {g.doneReason === "context" && <span className="muted small">Stopped: the 256-token context is full.</span>}
          </div>
          {g.status.kind === "loading" && g.status.total > 0 && (
            <div className="progress">
              <i style={{ width: `${(g.status.loaded / g.status.total) * 100}%` }} />
            </div>
          )}
        </div>

        <div className="panel">
          <p className="page" onMouseLeave={() => setHover(null)}>
            {g.tokens.length === 0 && <span className="muted">The text appears here, one token at a time. Hover any word to see what Glance was looking at.</span>}
            {g.tokens.map((t, i) => {
              const w = weights && sel !== null && i <= sel ? weights[i] / (maxW || 1) : 0;
              return (
                <span
                  key={i}
                  className={`t${t.isPrompt ? " prompt" : ""}${i === sel ? " sel" : ""}`}
                  style={w > 0.02 && i !== sel ? { backgroundColor: `rgba(242, 181, 68, ${(0.08 + w * 0.82).toFixed(3)})`, color: w > 0.5 ? "#231803" : undefined } : undefined}
                  onMouseEnter={() => setHover(i)}
                  onClick={() => setPinned(i === pinned ? null : i)}
                >
                  {t.piece}
                </span>
              );
            })}
            {running && <span className="caret" aria-hidden="true" />}
          </p>
        </div>
      </section>

      <aside>
        <div className="panel">
          <h3 style={{ fontSize: "1.3rem" }}>{selTok ? <>Writing “{show(selTok.piece)}”</> : "Attention"}</h3>
          <p className="muted small" style={{ marginTop: 6 }}>
            {selTok
              ? selTok.isPrompt
                ? "A word from your prompt. The highlights show the earlier words it attended to."
                : `The model gave this token a ${pct(selTok.prob ?? 0)} chance. Highlighted words are what it looked at while choosing it.`
              : "Hover or click a token in the text."}
          </p>
          {g.config && (
            <>
              <div style={{ marginTop: 16 }}>
                <label>Layer (1 is closest to the words, {g.config.n_layer} is closest to the prediction)</label>
                <div className="seg">
                  {Array.from({ length: g.config.n_layer }, (_, l) => (
                    <button key={l} type="button" aria-pressed={layer === l} onClick={() => setLayer(l)}>
                      {l + 1}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ marginTop: 12 }}>
                <label>Attention head</label>
                <div className="seg">
                  <button type="button" aria-pressed={head === "mean"} onClick={() => setHead("mean")}>
                    Average
                  </button>
                  {Array.from({ length: g.config.n_head }, (_, h) => (
                    <button key={h} type="button" aria-pressed={head === h} onClick={() => setHead(h)}>
                      {h + 1}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
          {selTok && weights && (
            <ul className="cands" style={{ marginTop: 16 }} aria-label="Most attended earlier tokens">
              {strongest.map((i) => (
                <li key={i}>
                  <code>{show(g.tokens[i].piece)}</code>
                  <span className="bar">
                    <i style={{ width: `${(weights[i] / (maxW || 1)) * 100}%`, background: "var(--heat)" }} />
                  </span>
                  <span className="pct">{pct(weights[i])}</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        {sel !== null && g.candidates[sel - 1] && (
          <div className="panel">
            <h3 style={{ fontSize: "1.15rem" }}>What it considered here</h3>
            <p className="muted small" style={{ marginTop: 6, marginBottom: 12 }}>
              The model&apos;s top guesses for this position, before sampling picked one.
            </p>
            <ul className="cands">
              {g.candidates[sel - 1].map((c) => (
                <li key={c.id} className={c.id === selTok?.id ? "chosen" : ""}>
                  <code>{show(c.piece)}</code>
                  <span className="bar">
                    <i style={{ width: `${c.p * 100}%` }} />
                  </span>
                  <span className="pct">{pct(c.p)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {g.config && g.tokens.length > 1 && (
          <div className="panel">
            <h3 style={{ fontSize: "1.15rem" }}>Attention map</h3>
            <p className="muted small" style={{ marginTop: 6, marginBottom: 12 }}>
              Each row is a token, each column an earlier token. Click a row to inspect it.
            </p>
            <AttentionMatrix tokens={g.tokens} cfg={g.config} layer={Math.min(layer, g.config.n_layer - 1)} head={head} selected={sel} onPick={setPinned} />
          </div>
        )}
      </aside>
    </div>
  );
}
