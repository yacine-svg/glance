"use client";

import { useMemo, useState } from "react";
import AttentionMatrix from "@/components/AttentionMatrix";
import GlanceLines from "@/components/GlanceLines";
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
  const [pageEl, setPageEl] = useState<HTMLElement | null>(null);

  const running = g.status.kind === "running";
  const ready = g.status.kind === "ready" || running;
  const sel = hover ?? pinned ?? (g.tokens.length && !running ? g.tokens.length - 1 : null);
  const selTok = sel !== null ? g.tokens[sel] : undefined;
  const L = g.config ? Math.min(layer, g.config.n_layer - 1) : 0;
  const weights = useMemo(() => (selTok && g.config ? attentionOf(selTok, g.config, L, head) : null), [selTok, g.config, L, head]);
  const maxW = weights ? Math.max(...weights) : 1;
  const strongest = useMemo(() => {
    if (!weights) return [];
    return Array.from(weights.keys())
      .sort((a, b) => weights[b] - weights[a])
      .slice(0, 6)
      .map((index) => ({ index, weight: weights[index] }));
  }, [weights]);

  const start = () => {
    setPinned(null);
    setHover(null);
    g.generate({ prompt, max, temperature, topK, topP, seed: (Math.random() * 2 ** 31) | 0 });
  };

  if (g.status.kind === "missing") {
    return (
      <div className="wrap">
        <div className="panel">
          <h2>The model hasn&apos;t been trained yet</h2>
          <p className="muted" style={{ marginTop: 12, maxWidth: "60ch" }}>
            Run <code>train/Glance.ipynb</code> on Google Colab, then upload <code>glance.bin</code>, <code>tokenizer.json</code> and{" "}
            <code>training_log.json</code> to <code>public/model/</code>.
          </p>
        </div>
      </div>
    );
  }

  const loading = g.status.kind === "loading";
  const loadPct = loading && g.status.kind === "loading" && g.status.total ? g.status.loaded / g.status.total : 0;

  return (
    <div className="wrap stage">
      <section>
        <form
          className="compose"
          onSubmit={(e) => {
            e.preventDefault();
            if (running) g.stop();
            else if (ready && prompt.trim()) start();
          }}
        >
          <label htmlFor="prompt" className="sr">
            Start a sentence
          </label>
          <textarea
            id="prompt"
            value={prompt}
            rows={2}
            maxLength={600}
            placeholder="Begin a sentence, Glance continues it…"
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                if (ready && !running && prompt.trim()) start();
              }
            }}
          />
          <button type="submit" className={running ? "" : "primary"} disabled={!running && (!ready || !prompt.trim())}>
            {running ? "Stop" : loading ? "Loading…" : "Write"}
          </button>
        </form>
        <div className="row starters">
          {PROMPTS.map((p) => (
            <button key={p} type="button" className="starter" onClick={() => setPrompt(p)}>
              {p}
            </button>
          ))}
        </div>

        <details className="settings">
          <summary>Sampling settings</summary>
          <div className="sliders">
            <div>
              <label htmlFor="temp">
                Temperature <b>{temperature.toFixed(2)}</b>
              </label>
              <input id="temp" type="range" min={0.1} max={1.5} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} />
              <p className="hint">Low: safe and repetitive. High: surprising, less coherent.</p>
            </div>
            <div>
              <label htmlFor="topk">
                Top-k <b>{topK === 0 ? "off" : topK}</b>
              </label>
              <input id="topk" type="range" min={0} max={200} step={5} value={topK} onChange={(e) => setTopK(+e.target.value)} />
              <p className="hint">Only the k most likely tokens can be picked.</p>
            </div>
            <div>
              <label htmlFor="topp">
                Top-p <b>{topP.toFixed(2)}</b>
              </label>
              <input id="topp" type="range" min={0.5} max={1} step={0.01} value={topP} onChange={(e) => setTopP(+e.target.value)} />
              <p className="hint">Keep the smallest set of tokens covering this much probability.</p>
            </div>
            <div>
              <label htmlFor="len">
                Length <b>{max} tokens</b>
              </label>
              <input id="len" type="range" min={20} max={220} step={10} value={max} onChange={(e) => setMax(+e.target.value)} />
              <p className="hint">The context holds 256 tokens, prompt included.</p>
            </div>
          </div>
        </details>

        <article className="book">
          <header className="book-head">
            <span>Glance</span>
            <span>
              {loading
                ? loadPct
                  ? `Loading the model, ${Math.round(loadPct * 100)}%`
                  : "Loading the model…"
                : running
                  ? "Writing…"
                  : g.tokens.length
                    ? `${g.tokens.filter((t) => !t.isPrompt).length} tokens written`
                    : "Ready"}
            </span>
          </header>
          {loading && (
            <div className="progress" aria-hidden="true">
              <i style={{ width: `${loadPct * 100}%` }} />
            </div>
          )}
          <div className="page" ref={setPageEl} onMouseLeave={() => setHover(null)}>
            {g.tokens.length === 0 ? (
              <p className="empty">
                {loading
                  ? "Downloading 12.5 MB of weights. After that, everything runs on your device."
                  : "Press Write. The text appears here one token at a time; hover any word to see what Glance was looking at when it wrote it."}
              </p>
            ) : (
              <p className="prose">
                {g.tokens.map((t, i) => {
                  const w = weights && sel !== null && i < sel ? weights[i] / (maxW || 1) : 0;
                  return (
                    <span
                      key={i}
                      data-i={i}
                      className={`t${t.isPrompt ? " prompt" : ""}${i === sel ? " sel" : ""}`}
                      style={w > 0.04 ? { backgroundColor: `rgba(246, 180, 94, ${(0.06 + w * 0.6).toFixed(3)})` } : undefined}
                      onMouseEnter={() => setHover(i)}
                      onClick={() => setPinned(i === pinned ? null : i)}
                    >
                      {t.piece}
                    </span>
                  );
                })}
                {running && <span className="caret" aria-hidden="true" />}
              </p>
            )}
            <GlanceLines
              container={pageEl}
              selected={sel}
              targets={strongest}
              version={`${sel}-${L}-${head}-${g.tokens.length}-${strongest.map((s) => s.index).join(".")}`}
            />
          </div>
          {g.doneReason === "context" && <p className="book-foot">The 256-token context is full, so Glance stopped here.</p>}
          {g.status.kind === "error" && <p className="book-foot err">{g.status.message}</p>}
        </article>
      </section>

      <aside className="inspector">
        <div className="panel">
          {selTok ? (
            <>
              <p className="eyebrow">{selTok.isPrompt ? "From your prompt" : "Written by Glance"}</p>
              <h3 className="word">{show(selTok.piece)}</h3>
              <p className="muted small">
                {selTok.isPrompt || selTok.prob === null
                  ? "The arcs show which earlier words this token attended to."
                  : `It gave this token a ${pct(selTok.prob)} chance. The arcs show which earlier words it looked at while choosing it.`}
              </p>
            </>
          ) : (
            <>
              <h3 className="word" style={{ fontSize: "1.6rem" }}>
                Hover a word
              </h3>
              <p className="muted small">
                Each word was written by looking back at the earlier ones. The arcs on the page show where this model looked, layer by layer and head
                by head.
              </p>
            </>
          )}

          {g.config && (
            <div className="pickers">
              <div>
                <label>Layer</label>
                <div className="seg">
                  {Array.from({ length: g.config.n_layer }, (_, l) => (
                    <button key={l} type="button" aria-pressed={L === l} onClick={() => setLayer(l)}>
                      {l + 1}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label>Head</label>
                <div className="seg">
                  <button type="button" aria-pressed={head === "mean"} onClick={() => setHead("mean")}>
                    All
                  </button>
                  {Array.from({ length: g.config.n_head }, (_, h) => (
                    <button key={h} type="button" aria-pressed={head === h} onClick={() => setHead(h)}>
                      {h + 1}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {selTok && weights && (
            <ul className="cands" aria-label="Earlier tokens it attended to most">
              {strongest.map(({ index, weight }) => (
                <li key={index} onMouseEnter={() => setHover(index)}>
                  <code>{show(g.tokens[index].piece)}</code>
                  <span className="bar">
                    <i style={{ width: `${(weight / (maxW || 1)) * 100}%`, background: "var(--heat)" }} />
                  </span>
                  <span className="pct">{pct(weight)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        {sel !== null && g.candidates[sel - 1] && !selTok?.isPrompt && (
          <div className="panel">
            <h3 className="panel-title">What it considered</h3>
            <p className="muted small" style={{ margin: "6px 0 14px" }}>
              Its top guesses at this spot, before one was drawn at random.
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
            <h3 className="panel-title">Attention map</h3>
            <p className="muted small" style={{ margin: "6px 0 14px" }}>
              Rows are tokens, columns the earlier tokens they look at. The bright diagonal is a head reading the previous words. Click a row to
              inspect it.
            </p>
            <AttentionMatrix tokens={g.tokens} cfg={g.config} layer={L} head={head} selected={sel} onPick={setPinned} />
          </div>
        )}
      </aside>
    </div>
  );
}
