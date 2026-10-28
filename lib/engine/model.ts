/*
 * Glance inference engine: the transformer forward pass in plain TypeScript,
 * one token at a time with a key/value cache, mirroring train/model.py.
 * No ML library: every matrix product, LayerNorm, softmax and GELU is here,
 * which is also what lets the UI show the attention weights of every head.
 */

export type GlanceConfig = { vocab_size: number; ctx: number; d: number; n_layer: number; n_head: number; dropout?: number };
export type GlanceMeta = {
  params?: number;
  best_step?: number;
  val_loss?: number;
  perplexity?: number;
  tokens_seen?: number;
  books?: string[];
  train_minutes?: number;
  device?: string;
};
type TensorEntry = { name: string; shape: number[]; dtype: "int8" | "f32"; offset: number; scales?: number };

export function parseWeights(buf: ArrayBuffer): { config: GlanceConfig; meta: GlanceMeta; tensors: Map<string, Float32Array> } {
  const bytes = new Uint8Array(buf);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== "GLNC") throw new Error("This is not a Glance weights file.");
  const hlen = new DataView(buf).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + hlen))) as {
    config: GlanceConfig;
    meta: GlanceMeta;
    tensors: TensorEntry[];
  };
  const base = 8 + hlen;
  const tensors = new Map<string, Float32Array>();
  for (const t of header.tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === "f32") {
      tensors.set(t.name, new Float32Array(buf.slice(base + t.offset, base + t.offset + n * 4)));
    } else {
      // int8 with one float32 scale per row: w = q · scale
      const q = new Int8Array(buf, base + t.offset, n);
      const s = new Float32Array(buf.slice(base + t.scales!, base + t.scales! + t.shape[0] * 4));
      const cols = t.shape[1];
      const w = new Float32Array(n);
      for (let r = 0; r < t.shape[0]; r++) {
        const sc = s[r];
        const o = r * cols;
        for (let c = 0; c < cols; c++) w[o + c] = q[o + c] * sc;
      }
      tensors.set(t.name, w);
    }
  }
  return { config: header.config, meta: header.meta ?? {}, tensors };
}

/** y = W·x + b, with W stored row-major as [out, in]. */
function matvec(W: Float32Array, x: Float32Array, b: Float32Array | null, out: Float32Array, rows: number, cols: number, outOffset = 0) {
  for (let r = 0; r < rows; r++) {
    let s = b ? b[r] : 0;
    const o = r * cols;
    let c = 0;
    // 4-way unrolled dot product: noticeably faster in V8
    for (; c + 3 < cols; c += 4) s += W[o + c] * x[c] + W[o + c + 1] * x[c + 1] + W[o + c + 2] * x[c + 2] + W[o + c + 3] * x[c + 3];
    for (; c < cols; c++) s += W[o + c] * x[c];
    out[outOffset + r] = s;
  }
}

function layerNorm(x: Float32Array, w: Float32Array, b: Float32Array, out: Float32Array) {
  const n = x.length;
  let mu = 0;
  for (let i = 0; i < n; i++) mu += x[i];
  mu /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mu) ** 2;
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mu) * inv * w[i] + b[i];
}

const GELU_K = Math.sqrt(2 / Math.PI);
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(GELU_K * (x + 0.044715 * x * x * x)));

export class GlanceModel {
  readonly cfg: GlanceConfig;
  readonly meta: GlanceMeta;
  private W: Map<string, Float32Array>;
  private kCache: Float32Array[];
  private vCache: Float32Array[];
  length = 0; // tokens currently in the cache

  // scratch buffers, allocated once
  private x: Float32Array;
  private h: Float32Array;
  private qkv: Float32Array;
  private y: Float32Array;
  private ff: Float32Array;
  private tmp: Float32Array;

  constructor(buf: ArrayBuffer) {
    const { config, meta, tensors } = parseWeights(buf);
    this.cfg = config;
    this.meta = meta;
    this.W = tensors;
    const { d, ctx, n_layer } = config;
    this.kCache = Array.from({ length: n_layer }, () => new Float32Array(ctx * d));
    this.vCache = Array.from({ length: n_layer }, () => new Float32Array(ctx * d));
    this.x = new Float32Array(d);
    this.h = new Float32Array(d);
    this.qkv = new Float32Array(3 * d);
    this.y = new Float32Array(d);
    this.ff = new Float32Array(4 * d);
    this.tmp = new Float32Array(d);
  }

  private w(name: string): Float32Array {
    const t = this.W.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    return t;
  }

  reset() {
    this.length = 0;
  }

  /**
   * Feeds one token at position `length`. Returns the next-token logits and,
   * for every layer, the attention weights of every head over positions 0..pos
   * (laid out head-major: attn[layer][head * (pos + 1) + t]).
   */
  step(token: number): { logits: Float32Array; attn: Float32Array[] } {
    const { d, n_head: H, n_layer, ctx, vocab_size: V } = this.cfg;
    const pos = this.length;
    if (pos >= ctx) throw new Error("context is full");
    const hd = d / H;
    const scale = 1 / Math.sqrt(hd);
    const { x, h, qkv, y, ff, tmp } = this;

    const wte = this.w("wte.weight");
    const wpe = this.w("wpe.weight");
    for (let i = 0; i < d; i++) x[i] = wte[token * d + i] + wpe[pos * d + i];

    const attn: Float32Array[] = [];
    const T = pos + 1;
    for (let l = 0; l < n_layer; l++) {
      const p = `blocks.${l}.`;
      layerNorm(x, this.w(p + "ln1.weight"), this.w(p + "ln1.bias"), h);
      matvec(this.w(p + "attn.qkv.weight"), h, this.w(p + "attn.qkv.bias"), qkv, 3 * d, d);
      const K = this.kCache[l];
      const Vc = this.vCache[l];
      K.set(qkv.subarray(d, 2 * d), pos * d);
      Vc.set(qkv.subarray(2 * d, 3 * d), pos * d);

      const probs = new Float32Array(H * T);
      for (let hh = 0; hh < H; hh++) {
        const qo = hh * hd;
        let max = -Infinity;
        for (let t = 0; t < T; t++) {
          let s = 0;
          const ko = t * d + qo;
          for (let i = 0; i < hd; i++) s += qkv[qo + i] * K[ko + i];
          s *= scale;
          probs[hh * T + t] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let t = 0; t < T; t++) {
          const e = Math.exp(probs[hh * T + t] - max);
          probs[hh * T + t] = e;
          sum += e;
        }
        for (let i = 0; i < hd; i++) y[qo + i] = 0;
        for (let t = 0; t < T; t++) {
          const a = probs[hh * T + t] / sum;
          probs[hh * T + t] = a;
          const vo = t * d + qo;
          for (let i = 0; i < hd; i++) y[qo + i] += a * Vc[vo + i];
        }
      }
      attn.push(probs);

      matvec(this.w(p + "attn.proj.weight"), y, this.w(p + "attn.proj.bias"), tmp, d, d);
      for (let i = 0; i < d; i++) x[i] += tmp[i];

      layerNorm(x, this.w(p + "ln2.weight"), this.w(p + "ln2.bias"), h);
      matvec(this.w(p + "mlp.fc.weight"), h, this.w(p + "mlp.fc.bias"), ff, 4 * d, d);
      for (let i = 0; i < 4 * d; i++) ff[i] = gelu(ff[i]);
      matvec(this.w(p + "mlp.proj.weight"), ff, this.w(p + "mlp.proj.bias"), tmp, d, 4 * d);
      for (let i = 0; i < d; i++) x[i] += tmp[i];
    }

    layerNorm(x, this.w("ln_f.weight"), this.w("ln_f.bias"), h);
    const logits = new Float32Array(V);
    matvec(wte, h, null, logits, V, d); // tied head: logits = wte · h
    this.length = pos + 1;
    return { logits, attn };
  }
}

/* ---------------- sampling ---------------- */

export type SampleOptions = { temperature: number; topK: number; topP: number };

/** Small seeded RNG (mulberry32) so a seed reproduces the same text. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function softmax(logits: Float32Array, temperature = 1): Float32Array {
  const T = Math.max(temperature, 1e-4);
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp((logits[i] - max) / T);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export function topCandidates(probs: Float32Array, k: number): { id: number; p: number }[] {
  const idx = Array.from(probs.keys());
  idx.sort((a, b) => probs[b] - probs[a]);
  return idx.slice(0, k).map((id) => ({ id, p: probs[id] }));
}

/** Temperature → top-k → nucleus (top-p) → draw. */
export function sample(logits: Float32Array, o: SampleOptions, random: () => number): number {
  const probs = softmax(logits, o.temperature);
  let cands = topCandidates(probs, o.topK > 0 ? Math.min(o.topK, probs.length) : probs.length);
  if (o.topP < 1) {
    let acc = 0;
    const kept: typeof cands = [];
    for (const c of cands) {
      kept.push(c);
      acc += c.p;
      if (acc >= o.topP) break;
    }
    cands = kept;
  }
  const total = cands.reduce((s, c) => s + c.p, 0);
  let r = random() * total;
  for (const c of cands) {
    r -= c.p;
    if (r <= 0) return c.id;
  }
  return cands[cands.length - 1].id;
}
