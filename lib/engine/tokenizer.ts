/*
 * Byte-level BPE, mirroring train/tokenizer.py exactly (same pattern, same
 * normalisation, same merge order), so the browser encodes prompts the way
 * the model saw text during training.
 */

export type TokenizerFile = { type: "byte-bpe"; pattern: string; eot: number; merges: [number, number][] };

const REPLACE: Record<string, string> = {
  "‘": "'",
  "’": "'",
  "“": '"',
  "”": '"',
  " ": " ",
  "\t": " ",
  "\r": "",
  "\f": " ",
  "\v": " ",
  "﻿": "",
};
const REPLACE_RE = /[‘’“” \t\r\f\v﻿]/g;

export function normalize(text: string): string {
  return text.normalize("NFC").replace(REPLACE_RE, (c) => REPLACE[c] ?? c);
}

export class Tokenizer {
  readonly eot: number;
  private readonly merges: [number, number][];
  private readonly ranks = new Map<number, number>(); // key = a * 65536 + b
  private readonly vocab: Uint8Array[] = [];
  private readonly pattern: RegExp;
  private readonly cache = new Map<string, number[]>();

  constructor(file: TokenizerFile) {
    this.eot = file.eot;
    this.merges = file.merges;
    this.pattern = new RegExp(file.pattern, "g");
    for (let i = 0; i < 256; i++) this.vocab.push(Uint8Array.of(i));
    file.merges.forEach(([a, b], i) => {
      this.ranks.set(a * 65536 + b, i);
      const x = this.vocab[a];
      const y = this.vocab[b];
      const z = new Uint8Array(x.length + y.length);
      z.set(x);
      z.set(y, x.length);
      this.vocab.push(z);
    });
  }

  get vocabSize() {
    return this.eot + 1;
  }

  private encodeChunk(chunk: string): number[] {
    const hit = this.cache.get(chunk);
    if (hit) return hit;
    let ids = Array.from(new TextEncoder().encode(chunk));
    while (ids.length >= 2) {
      let bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < bestRank) bestRank = r;
      }
      if (bestRank === Infinity) break;
      const [a, b] = this.merges[bestRank];
      const id = 256 + bestRank;
      const out: number[] = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
          out.push(id);
          i++;
        } else out.push(ids[i]);
      }
      ids = out;
    }
    if (this.cache.size < 50000) this.cache.set(chunk, ids);
    return ids;
  }

  encode(text: string): number[] {
    const out: number[] = [];
    for (const m of normalize(text).matchAll(this.pattern)) out.push(...this.encodeChunk(m[0]));
    return out;
  }

  /** Raw bytes of a token (a token can be part of a multi-byte character). */
  bytes(id: number): Uint8Array {
    return id === this.eot ? new Uint8Array(0) : this.vocab[id];
  }

  decode(ids: number[]): string {
    const parts = ids.map((i) => this.bytes(i));
    const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) {
      all.set(p, o);
      o += p.length;
    }
    return new TextDecoder().decode(all);
  }

  /** Readable form of one token for the UI (partial UTF-8 shown as hex). */
  piece(id: number): string {
    if (id === this.eot) return "⏎end";
    const b = this.bytes(id);
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(b);
    } catch {
      return Array.from(b, (x) => `\\x${x.toString(16).padStart(2, "0")}`).join("");
    }
  }
}
