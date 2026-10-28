import { beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GlanceModel, rng, sample, softmax } from "../lib/engine/model";
import { Tokenizer } from "../lib/engine/tokenizer";

/*
 * Cross-language tests: the Python side (training code, NumPy reference model)
 * writes fixtures, the browser engine must reproduce them exactly.
 */

const dir = mkdtempSync(join(tmpdir(), "glance-"));

beforeAll(() => {
  const r = Bun.spawnSync(["python3", join(import.meta.dir, "make_fixtures.py"), dir]);
  if (r.exitCode !== 0) throw new Error(r.stderr.toString());
});

describe("engine vs NumPy reference", () => {
  test("logits and attention match for every position", async () => {
    const exp = await Bun.file(join(dir, "tiny.json")).json();
    const m = new GlanceModel(await Bun.file(join(dir, "tiny.bin")).arrayBuffer());
    let maxDiff = 0;
    let attnDiff = 0;
    exp.ids.forEach((id: number, t: number) => {
      const { logits, attn } = m.step(id);
      for (let v = 0; v < logits.length; v++) maxDiff = Math.max(maxDiff, Math.abs(logits[v] - exp.logits[t][v]));
      const T = t + 1;
      attn.forEach((layer, l) => {
        for (let h = 0; h < m.cfg.n_head; h++)
          for (let k = 0; k < T; k++) attnDiff = Math.max(attnDiff, Math.abs(layer[h * T + k] - exp.attn[l][h][t][k]));
      });
    });
    expect(maxDiff).toBeLessThan(1e-3);
    expect(attnDiff).toBeLessThan(1e-4);
  });

  test("attention rows are probability distributions", async () => {
    const m = new GlanceModel(await Bun.file(join(dir, "tiny.bin")).arrayBuffer());
    for (let t = 0; t < 6; t++) {
      const { attn } = m.step(t + 1);
      for (const layer of attn)
        for (let h = 0; h < m.cfg.n_head; h++) {
          const row = layer.subarray(h * (t + 1), (h + 1) * (t + 1));
          expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
        }
    }
  });

  test("int8 quantisation error stays small", async () => {
    const exp = await Bun.file(join(dir, "tiny.json")).json();
    expect(exp.quant_rel_error).toBeLessThan(0.05);
  });

  test("context limit is enforced", async () => {
    const m = new GlanceModel(await Bun.file(join(dir, "tiny.bin")).arrayBuffer());
    for (let i = 0; i < m.cfg.ctx; i++) m.step(1);
    expect(() => m.step(1)).toThrow();
  });
});

describe("tokenizer vs Python", () => {
  test("encodes exactly like the training tokenizer and round-trips", async () => {
    const tok = new Tokenizer(await Bun.file(join(dir, "tokenizer.json")).json());
    const cases = await Bun.file(join(dir, "tokens.json")).json();
    for (const c of cases) {
      expect(tok.encode(c.text)).toEqual(c.ids);
      expect(tok.decode(c.ids)).toBe(c.decoded);
    }
  });
});

describe("sampling", () => {
  const logits = Float32Array.from([2, 1, 0.5, -1, -3]);
  test("low temperature is almost greedy", () => {
    const r = rng(1);
    const picks = Array.from({ length: 50 }, () => sample(logits, { temperature: 0.05, topK: 0, topP: 1 }, r));
    expect(picks.every((p) => p === 0)).toBe(true);
  });
  test("top-k 2 never picks beyond the 2 best", () => {
    const r = rng(2);
    for (let i = 0; i < 200; i++) expect(sample(logits, { temperature: 2, topK: 2, topP: 1 }, r)).toBeLessThan(2);
  });
  test("seeded RNG is reproducible and softmax sums to 1", () => {
    const a = rng(42);
    const b = rng(42);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
    expect(softmax(logits).reduce((s, x) => s + x, 0)).toBeCloseTo(1, 6);
  });
});
