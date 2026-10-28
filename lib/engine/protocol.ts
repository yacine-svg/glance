import type { GlanceConfig, GlanceMeta } from "./model";

export type ToWorker =
  | { type: "load"; base: string }
  | { type: "generate"; prompt: string; max: number; temperature: number; topK: number; topP: number; seed: number }
  | { type: "stop" };

export type Candidate = { id: number; p: number; piece: string };

export type FromWorker =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; config: GlanceConfig; meta: GlanceMeta }
  /** attn: every layer's attention of this token, head-major, flattened: [layer][head][0..index] */
  | { type: "token"; index: number; id: number; piece: string; isPrompt: boolean; prob: number | null; attn: Float32Array }
  /** the model's top guesses for the token after `index` */
  | { type: "candidates"; index: number; top: Candidate[] }
  | { type: "done"; reason: "end" | "length" | "context" | "stopped" }
  | { type: "error"; message: string };
