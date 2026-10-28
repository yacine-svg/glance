"use client";

import { useEffect, useRef } from "react";
import type { GlanceConfig } from "@/lib/engine/model";
import { attentionOf, type Tok } from "@/lib/useGlance";

/**
 * Rows = the token being written, columns = the earlier tokens it can see.
 * The bright diagonal band is the head looking at the previous few words; the
 * strong first column is the "attention sink" models dump spare attention into.
 */
export default function AttentionMatrix({
  tokens,
  cfg,
  layer,
  head,
  selected,
  onPick,
}: {
  tokens: Tok[];
  cfg: GlanceConfig;
  layer: number;
  head: number | "mean";
  selected: number | null;
  onPick: (i: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const n = Math.max(1, tokens.length);
    c.width = n;
    c.height = n;
    const g = c.getContext("2d");
    if (!g) return;
    const img = g.createImageData(n, n);
    tokens.forEach((tok, r) => {
      const a = attentionOf(tok, cfg, layer, head);
      for (let col = 0; col <= r; col++) {
        const v = Math.sqrt(a[col]); // sqrt makes weak links visible
        const o = (r * n + col) * 4;
        // from night blue to horizon amber
        img.data[o] = 22 + (246 - 22) * v;
        img.data[o + 1] = 34 + (180 - 34) * v;
        img.data[o + 2] = 68 + (94 - 68) * v;
        img.data[o + 3] = 255;
      }
      if (selected === r) for (let col = r + 1; col < n; col++) img.data.set([237, 240, 255, 40], (r * n + col) * 4);
    });
    g.putImageData(img, 0, 0);
  }, [tokens, cfg, layer, head, selected]);

  return (
    <canvas
      ref={canvas}
      className="matrix"
      aria-label="Attention matrix: each row is a token, each column an earlier token it attends to"
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const row = Math.floor(((e.clientY - r.top) / r.height) * tokens.length);
        if (row >= 0 && row < tokens.length) onPick(row);
      }}
    />
  );
}
