"use client";

import { useLayoutEffect, useState } from "react";

/*
 * The signature visual: thin amber arcs from the word being written to the
 * earlier words it attends to most, like an eye-tracking trace over a page.
 * Thickness and brightness follow the attention weight.
 */

type Arc = { d: string; w: number; x: number; y: number; key: string };

export default function GlanceLines({
  container,
  selected,
  targets,
  version,
}: {
  container: HTMLElement | null;
  selected: number | null;
  targets: { index: number; weight: number }[];
  version: string; // changes whenever the arcs must be redrawn
}) {
  const [arcs, setArcs] = useState<Arc[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    if (!container || selected === null) {
      setArcs([]);
      return;
    }
    const draw = () => {
      const box = container.getBoundingClientRect();
      const at = (i: number) => {
        const el = container.querySelector<HTMLElement>(`[data-i="${i}"]`);
        if (!el) return null;
        const r = el.getClientRects()[0] ?? el.getBoundingClientRect();
        return { x: r.left - box.left + r.width / 2, y: r.top - box.top + 4 };
      };
      const from = at(selected);
      if (!from) return setArcs([]);
      const max = Math.max(...targets.map((t) => t.weight), 1e-6);
      const out: Arc[] = [];
      for (const t of targets) {
        if (t.index === selected) continue;
        const to = at(t.index);
        if (!to) continue;
        const lift = 16 + Math.min(70, Math.abs(from.x - to.x) * 0.18 + Math.abs(from.y - to.y) * 0.25);
        const cx = (from.x + to.x) / 2;
        const cy = Math.min(from.y, to.y) - lift;
        out.push({ d: `M${from.x.toFixed(1)},${from.y.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${to.x.toFixed(1)},${to.y.toFixed(1)}`, w: t.weight / max, x: to.x, y: to.y, key: `${selected}-${t.index}` });
      }
      setSize({ w: box.width, h: box.height });
      setArcs(out);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(container);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [container, selected, version]);

  if (!arcs.length) return null;
  return (
    <svg className="lines" width={size.w} height={size.h} viewBox={`0 0 ${size.w} ${size.h}`} aria-hidden="true">
      {arcs.map((a) => (
        <g key={a.key} style={{ opacity: 0.35 + 0.65 * a.w }}>
          <path d={a.d} pathLength={1} className="arc" strokeWidth={0.8 + 2.6 * a.w} />
          <circle cx={a.x} cy={a.y} r={1.5 + 2.5 * a.w} className="arc-dot" />
        </g>
      ))}
    </svg>
  );
}
