"use client";

import { useEffect, useRef } from "react";

/*
 * Fixed night-sky background that travels through the WHOLE image as you
 * scroll: stars and the Milky Way at the top of the page, the mountains in the
 * middle, the lake's reflection at the bottom. The image is sized to the
 * screen's width (or height, if that's larger), and moved by exactly the part
 * that doesn't fit. Lenis drives native scroll events, so the movement follows
 * the smooth scrolling. With reduced motion it stays still.
 */
export default function Sky() {
  const img = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const el = img.current;
    if (!el) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    const update = () => {
      raf = 0;
      const travel = Math.max(0, el.offsetHeight - window.innerHeight); // part of the image below the screen
      const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      const p = reduced ? 0.35 : Math.min(1, Math.max(0, window.scrollY / max));
      el.style.transform = `translate3d(0, ${(-p * travel).toFixed(1)}px, 0)`;
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    el.addEventListener("load", schedule);
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    // the page grows as text is generated: keep the mapping right
    const ro = new ResizeObserver(schedule);
    ro.observe(document.body);
    return () => {
      cancelAnimationFrame(raf);
      el.removeEventListener("load", schedule);
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      ro.disconnect();
    };
  }, []);

  return (
    <div className="sky" aria-hidden="true">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img ref={img} className="sky-img" src="/bg/night.jpg" alt="" decoding="async" fetchPriority="high" />
      <div className="sky-shade" />
    </div>
  );
}
