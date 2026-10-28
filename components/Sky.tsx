"use client";

import { useEffect, useRef } from "react";

/*
 * Fixed night-sky background (portrait image on phones, landscape crop on wide
 * screens), with a gentle parallax as the page scrolls. Lenis drives native
 * scroll events, so this follows the smooth scrolling. Static for reduced motion.
 */
export default function Sky() {
  const el = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let raf = 0;
    const on = () => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        const max = Math.max(1, document.documentElement.scrollHeight - innerHeight);
        const p = Math.min(1, window.scrollY / max);
        if (el.current) el.current.style.transform = `translate3d(0, ${(-p * 6).toFixed(2)}%, 0) scale(1.08)`;
      });
    };
    on();
    window.addEventListener("scroll", on, { passive: true });
    window.addEventListener("resize", on);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", on);
      window.removeEventListener("resize", on);
    };
  }, []);
  return (
    <div className="sky" aria-hidden="true">
      <div ref={el} className="sky-img" />
      <div className="sky-shade" />
    </div>
  );
}
