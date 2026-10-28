"use client";

import "lenis/dist/lenis.css";
import { ReactLenis } from "lenis/react";
import { useEffect, useState } from "react";

/** Lenis smooth scrolling; plain native scrolling for people who prefer reduced motion. */
export default function SmoothScroll({ children }: { children: React.ReactNode }) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const q = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setOn(!q.matches);
    update();
    q.addEventListener("change", update);
    return () => q.removeEventListener("change", update);
  }, []);
  if (!on) return <>{children}</>;
  return (
    <ReactLenis root options={{ lerp: 0.1, smoothWheel: true }}>
      {children}
    </ReactLenis>
  );
}
