"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Nav() {
  const path = usePathname();
  const link = (href: string, label: string) => (
    <Link href={href} aria-current={path === href ? "page" : undefined}>
      {label}
    </Link>
  );
  return (
    <header className="top">
      <Link href="/" className="brand">
        <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
          <circle cx="11" cy="11" r="9" fill="none" stroke="#edf0ff" strokeWidth="2" />
          <circle cx="14" cy="11" r="4" fill="#f6b45e" />
        </svg>
        Glance
      </Link>
      <nav aria-label="Main">
        {link("/", "Write with it")}
        {link("/training", "How it learned")}
        {link("/inside", "Inside the model")}
      </nav>
    </header>
  );
}
