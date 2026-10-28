import type { Metadata } from "next";
import { IBM_Plex_Sans, Newsreader } from "next/font/google";
import Nav from "@/components/Nav";
import SmoothScroll from "@/components/SmoothScroll";
import "./globals.css";

const serif = Newsreader({ subsets: ["latin"], variable: "--font-serif", display: "swap" });
const sans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-sans", display: "swap" });

export const metadata: Metadata = {
  title: { default: "Glance: a language model you can see think", template: "%s · Glance" },
  description:
    "A 12-million-parameter language model trained from scratch on classic English novels. It runs in your browser, and shows which earlier words it looks at for every word it writes.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable}`}>
      <body>
        <SmoothScroll>
          <Nav />
          {children}
          <footer>
          Glance was built and trained from scratch by Sekhri Mohammed Yassine.{" "}
          <a href="https://github.com/yacine-svg/glance">Source on GitHub</a>,{" "}
          <a href="https://yassine-sekhri.vercel.app">Portfolio</a>
          </footer>
        </SmoothScroll>
      </body>
    </html>
  );
}
