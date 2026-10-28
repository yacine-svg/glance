import Playground from "@/components/Playground";

export default function Home() {
  return (
    <main>
      <div className="wrap" style={{ paddingTop: 36 }}>
        <h1 style={{ maxWidth: "18ch" }}>A language model you can watch think.</h1>
        <p className="muted" style={{ marginTop: 14, maxWidth: "64ch", fontSize: "1.1rem" }}>
          Glance is a 12-million-parameter transformer trained from scratch on 25 classic English novels. It runs entirely in your browser.
          Hover any word it writes to see which earlier words it paid attention to.
        </p>
      </div>
      <Playground />
    </main>
  );
}
