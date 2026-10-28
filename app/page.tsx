import Playground from "@/components/Playground";

export default function Home() {
  return (
    <main>
      <section className="wrap hero">
        <h1>A language model you can watch think.</h1>
        <p className="lede">
          Glance learned to write by reading 25 classic novels. It runs entirely in your browser, and for every word it writes, it shows you which
          earlier words it looked at.
        </p>
        <div className="facts">
          <div>
            <b>12.2 M</b>
            <span>parameters, trained from scratch</span>
          </div>
          <div>
            <b>6.2 M</b>
            <span>tokens of Austen, Dickens, the Brontës…</span>
          </div>
          <div>
            <b>8.5 min</b>
            <span>of training on one free GPU</span>
          </div>
        </div>
      </section>
      <Playground />
    </main>
  );
}
