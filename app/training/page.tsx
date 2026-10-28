import Training from "@/components/Training";

export const metadata = { title: "How it learned" };

export default function TrainingPage() {
  return (
    <main className="wrap doc">
      <h1>How it learned</h1>
      <p className="lede">
        Glance started with random numbers. It was shown passages from novels and asked, over and over, to guess the next word. Here is what that
        looked like.
      </p>
      <Training />
    </main>
  );
}
