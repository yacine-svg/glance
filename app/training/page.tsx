import Training from "@/components/Training";

export const metadata = { title: "How it learned" };

export default function TrainingPage() {
  return (
    <main className="wrap" style={{ paddingTop: 36, paddingBottom: 60 }}>
      <h1>How it learned</h1>
      <p className="muted" style={{ marginTop: 14, maxWidth: "64ch", fontSize: "1.1rem", marginBottom: 28 }}>
        Glance started with random numbers. It was shown passages from novels and asked, over and over, to guess the next word. Here is what that
        looked like.
      </p>
      <Training />
    </main>
  );
}
