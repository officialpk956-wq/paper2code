import { PageSkeleton } from "@/components/skeleton";

export default function PlaygroundPage() {
  return (
    <div className="page-container">
      <div className="section-header">
        <h1>Playground</h1>
        <p>
          Interactive tools: tensor visualizer, attention tracer, FLOPs
          calculator.
        </p>
      </div>

      <PageSkeleton />
    </div>
  );
}
