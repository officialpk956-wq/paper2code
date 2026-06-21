import { PageSkeleton } from "@/components/skeleton";

export default function MathPage() {
  return (
    <div className="page-container">
      <div className="section-header">
        <h1>Mathematics</h1>
        <p>Linear algebra, calculus, probability, and information theory.</p>
      </div>

      <PageSkeleton />
    </div>
  );
}
