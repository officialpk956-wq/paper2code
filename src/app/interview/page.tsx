import { PageSkeleton } from "@/components/skeleton";

export default function InterviewPage() {
  return (
    <div className="page-container">
      <div className="section-header">
        <h1>Interview Prep</h1>
        <p>
          Ace your AI engineering interviews with guided practice and company
          guides.
        </p>
      </div>

      <PageSkeleton />
    </div>
  );
}
