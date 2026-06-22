import { DashboardHero } from '@/components/dashboard/dashboard-hero';
import { QuickActionGrid } from '@/components/dashboard/quick-actions';
import { KnowledgeMap } from '@/components/dashboard/knowledge-map';
import { LearningProgress } from '@/components/dashboard/learning-progress';
import { ArchitectureJourney } from '@/components/dashboard/architecture-journey';
import { SystemDesignJourney } from '@/components/dashboard/system-design-journey';
import { ResearchJourney } from '@/components/dashboard/research-journey';
import { PracticeCenter } from '@/components/dashboard/practice-center';
import { InterviewReadiness } from '@/components/dashboard/interview-readiness';
import { RecommendedSteps } from '@/components/dashboard/recommended-steps';
import { RightInsightsPanel } from '@/components/dashboard/right-insights';

export default function DashboardPage() {
  return (
    <div className="h-full flex overflow-hidden" style={{ background: '#070B18' }}>
      {/* ── Scrollable main content ── */}
      <div
        className="flex-1 overflow-y-auto min-w-0"
        style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(139,92,246,0.2) transparent' }}
      >
        <DashboardHero />

        <div className="px-6 pb-8 space-y-5 mt-4">
          <QuickActionGrid />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <KnowledgeMap />
            <LearningProgress />
          </div>

          <ArchitectureJourney />
          <SystemDesignJourney />
          <ResearchJourney />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <PracticeCenter />
            <InterviewReadiness />
          </div>

          <RecommendedSteps />
        </div>
      </div>

      {/* ── Fixed right insights panel ── */}
      <RightInsightsPanel />
    </div>
  );
}
