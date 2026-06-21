import { getAllContent, getAllSlugs } from '@/lib/content/loader';
import { getTopicCount } from '@/data/topics';
import { LandingNav } from '@/components/landing/landing-nav';
import { HeroSection } from '@/components/landing/hero-section';
import { FeatureShowcase } from '@/components/landing/feature-showcase';
import { LearningJourney } from '@/components/landing/learning-journey';
import { DomainExplorer } from '@/components/landing/domain-explorer';
import { ResearchPipeline } from '@/components/landing/research-pipeline';
import { SystemDesignSection } from '@/components/landing/system-design-section';
import { ComparisonTable } from '@/components/landing/comparison-table';
import { PlatformStats } from '@/components/landing/platform-stats';
import { FinalCTA } from '@/components/landing/final-cta';
import { LandingFooter } from '@/components/landing/landing-footer';

export default async function Home() {
  // ── Real content counts (filesystem-derived, never hardcoded) ────────────
  const paperCount     = getAllContent('paper').length;
  const archCount      = getAllSlugs('architecture').length;
  const sysDesignCount = getAllSlugs('system-design').length;
  const problemCount   = getAllSlugs('problem').length;
  const topicCount     = getTopicCount();

  const realStats = { paperCount, archCount, sysDesignCount, problemCount, topicCount };

  return (
    <div
      className="relative min-h-screen overflow-x-hidden"
      style={{ background: '#050816', color: '#ffffff' }}
    >
      <LandingNav />
      <HeroSection stats={realStats} />
      <FeatureShowcase />
      <LearningJourney />
      <DomainExplorer />
      <ResearchPipeline />
      <SystemDesignSection />
      <ComparisonTable />
      <PlatformStats stats={realStats} />
      <FinalCTA />
      <LandingFooter />
    </div>
  );
}
