'use client';

import { useParams } from 'next/navigation';
import { getDomainData } from '@/data/domains';
import { DomainHero } from '@/components/domain/DomainHero';
import { ProgressOverview } from '@/components/domain/ProgressOverview';
import { LearningRoadmap } from '@/components/domain/LearningRoadmap';
import { TopicClusters } from '@/components/domain/TopicClusters';
import { FeaturedLessons } from '@/components/domain/FeaturedLessons';
import { DomainKnowledgeGraph } from '@/components/domain/DomainKnowledgeGraph';
import { ProjectShowcase } from '@/components/domain/ProjectShowcase';
import { ResearchConnections } from '@/components/domain/ResearchConnections';

export default function DomainPage() {
  const { domain } = useParams<{ domain: string }>();
  const data = getDomainData(domain);

  if (!data) {
    return (
      <div className="h-full flex overflow-hidden" style={{ background: 'var(--bg-body)' }}>
        <div className="flex-1 overflow-y-auto min-w-0 flex items-center justify-center">
          <div className="text-center px-6">
            <div
              className="text-6xl font-black mb-4 gradient-text"
              aria-hidden="true"
            >
              404
            </div>
            <h1 className="text-xl font-bold mb-2" style={{ color: 'var(--color-text-primary)' }}>
              Domain not found
            </h1>
            <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
              The domain &ldquo;{domain}&rdquo; doesn&apos;t exist yet.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex overflow-hidden" style={{ background: 'var(--bg-body)' }}>
      <div
        className="flex-1 overflow-y-auto min-w-0"
        style={{
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(124,58,237,0.2) transparent',
        }}
      >
        <DomainHero data={data} />

        <div className="px-6 pb-12">
          <ProgressOverview progress={data.progress} color={data.color} />
          <LearningRoadmap stages={data.roadmap} domainSlug={data.slug} color={data.color} />
          <TopicClusters clusters={data.topicClusters} />
          <FeaturedLessons lessons={data.featuredLessons} domainSlug={data.slug} color={data.color} />
          <DomainKnowledgeGraph
            nodes={data.kgNodes}
            edges={data.kgEdges}
            domainSlug={data.slug}
            color={data.color}
          />
          <ProjectShowcase projects={data.projects} domainSlug={data.slug} />
          <ResearchConnections papers={data.papers} />
        </div>
      </div>
    </div>
  );
}
