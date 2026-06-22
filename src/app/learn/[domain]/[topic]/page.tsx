'use client';

import { useEffect, useRef, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { getTopicData, resolveTopicPath } from '@/data/topics';
import type { SectionId } from '@/components/topic/TopicSidebar';
import { CHAPTER_SECTIONS } from '@/components/topic/TopicSidebar';

import { TopicHero }           from '@/components/topic/TopicHero';
import { TopicSidebar }        from '@/components/topic/TopicSidebar';
import { StudyAssistant }      from '@/components/topic/StudyAssistant';
import { MotivationSection }   from '@/components/topic/MotivationSection';
import { IntuitionSection }    from '@/components/topic/IntuitionSection';
import { MathSection }         from '@/components/topic/MathSection';
import { ArchitectureSection } from '@/components/topic/ArchitectureSection';
import { CodeWalkthrough }     from '@/components/topic/CodeWalkthrough';
import { ApplicationsSection } from '@/components/topic/ApplicationsSection';
import { ResearchTimeline }    from '@/components/topic/ResearchTimeline';
import { RelatedPapers }       from '@/components/topic/RelatedPapers';
import { InterviewNotes }      from '@/components/topic/InterviewNotes';
import { PracticePreview }        from '@/components/topic/PracticePreview';
import { SummarySection }         from '@/components/topic/SummarySection';
import { RecommendedProblems }    from '@/components/topic/RecommendedProblems';

export default function TopicPage() {
  const { domain, topic } = useParams<{ domain: string; topic: string }>();
  const data = getTopicData(domain, topic);

  const [activeSection, setActiveSection] = useState<SectionId | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Scroll spy via IntersectionObserver on the scroll container
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const sectionIds = CHAPTER_SECTIONS.map(s => s.id);
    const observers: IntersectionObserver[] = [];

    for (const id of sectionIds) {
      const el = container.querySelector(`[id="${id}"]`);
      if (!el) continue;

      const obs = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) setActiveSection(id as SectionId);
        },
        {
          root: container,
          rootMargin: '-10% 0px -65% 0px',
          threshold: 0,
        }
      );
      obs.observe(el);
      observers.push(obs);
    }

    return () => { observers.forEach(o => o.disconnect()); };
  }, [data]);

  const handleNavigate = (id: SectionId) => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const el = container.querySelector(`[id="${id}"]`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  if (!data) {
    return (
      <div
        className="h-full flex items-center justify-center"
        style={{ background: 'var(--bg-body)' }}
      >
        <div className="text-center">
          <div className="text-4xl mb-4" aria-hidden="true">🔍</div>
          <h1 className="text-xl font-black mb-2" style={{ color: 'var(--color-text-primary)' }}>
            Topic Not Found
          </h1>
          <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            &ldquo;{topic}&rdquo; doesn&apos;t exist in &ldquo;{domain}&rdquo;.
          </p>
        </div>
      </div>
    );
  }

  const color = data.meta.domainColor;

  let nextTopicUrl = data.meta.nextTopic ? resolveTopicPath(data.meta.nextTopic) : null;
  if (!nextTopicUrl && data.relatedTopics.length > 0) {
    nextTopicUrl = resolveTopicPath(data.relatedTopics[0].slug);
  }

  return (
    <div
      className="h-full flex overflow-hidden"
      style={{ background: 'var(--bg-body)' }}
    >
      {/* Left sidebar — chapter nav */}
      <aside
        className="hidden lg:flex flex-col w-52 xl:w-56 flex-none border-r overflow-y-auto"
        style={{
          borderColor: 'rgba(255,255,255,0.06)',
          background: 'var(--bg-sidebar)',
        }}
        aria-label="Chapter navigation"
      >
        <div className="sticky top-0 p-4">
          <TopicSidebar
            activeSection={activeSection}
            topicSlug={data.meta.slug}
            durationMinutes={data.meta.durationMinutes}
            difficulty={data.meta.difficulty}
            onNavigate={handleNavigate}
          />
        </div>
      </aside>

      {/* Center — scrollable content */}
      <main
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto min-w-0"
        id="topic-main-content"
        tabIndex={-1}
        aria-label="Topic learning content"
        style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.1) transparent' }}
      >
        <TopicHero meta={data.meta} />

        <div className="px-5 xl:px-8 pb-16">
          <MotivationSection     cards={data.motivation} />
          <IntuitionSection      intuition={data.intuition} color={color} />
          <MathSection           formulas={data.formulas} />
          <ArchitectureSection   nodes={data.archNodes} edges={data.archEdges} color={color} />
          <CodeWalkthrough       snippets={data.codeSnippets} />
          <ApplicationsSection   applications={data.applications} />
          <ResearchTimeline      events={data.timeline} />
          <RelatedPapers         papers={data.relatedPapers} />
          <InterviewNotes        questions={data.interviewQuestions} />
          <PracticePreview
            questions={data.practice}
            domainSlug={domain}
            topicSlug={topic}
          />
          <SummarySection        points={data.summary} color={color} />
          <RecommendedProblems   topicSlug={topic} />
          
          {nextTopicUrl && (
            <div className="mt-12 pt-8 border-t" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
              <Link
                href={nextTopicUrl}
                className="group flex flex-col items-center justify-center p-6 rounded-2xl transition-all"
                style={{
                  background: 'var(--bg-surface)',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'var(--accent-primary)';
                  (e.currentTarget as HTMLElement).style.background = 'rgba(124,58,237,0.05)';
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.06)';
                  (e.currentTarget as HTMLElement).style.background = 'var(--bg-surface)';
                }}
              >
                <span className="text-sm font-semibold mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                  Next Up
                </span>
                <span className="text-xl font-bold flex items-center gap-2" style={{ color: 'var(--color-text-primary)' }}>
                  Continue Learning
                  <span className="group-hover:translate-x-1 transition-transform" style={{ color: 'var(--accent-primary)' }}>→</span>
                </span>
              </Link>
            </div>
          )}
        </div>
      </main>

      {/* Right sidebar — study tools */}
      <aside
        className="hidden xl:flex flex-col w-64 flex-none border-l overflow-y-auto"
        style={{
          borderColor: 'rgba(255,255,255,0.06)',
          background: 'var(--bg-sidebar)',
        }}
        aria-label="Study tools"
      >
        <div className="sticky top-0 p-4">
          <StudyAssistant data={data} />
        </div>
      </aside>
    </div>
  );
}
