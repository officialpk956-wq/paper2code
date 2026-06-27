import { getAllContent, getAllSlugs } from '@/lib/content/loader';
import { getTopicCount } from '@/data/topics';

import { LearnHero } from '@/components/learn/LearnHero';
import { ContinueLearningCard } from '@/components/learn/ContinueLearningCard';
import { LearningPaths } from '@/components/learn/LearningPaths';
import { DomainGrid } from '@/components/learn/DomainGrid';
import { TrendingTopics } from '@/components/learn/TrendingTopics';
import { Recommendations } from '@/components/learn/Recommendations';
import { RecentlyAdded } from '@/components/learn/RecentlyAdded';
import { KnowledgeGraphPreview } from '@/components/learn/KnowledgeGraphPreview';
import { DOMAINS } from '@/data/learn/domains';
import { LEARNING_PATHS } from '@/data/learn/paths';
import { TRENDING_TOPICS } from '@/data/learn/topics';
import {
  CONTINUE_LEARNING,
  RECOMMENDATIONS,
  RECENTLY_ADDED,
} from '@/data/learn/recommendations';

export default function LearnPage() {
  const paperCount = getAllContent('paper').length;
  const archCount = getAllSlugs('architecture').length;
  const problemCount = getAllSlugs('problem').length;
  const topicCount = getTopicCount();
  const realStats = { paperCount, archCount, problemCount, topicCount };

  return (
    <div
      className="h-full flex overflow-hidden"
      style={{ background: 'var(--bg-body)' }}
    >
      <div
        className="flex-1 overflow-y-auto min-w-0"
        style={{
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(124,58,237,0.2) transparent',
        }}
      >
        {/* Section 1: Hero */}
        <LearnHero stats={realStats} />

        <div className="px-6 pb-12">
          {/* Section 2: Continue Learning */}
          <ContinueLearningCard data={CONTINUE_LEARNING} />

          {/* Section 3: Learning Paths */}
          <LearningPaths paths={LEARNING_PATHS} />

          {/* Section 4: Domains */}
          <DomainGrid domains={DOMAINS} />

          {/* Section 5: Trending Topics */}
          <TrendingTopics topics={TRENDING_TOPICS} />

          {/* Section 6: Recommended For You */}
          <Recommendations items={RECOMMENDATIONS} />

          {/* Section 7: Recently Added */}
          <RecentlyAdded items={RECENTLY_ADDED} />

          {/* Section 8: Knowledge Graph Preview */}
          <KnowledgeGraphPreview />
        </div>
      </div>
    </div>
  );
}
