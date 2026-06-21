/* ─── Domain ───────────────────────────────────────────────── */
export interface Domain {
  id: string;
  label: string;
  icon: string;           // lucide icon name
  color: string;          // hex accent
  glow: string;           // rgba glow
  lessonCount: number;
  estimatedHours: number;
  progress: number;       // 0-100
  href: string;
  description: string;
  category: 'foundation' | 'core' | 'advanced' | 'specialized';
}

/* ─── Track ────────────────────────────────────────────────── */
export interface Track {
  id: string;
  title: string;
  role: string;
  color: string;
  glow: string;
  modules: number;
  estimatedHours: number;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
  progress: number;
  enrolled: number;       // learner count
  domains: string[];      // domain ids covered
  href: string;
  badge: string;          // e.g. "Most Popular"
}

/* ─── Trending Topic ───────────────────────────────────────── */
export interface TrendingTopic {
  id: string;
  title: string;
  domain: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  diffColor: string;
  estimatedMinutes: number;
  popularity: number;     // 0-100
  completionRate: number; // 0-100
  color: string;
  href: string;
  tags: string[];
}

/* ─── Lesson ───────────────────────────────────────────────── */
export interface Lesson {
  id: string;
  title: string;
  domain: string;
  domainColor: string;
  durationMinutes: number;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  isNew: boolean;
  addedDaysAgo: number;
  href: string;
  type: 'lesson' | 'lab' | 'paper' | 'project';
}

/* ─── Recommendation ───────────────────────────────────────── */
export interface Recommendation {
  id: string;
  title: string;
  reason: string;
  domain: string;
  domainColor: string;
  confidence: number;     // 0-100
  type: 'lesson' | 'track' | 'paper' | 'lab';
  estimatedMinutes: number;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  href: string;
}

/* ─── Knowledge Graph ──────────────────────────────────────── */
export interface KGNode {
  id: string;
  label: string;
  x: number;
  y: number;
  r: number;
  color: string;
  glow: string;
  lessonCount: number;
  floatDelay: number;     // animation delay seconds
}

export interface KGEdge {
  from: string;
  to: string;
  weight: number;         // 1-3, affects stroke width
}

/* ─── Progress ─────────────────────────────────────────────── */
export interface DailyTask {
  label: string;
  done: boolean;
  xp: number;
}

export interface Achievement {
  label: string;
  icon: string;
  color: string;
  earnedAt: string;
}

export interface ProgressStats {
  overall: number;
  topicsCovered: number;
  topicsTotal: number;
  lessonsCompleted: number;
  timeSpentHours: number;
  streakDays: number;
  weeklyGoalLessons: number;
  weeklyCompletedLessons: number;
  xp: number;
  level: number;
  tasks: DailyTask[];
  achievements: Achievement[];
}

/* ─── Current Lesson ───────────────────────────────────────── */
export interface CurrentLesson {
  trackTitle: string;
  module: string;
  lesson: string;
  progress: number;
  timeRemainingMinutes: number;
  domain: string;
  domainColor: string;
}

/* ─── Search ───────────────────────────────────────────────── */
export type SearchCategory = 'all' | 'lessons' | 'architectures' | 'papers' | 'system-design' | 'roadmaps';

export interface SearchResult {
  id: string;
  title: string;
  category: Omit<SearchCategory, 'all'>;
  domain: string;
  href: string;
}
