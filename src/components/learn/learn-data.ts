import type {
  Domain, Track, TrendingTopic, Lesson,
  Recommendation, KGNode, KGEdge, CurrentLesson,
} from './learn-types';

/* ─── Domains ──────────────────────────────────────────────── */
export const DOMAINS: Domain[] = [
  {
    id: 'math', label: 'Mathematics', icon: 'Sigma', color: '#8B5CF6', glow: 'rgba(139,92,246,0.25)',
    lessonCount: 48, estimatedHours: 40, progress: 72, href: '/learn/mathematics',
    description: 'Linear algebra, calculus, probability theory and statistics for AI.',
    category: 'foundation',
  },
  {
    id: 'ml', label: 'Machine Learning', icon: 'Brain', color: '#7C3AED', glow: 'rgba(124,58,237,0.25)',
    lessonCount: 64, estimatedHours: 55, progress: 45, href: '/learn/machine-learning',
    description: 'Supervised, unsupervised and semi-supervised learning algorithms.',
    category: 'core',
  },
  {
    id: 'dl', label: 'Deep Learning', icon: 'Layers3', color: '#06B6D4', glow: 'rgba(6,182,212,0.25)',
    lessonCount: 72, estimatedHours: 65, progress: 34, href: '/learn/deep-learning',
    description: 'Neural networks, backpropagation, optimization and architectures.',
    category: 'core',
  },
  {
    id: 'cv', label: 'Computer Vision', icon: 'Eye', color: '#10B981', glow: 'rgba(16,185,129,0.25)',
    lessonCount: 56, estimatedHours: 48, progress: 60, href: '/learn/computer-vision',
    description: 'CNNs, object detection, segmentation, and generative vision models.',
    category: 'core',
  },
  {
    id: 'nlp', label: 'Natural Language Processing', icon: 'MessageSquare', color: '#0284C7', glow: 'rgba(2,132,199,0.25)',
    lessonCount: 58, estimatedHours: 52, progress: 22, href: '/learn/nlp',
    description: 'Tokenization, embeddings, sequence models and language understanding.',
    category: 'core',
  },
  {
    id: 'llm', label: 'Large Language Models', icon: 'Sparkles', color: '#A78BFA', glow: 'rgba(167,139,250,0.25)',
    lessonCount: 44, estimatedHours: 38, progress: 18, href: '/learn/llms',
    description: 'Transformer architecture, pre-training, fine-tuning and alignment.',
    category: 'advanced',
  },
  {
    id: 'agents', label: 'AI Agents', icon: 'Bot', color: '#F59E0B', glow: 'rgba(245,158,11,0.25)',
    lessonCount: 32, estimatedHours: 28, progress: 0, href: '/learn/agents',
    description: 'Autonomous agents, tool use, multi-agent systems and planning.',
    category: 'advanced',
  },
  {
    id: 'rag', label: 'RAG Systems', icon: 'Database', color: '#EC4899', glow: 'rgba(236,72,153,0.25)',
    lessonCount: 28, estimatedHours: 24, progress: 0, href: '/learn/rag',
    description: 'Retrieval-augmented generation, vector stores and hybrid search.',
    category: 'advanced',
  },
  {
    id: 'rl', label: 'Reinforcement Learning', icon: 'Target', color: '#EF4444', glow: 'rgba(239,68,68,0.25)',
    lessonCount: 40, estimatedHours: 36, progress: 0, href: '/learn/reinforcement-learning',
    description: 'Markov decision processes, policy gradient and deep RL.',
    category: 'advanced',
  },
  {
    id: 'mlops', label: 'MLOps', icon: 'Settings2', color: '#64748B', glow: 'rgba(100,116,139,0.25)',
    lessonCount: 36, estimatedHours: 32, progress: 5, href: '/learn/mlops',
    description: 'Model deployment, monitoring, CI/CD and production ML systems.',
    category: 'specialized',
  },
  {
    id: 'research', label: 'Research Methodology', icon: 'Microscope', color: '#34D399', glow: 'rgba(52,211,153,0.25)',
    lessonCount: 24, estimatedHours: 20, progress: 10, href: '/learn/research',
    description: 'Paper reading, experimental design, ablations and scientific writing.',
    category: 'specialized',
  },
  {
    id: 'ai-eng', label: 'AI Engineering', icon: 'Cpu', color: '#22D3EE', glow: 'rgba(34,211,238,0.25)',
    lessonCount: 52, estimatedHours: 45, progress: 8, href: '/learn/ai-engineering',
    description: 'System design for AI, inference infrastructure and production pipelines.',
    category: 'specialized',
  },
];

/* ─── Featured Tracks ──────────────────────────────────────── */
export const TRACKS: Track[] = [
  {
    id: 'ml-engineer', title: 'Become an ML Engineer', role: 'ML Engineer',
    color: '#8B5CF6', glow: 'rgba(139,92,246,0.2)',
    modules: 14, estimatedHours: 120, difficulty: 'Intermediate', progress: 34,
    enrolled: 4820, domains: ['math', 'ml', 'dl', 'mlops'],
    href: '/roadmaps/ml-engineer', badge: 'Most Popular',
  },
  {
    id: 'llm-engineer', title: 'Become an LLM Engineer', role: 'LLM Engineer',
    color: '#A78BFA', glow: 'rgba(167,139,250,0.2)',
    modules: 12, estimatedHours: 95, difficulty: 'Advanced', progress: 18,
    enrolled: 3640, domains: ['dl', 'nlp', 'llm', 'rag', 'agents'],
    href: '/roadmaps/llm-engineer', badge: 'Trending',
  },
  {
    id: 'research-sci', title: 'Become a Research Scientist', role: 'Research Scientist',
    color: '#34D399', glow: 'rgba(52,211,153,0.2)',
    modules: 16, estimatedHours: 140, difficulty: 'Expert', progress: 10,
    enrolled: 2180, domains: ['math', 'ml', 'dl', 'research', 'rl'],
    href: '/roadmaps/research-scientist', badge: '',
  },
  {
    id: 'ai-engineer', title: 'Become an AI Engineer', role: 'AI Engineer',
    color: '#22D3EE', glow: 'rgba(34,211,238,0.2)',
    modules: 18, estimatedHours: 160, difficulty: 'Advanced', progress: 8,
    enrolled: 5120, domains: ['ml', 'dl', 'llm', 'rag', 'agents', 'ai-eng', 'mlops'],
    href: '/roadmaps/ai-engineer', badge: 'Comprehensive',
  },
  {
    id: 'mlops-engineer', title: 'Become an MLOps Engineer', role: 'MLOps Engineer',
    color: '#F59E0B', glow: 'rgba(245,158,11,0.2)',
    modules: 10, estimatedHours: 80, difficulty: 'Intermediate', progress: 5,
    enrolled: 1960, domains: ['ml', 'mlops', 'ai-eng'],
    href: '/roadmaps/mlops-engineer', badge: '',
  },
  {
    id: 'cv-engineer', title: 'Become a CV Engineer', role: 'Computer Vision Engineer',
    color: '#10B981', glow: 'rgba(16,185,129,0.2)',
    modules: 11, estimatedHours: 90, difficulty: 'Intermediate', progress: 60,
    enrolled: 2400, domains: ['math', 'dl', 'cv'],
    href: '/roadmaps/cv-engineer', badge: '',
  },
];

/* ─── Trending Topics ──────────────────────────────────────── */
export const TRENDING: TrendingTopic[] = [
  { id: 't1', title: 'Attention Mechanism', domain: 'Deep Learning', difficulty: 'Intermediate', diffColor: '#F59E0B', estimatedMinutes: 45, popularity: 97, completionRate: 84, color: '#8B5CF6', href: '/learn/deep-learning/attention', tags: ['transformers', 'nlp'] },
  { id: 't2', title: 'Transformers', domain: 'Deep Learning', difficulty: 'Advanced', diffColor: '#EF4444', estimatedMinutes: 90, popularity: 95, completionRate: 71, color: '#06B6D4', href: '/learn/deep-learning/transformers', tags: ['architecture', 'attention'] },
  { id: 't3', title: 'RAG Architecture', domain: 'LLMs', difficulty: 'Intermediate', diffColor: '#F59E0B', estimatedMinutes: 60, popularity: 92, completionRate: 78, color: '#EC4899', href: '/learn/rag', tags: ['retrieval', 'llm'] },
  { id: 't4', title: 'AI Agents', domain: 'LLMs', difficulty: 'Advanced', diffColor: '#EF4444', estimatedMinutes: 75, popularity: 90, completionRate: 65, color: '#F59E0B', href: '/learn/agents', tags: ['autonomous', 'tools'] },
  { id: 't5', title: 'Diffusion Models', domain: 'Computer Vision', difficulty: 'Advanced', diffColor: '#EF4444', estimatedMinutes: 80, popularity: 88, completionRate: 62, color: '#A78BFA', href: '/learn/computer-vision/diffusion-models', tags: ['generative', 'cv'] },
  { id: 't6', title: 'RLHF', domain: 'LLMs', difficulty: 'Advanced', diffColor: '#EF4444', estimatedMinutes: 70, popularity: 85, completionRate: 58, color: '#34D399', href: '/learn/llms', tags: ['alignment', 'rl'] },
  { id: 't7', title: 'Mixture of Experts', domain: 'LLMs', difficulty: 'Advanced', diffColor: '#EF4444', estimatedMinutes: 65, popularity: 83, completionRate: 55, color: '#22D3EE', href: '/learn/llms/mixture-of-experts', tags: ['scaling', 'architecture'] },
  { id: 't8', title: 'Vision Transformers', domain: 'Computer Vision', difficulty: 'Intermediate', diffColor: '#F59E0B', estimatedMinutes: 55, popularity: 80, completionRate: 72, color: '#10B981', href: '/learn/computer-vision/vision-transformers', tags: ['vit', 'attention'] },
  { id: 't9', title: 'State Space Models', domain: 'Deep Learning', difficulty: 'Advanced', diffColor: '#EF4444', estimatedMinutes: 70, popularity: 78, completionRate: 50, color: '#60A5FA', href: '/learn/deep-learning', tags: ['mamba', 'sequence'] },
];

/* ─── Recently Added ───────────────────────────────────────── */
export const RECENT_LESSONS: Lesson[] = [
  { id: 'l1', title: 'KV Cache Optimization in LLMs', domain: 'LLMs', domainColor: '#A78BFA', durationMinutes: 40, difficulty: 'Advanced', isNew: true, addedDaysAgo: 1, href: '/learn/kv-cache', type: 'lesson' },
  { id: 'l2', title: 'Flash Attention Deep Dive', domain: 'Deep Learning', domainColor: '#06B6D4', durationMinutes: 55, difficulty: 'Advanced', isNew: true, addedDaysAgo: 2, href: '/learn/flash-attention', type: 'lesson' },
  { id: 'l3', title: 'Build a RAG Pipeline from Scratch', domain: 'RAG Systems', domainColor: '#EC4899', durationMinutes: 90, difficulty: 'Intermediate', isNew: true, addedDaysAgo: 3, href: '/learn/rag-pipeline', type: 'lab' },
  { id: 'l4', title: 'Constitutional AI: Anthropic\'s Alignment', domain: 'Research', domainColor: '#34D399', durationMinutes: 35, difficulty: 'Advanced', isNew: true, addedDaysAgo: 4, href: '/learn/constitutional-ai', type: 'paper' },
  { id: 'l5', title: 'Mamba: Linear-Time Sequence Modeling', domain: 'Deep Learning', domainColor: '#06B6D4', durationMinutes: 60, difficulty: 'Advanced', isNew: true, addedDaysAgo: 5, href: '/learn/mamba', type: 'paper' },
  { id: 'l6', title: 'Multi-Agent Systems with LangGraph', domain: 'AI Agents', domainColor: '#F59E0B', durationMinutes: 75, difficulty: 'Intermediate', isNew: false, addedDaysAgo: 7, href: '/learn/langgraph', type: 'lab' },
];

/* ─── Recommendations ──────────────────────────────────────── */
export const RECOMMENDATIONS: Recommendation[] = [
  { id: 'r1', title: 'Multi-Head Attention Implementation', reason: 'Next in your Transformers track', domain: 'Deep Learning', domainColor: '#06B6D4', confidence: 97, type: 'lesson', estimatedMinutes: 45, difficulty: 'Intermediate', href: '/learn/mha' },
  { id: 'r2', title: 'Attention is All You Need (Paper)', reason: 'Pairs with current module', domain: 'Research', domainColor: '#34D399', confidence: 94, type: 'paper', estimatedMinutes: 30, difficulty: 'Advanced', href: '/papers/attention-is-all-you-need' },
  { id: 'r3', title: 'Positional Encoding Variants', reason: 'Unlocked by previous lesson', domain: 'NLP', domainColor: '#0284C7', confidence: 89, type: 'lesson', estimatedMinutes: 35, difficulty: 'Intermediate', href: '/learn/positional-encoding' },
  { id: 'r4', title: 'Implement GPT-2 from Scratch', reason: 'Matches your CV skill level', domain: 'LLMs', domainColor: '#A78BFA', confidence: 85, type: 'lab', estimatedMinutes: 120, difficulty: 'Advanced', href: '/learn/gpt2-scratch' },
  { id: 'r5', title: 'BERT Pre-training Objectives', reason: 'Expands your NLP knowledge', domain: 'NLP', domainColor: '#0284C7', confidence: 82, type: 'lesson', estimatedMinutes: 40, difficulty: 'Advanced', href: '/learn/bert-pretraining' },
];

/* ─── Knowledge Graph Nodes (hero + section 9) ─────────────── */
export const KG_NODES: KGNode[] = [
  { id: 'math',    label: 'Math',        x: 60,  y: 175, r: 28, color: '#8B5CF6', glow: 'rgba(139,92,246,0.5)', lessonCount: 48, floatDelay: 0 },
  { id: 'ml',      label: 'ML',          x: 165, y: 95,  r: 26, color: '#7C3AED', glow: 'rgba(124,58,237,0.5)', lessonCount: 64, floatDelay: 0.4 },
  { id: 'dl',      label: 'Deep L.',     x: 260, y: 165, r: 30, color: '#06B6D4', glow: 'rgba(6,182,212,0.5)',  lessonCount: 72, floatDelay: 0.8 },
  { id: 'nlp',     label: 'NLP',         x: 350, y: 90,  r: 24, color: '#0284C7', glow: 'rgba(2,132,199,0.5)', lessonCount: 58, floatDelay: 0.2 },
  { id: 'llm',     label: 'LLMs',        x: 420, y: 160, r: 28, color: '#A78BFA', glow: 'rgba(167,139,250,0.5)',lessonCount: 44, floatDelay: 1.0 },
  { id: 'agents',  label: 'Agents',      x: 470, y: 72,  r: 22, color: '#F59E0B', glow: 'rgba(245,158,11,0.5)', lessonCount: 32, floatDelay: 0.6 },
  { id: 'cv',      label: 'CV',          x: 160, y: 270, r: 24, color: '#10B981', glow: 'rgba(16,185,129,0.5)', lessonCount: 56, floatDelay: 1.2 },
  { id: 'rag',     label: 'RAG',         x: 345, y: 260, r: 22, color: '#EC4899', glow: 'rgba(236,72,153,0.5)', lessonCount: 28, floatDelay: 0.5 },
  { id: 'mlops',   label: 'MLOps',       x: 440, y: 270, r: 20, color: '#64748B', glow: 'rgba(100,116,139,0.5)',lessonCount: 36, floatDelay: 0.9 },
  { id: 'research',label: 'Research',    x: 78,  y: 320, r: 20, color: '#34D399', glow: 'rgba(52,211,153,0.5)', lessonCount: 24, floatDelay: 1.4 },
  { id: 'ai-eng',  label: 'AI Eng.',     x: 258, y: 340, r: 32, color: '#22D3EE', glow: 'rgba(34,211,238,0.5)', lessonCount: 52, floatDelay: 0.7 },
];

export const KG_EDGES: KGEdge[] = [
  { from: 'math',    to: 'ml',       weight: 3 },
  { from: 'math',    to: 'cv',       weight: 2 },
  { from: 'math',    to: 'research', weight: 2 },
  { from: 'ml',      to: 'dl',       weight: 3 },
  { from: 'ml',      to: 'mlops',    weight: 1 },
  { from: 'dl',      to: 'nlp',      weight: 2 },
  { from: 'dl',      to: 'cv',       weight: 2 },
  { from: 'dl',      to: 'llm',      weight: 3 },
  { from: 'nlp',     to: 'llm',      weight: 3 },
  { from: 'nlp',     to: 'rag',      weight: 2 },
  { from: 'llm',     to: 'agents',   weight: 2 },
  { from: 'llm',     to: 'rag',      weight: 2 },
  { from: 'agents',  to: 'ai-eng',   weight: 2 },
  { from: 'cv',      to: 'ai-eng',   weight: 1 },
  { from: 'rag',     to: 'ai-eng',   weight: 2 },
  { from: 'mlops',   to: 'ai-eng',   weight: 1 },
  { from: 'research',to: 'ai-eng',   weight: 1 },
];


/* ─── Current Lesson ───────────────────────────────────────── */
export const CURRENT_LESSON: CurrentLesson = {
  trackTitle: 'Become an ML Engineer',
  module: 'Module 5: Attention Mechanisms',
  lesson: 'Scaled Dot-Product Attention',
  progress: 62,
  timeRemainingMinutes: 18,
  domain: 'Deep Learning',
  domainColor: '#06B6D4',
};
