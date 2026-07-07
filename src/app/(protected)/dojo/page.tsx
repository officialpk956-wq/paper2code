'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { apiGet, isLoggedIn } from '@/lib/api';
import { PROBLEMS } from '@/data/problems';
import { motion } from 'framer-motion';

type ProblemRow = {
  id: number; slug: string; title: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  topics: string[]; acceptance: string; solved: boolean;
};

// Matches GET /api/leaderboard's real response: {leaders: [{rank, user_id,
// name, points, xp_level, streak, problems_solved, avatar_url}]}
type LeaderboardEntry = {
  rank: number; user_id: number; name: string;
  points: number; xp_level: number; streak: number; problems_solved: number;
  avatar_url?: string | null;
};

// Matches GET /api/dojo/submissions rows: {problem_id, passed, created_at, ...}
type SubmissionRow = {
  problem_id: string; passed: boolean;
  [key: string]: unknown;
};

// Derived from the single source of truth in src/data/problems.ts —
// every listed row is a real, solvable problem by construction.
const INITIAL_PROBLEMS: ProblemRow[] = PROBLEMS.map((p, i) => ({
  id: i + 1,
  slug: p.slug,
  title: p.title,
  difficulty: p.difficulty,
  topics: p.topics,
  acceptance: p.acceptance,
  solved: false,
}));

const TOPIC_CHIPS = ['All', ...Array.from(new Set(PROBLEMS.flatMap(p => p.topics)))];

const DIFF_TEXT: Record<ProblemRow['difficulty'], string> = {
  Easy: 'text-[#4ADE80]', Medium: 'text-[#FACC15]', Hard: 'text-[#F87171]',
};

const RANK_BORDER: Record<number, string> = { 1: 'border-l-[#FACC15]', 2: 'border-l-[#D4D4D8]', 3: 'border-l-[#B45309]' };

function useCountdown() {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  if (!now) return '--:--:--';
  const end  = new Date(now); end.setHours(24, 0, 0, 0);
  const diff = Math.max(0, end.getTime() - now.getTime());
  const h = Math.floor(diff / 3_600_000);
  const m = Math.floor((diff % 3_600_000) / 60_000);
  const s = Math.floor((diff % 60_000) / 1000);
  return [h, m, s].map(n => String(n).padStart(2, '0')).join(':');
}

function fmtId(id: number) { return '#' + String(id).padStart(3, '0'); }

export default function DojoPage() {
  const [tab,        setTab]        = useState<'problems' | 'leaderboard'>('problems');
  const [topic,      setTopic]      = useState('All');
  const [search,     setSearch]     = useState('');
  const [status,     setStatus]     = useState('All');
  const [difficulty, setDifficulty] = useState('All');
  const [board,      setBoard]      = useState<'Weekly' | 'All-time'>('Weekly');
  const countdown = useCountdown();

  const [problems, setProblems] = useState<ProblemRow[]>(INITIAL_PROBLEMS);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  const [userProfile, setUserProfile] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const p = localStorage.getItem('user_profile');
      if (p) {
        try { setUserProfile(JSON.parse(p)); } catch (e) {}
      }
    }
    
    const fetchData = async () => {
      setError('');
      setLoading(true);
      try {
        // NOTE: leaderboard returns an OBJECT {leaders: [...]}, not an array,
        // and `category=all` would ILIKE-filter for the literal string "all".
        const [lbData, subData] = await Promise.all([
          apiGet<{ leaders: LeaderboardEntry[] }>('/api/leaderboard?limit=10').catch(() => null),
          isLoggedIn()
            ? apiGet<{ submissions: SubmissionRow[] }>('/api/dojo/submissions?limit=100').catch(() => null)
            : Promise.resolve(null),
        ]);

        setLeaderboard(lbData?.leaders ?? []);

        const rows = subData?.submissions ?? [];
        if (rows.length > 0) {
          const solvedSlugs = new Set(
            rows.filter(s => s.passed).map(s => s.problem_id)
          );

          setProblems(INITIAL_PROBLEMS.map(p => ({
            ...p,
            solved: solvedSlugs.has(p.slug),
          })));
        }
      } catch (err: unknown) {
        setError((err as Error).message || 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);

  const filtered = useMemo(() => problems.filter(p => {
    if (topic !== 'All' && !p.topics.includes(topic)) return false;
    if (search && !p.title.toLowerCase().includes(search.toLowerCase())) return false;
    if (status === 'Solved'   && !p.solved) return false;
    if (status === 'Unsolved' &&  p.solved) return false;
    if (difficulty !== 'All'  && p.difficulty !== difficulty) return false;
    return true;
  }), [problems, topic, search, status, difficulty]);

  return (
    <div className="flex overflow-hidden bg-transparent text-white" style={{ height: 'calc(100vh - 56px)' }}>
      {/* LEFT SIDEBAR */}
      <aside className="flex w-[260px] flex-shrink-0 flex-col gap-3 overflow-y-auto border-r border-[#1A1A1A] bg-[#0A0A0A] p-4">
        {/* POTD */}
        <Link href="/dojo/ml-attention" className="block">
          <motion.div 
            className="rounded-xl border border-[#A78BFA]/25 bg-[#0F2418] p-4 hover:border-[#A78BFA]/30 transition-colors h-full"
            whileHover={{ scale: 1.015, y: -3 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 400, damping: 25 }}
          >
            <div className="text-[9px] font-bold uppercase tracking-[0.12em] text-[#A78BFA]">Problem of the Day</div>
            <div className="mt-1 text-[13px] font-semibold text-white">Scaled Dot-Product Attention</div>
            <div className="mt-3 flex items-center justify-between">
              <span className="rounded-md bg-[#F87171]/12 px-2 py-0.5 text-[11px] font-semibold text-[#F87171]">Hard</span>
              <span className="font-mono text-xs text-[#A78BFA]">{countdown}</span>
            </div>
          </motion.div>
        </Link>

        {/* PROGRESS — real counts derived from submission history */}
        <div className="rounded-xl border border-[#262626] bg-[#111111] p-4">
          <div className="mb-3 text-[12px] font-semibold text-white">Your Progress</div>
          <div className="grid grid-cols-3 gap-2">
            {(['Easy', 'Medium', 'Hard'] as const).map(level => {
              const color = level === 'Easy' ? 'text-[#4ADE80]' : level === 'Medium' ? 'text-[#FACC15]' : 'text-[#F87171]';
              const n = problems.filter(p => p.difficulty === level && p.solved).length;
              return (
                <div key={level} className="rounded-lg border border-[#1A1A1A] bg-transparent p-2 text-center">
                  <div className={'text-xl font-bold ' + color}>{n}</div>
                  <div className="mt-0.5 text-[10px] text-[#525252]">{level}</div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="mt-auto flex gap-2 pt-4">
          {(['problems', 'leaderboard'] as const).map(t => (
            <button key={t} type="button" onClick={() => setTab(t)}
              className={'flex-1 rounded-lg px-4 py-2 text-xs ' +
                (tab === t
                  ? 'bg-[#A78BFA] font-semibold text-black'
                  : 'border border-[#262626] bg-[#111111] text-[#A3A3A3] hover:text-white')}>
              {t === 'problems' ? 'Problems' : 'Leaderboard'}
            </button>
          ))}
        </div>
      </aside>

      {/* MAIN */}
      <main className="flex flex-1 flex-col overflow-hidden relative">
        {error && (
          <div className="bg-red-500/10 border-b border-red-500/20 px-8 py-2 text-xs text-red-500 flex justify-center">
            Could not load data — retrying… ({error})
          </div>
        )}
        {tab === 'problems' ? (
          <>
            {/* Topic chips */}
            <div className="flex flex-wrap gap-2 border-b border-[#1A1A1A] px-5 py-3">
              {TOPIC_CHIPS.map(t => (
                <button key={t} type="button" onClick={() => setTopic(t)}
                  className={'rounded-full px-3 py-1 text-[11px] transition-colors ' +
                    (topic === t
                      ? 'bg-[#A78BFA] font-semibold text-black'
                      : 'border border-[#262626] bg-[#111111] text-[#A3A3A3] hover:text-white')}>
                  {t}
                </button>
              ))}
            </div>

            {/* Filter bar */}
            <div className="flex items-center gap-3 border-b border-[#1A1A1A] px-5 py-3">
              <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search problems..."
                className="w-[220px] rounded-lg border border-[#262626] bg-[#111111] px-3 py-2 text-[13px] text-white placeholder:text-[#525252] outline-none focus:border-[#A78BFA]" />
              <select value={status} onChange={e => setStatus(e.target.value)}
                className="w-[120px] rounded-lg border border-[#262626] bg-[#111111] px-3 py-2 text-[13px] text-white outline-none">
                <option value="All">All Status</option>
                <option>Solved</option><option>Unsolved</option>
              </select>
              <select value={difficulty} onChange={e => setDifficulty(e.target.value)}
                className="w-[120px] rounded-lg border border-[#262626] bg-[#111111] px-3 py-2 text-[13px] text-white outline-none">
                <option value="All">All</option>
                <option>Easy</option><option>Medium</option><option>Hard</option>
              </select>
              <div className="ml-auto text-[12px] text-[#525252]">{filtered.length} problems</div>
            </div>

            {/* Table */}
            <div className="flex-1 overflow-y-auto">
              <div className="sticky top-0 z-10 flex h-10 items-center border-b border-[#1A1A1A] bg-[#0A0A0A] px-4 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#525252]">
                <div className="w-7 flex-shrink-0"></div>
                <div className="min-w-0 flex-1"># Title</div>
                <div className="w-[68px] flex-shrink-0 text-right">Difficulty</div>
                <div className="w-[60px] flex-shrink-0 text-right">Accept.</div>
              </div>
              {filtered.map(p => {
                const Row = (
                  <motion.div 
                    className="flex items-center border-b border-[#1A1A1A]/60 px-4 py-2.5 hover:bg-[#111111] cursor-pointer min-h-[52px]"
                    whileHover={{ scale: 1.005, x: 2 }}
                    whileTap={{ scale: 0.995 }}
                    transition={{ type: 'spring', stiffness: 400, damping: 25 }}
                  >
                    <div className="w-7 flex-shrink-0">
                      <span className="inline-block h-2 w-2 rounded-full"
                        style={{ background: p.solved ? '#4ADE80' : '#262626' }} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="flex-shrink-0 text-[11px] text-[#525252]">{fmtId(p.id)}</span>
                        <span className="truncate text-[13px] font-semibold text-white">{p.title}</span>
                      </div>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {p.topics.slice(0, 2).map(t => (
                          <span key={t} className="rounded bg-[#1A1A1A] px-1.5 py-px text-[10px] text-[#525252]">{t}</span>
                        ))}
                      </div>
                    </div>
                    <div className={'w-[68px] flex-shrink-0 text-right text-[12px] font-semibold ' + DIFF_TEXT[p.difficulty]}>{p.difficulty}</div>
                    <div className="w-[60px] flex-shrink-0 text-right text-[11px] text-[#525252]">{p.acceptance}</div>
                  </motion.div>
                );
                return <Link key={p.slug} href={`/dojo/${p.slug}`} className="block">{Row}</Link>;
              })}
              {filtered.length === 0 && (
                <div className="py-20 text-center text-[#525252] border border-[#262626] border-dashed rounded-xl mx-4 mt-4">
                  No problems match your filters.
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col overflow-hidden">
            <div className="flex items-center justify-between border-b border-[#1A1A1A] px-5 py-4">
              <h2 className="text-[18px] font-bold text-white">🏆 Leaderboard</h2>
              <div className="flex gap-1 rounded-lg border border-[#262626] bg-[#111111] p-1">
                {(['Weekly', 'All-time'] as const).map(b => (
                  <button key={b} type="button" onClick={() => setBoard(b)}
                    className={'rounded-md px-3 py-1 text-xs ' +
                      (board === b ? 'bg-[#A78BFA] font-semibold text-black' : 'text-[#A3A3A3] hover:text-white')}>
                    {b}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex-1 overflow-y-auto">
              <div className="sticky top-0 z-10 flex h-10 items-center border-b border-[#1A1A1A] bg-[#0A0A0A] px-5 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#525252]">
                <div className="w-16">Rank</div><div className="flex-1">User</div>
                <div className="w-24">Solved</div><div className="w-24">XP</div><div className="w-24">Streak</div>
              </div>
              
              {loading && leaderboard.length === 0 ? (
                <div className="p-4 space-y-4">
                  {[1, 2, 3, 4, 5].map(i => (
                    <div key={i} className="flex h-[40px] items-center border border-[#1A1A1A] bg-[#111111] animate-pulse rounded px-5">
                      <div className="w-16 h-4 bg-[#262626] rounded"></div>
                      <div className="flex flex-1 items-center gap-3">
                        <div className="w-8 h-8 bg-[#262626] rounded-full"></div>
                        <div className="w-32 h-4 bg-[#262626] rounded"></div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : leaderboard.length === 0 ? (
                <div className="p-8 text-center text-[#525252] text-sm">
                  No leaderboard data available.
                </div>
              ) : (
                leaderboard.map(row => {
                  const isYou = typeof userProfile?.name === 'string' && row.name === userProfile.name;
                  return (
                    <div key={row.rank}
                      className={'flex h-[56px] items-center border-b border-[#1A1A1A]/60 px-5 ' +
                        (RANK_BORDER[row.rank] ? 'border-l-4 ' + RANK_BORDER[row.rank] + ' ' : '') +
                        (isYou ? 'bg-[#A78BFA]/10' : 'hover:bg-[#111111]')}>
                      <div className="w-16 text-[13px] font-semibold text-white">#{row.rank}</div>
                      <div className="flex flex-1 items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#1A1A1A] text-xs font-bold text-white">
                          {row.name ? row.name.charAt(0).toUpperCase() : '?'}
                        </div>
                        <span className={'text-[13px] ' + (isYou ? 'font-semibold text-[#A78BFA]' : 'text-white')}>
                          {isYou ? 'You' : row.name}
                        </span>
                      </div>
                      <div className="w-24 text-[13px] text-white">{row.problems_solved}</div>
                      <div className="w-24 text-[13px] text-[#A78BFA]">{row.points?.toLocaleString()}</div>
                      <div className="w-24 text-[13px] text-[#F59E0B]">🔥 {row.streak}</div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
