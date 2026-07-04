'use client';

import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useState, useEffect } from 'react';
import {
  ArrowLeft, ChevronLeft, ChevronRight,
  Play, Send, RotateCcw, Copy, Clock,
  Lock, CheckCircle, XCircle, Loader2,
} from 'lucide-react';
import { PROBLEMS, getProblemBySlug, getProblemIndex } from '@/data/problems';
import { apiGet, apiPost } from '@/lib/api';

const MonacoEditor = dynamic(() => import('@monaco-editor/react'), { ssr: false });

type SlugMeta = { archSlug?: string; archName?: string; paperSlug?: string; paperTitle?: string; learnPath?: string; learnName?: string };
const SLUG_META: Record<string, SlugMeta> = {
  'numpy-array-creation': { learnPath: '/learn/mathematics', learnName: 'Mathematics' },
  'ml-sigmoid':           { archSlug: 'transformer', archName: 'Transformer', learnPath: '/learn/deep-learning', learnName: 'Deep Learning' },
  'ml-relu':              { archSlug: 'resnet', archName: 'ResNet', paperSlug: 'resnet', paperTitle: 'Deep Residual Learning', learnPath: '/learn/deep-learning', learnName: 'Deep Learning' },
  'ml-mse':               { archSlug: 'resnet', archName: 'ResNet', paperSlug: 'resnet', paperTitle: 'Deep Residual Learning', learnPath: '/learn/deep-learning', learnName: 'Deep Learning' },
  'numpy-dot-product':    { paperSlug: 'lora', paperTitle: 'LoRA', learnPath: '/learn/mathematics', learnName: 'Mathematics' },
  'ml-softmax':           { archSlug: 'transformer', archName: 'Transformer', paperSlug: 'attention-is-all-you-need', paperTitle: 'Attention Is All You Need', learnPath: '/learn/deep-learning', learnName: 'Deep Learning' },
  'ml-normalize':         { archSlug: 'vit', archName: 'ViT', paperSlug: 'vit', paperTitle: 'An Image is Worth 16×16 Words', learnPath: '/learn/computer-vision', learnName: 'Computer Vision' },
  'ml-cross-entropy':     { archSlug: 'bert', archName: 'BERT', paperSlug: 'bert', paperTitle: 'BERT', learnPath: '/learn/natural-language-processing', learnName: 'Natural Language Processing' },
  'ml-gradient-descent':  { paperSlug: 'lora', paperTitle: 'LoRA', learnPath: '/learn/deep-learning', learnName: 'Deep Learning' },
  'ml-attention':         { archSlug: 'transformer', archName: 'Transformer', paperSlug: 'attention-is-all-you-need', paperTitle: 'Attention Is All You Need', learnPath: '/learn/natural-language-processing', learnName: 'Natural Language Processing' },
};

type LeftTab   = 'description' | 'submissions' | 'notes';
type ConsoleTab = 'testcase' | 'result';
type RunState  = 'idle' | 'running' | 'passed' | 'failed' | 'error';

/* ── Timer ─────────────────────────────────────────────── */
function Timer() {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSecs(s => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  const m = String(Math.floor(secs / 60)).padStart(2, '0');
  const s = String(secs % 60).padStart(2, '0');
  return <span className="font-mono text-[13px] text-white tabular-nums">{m}:{s}</span>;
}

/* ── Difficulty badge ───────────────────────────────────── */
function DiffBadge({ d }: { d: string }) {
  const cls =
    d === 'Easy'   ? 'bg-[#4ADE80]/12 text-[#4ADE80]' :
    d === 'Medium' ? 'bg-[#FACC15]/12 text-[#FACC15]' :
                     'bg-[#F87171]/12 text-[#F87171]';
  return (
    <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-md ${cls}`}>{d}</span>
  );
}

/* ══════════════════════════════════════════════════════════
   PAGE
══════════════════════════════════════════════════════════ */
export default function ProblemPage() {
  const params        = useParams<{ slug: string }>();
  const slug          = params?.slug ?? '';
  const problem       = getProblemBySlug(slug);
  const problemIndex  = getProblemIndex(slug);
  const prevProblem   = PROBLEMS[problemIndex - 1];
  const nextProblem   = PROBLEMS[problemIndex + 1];

  const [leftTab,    setLeftTab]    = useState<LeftTab>('description');
  const [consoleTab, setConsoleTab] = useState<ConsoleTab>('testcase');
  const [code,       setCode]       = useState('');
  const [testInput,  setTestInput]  = useState('');
  const [runState,   setRunState]   = useState<RunState>('idle');
  const [stdout,     setStdout]     = useState('');
  const [expected,   setExpected]   = useState('');
  const [notes,      setNotes]      = useState('');
  const [submissions, setSubmissions] = useState<{status:string; created_at:string}[]>([]);

  /* load saved state on slug change */
  useEffect(() => {
    if (!problem) return;
    setCode(localStorage.getItem(`code_${slug}`) ?? problem.starter_code);
    setNotes(localStorage.getItem(`notes_${slug}`) ?? '');
    setTestInput(problem.test_input);
    setRunState('idle');
    setStdout('');
    setExpected('');

    // Fetch prior submissions
    apiGet<{status:string; created_at:string}[]>(`/api/dojo/submissions?problem_slug=${slug}`)
      .then(data => {
        if (Array.isArray(data)) setSubmissions(data);
      })
      .catch(err => console.error('Failed to load submissions:', err));
  }, [slug, problem]);

  /* auto-save code */
  useEffect(() => {
    if (!code || !slug) return;
    const t = setTimeout(() => localStorage.setItem(`code_${slug}`, code), 800);
    return () => clearTimeout(t);
  }, [code, slug]);

  /* ── Run ─────────────────────────────────────────────── */
  async function handleRun() {
    setRunState('running');
    setConsoleTab('result');
    try {
      const data = await apiPost<any>('/api/dojo/run', {
        problem_slug: slug, code, language: 'python', test_input: testInput
      });
      if (data.passed || data.status === 'passed') {
        setRunState('passed');
        setStdout(data.stdout ?? data.output ?? 'All tests passed!');
        setExpected('');
      } else {
        setRunState('failed');
        setStdout(data.stdout ?? data.error ?? data.output ?? 'Wrong answer');
        setExpected(data.expected ?? '');
      }
    } catch (err: any) {
      setRunState('error');
      setStdout(err.message || 'Could not reach the server. Is the backend running?');
    }
  }

  /* ── Submit ──────────────────────────────────────────── */
  async function handleSubmit() {
    setRunState('running');
    setConsoleTab('result');
    try {
      const data = await apiPost<any>('/api/dojo/submit', {
        problem_slug: slug, code, language: 'python'
      });
      if (data.passed || data.status === 'accepted') {
        setRunState('passed');
        setStdout(
          data.tests_passed !== undefined
            ? `${data.tests_passed} / ${data.tests_total} test cases passed`
            : (data.stdout ?? 'Accepted!'),
        );
        setSubmissions(prev => [{ status: 'accepted', created_at: new Date().toLocaleString() }, ...prev]);
      } else {
        setRunState('failed');
        setStdout(data.error ?? data.stdout ?? 'Some test cases failed');
        setExpected(data.expected ?? '');
        setSubmissions(prev => [{ status: 'wrong_answer', created_at: new Date().toLocaleString() }, ...prev]);
      }
    } catch (err: any) {
      setRunState('error');
      setStdout(err.message || 'Could not reach the server. Is the backend running?');
    }
  }

  /* ── Problem not found ───────────────────────────────── */
  if (!problem) {
    return (
      <div
        className="flex h-screen items-center justify-center flex-col gap-4"
        style={{ background: 'transparent' }}
      >
        <p style={{ color: '#525252' }}>Problem "{slug}" not found.</p>
        <Link href="/dojo" style={{ color: '#34D399', fontSize: 13 }}>
          ← Back to Dojo
        </Link>
      </div>
    );
  }

  /* ── Layout ──────────────────────────────────────────── */
  return (
    <div
      className="flex flex-col"
      style={{ height: '100vh', background: 'transparent', overflow: 'hidden' }}
    >
      {/* ── TOP BAR ──────────────────────────────────── */}
      <div
        className="flex items-center gap-3 px-4 flex-shrink-0"
        style={{
          height: 52,
          background: 'transparent',
          borderBottom: '1px solid #1B2A20',
        }}
      >
        <Link
          href="/dojo"
          className="flex items-center gap-1.5 transition-colors"
          style={{ color: '#525252', fontSize: 13 }}
          onMouseEnter={e => (e.currentTarget.style.color = '#FAFAFA')}
          onMouseLeave={e => (e.currentTarget.style.color = '#525252')}
        >
          <ArrowLeft size={14} />
          Problems
        </Link>

        <div style={{ width: 1, height: 16, background: '#223429' }} />

        <span style={{ fontSize: 13, fontWeight: 600, color: '#FAFAFA' }}>
          #{String(problemIndex + 1).padStart(3, '0')} · {problem.title}
        </span>

        <DiffBadge d={problem.difficulty} />

        <div style={{ flex: 1 }} />

        {/* prev / next */}
        <div className="flex items-center gap-1.5" style={{ color: '#525252', fontSize: 12 }}>
          <Link
            href={prevProblem ? `/dojo/${prevProblem.slug}` : '#'}
            className="transition-colors p-1 rounded"
            style={{ opacity: prevProblem ? 1 : 0.3, pointerEvents: prevProblem ? 'auto' : 'none' }}
            onMouseEnter={e => (e.currentTarget.style.color = '#FAFAFA')}
            onMouseLeave={e => (e.currentTarget.style.color = '#525252')}
          >
            <ChevronLeft size={16} />
          </Link>
          <span style={{ color: '#525252' }}>{problemIndex + 1} / {PROBLEMS.length}</span>
          <Link
            href={nextProblem ? `/dojo/${nextProblem.slug}` : '#'}
            className="transition-colors p-1 rounded"
            style={{ opacity: nextProblem ? 1 : 0.3, pointerEvents: nextProblem ? 'auto' : 'none' }}
            onMouseEnter={e => (e.currentTarget.style.color = '#FAFAFA')}
            onMouseLeave={e => (e.currentTarget.style.color = '#525252')}
          >
            <ChevronRight size={16} />
          </Link>
        </div>

        {/* timer */}
        <div
          className="flex items-center gap-2 px-3 py-1.5 rounded-md"
          style={{ background: '#16241B', border: '1px solid #223429' }}
        >
          <Clock size={12} style={{ color: '#525252' }} />
          <Timer />
        </div>
      </div>

      {/* ── SPLIT PANE ───────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* LEFT PANEL */}
        <div
          className="flex flex-col overflow-hidden"
          style={{ width: '45%', borderRight: '1px solid #1B2A20' }}
        >
          {/* tab bar */}
          <div
            className="flex items-end gap-1 px-4 flex-shrink-0"
            style={{ height: 44, borderBottom: '1px solid #1B2A20' }}
          >
            {(['description', 'submissions', 'notes'] as LeftTab[]).map(t => (
              <button
                key={t}
                onClick={() => setLeftTab(t)}
                className="capitalize transition-colors"
                style={{
                  padding: '0 12px 10px',
                  fontSize: 13,
                  color: leftTab === t ? '#34D399' : '#525252',
                  borderBottom: leftTab === t ? '2px solid #34D399' : '2px solid transparent',
                  background: 'none',
                  cursor: 'pointer',
                }}
                onMouseEnter={e => { if (leftTab !== t) e.currentTarget.style.color = '#A3A3A3'; }}
                onMouseLeave={e => { if (leftTab !== t) e.currentTarget.style.color = '#525252'; }}
              >
                {t}
              </button>
            ))}
            {/* locked solution tab */}
            <button
              style={{
                padding: '0 12px 10px',
                fontSize: 13,
                color: '#525252',
                borderBottom: '2px solid transparent',
                background: 'none',
                cursor: 'default',
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                paddingBottom: 10,
              }}
            >
              Solution <Lock size={10} />
            </button>
          </div>

          {/* tab content */}
          <div className="flex-1 overflow-y-auto" style={{ scrollbarWidth: 'thin', scrollbarColor: '#223429 transparent' }}>

            {/* DESCRIPTION */}
            {leftTab === 'description' && (
              <div className="p-6">
                <h1 style={{ fontSize: 19, fontWeight: 700, color: '#FAFAFA' }}>{problem.title}</h1>

                <div className="flex flex-wrap gap-2 mt-3">
                  {problem.topics.map(t => (
                    <span
                      key={t}
                      style={{
                        background: '#1B2C21', color: '#60A5FA',
                        fontSize: 11, padding: '4px 8px', borderRadius: 6,
                      }}
                    >{t}</span>
                  ))}
                </div>

                <hr style={{ borderColor: '#1B2A20', margin: '16px 0' }} />

                <p style={{ fontSize: 13, color: '#A3A3A3', lineHeight: 1.85, whiteSpace: 'pre-line' }}>
                  {problem.description}
                </p>

                {problem.examples.map((ex, i) => (
                  <div key={i} className="mt-5">
                    <p style={{ fontSize: 11, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                      Example {i + 1}
                    </p>
                    <div
                      className="space-y-1"
                      style={{ background: '#0E1811', borderRadius: 12, padding: 16, fontFamily: 'JetBrains Mono, monospace', fontSize: 12 }}
                    >
                      <div>
                        <span style={{ color: '#4ADE80' }}>Input: </span>
                        <span style={{ color: '#FAFAFA' }}>{ex.input}</span>
                      </div>
                      <div>
                        <span style={{ color: '#60A5FA' }}>Output: </span>
                        <span style={{ color: '#FAFAFA' }}>{ex.output}</span>
                      </div>
                      {ex.explanation && (
                        <div style={{ color: '#525252', marginTop: 4 }}># {ex.explanation}</div>
                      )}
                    </div>
                  </div>
                ))}

                <div
                  style={{
                    marginTop: 20, background: '#0E1811',
                    border: '1px solid #223429', borderRadius: 12, padding: 16,
                  }}
                >
                  <p style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.12em', color: '#34D399', textTransform: 'uppercase', marginBottom: 12 }}>
                    Constraints
                  </p>
                  <ul className="space-y-2">
                    {problem.constraints.map((c, i) => (
                      <li key={i} className="flex items-start gap-2" style={{ fontSize: 12, color: '#A3A3A3' }}>
                        <span style={{ color: '#525252', marginTop: 2 }}>•</span>
                        {c}
                      </li>
                    ))}
                  </ul>
                </div>

                {problem.hints.length > 0 && (
                  <div
                    style={{
                      marginTop: 16, border: '1px solid #223429',
                      borderRadius: 12, padding: 16,
                    }}
                  >
                    <p style={{ fontSize: 10, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>
                      Hints
                    </p>
                    {problem.hints.map((h, i) => (
                      <p key={i} style={{ fontSize: 12, color: '#A3A3A3', marginTop: 6 }}>💡 {h}</p>
                    ))}
                  </div>
                )}

                {/* RELATED */}
                {(SLUG_META[slug]?.archSlug || SLUG_META[slug]?.paperSlug || SLUG_META[slug]?.learnPath) && (
                  <div style={{ marginTop: 20, border: '1px solid #223429', borderRadius: 12, padding: 16 }}>
                    <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', color: '#A78BFA', textTransform: 'uppercase', marginBottom: 12 }}>
                      Related
                    </p>
                    <div className="space-y-2">
                      {SLUG_META[slug]?.learnPath && (
                        <Link href={SLUG_META[slug].learnPath!}
                          className="flex items-center justify-between rounded-lg px-3 py-2.5 transition-colors hover:bg-[#121D16]"
                          style={{ border: '1px solid #1B2A20' }}>
                          <span style={{ fontSize: 12, color: '#60A5FA' }}>📚 {SLUG_META[slug].learnName} Path</span>
                          <span style={{ fontSize: 11, color: '#525252' }}>Learn →</span>
                        </Link>
                      )}
                      {SLUG_META[slug]?.archSlug && (
                        <Link href={`/architectures/${SLUG_META[slug].archSlug}`}
                          className="flex items-center justify-between rounded-lg px-3 py-2.5 transition-colors hover:bg-[#121D16]"
                          style={{ border: '1px solid #1B2A20' }}>
                          <span style={{ fontSize: 12, color: '#A3E635' }}>⚡ {SLUG_META[slug].archName} Architecture</span>
                          <span style={{ fontSize: 11, color: '#525252' }}>Explore →</span>
                        </Link>
                      )}
                      {SLUG_META[slug]?.paperSlug && (
                        <Link href={`/papers/${SLUG_META[slug].paperSlug}`}
                          className="flex items-center justify-between rounded-lg px-3 py-2.5 transition-colors hover:bg-[#121D16]"
                          style={{ border: '1px solid #1B2A20' }}>
                          <span style={{ fontSize: 12, color: '#A78BFA' }}>📄 {SLUG_META[slug].paperTitle}</span>
                          <span style={{ fontSize: 11, color: '#525252' }}>Read →</span>
                        </Link>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* SUBMISSIONS */}
            {leftTab === 'submissions' && (
              <div className="p-4">
                <p style={{ fontSize: 12, color: '#525252', marginBottom: 16 }}>
                  Your submissions for this problem
                </p>
                {submissions.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <CheckCircle size={32} style={{ color: '#223429', marginBottom: 12 }} />
                    <p style={{ fontSize: 13, color: '#525252' }}>No submissions yet</p>
                    <p style={{ fontSize: 11, color: '#525252', marginTop: 4 }}>
                      Hit Submit to record your solution
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {submissions.map((s, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between px-4 py-3 rounded-lg"
                        style={{ background: '#121D16', border: '1px solid #223429' }}
                      >
                        <span
                          style={{
                            fontSize: 12, fontWeight: 600,
                            color: s.status === 'accepted' ? '#4ADE80' : '#F87171',
                          }}
                        >
                          {s.status === 'accepted' ? '✓ Accepted' : '✗ Wrong Answer'}
                        </span>
                        <span style={{ fontSize: 11, color: '#525252' }}>{s.created_at}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* NOTES */}
            {leftTab === 'notes' && (
              <div style={{ padding: 16, height: '100%' }}>
                <textarea
                  style={{
                    width: '100%', height: 420, background: 'transparent',
                    resize: 'none', fontSize: 13, color: '#A3A3A3',
                    border: 'none', outline: 'none', lineHeight: 1.7,
                    fontFamily: 'Inter, system-ui, sans-serif',
                  }}
                  placeholder="Write your notes here... (auto-saved)"
                  value={notes}
                  onChange={e => {
                    setNotes(e.target.value);
                    localStorage.setItem(`notes_${slug}`, e.target.value);
                  }}
                />
              </div>
            )}
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div className="flex flex-col flex-1 overflow-hidden">

          {/* editor toolbar */}
          <div
            className="flex items-center gap-2 px-4 flex-shrink-0"
            style={{ height: 44, background: '#0E1811', borderBottom: '1px solid #1B2A20' }}
          >
            <span
              style={{
                background: '#0F2418', border: '1px solid rgba(52,211,153,0.4)',
                color: '#34D399', fontSize: 12, fontWeight: 600,
                padding: '4px 12px', borderRadius: 6,
              }}
            >
              Python 3.11
            </span>
            <div style={{ flex: 1 }} />
            <button
              onClick={() => { setCode(problem.starter_code); localStorage.removeItem(`code_${slug}`); }}
              className="flex items-center gap-1.5 transition-colors"
              style={{
                background: '#16241B', border: '1px solid #223429', borderRadius: 6,
                padding: '6px 10px', fontSize: 11, color: '#A3A3A3', cursor: 'pointer',
              }}
              onMouseEnter={e => (e.currentTarget.style.color = '#FAFAFA')}
              onMouseLeave={e => (e.currentTarget.style.color = '#A3A3A3')}
            >
              <RotateCcw size={11} /> Reset
            </button>
            <button
              onClick={() => navigator.clipboard.writeText(code)}
              className="flex items-center gap-1.5 transition-colors"
              style={{
                background: '#16241B', border: '1px solid #223429', borderRadius: 6,
                padding: '6px 10px', fontSize: 11, color: '#A3A3A3', cursor: 'pointer',
              }}
              onMouseEnter={e => (e.currentTarget.style.color = '#FAFAFA')}
              onMouseLeave={e => (e.currentTarget.style.color = '#A3A3A3')}
            >
              <Copy size={11} /> Copy
            </button>
          </div>

          {/* Monaco editor */}
          <div className="flex-1 min-h-0">
            <MonacoEditor
              height="100%"
              language="python"
              theme="vs-dark"
              value={code}
              onChange={v => setCode(v ?? '')}
              options={{
                fontSize: 13,
                fontFamily: '"JetBrains Mono", "Fira Code", monospace',
                fontLigatures: true,
                lineHeight: 22,
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                padding: { top: 16, bottom: 16 },
                renderLineHighlight: 'all',
                overviewRulerLanes: 0,
                lineDecorationsWidth: 6,
                glyphMargin: false,
                folding: true,
                wordWrap: 'on',
              }}
            />
          </div>

          {/* action bar */}
          <div
            className="flex items-center justify-end gap-3 px-4 flex-shrink-0"
            style={{ height: 52, background: '#0E1811', borderTop: '1px solid #1B2A20' }}
          >
            <button
              onClick={handleRun}
              disabled={runState === 'running'}
              className="flex items-center gap-2 transition-colors"
              style={{
                background: '#16241B', border: '1px solid #223429', borderRadius: 8,
                padding: '8px 20px', fontSize: 13, color: '#FAFAFA', cursor: 'pointer',
                opacity: runState === 'running' ? 0.5 : 1,
              }}
              onMouseEnter={e => { if (runState !== 'running') e.currentTarget.style.background = '#1B2C21'; }}
              onMouseLeave={e => { e.currentTarget.style.background = '#16241B'; }}
            >
              {runState === 'running'
                ? <Loader2 size={13} className="animate-spin" />
                : <Play size={13} />}
              Run
            </button>
            <button
              onClick={handleSubmit}
              disabled={runState === 'running'}
              className="flex items-center gap-2 transition-all"
              style={{
                background: '#34D399', borderRadius: 8,
                padding: '8px 24px', fontSize: 13, fontWeight: 600,
                color: '#000', cursor: 'pointer', border: 'none',
                opacity: runState === 'running' ? 0.5 : 1,
              }}
              onMouseEnter={e => { if (runState !== 'running') e.currentTarget.style.filter = 'brightness(1.1)'; }}
              onMouseLeave={e => { e.currentTarget.style.filter = 'none'; }}
            >
              {runState === 'running'
                ? <Loader2 size={13} className="animate-spin" />
                : <Send size={13} />}
              Submit
            </button>
          </div>

          {/* CONSOLE PANEL */}
          <div
            className="flex flex-col flex-shrink-0"
            style={{ height: 220, borderTop: '1px solid #1B2A20', background: '#0C160F' }}
          >
            {/* drag handle */}
            <div
              style={{ height: 3, background: '#1B2A20', cursor: 'row-resize', flexShrink: 0, transition: 'background 150ms' }}
              onMouseEnter={e => (e.currentTarget.style.background = 'rgba(52,211,153,0.4)')}
              onMouseLeave={e => (e.currentTarget.style.background = '#1B2A20')}
            />

            {/* console tabs */}
            <div
              className="flex items-end gap-1 px-4 flex-shrink-0"
              style={{ height: 40, borderBottom: '1px solid #1B2A20' }}
            >
              {([['testcase', 'Testcase'], ['result', 'Test Result']] as [ConsoleTab, string][]).map(([k, label]) => (
                <button
                  key={k}
                  onClick={() => setConsoleTab(k)}
                  style={{
                    padding: '0 10px 8px', fontSize: 12,
                    color: consoleTab === k ? '#34D399' : '#525252',
                    borderBottom: consoleTab === k ? '2px solid #34D399' : '2px solid transparent',
                    background: 'none', cursor: 'pointer', transition: 'color 150ms',
                  }}
                  onMouseEnter={e => { if (consoleTab !== k) e.currentTarget.style.color = '#A3A3A3'; }}
                  onMouseLeave={e => { if (consoleTab !== k) e.currentTarget.style.color = '#525252'; }}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* console content */}
            <div className="flex-1 overflow-y-auto" style={{ scrollbarWidth: 'thin', scrollbarColor: '#223429 transparent' }}>

              {consoleTab === 'testcase' && (
                <div style={{ padding: 16 }}>
                  <p style={{ fontSize: 10, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                    Input
                  </p>
                  <textarea
                    style={{
                      width: '100%', height: 120, background: 'transparent',
                      fontFamily: 'JetBrains Mono, monospace', fontSize: 12,
                      color: '#A3A3A3', resize: 'none', border: 'none', outline: 'none',
                    }}
                    value={testInput}
                    onChange={e => setTestInput(e.target.value)}
                  />
                </div>
              )}

              {consoleTab === 'result' && (
                <div style={{ padding: 16 }}>
                  {runState === 'idle' && (
                    <p style={{ fontSize: 13, color: '#525252', textAlign: 'center', marginTop: 24 }}>
                      Run your code to see output
                    </p>
                  )}

                  {runState === 'running' && (
                    <div className="flex items-center justify-center gap-3" style={{ marginTop: 24 }}>
                      <div className="flex gap-1">
                        {[0, 1, 2].map(i => (
                          <div
                            key={i}
                            className="animate-bounce"
                            style={{
                              width: 6, height: 6, borderRadius: '50%',
                              background: '#34D399',
                              animationDelay: `${i * 0.15}s`,
                            }}
                          />
                        ))}
                      </div>
                      <span style={{ fontSize: 13, color: '#525252' }}>Running your code…</span>
                    </div>
                  )}

                  {runState === 'passed' && (
                    <div>
                      <div
                        className="flex items-center gap-2 rounded-lg px-3 py-2.5"
                        style={{
                          background: 'rgba(74,222,128,0.08)',
                          border: '1px solid rgba(74,222,128,0.25)',
                          color: '#4ADE80', fontSize: 13, fontWeight: 600, marginBottom: 12,
                        }}
                      >
                        <CheckCircle size={14} /> Accepted
                      </div>
                      {stdout && (
                        <>
                          <p style={{ fontSize: 10, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Output</p>
                          <pre style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#A3A3A3', whiteSpace: 'pre-wrap' }}>{stdout}</pre>
                        </>
                      )}
                    </div>
                  )}

                  {(runState === 'failed' || runState === 'error') && (
                    <div>
                      <div
                        className="flex items-center gap-2 rounded-lg px-3 py-2.5"
                        style={{
                          background: 'rgba(248,113,113,0.08)',
                          border: '1px solid rgba(248,113,113,0.25)',
                          color: '#F87171', fontSize: 13, fontWeight: 600, marginBottom: 12,
                        }}
                      >
                        <XCircle size={14} />
                        {runState === 'error' ? 'Runtime Error' : 'Wrong Answer'}
                      </div>
                      {stdout && (
                        <>
                          <p style={{ fontSize: 10, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Output</p>
                          <pre style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#A3A3A3', whiteSpace: 'pre-wrap' }}>{stdout}</pre>
                        </>
                      )}
                      {expected && (
                        <div style={{ marginTop: 10 }}>
                          <p style={{ fontSize: 10, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Expected</p>
                          <pre style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#4ADE80', whiteSpace: 'pre-wrap' }}>{expected}</pre>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
