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
import { apiGet, apiPost, isLoggedIn } from '@/lib/api';

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
export default function DojoEditor({ children }: { children?: React.ReactNode }) {
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
    if (isLoggedIn()) {
      apiGet<{ submissions: { problem_id: string; passed: boolean; created_at: string | null }[] }>('/api/dojo/submissions?limit=100')
        .then(data => {
          const rows = (data?.submissions ?? [])
            .filter(s => s.problem_id === slug)
            .map(s => ({
              status: s.passed ? 'accepted' : 'wrong_answer',
              created_at: s.created_at ? new Date(s.created_at).toLocaleString() : '',
            }));
          setSubmissions(rows);
        })
        .catch(() => {});
    }
  }, [slug, problem]);

  /* save code + notes */
  useEffect(() => {
    if (slug && code) localStorage.setItem(`code_${slug}`, code);
  }, [code, slug]);
  useEffect(() => {
    if (slug) localStorage.setItem(`notes_${slug}`, notes);
  }, [notes, slug]);

  /* ── Run / Submit ─────────────────────────────────────── */
  async function handleRun(submit: boolean) {
    if (!problem) return;
    setRunState('running');
    setConsoleTab('result');
    setStdout('');
    setExpected('');
    try {
      const res = await apiPost<{
        passed: boolean; stdout: string; stderr: string;
        expected_output?: string; error?: string;
      }>(submit ? '/api/dojo/submit' : '/api/dojo/run', {
        problem_id: slug,
        code,
        test_input: submit ? problem.test_input : testInput,
      });
      if (res?.error) {
        setRunState('error');
        setStdout(res.error);
      } else {
        setRunState(res?.passed ? 'passed' : 'failed');
        setStdout(res?.stdout ?? res?.stderr ?? '');
        setExpected(res?.expected_output ?? '');
        if (submit && res?.passed) {
          setSubmissions(prev => [{ status: 'accepted', created_at: new Date().toLocaleString() }, ...prev]);
        }
      }
    } catch (e: unknown) {
      setRunState('error');
      setStdout(e instanceof Error ? e.message : String(e));
    }
  }

  if (!problem) {
    return (
      <div className="min-h-screen bg-[#0A0A0A] text-white flex flex-col items-center justify-center gap-4">
        <p className="text-[#525252]">Problem not found.</p>
        <Link href="/dojo" className="text-[#A78BFA] hover:underline text-[13px]">← Back to Dojo</Link>
      </div>
    );
  }

  /* ── Layout ───────────────────────────────────────────── */
  return (
    <div className="flex flex-col" style={{ height: 'calc(100vh - 56px)', background: '#0A0A0A', color: '#fff' }}>
      {/* TOP BAR */}
      <div style={{
        height: 44, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 16px', borderBottom: '1px solid #1A1A1A', flexShrink: 0,
      }}>
        <div className="flex items-center gap-2">
          <Link href="/dojo" style={{ color: '#525252', display: 'flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
            <ArrowLeft size={14} /> Problem List
          </Link>
          <span style={{ color: '#262626', fontSize: 12 }}>|</span>
          <div className="flex items-center gap-1">
            {prevProblem ? (
              <Link href={`/dojo/${prevProblem.slug}`} style={{ color: '#525252', padding: '2px 6px' }}>
                <ChevronLeft size={14} />
              </Link>
            ) : (
              <span style={{ color: '#2A2A2A', padding: '2px 6px' }}><ChevronLeft size={14} /></span>
            )}
            {nextProblem ? (
              <Link href={`/dojo/${nextProblem.slug}`} style={{ color: '#525252', padding: '2px 6px' }}>
                <ChevronRight size={14} />
              </Link>
            ) : (
              <span style={{ color: '#2A2A2A', padding: '2px 6px' }}><ChevronRight size={14} /></span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Timer />
          <button
            onClick={() => handleRun(false)}
            disabled={runState === 'running'}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              background: '#1A1A1A', border: '1px solid #262626',
              borderRadius: 8, padding: '5px 12px', fontSize: 12, color: '#D4D4D4', cursor: 'pointer',
            }}
          >
            {runState === 'running' ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
            Run
          </button>
          <button
            onClick={() => handleRun(true)}
            disabled={runState === 'running'}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              background: '#A78BFA', border: 'none',
              borderRadius: 8, padding: '5px 12px', fontSize: 12, fontWeight: 600, color: '#000', cursor: 'pointer',
            }}
          >
            <Send size={13} /> Submit
          </button>
        </div>
      </div>

      {/* MAIN PANELS */}
      <div className="flex flex-1 overflow-hidden">
        {/* LEFT PANEL — problem description */}
        <div style={{
          width: '38%', minWidth: 300, borderRight: '1px solid #1A1A1A',
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
          {/* tabs */}
          <div style={{
            display: 'flex', borderBottom: '1px solid #1A1A1A',
            padding: '0 16px', flexShrink: 0, gap: 2,
          }}>
            {(['description', 'submissions', 'notes'] as LeftTab[]).map(t => (
              <button
                key={t}
                onClick={() => setLeftTab(t)}
                className="capitalize transition-colors"
                style={{
                  padding: '0 12px 10px',
                  fontSize: 13,
                  color: leftTab === t ? '#A78BFA' : '#525252',
                  borderBottom: leftTab === t ? '2px solid #A78BFA' : '2px solid transparent',
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
          <div className="flex-1 overflow-y-auto" style={{ scrollbarWidth: 'thin', scrollbarColor: '#262626 transparent' }}>

            {/* DESCRIPTION */}
            {leftTab === 'description' && (
              <div className="p-6">
                <h1 style={{ fontSize: 19, fontWeight: 700, color: '#FAFAFA' }}>{problem.title}</h1>

                <div className="flex flex-wrap gap-2 mt-3">
                  {problem.topics.map(t => (
                    <span
                      key={t}
                      style={{
                        background: '#1A1A1A', color: '#60A5FA',
                        fontSize: 11, padding: '4px 8px', borderRadius: 6,
                      }}
                    >{t}</span>
                  ))}
                </div>

                <hr style={{ borderColor: '#1A1A1A', margin: '16px 0' }} />

                {children ? (
                  <div className="mdx-content text-[13px] leading-[1.85] text-[#A3A3A3]">
                    {children}
                  </div>
                ) : (
                  <p style={{ fontSize: 13, color: '#A3A3A3', lineHeight: 1.85, whiteSpace: 'pre-line' }}>
                    {problem.description}
                  </p>
                )}

                {problem.examples.map((ex, i) => (
                  <div key={i} className="mt-5">
                    <p style={{ fontSize: 11, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                      Example {i + 1}
                    </p>
                    <div
                      className="space-y-1"
                      style={{ background: '#0A0A0A', borderRadius: 12, padding: 16, fontFamily: 'JetBrains Mono, monospace', fontSize: 12 }}
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
                    marginTop: 20, background: '#0A0A0A',
                    border: '1px solid #262626', borderRadius: 12, padding: 16,
                  }}
                >
                  <p style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.12em', color: '#A78BFA', textTransform: 'uppercase', marginBottom: 12 }}>
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
                      marginTop: 16, border: '1px solid #262626',
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
                  <div style={{ marginTop: 20, border: '1px solid #262626', borderRadius: 12, padding: 16 }}>
                    <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', color: '#A78BFA', textTransform: 'uppercase', marginBottom: 12 }}>
                      Related
                    </p>
                    <div className="space-y-2">
                      {SLUG_META[slug]?.learnPath && (
                        <Link href={SLUG_META[slug].learnPath!}
                          style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#A3A3A3', textDecoration: 'none' }}
                          onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
                          onMouseLeave={e => (e.currentTarget.style.color = '#A3A3A3')}
                        >
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#60A5FA', flexShrink: 0 }} />
                          Learn: {SLUG_META[slug].learnName}
                        </Link>
                      )}
                      {SLUG_META[slug]?.archSlug && (
                        <Link href={`/architectures/${SLUG_META[slug].archSlug}`}
                          style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#A3A3A3', textDecoration: 'none' }}
                          onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
                          onMouseLeave={e => (e.currentTarget.style.color = '#A3A3A3')}
                        >
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#A78BFA', flexShrink: 0 }} />
                          Architecture: {SLUG_META[slug].archName}
                        </Link>
                      )}
                      {SLUG_META[slug]?.paperSlug && (
                        <Link href={`/papers/${SLUG_META[slug].paperSlug}`}
                          style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#A3A3A3', textDecoration: 'none' }}
                          onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
                          onMouseLeave={e => (e.currentTarget.style.color = '#A3A3A3')}
                        >
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#F97316', flexShrink: 0 }} />
                          Paper: {SLUG_META[slug].paperTitle}
                        </Link>
                      )}
                    </div>
                  </div>
                )}

                <div style={{ marginTop: 16 }}>
                  <DiffBadge d={problem.difficulty} />
                  <span style={{ marginLeft: 8, fontSize: 11, color: '#525252' }}>
                    <Clock size={11} style={{ display: 'inline', verticalAlign: 'middle' }} /> {problem.acceptance} acceptance
                  </span>
                </div>
              </div>
            )}

            {/* SUBMISSIONS */}
            {leftTab === 'submissions' && (
              <div className="p-6">
                <p style={{ fontSize: 12, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
                  Your Submissions
                </p>
                {submissions.length === 0 ? (
                  <p style={{ fontSize: 13, color: '#525252' }}>No submissions yet.</p>
                ) : (
                  <div className="space-y-2">
                    {submissions.map((s, i) => (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#111', borderRadius: 8, padding: '8px 12px' }}>
                        {s.status === 'accepted' ? (
                          <CheckCircle size={14} style={{ color: '#4ADE80', flexShrink: 0 }} />
                        ) : (
                          <XCircle size={14} style={{ color: '#F87171', flexShrink: 0 }} />
                        )}
                        <span style={{ fontSize: 12, color: s.status === 'accepted' ? '#4ADE80' : '#F87171', fontWeight: 600, textTransform: 'capitalize', flexShrink: 0 }}>
                          {s.status.replace('_', ' ')}
                        </span>
                        <span style={{ fontSize: 11, color: '#525252', marginLeft: 'auto' }}>{s.created_at}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* NOTES */}
            {leftTab === 'notes' && (
              <div className="p-6 h-full flex flex-col">
                <p style={{ fontSize: 12, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
                  Your Notes
                </p>
                <textarea
                  value={notes}
                  onChange={e => setNotes(e.target.value)}
                  placeholder="Jot down your approach, edge cases, or key insights…"
                  style={{
                    flex: 1,
                    background: '#111', border: '1px solid #262626', borderRadius: 10,
                    color: '#D4D4D4', fontSize: 13, lineHeight: 1.7,
                    padding: 12, resize: 'none', outline: 'none', fontFamily: 'inherit',
                    scrollbarWidth: 'thin', scrollbarColor: '#262626 transparent',
                  }}
                />
              </div>
            )}
          </div>
        </div>

        {/* RIGHT PANEL — editor + console */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Editor toolbar */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '0 12px', height: 36, borderBottom: '1px solid #1A1A1A', flexShrink: 0,
          }}>
            <span style={{ fontSize: 11, color: '#525252', fontFamily: 'monospace' }}>Python 3</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => { if (problem) setCode(problem.starter_code); }}
                style={{ background: 'none', border: 'none', color: '#525252', cursor: 'pointer', padding: '2px 6px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}
                title="Reset to starter"
              >
                <RotateCcw size={12} /> Reset
              </button>
              <button
                onClick={() => navigator.clipboard.writeText(code)}
                style={{ background: 'none', border: 'none', color: '#525252', cursor: 'pointer', padding: '2px 6px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}
                title="Copy code"
              >
                <Copy size={12} /> Copy
              </button>
            </div>
          </div>

          {/* Monaco editor */}
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <MonacoEditor
              language="python"
              theme="vs-dark"
              value={code}
              onChange={v => setCode(v ?? '')}
              options={{
                fontSize: 13,
                fontFamily: 'JetBrains Mono, monospace',
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                lineNumbers: 'on',
                tabSize: 4,
                wordWrap: 'on',
                automaticLayout: true,
              }}
            />
          </div>

          {/* Console panel */}
          <div style={{ height: 220, borderTop: '1px solid #1A1A1A', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
            {/* Console tabs */}
            <div style={{ display: 'flex', borderBottom: '1px solid #1A1A1A', padding: '0 12px', flexShrink: 0 }}>
              {(['testcase', 'result'] as ConsoleTab[]).map(t => (
                <button key={t} onClick={() => setConsoleTab(t)}
                  style={{
                    padding: '6px 12px 8px', fontSize: 12, background: 'none',
                    color: consoleTab === t ? '#A78BFA' : '#525252',
                    borderBottom: consoleTab === t ? '2px solid #A78BFA' : '2px solid transparent',
                    cursor: 'pointer', textTransform: 'capitalize',
                  }}>
                  {t === 'testcase' ? 'Test Case' : 'Result'}
                </button>
              ))}
            </div>

            <div style={{ flex: 1, overflow: 'auto', padding: 12, scrollbarWidth: 'thin', scrollbarColor: '#262626 transparent' }}>
              {consoleTab === 'testcase' && (
                <div>
                  <p style={{ fontSize: 10, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>stdin</p>
                  <textarea
                    value={testInput}
                    onChange={e => setTestInput(e.target.value)}
                    style={{
                      width: '100%', background: '#111', border: '1px solid #262626',
                      borderRadius: 6, color: '#D4D4D4', fontSize: 12,
                      fontFamily: 'JetBrains Mono, monospace', padding: 8, resize: 'none',
                      outline: 'none', height: 120, scrollbarWidth: 'thin',
                    }}
                  />
                </div>
              )}

              {consoleTab === 'result' && (
                <div>
                  {runState === 'idle' && (
                    <p style={{ fontSize: 13, color: '#525252' }}>Run your code to see results here.</p>
                  )}
                  {runState === 'running' && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#A78BFA' }}>
                      <Loader2 size={14} className="animate-spin" />
                      <span style={{ fontSize: 13 }}>Running…</span>
                    </div>
                  )}
                  {(runState === 'passed' || runState === 'failed' || runState === 'error') && (
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                        {runState === 'passed' && <><CheckCircle size={16} style={{ color: '#4ADE80' }} /><span style={{ fontSize: 13, fontWeight: 600, color: '#4ADE80' }}>Accepted</span></>}
                        {runState === 'failed' && <><XCircle size={16} style={{ color: '#F87171' }} /><span style={{ fontSize: 13, fontWeight: 600, color: '#F87171' }}>Wrong Answer</span></>}
                        {runState === 'error' && <><XCircle size={16} style={{ color: '#F59E0B' }} /><span style={{ fontSize: 13, fontWeight: 600, color: '#F59E0B' }}>Runtime Error</span></>}
                      </div>
                      {stdout && (
                        <div>
                          <p style={{ fontSize: 10, fontWeight: 600, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Output</p>
                          <pre style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: runState === 'passed' ? '#4ADE80' : '#F87171', whiteSpace: 'pre-wrap' }}>{stdout}</pre>
                        </div>
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
