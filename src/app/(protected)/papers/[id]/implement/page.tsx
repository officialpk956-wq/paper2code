'use client';



import { use, useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import dynamic from 'next/dynamic';

import Link from 'next/link';

import { apiPost } from '@/lib/api';

import { FLAGSHIP_META } from '@/data/flagship-papers';

import { getImpl, type ImplementStep } from '@/data/implementations';

import { loader } from '@monaco-editor/react';



loader.config({ paths: { vs: '/monaco-editor/vs' } });

const MonacoEditor = dynamic(() => import('@monaco-editor/react'), { ssr: false });



type RunState = 'idle' | 'running' | 'passed' | 'failed' | 'error';



const STORAGE_KEY = (id: string) => `impl_${id}_v1`;



function loadProgress(id: string): Set<number> {

  try {

    const raw = localStorage.getItem(STORAGE_KEY(id));

    if (!raw) return new Set();

    const { completed } = JSON.parse(raw) as { completed: number[] };

    return new Set(completed);

  } catch {

    return new Set();

  }

}



function saveProgress(id: string, completed: Set<number>) {

  localStorage.setItem(STORAGE_KEY(id), JSON.stringify({ completed: [...completed] }));

}



function PassCelebration() {
  const dots = [
    { x: -28, y: -24, color: '#4ADE80' },
    { x:  28, y: -24, color: '#A78BFA' },
    { x: -32, y:   8, color: '#00E5FF' },
    { x:  32, y:   8, color: '#FACC15' },
    { x:   0, y: -32, color: '#4ADE80' },
    { x:   0, y:  20, color: '#A78BFA' },
  ]
  return (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
      {dots.map((d, i) => (
        <motion.div
          key={i}
          className="absolute w-2 h-2 rounded-full"
          style={{ backgroundColor: d.color }}
          initial={{ x: 0, y: 0, opacity: 1, scale: 0 }}
          animate={{ x: d.x, y: d.y, opacity: 0, scale: 1 }}
          transition={{ duration: 0.55, ease: 'easeOut', delay: i * 0.04 }}
        />
      ))}
    </div>
  )
}

function LoadingDots() {
  return (
    <span className="inline-flex items-center gap-[3px]">
      {[0, 0.12, 0.24].map((delay, i) => (
        <motion.span
          key={i}
          className="block w-[4px] h-[4px] rounded-full bg-current"
          animate={{ y: [0, -4, 0] }}
          transition={{ duration: 0.5, repeat: Infinity, ease: 'easeInOut', delay }}
        />
      ))}
    </span>
  )
}

export default function GuidedImplementPage({ params }: { params: Promise<{ id: string }> }) {

  const { id } = use(params);

  const impl = getImpl(id);

  const paperMeta = FLAGSHIP_META[id];



  const [stepIdx, setStepIdx] = useState(0);

  const [code, setCode] = useState('');

  const [runState, setRunState] = useState<RunState>('idle');

  const [stdout, setStdout] = useState('');

  const [completed, setCompleted] = useState<Set<number>>(new Set());

  const [showHint, setShowHint] = useState(-1);

  const [editorCodes, setEditorCodes] = useState<Record<number, string>>({});



  useEffect(() => {

    setCompleted(loadProgress(id));

  }, [id]);



  const currentStep: ImplementStep | undefined = impl?.steps[stepIdx];



  useEffect(() => {

    if (!currentStep) return;

    setCode(editorCodes[stepIdx] ?? currentStep.starterCode);

    setRunState('idle');

    setStdout('');

    setShowHint(-1);

  }, [stepIdx, currentStep]);



  const handleRun = useCallback(async () => {

    if (!currentStep || runState === 'running') return;

    const fullCode = code + '\n\n' + currentStep.testCode;

    setRunState('running');

    setStdout('');

    try {

      const result = await apiPost<{ passed: boolean; stdout: string; stderr: string }>(

        '/api/dojo/execute',

        { code: fullCode, stdin: '' }

      );

      const out = result.stdout || result.stderr || '';

      setStdout(out);

      if (out.includes('ALL_TESTS_PASSED')) {

        setRunState('passed');

        setCompleted(prev => {

          const next = new Set(prev).add(stepIdx);

          saveProgress(id, next);

          return next;

        });

      } else {

        setRunState('failed');

      }

    } catch (err: unknown) {

      setStdout((err as Error).message || 'Execution failed');

      setRunState('error');

    }

  }, [code, currentStep, runState, stepIdx, id]);



  useEffect(() => {

    const handler = (e: KeyboardEvent) => {

      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {

        e.preventDefault();

        handleRun();

      }

    };

    window.addEventListener('keydown', handler);

    return () => window.removeEventListener('keydown', handler);

  }, [handleRun]);



  if (!impl) {

    return (

      <div className="flex items-center justify-center h-screen bg-[#0A0A0A] text-white">

        <div className="text-center">

          <div className="text-[#525252] text-sm mb-4">No guided implementation available for this paper yet.</div>

          <Link href={`/papers/${id}`} className="text-[#A78BFA] text-sm hover:underline">← Back to Paper</Link>

        </div>

      </div>

    );

  }



  const paperTitle = paperMeta?.title ?? impl.paperTitle;

  const accentColor = paperMeta?.color ?? '#A78BFA';

  const isUnlocked = (idx: number) => idx === 0 || completed.has(idx - 1);



  const statusIcon = (idx: number) => {

    if (completed.has(idx)) return '✓';

    if (idx === stepIdx) return '→';

    if (!isUnlocked(idx)) return '🔒';

    return String(idx + 1);

  };



  const consoleColor: Record<RunState, string> = {

    idle: '#525252',

    running: '#FACC15',

    passed: '#4ADE80',

    failed: '#F87171',

    error: '#F87171',

  };



  return (

    <div className="flex flex-col bg-[#0A0A0A] text-white" style={{ height: '100vh' }}>

      {/* Top bar */}

      <div className="flex-shrink-0 border-b border-[#1A1A1A] bg-[#0A0A0A] px-6 py-3 flex items-center justify-between">

        <div className="flex items-center gap-4">

          <Link href={`/papers/${id}`} className="text-[11px] text-[#525252] hover:text-white">← {paperTitle}</Link>

          <div className="h-0.5 w-6 rounded-full" style={{ backgroundColor: accentColor }} />

          <span className="text-[13px] font-semibold text-white">Guided Implementation</span>

        </div>

        <div className="text-[11px] text-[#525252]">

          {completed.size}/{impl.totalSteps} steps complete

        </div>

      </div>



      {/* Body: 3 columns */}

      <div className="flex flex-1 overflow-hidden">

        {/* Column 1: Step sidebar */}

        <div className="w-[200px] flex-shrink-0 border-r border-[#1A1A1A] overflow-y-auto py-4">

          {impl.steps.map((step, idx) => {

            const unlocked = isUnlocked(idx);

            const active = idx === stepIdx;

            const done = completed.has(idx);

            return (

              <button

                key={step.id}

                disabled={!unlocked}

                onClick={() => {

                  if (unlocked) {

                    setEditorCodes(prev => ({ ...prev, [stepIdx]: code }));

                    setStepIdx(idx);

                  }

                }}

                className={

                  'w-full text-left px-4 py-3 text-[12px] border-l-2 transition-colors ' +

                  (active

                    ? 'border-l-[#A78BFA] bg-[#1A1A1A] text-white'

                    : done

                    ? 'border-l-[#4ADE80] text-[#4ADE80] hover:bg-[#111111]'

                    : unlocked

                    ? 'border-l-transparent text-[#A3A3A3] hover:bg-[#111111] hover:text-white'

                    : 'border-l-transparent text-[#3A3A3A] cursor-not-allowed')

                }

              >

                <span className="mr-2 text-[10px]">{statusIcon(idx)}</span>

                {step.title}

              </button>

            );

          })}

        </div>



        {/* Column 2: Editor + console */}

        {currentStep && (

          <div className="flex flex-col flex-1 overflow-hidden">

            <div className="flex-1 overflow-hidden">

              <MonacoEditor

                height="100%"

                language="python"

                theme="vs-dark"

                value={code}

                onChange={v => setCode(v ?? '')}

                options={{

                  fontSize: 13,

                  minimap: { enabled: false },

                  scrollBeyondLastLine: false,

                  fontFamily: 'JetBrains Mono, monospace',

                }}

              />

            </div>



            {/* Console */}

            <div className="flex-shrink-0 border-t border-[#1A1A1A] bg-[#0A0A0A]" style={{ height: 160 }}>

              <div className="relative flex items-center justify-between px-4 py-2 border-b border-[#1A1A1A]">
                <AnimatePresence>
                  {runState === 'passed' && <PassCelebration key="celebrate" />}
                </AnimatePresence>

                <span className="text-[11px] font-semibold" style={{ color: consoleColor[runState] }}>

                  {runState === 'idle' ? 'Console' :

                   runState === 'running' ? 'Running...' :

                   runState === 'passed' ? (
                     <motion.span
                       animate={{ scale: [1, 1.1, 1] }}
                       transition={{ duration: 0.4, delay: 0.1 }}
                       style={{ display: 'inline-block' }}
                     >
                       All tests passed!
                     </motion.span>
                   ) :

                   runState === 'failed' ? 'Tests failed' : 'Error'}

                </span>

                <div className="flex items-center gap-2">

                  {runState === 'passed' && stepIdx < impl.steps.length - 1 && (

                    <button

                      onClick={() => {

                        setEditorCodes(prev => ({ ...prev, [stepIdx]: code }));

                        setStepIdx(stepIdx + 1);

                      }}

                      className="text-[12px] px-3 py-1 rounded-lg bg-[#4ADE80] text-black font-semibold hover:brightness-110"

                    >

                      Next Step →

                    </button>

                  )}

                  {runState === 'passed' && stepIdx === impl.steps.length - 1 && (

                    <span className="text-[12px] text-[#4ADE80] font-semibold">Implementation complete!</span>

                  )}

                  <button

                    onClick={handleRun}

                    disabled={runState === 'running'}

                    className="text-[12px] px-4 py-1 rounded-lg bg-[#A78BFA] text-black font-semibold hover:brightness-110 disabled:opacity-50"

                  >

                    {runState === 'running' ? <LoadingDots /> : 'Run ⌘↵'}

                  </button>

                </div>

              </div>

              <pre className="px-4 py-2 text-[12px] font-mono text-[#A3A3A3] overflow-auto h-[112px] whitespace-pre-wrap">

                {stdout || (runState === 'idle' ? 'Press Run or Cmd+Enter to execute' : '')}

              </pre>

            </div>

          </div>

        )}



        {/* Column 3: Description + hints */}

        {currentStep && (

          <div className="w-[280px] flex-shrink-0 border-l border-[#1A1A1A] overflow-y-auto p-5 space-y-5">

            <div>

              <div className="text-[10px] uppercase font-bold tracking-wider mb-2" style={{ color: accentColor }}>

                Step {stepIdx + 1} of {impl.totalSteps}

              </div>

              <div className="text-[14px] font-bold text-white mb-3">{currentStep.title}</div>

              <p className="text-[12px] leading-relaxed text-[#A3A3A3]">{currentStep.description}</p>

            </div>



            {currentStep.concepts.length > 0 && (

              <div>

                <div className="text-[10px] uppercase font-bold text-[#525252] tracking-wider mb-2">Concepts</div>

                <div className="flex flex-wrap gap-1">

                  {currentStep.concepts.map(c => (

                    <span key={c} className="text-[11px] px-2 py-0.5 rounded-full bg-[#1A1A1A] border border-[#262626] text-[#A3A3A3]">

                      {c}

                    </span>

                  ))}

                </div>

              </div>

            )}



            <div>

              <div className="text-[10px] uppercase font-bold text-[#525252] tracking-wider mb-2">Hints</div>

              <div className="space-y-2">

                {currentStep.hints.map((hint, i) => (

                  <div key={i}>

                    {showHint >= i ? (

                      <div className="text-[12px] text-[#A3A3A3] bg-[#1A1A1A] rounded-lg px-3 py-2">{hint}</div>

                    ) : (

                      <button

                        onClick={() => setShowHint(i)}

                        className="text-[12px] text-[#525252] hover:text-[#A78BFA] underline"

                      >

                        Reveal hint {i + 1}

                      </button>

                    )}

                  </div>

                ))}

              </div>

            </div>



            <div className="pt-4 border-t border-[#1A1A1A]">

              <Link href={`/papers/${id}`} className="text-[12px] text-[#525252] hover:text-white">

                ← Back to {paperTitle}

              </Link>

            </div>

          </div>

        )}

      </div>

    </div>

  );

}