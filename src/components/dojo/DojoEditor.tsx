'use client';

import { useRef, useState, useEffect } from 'react';
import { Spinner } from '@/components/ui/Spinner';
import dynamic from 'next/dynamic';
import type { OnMount } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';

const MonacoEditor = dynamic(() => import('@monaco-editor/react'), {
  ssr: false,
  loading: () => (
    <div
      className="w-full h-full flex items-center justify-center"
      style={{ background: 'var(--bg-editor)' }}
    >
      <span className="text-sm" style={{ color: 'rgba(255,255,255,0.3)' }}>
        Loading editor…
      </span>
    </div>
  ),
});

const MONACO_OPTIONS: editor.IStandaloneEditorConstructionOptions = {
  fontSize: 14,
  fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  lineHeight: 22,
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  automaticLayout: true,
  tabSize: 4,
  insertSpaces: true,
  wordWrap: 'on',
  folding: true,
  renderLineHighlight: 'line',
  smoothScrolling: true,
  cursorBlinking: 'smooth',
  formatOnPaste: true,
  suggestOnTriggerCharacters: true,
  acceptSuggestionOnEnter: 'smart',
  padding: { top: 12, bottom: 12 },
};

export type RunState = 'idle' | 'running' | 'done' | 'error';

interface DojoEditorProps {
  initialCode: string;
  onRun: (code: string) => void;
  onSubmit: (code: string) => void;
  runState: RunState;
  submitState: RunState;
}

export function DojoEditor({
  initialCode,
  onRun,
  onSubmit,
  runState,
  submitState,
}: DojoEditorProps) {
  const [code, setCode] = useState(initialCode);
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  
  const [editorReady, setEditorReady] = useState(false);
  const [editorFailed, setEditorFailed] = useState(false);

  // Stable ref to always call latest onRun without stale closure
  const onRunRef = useRef(onRun);
  onRunRef.current = onRun;

  useEffect(() => {
    const timer = setTimeout(() => {
      if (!editorReady) {
        setEditorFailed(true);
      }
    }, 10000);
    return () => clearTimeout(timer);
  }, [editorReady]);

  const handleEditorMount: OnMount = (ed, monaco) => {
    editorRef.current = ed;
    setEditorReady(true);
    ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
      onRunRef.current(ed.getValue());
    });
  };

  const isRunning = runState === 'running';
  const isSubmitting = submitState === 'running';

  return (
    <div className="flex flex-col h-full">
      {/* Editor Toolbar */}
      <div
        className="flex items-center justify-between px-4 py-2 border-b flex-shrink-0"
        style={{ background: 'var(--bg-editor)', borderColor: 'rgba(255,255,255,0.08)' }}
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium" style={{ color: 'rgba(255,255,255,0.5)' }}>
            Python 3
          </span>
          <span style={{ color: 'rgba(255,255,255,0.15)' }}>·</span>
          <span className="text-xs" style={{ color: 'rgba(255,255,255,0.3)' }}>
            Ctrl+Enter to run
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onRun(code)}
            disabled={isRunning || isSubmitting}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors"
            style={{
              background: isRunning
                ? 'rgba(var(--color-success-rgb), 0.15)'
                : 'rgba(var(--color-success-rgb), 0.20)',
              color: 'var(--color-success)',
              border: '1px solid rgba(var(--color-success-rgb), 0.30)',
              opacity: isRunning || isSubmitting ? 0.7 : 1,
              cursor: isRunning || isSubmitting ? 'not-allowed' : 'pointer',
            }}
          >
            {isRunning ? (
              <>
                <Spinner size={12} color="var(--color-success)" />
                Running…
              </>
            ) : (
              '▶ Run'
            )}
          </button>
          <button
            onClick={() => onSubmit(code)}
            disabled={isRunning || isSubmitting}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors"
            style={{
              background: isSubmitting
                ? 'rgba(var(--accent-transformer-rgb), 0.15)'
                : 'rgba(var(--accent-transformer-rgb), 0.20)',
              color: 'var(--accent-transformer-light)',
              border: '1px solid rgba(var(--accent-transformer-rgb), 0.30)',
              opacity: isRunning || isSubmitting ? 0.7 : 1,
              cursor: isRunning || isSubmitting ? 'not-allowed' : 'pointer',
            }}
          >
            {isSubmitting ? (
              <>
                <Spinner size={12} color="var(--accent-transformer-light)" />
                Submitting…
              </>
            ) : (
              '↑ Submit'
            )}
          </button>
        </div>
      </div>

      {/* Monaco Editor */}
      <div className="flex-1 min-h-0 flex flex-col">
        {editorFailed ? (
          <div className="flex-1 flex flex-col min-h-0">
            <div className="bg-yellow-500/20 text-yellow-200 text-xs px-4 py-2 border-b border-yellow-500/30 flex-shrink-0">
              Code editor unavailable (offline) — using a basic editor.
            </div>
            <textarea
              className="flex-1 w-full p-4 bg-[--bg-editor] text-[--color-text-primary] resize-none focus:outline-none"
              style={{
                fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
                fontSize: 14,
                lineHeight: '22px',
              }}
              value={code}
              onChange={(e) => setCode(e.target.value)}
              spellCheck={false}
              onKeyDown={(e) => {
                if (e.key === 'Tab') {
                  e.preventDefault();
                  const target = e.target as HTMLTextAreaElement;
                  const start = target.selectionStart;
                  const end = target.selectionEnd;
                  setCode(code.substring(0, start) + '    ' + code.substring(end));
                  setTimeout(() => {
                    target.selectionStart = target.selectionEnd = start + 4;
                  }, 0);
                } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  onRunRef.current(code);
                }
              }}
            />
          </div>
        ) : (
          <MonacoEditor
            height="100%"
            defaultLanguage="python"
            value={code}
            onChange={(val) => setCode(val ?? '')}
            onMount={handleEditorMount}
            theme="vs-dark"
            options={MONACO_OPTIONS}
          />
        )}
      </div>
    </div>
  );
}
