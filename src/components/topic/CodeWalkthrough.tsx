'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Copy, Check, ChevronRight } from 'lucide-react';
import type { CodeSnippet, CodeLanguage } from '@/data/topics/types';

interface CodeWalkthroughProps {
  snippets: CodeSnippet[];
}

const LANGUAGE_COLORS: Record<CodeLanguage, string> = {
  pytorch:     '#EE4C2C',
  tensorflow:  '#FF6F00',
  pseudocode:  '#7C3AED',
};

const KW_COLOR = '#A78BFA';
const STR_COLOR = '#86EFAC';
const NUM_COLOR = '#FCD34D';
const COMMENT_COLOR = '#475569';
const FUNC_COLOR = '#67E8F9';
const BUILTIN_COLOR = '#F9A8D4';

const KEYWORDS = new Set([
  'def','class','import','from','return','if','else','elif','for','while',
  'in','not','and','or','True','False','None','self','super','with','as',
  'lambda','yield','pass','break','continue','raise','try','except',
  'finally','assert','global','nonlocal',
]);

const BUILTINS = new Set(['print','len','range','int','str','float','list','dict','set','tuple','bool','type','isinstance']);

function tokenizeLine(line: string): React.ReactNode {
  // Simple state-machine: handles strings, comments, keywords, numbers, func calls
  const parts: React.ReactNode[] = [];
  let i = 0;
  const n = line.length;

  while (i < n) {
    // Comment
    if (line[i] === '#') {
      parts.push(<span key={i} style={{ color: COMMENT_COLOR }}>{line.slice(i)}</span>);
      break;
    }

    // String: ", ', """, '''
    const tripleQ = line.slice(i, i + 3);
    if (tripleQ === '"""' || tripleQ === "'''") {
      const delim = tripleQ;
      let j = i + 3;
      while (j < n - 2 && line.slice(j, j + 3) !== delim) j++;
      const end = j + 3;
      parts.push(<span key={i} style={{ color: STR_COLOR }}>{line.slice(i, end)}</span>);
      i = end;
      continue;
    }

    if (line[i] === '"' || line[i] === "'") {
      const q = line[i];
      let j = i + 1;
      while (j < n && line[j] !== q) {
        if (line[j] === '\\') j++;
        j++;
      }
      const end = j + 1;
      parts.push(<span key={i} style={{ color: STR_COLOR }}>{line.slice(i, end)}</span>);
      i = end;
      continue;
    }

    // Number
    if (/\d/.test(line[i]) && (i === 0 || /\W/.test(line[i - 1]))) {
      let j = i;
      while (j < n && /[\d._]/.test(line[j])) j++;
      parts.push(<span key={i} style={{ color: NUM_COLOR }}>{line.slice(i, j)}</span>);
      i = j;
      continue;
    }

    // Word (keyword / builtin / function call / identifier)
    if (/[a-zA-Z_]/.test(line[i])) {
      let j = i;
      while (j < n && /\w/.test(line[j])) j++;
      const word = line.slice(i, j);
      // Check if followed by (
      const afterWord = line.slice(j).trimStart();
      if (KEYWORDS.has(word)) {
        parts.push(<span key={i} style={{ color: KW_COLOR }}>{word}</span>);
      } else if (BUILTINS.has(word)) {
        parts.push(<span key={i} style={{ color: BUILTIN_COLOR }}>{word}</span>);
      } else if (afterWord.startsWith('(')) {
        parts.push(<span key={i} style={{ color: FUNC_COLOR }}>{word}</span>);
      } else {
        parts.push(<span key={i} style={{ color: 'var(--color-text-primary)' }}>{word}</span>);
      }
      i = j;
      continue;
    }

    // Everything else
    parts.push(<span key={i} style={{ color: 'var(--color-text-secondary)' }}>{line[i]}</span>);
    i++;
  }

  return <>{parts}</>;
}

function HighlightedCode({ code, activeLines }: { code: string; activeLines: Set<number> }) {
  const lines = code.split('\n');

  return (
    <code style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
      {lines.map((line, i) => {
        const lineNum = i + 1;
        const isActive = activeLines.has(lineNum);

        return (
          <div
            key={i}
            className="flex items-start"
            style={{
              background: isActive ? 'rgba(124,58,237,0.1)' : 'transparent',
              borderLeft: `2px solid ${isActive ? 'var(--accent-primary)' : 'transparent'}`,
            }}
          >
            <span
              className="flex-none w-8 text-right pr-3 select-none text-[11px] pt-[1px]"
              style={{ color: isActive ? 'var(--accent-light)' : 'var(--color-text-muted)' }}
              aria-hidden="true"
            >
              {lineNum}
            </span>
            <span className="flex-1 whitespace-pre-wrap break-all">{tokenizeLine(line)}</span>
          </div>
        );
      })}
    </code>
  );
}

function parseLineRange(range: string): number[] {
  const result: number[] = [];
  for (const part of range.split(',')) {
    const dash = part.indexOf('-');
    if (dash >= 0) {
      const from = parseInt(part.slice(0, dash).trim(), 10);
      const to = parseInt(part.slice(dash + 1).trim(), 10);
      for (let i = from; i <= to; i++) result.push(i);
    } else {
      const n = parseInt(part.trim(), 10);
      if (!isNaN(n)) result.push(n);
    }
  }
  return result;
}

export function CodeWalkthrough({ snippets }: CodeWalkthroughProps) {
  const [activeTab, setActiveTab] = useState<CodeLanguage>(snippets[0]?.language ?? 'pytorch');
  const [activeLine, setActiveLine] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);

  const current = snippets.find(s => s.language === activeTab) ?? snippets[0];
  if (!current) return null;

  const activeLines = activeLine !== null
    ? new Set(
        current.lines
          .filter(l => parseLineRange(l.lineNumbers).includes(activeLine))
          .flatMap(l => parseLineRange(l.lineNumbers))
      )
    : new Set<number>();

  const handleCopy = () => {
    navigator.clipboard.writeText(current.code).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="code" aria-labelledby="code-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 id="code-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Code Walkthrough
        </h2>
        <p className="text-xs mb-5" style={{ color: 'var(--color-text-secondary)' }}>
          Click any explanation to highlight the corresponding lines.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.45, delay: 0.1 }}
        className="rounded-2xl overflow-hidden"
        style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.07)' }}
      >
        {/* Tab bar */}
        <div
          className="flex items-center gap-0.5 px-4 pt-3 pb-0"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
          role="tablist"
          aria-label="Code language"
        >
          {snippets.map(s => {
            const isActive = s.language === activeTab;
            return (
              <button
                key={s.language}
                role="tab"
                aria-selected={isActive}
                aria-controls={`tab-panel-${s.language}`}
                id={`tab-${s.language}`}
                onClick={() => { setActiveTab(s.language); setActiveLine(null); }}
                className="px-4 py-2 text-xs font-semibold rounded-t-lg transition-all focus-visible:outline-none focus-visible:ring-inset focus-visible:ring-2"
                style={{
                  background: isActive ? 'var(--bg-hover)' : 'transparent',
                  borderBottom: `2px solid ${isActive ? LANGUAGE_COLORS[s.language] : 'transparent'}`,
                  color: isActive ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                  marginBottom: '-1px',
                }}
              >
                {s.label}
              </button>
            );
          })}
          <div className="flex-1" />
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1.5 mb-2 rounded-lg text-xs transition-all focus-visible:outline-none focus-visible:ring-2"
            style={{
              background: copied ? 'rgba(16,185,129,0.12)' : 'rgba(255,255,255,0.05)',
              color: copied ? '#10B981' : 'var(--color-text-muted)',
            }}
            aria-label="Copy code to clipboard"
          >
            {copied ? <Check size={11} aria-hidden="true" /> : <Copy size={11} aria-hidden="true" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>

        {/* Code + annotations */}
        <div className="flex flex-col lg:flex-row">
          {/* Code block */}
          <div
            className="flex-1 overflow-x-auto p-4"
            id={`tab-panel-${current.language}`}
            role="tabpanel"
            aria-labelledby={`tab-${current.language}`}
          >
            <pre
              className="text-xs leading-6"
              style={{ background: 'transparent', margin: 0 }}
              aria-label={`${current.label} code example`}
            >
              <HighlightedCode code={current.code} activeLines={activeLines} />
            </pre>
          </div>

          {/* Line annotations */}
          <div
            className="lg:w-72 flex-none border-t lg:border-t-0 lg:border-l p-4"
            style={{ borderColor: 'rgba(255,255,255,0.06)' }}
          >
            <div className="text-[10px] font-bold uppercase tracking-wider mb-3" style={{ color: 'var(--color-text-muted)' }}>
              Line Explanations
            </div>
            <div className="flex flex-col gap-1.5" role="list" aria-label="Code line explanations">
              {current.lines.map((line, i) => {
                const lineNums = parseLineRange(line.lineNumbers);
                const isActive = activeLine !== null && lineNums.includes(activeLine);

                return (
                  <button
                    key={i}
                    role="listitem"
                    onClick={() => setActiveLine(lineNums[0] ?? null)}
                    onMouseEnter={() => setActiveLine(lineNums[0] ?? null)}
                    onMouseLeave={() => setActiveLine(null)}
                    className="flex items-start gap-2 text-left p-2 rounded-lg transition-all focus-visible:outline-none focus-visible:ring-1"
                    style={{
                      background: isActive ? 'rgba(124,58,237,0.1)' : 'transparent',
                      border: `1px solid ${isActive ? 'rgba(124,58,237,0.3)' : 'transparent'}`,
                    }}
                    aria-label={`Lines ${line.lineNumbers}: ${line.explanation}`}
                  >
                    <code
                      className="flex-none text-[10px] font-bold px-1.5 py-0.5 rounded"
                      style={{
                        background: isActive ? 'rgba(124,58,237,0.2)' : 'rgba(255,255,255,0.06)',
                        color: isActive ? 'var(--accent-light)' : 'var(--color-text-muted)',
                        fontFamily: 'var(--font-mono)',
                      }}
                    >
                      L{line.lineNumbers}
                    </code>
                    <span className="text-[11px] leading-relaxed" style={{ color: isActive ? 'var(--color-text-primary)' : 'var(--color-text-secondary)' }}>
                      {line.explanation}
                    </span>
                    {isActive && <ChevronRight size={10} style={{ color: 'var(--accent-primary)', flexShrink: 0, marginTop: 2 }} aria-hidden="true" />}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  );
}
