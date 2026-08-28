'use client';

import Link from 'next/link';
import { useState } from 'react';
import type { Methodology } from '@/lib/mdx';

type MethodologyTrackProps = {
  methodology: Methodology;
};

export default function MethodologyTrack({ methodology }: MethodologyTrackProps) {
  const [revealedAnswers, setRevealedAnswers] = useState<Set<number>>(() => new Set());

  const toggleAnswer = (partNumber: number) => {
    setRevealedAnswers((current) => {
      const next = new Set(current);
      if (next.has(partNumber)) next.delete(partNumber);
      else next.add(partNumber);
      return next;
    });
  };

  return (
    <section className="rounded-2xl border border-[#262626] bg-[#0A0A0A] p-6 sm:p-8">
      <div className="mb-7">
        <div className="mb-2 text-[11px] font-bold uppercase tracking-[0.16em] text-[#A78BFA]">
          Reconstruct This Paper
        </div>
        <h2 className="text-2xl font-semibold text-white">{methodology.title}</h2>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-[#A3A3A3]">{methodology.summary}</p>
      </div>

      <ol className="space-y-4">
        {methodology.parts.map((part) => {
          const answerVisible = part.kind === 'concept' && revealedAnswers.has(part.n);
          return (
            <li key={part.n} className="rounded-xl border border-[#262626] bg-[#111111] p-5">
              <div className="flex items-start gap-4">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#A78BFA]/15 text-sm font-bold text-[#C4B5FD]">
                  {part.n}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="mb-1 text-[10px] font-bold uppercase tracking-[0.14em] text-[#737373]">
                    {part.kind === 'concept' ? 'Concept check' : 'Graded implementation'}
                  </div>
                  <h3 className="text-base font-semibold text-white">{part.title}</h3>
                  <p className="mt-2 text-sm leading-6 text-[#A3A3A3]">{part.prompt}</p>

                  {part.kind === 'concept' ? (
                    <div className="mt-4 space-y-3">
                      <button
                        type="button"
                        aria-expanded={answerVisible}
                        onClick={() => toggleAnswer(part.n)}
                        className="rounded-lg border border-[#3A3154] px-3 py-2 text-xs font-semibold text-[#C4B5FD] transition-colors hover:border-[#A78BFA] hover:bg-[#A78BFA]/10"
                      >
                        {answerVisible ? 'Hide answer' : 'Show answer'}
                      </button>
                      {answerVisible && (
                        <div className="rounded-lg border border-[#2F2942] bg-[#A78BFA]/5 p-4 text-sm leading-6 text-[#D4D4D4]">
                          {part.answer}
                        </div>
                      )}
                      <details className="group rounded-lg border border-[#262626] bg-[#0A0A0A] px-4 py-3">
                        <summary className="cursor-pointer text-xs font-semibold text-[#A3A3A3] transition-colors hover:text-white">
                          Hints ({part.hints.length})
                        </summary>
                        <ul className="mt-3 list-disc space-y-2 pl-5 text-xs leading-5 text-[#737373]">
                          {part.hints.map((hint) => <li key={hint}>{hint}</li>)}
                        </ul>
                      </details>
                    </div>
                  ) : (
                    <Link
                      href={`/dojo/${part.dojoSlug}`}
                      className="mt-4 inline-flex rounded-lg bg-[#A78BFA] px-4 py-2 text-sm font-semibold text-black transition-colors hover:bg-[#C4B5FD]"
                    >
                      Solve in Dojo -&gt;
                    </Link>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
