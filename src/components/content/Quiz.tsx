'use client';
import React, { useState } from 'react';

export function Quiz({ question, options, answer }: { question: string; options: string[]; answer: number }) {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="p-6 rounded-xl border border-white/10 bg-white/5 my-6">
      <div className="font-bold text-lg mb-4">{question}</div>
      <div className="space-y-2">
        {options.map((opt, i) => {
          const isSelected = selected === i;
          const isWrong = isSelected && i !== answer;

          let btnClass = "w-full text-left p-3 rounded-lg border transition-colors ";
          if (selected === null) {
            btnClass += "border-white/10 hover:border-purple-500 hover:bg-white/5";
          } else if (i === answer) {
            btnClass += "border-green-500 bg-green-500/20 text-green-200";
          } else if (isWrong) {
            btnClass += "border-red-500 bg-red-500/20 text-red-200";
          } else {
            btnClass += "border-white/5 opacity-50";
          }

          return (
            <button
              key={i}
              className={btnClass}
              onClick={() => selected === null && setSelected(i)}
              disabled={selected !== null}
            >
              {opt}
            </button>
          );
        })}
      </div>
    </div>
  );
}
