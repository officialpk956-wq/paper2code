"use client";

import { TensorStep } from "@/data/tensor-traces";
import { Info, Code2, Link as LinkIcon, ExternalLink } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";
import remarkRehype from "remark-rehype";
import "katex/dist/katex.min.css"; // Ensure KaTeX styles are imported globally or here

interface Props {
  step: TensorStep;
}

export function OperationExplainer({ step }: Props) {
  const [activeTab, setActiveTab] = useState<"intuition" | "code">("intuition");
  const [renderedEquation, setRenderedEquation] = useState<string | null>(null);

  // Render LaTeX equation safely using unified + KaTeX
  useEffect(() => {
    if (step.equation) {
      unified()
        .use(remarkParse)
        .use(remarkMath)
        .use(remarkRehype)
        .use(rehypeKatex)
        .use(rehypeStringify)
        .process(`$$${step.equation}$$`)
        .then((file: unknown) => setRenderedEquation(String(file)))
        .catch(() => setRenderedEquation(null));
    } else {
      setRenderedEquation(null);
    }
  }, [step.equation]);

  // Reset tab on step change
  useEffect(() => {
    setActiveTab("intuition");
  }, [step.id]);

  return (
    <div className="bg-[--bg-surface] rounded-xl border border-[--color-border] overflow-hidden flex flex-col h-full animate-in fade-in slide-in-from-bottom-6 duration-500 delay-100">
      
      {/* Tabs */}
      <div className="flex border-b border-[--color-border] bg-[--bg-panel]">
        <button
          onClick={() => setActiveTab("intuition")}
          className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-semibold transition-colors ${
            activeTab === "intuition" 
              ? "text-[--accent-primary] border-b-2 border-[--accent-primary] bg-[--bg-surface]" 
              : "text-[--color-text-secondary] hover:bg-[--bg-surface] hover:text-[--color-text-primary]"
          }`}
        >
          <Info className="w-4 h-4" /> Intuition
        </button>
        {step.codeSnippet && (
          <button
            onClick={() => setActiveTab("code")}
            className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-semibold transition-colors ${
              activeTab === "code" 
                ? "text-[--accent-primary] border-b-2 border-[--accent-primary] bg-[--bg-surface]" 
                : "text-[--color-text-secondary] hover:bg-[--bg-surface] hover:text-[--color-text-primary]"
            }`}
          >
            <Code2 className="w-4 h-4" /> Implementation
          </button>
        )}
      </div>

      {/* Content */}
      <div className="p-6 flex-1 overflow-y-auto">
        {activeTab === "intuition" && (
          <div className="space-y-6">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-[--color-text-tertiary] mb-2">What happened?</h3>
              <p className="text-[--color-text-primary] leading-relaxed">{step.what}</p>
            </div>
            
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-[--color-text-tertiary] mb-2">Why?</h3>
              <p className="text-[--color-text-secondary] leading-relaxed bg-[--bg-panel] p-4 rounded-lg border border-[--color-border] border-l-4 border-l-[--accent-cyan]">
                {step.why}
              </p>
            </div>

            {renderedEquation && (
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-[--color-text-tertiary] mb-2">Mathematics</h3>
                <div 
                  className="bg-[#0A0A12] py-4 px-6 rounded-lg overflow-x-auto text-[--accent-light] border border-[--color-border] shadow-inner"
                  dangerouslySetInnerHTML={{ __html: renderedEquation }}
                />
              </div>
            )}
          </div>
        )}

        {activeTab === "code" && step.codeSnippet && (
          <div className="h-full">
            <pre className="bg-[#0A0A12] p-4 rounded-lg overflow-x-auto border border-[--color-border] text-sm text-[--accent-light] font-mono leading-loose">
              <code>{step.codeSnippet}</code>
            </pre>
          </div>
        )}
      </div>

      {/* Links Footer */}
      {(step.implementationLink || (step.mathLinks && step.mathLinks.length > 0)) && (
        <div className="bg-[--bg-panel] border-t border-[--color-border] p-4 px-6 flex flex-wrap gap-4 items-center">
          {step.implementationLink && (
            <Link 
              href={`/paper-to-code/${step.implementationLink.slug}`}
              className="inline-flex items-center gap-2 text-xs font-semibold text-[--accent-cyan] hover:underline"
            >
              <Code2 className="w-3.5 h-3.5" />
              Paper → Code: {step.implementationLink.milestone} <ExternalLink className="w-3 h-3" />
            </Link>
          )}
          
          {step.mathLinks?.map((link, idx) => (
            <Link 
              key={idx}
              href={link.href}
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-orange-400 hover:underline"
            >
              <LinkIcon className="w-3 h-3" />
              Math: {link.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
