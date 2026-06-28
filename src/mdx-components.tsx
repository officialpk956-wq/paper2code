import type { MDXComponents } from 'mdx/types';
import { Callout } from '@/components/content/Callout';
import { Quiz } from '@/components/content/Quiz';
import { SimpleChart } from '@/components/content/SimpleChart';
import { SVGDiagram } from '@/components/content/SVGDiagram';
import { YouTubeEmbed } from '@/components/content/YouTubeEmbed';
import { ObservableEmbed } from '@/components/content/ObservableEmbed';
import { LottieEmbed } from '@/components/content/LottieEmbed';

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    Callout,
    Quiz,
    SimpleChart,
    SVGDiagram,
    YouTubeEmbed,
    ObservableEmbed,
    LottieEmbed,
    h1: ({ children }) => (
      <h1 className="text-3xl font-black mb-6 mt-2 text-white">{children}</h1>
    ),
    h2: ({ children }) => (
      <h2 className="text-xl font-bold mb-4 mt-10 text-white border-b border-white/10 pb-2">
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className="text-lg font-semibold mb-3 mt-6 text-white/90">{children}</h3>
    ),
    p: ({ children }) => (
      <p className="text-white/75 leading-7 mb-4">{children}</p>
    ),
    ul: ({ children }) => (
      <ul className="list-disc list-inside text-white/75 mb-4 space-y-1 pl-2">{children}</ul>
    ),
    ol: ({ children }) => (
      <ol className="list-decimal list-inside text-white/75 mb-4 space-y-1 pl-2">{children}</ol>
    ),
    li: ({ children }) => <li className="leading-7">{children}</li>,
    strong: ({ children }) => (
      <strong className="font-semibold text-white">{children}</strong>
    ),
    code: ({ children }) => (
      <code className="bg-white/10 text-purple-300 px-1.5 py-0.5 rounded text-sm font-mono">
        {children}
      </code>
    ),
    pre: ({ children }) => (
      <pre className="bg-[#0d1117] border border-white/10 rounded-xl p-4 overflow-x-auto text-sm mb-6">
        {children}
      </pre>
    ),
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-purple-500 pl-4 italic text-white/60 my-4">
        {children}
      </blockquote>
    ),
    ...components,
  };
}
