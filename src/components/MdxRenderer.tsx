import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import Link from 'next/link';
import React from 'react';
import { FadeIn } from './FadeIn';

function extractText(node: any): string {
  if (!node) return '';
  if (typeof node === 'string') return node;
  if (typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(extractText).join('');
  if (node.props && node.props.children) return extractText(node.props.children);
  return '';
}

function slugify(node: any): string {
  const text = extractText(node);
  return text
    .toLowerCase()
    .trim()
    .replace(/\s+/g, '-')
    .replace(/[^\w\-]+/g, '')
    .replace(/\-\-+/g, '-');
}

const customComponents = {
  h2: ({ children, ...props }: any) => (
    <h2
      id={slugify(children)}
      className="text-[20px] font-bold text-white mt-10 mb-4 border-b border-[#262626] pb-2"
      {...props}
    >
      {children}
    </h2>
  ),
  h3: ({ children, ...props }: any) => (
    <h3 className="text-[16px] font-semibold text-[#A3A3A3] mt-6 mb-2" {...props}>
      {children}
    </h3>
  ),
  p: ({ children, ...props }: any) => (
    <p className="text-[14px] leading-[1.9] text-[#A3A3A3] mb-4" {...props}>
      {children}
    </p>
  ),
  ul: ({ children, ...props }: any) => (
    <ul className="list-disc text-[14px] text-[#A3A3A3] leading-[1.8] space-y-1.5 pl-5 mb-4" {...props}>
      {children}
    </ul>
  ),
  ol: ({ children, ...props }: any) => (
    <ol className="list-decimal text-[14px] text-[#A3A3A3] leading-[1.8] space-y-1.5 pl-5 mb-4" {...props}>
      {children}
    </ol>
  ),
  li: ({ children, ...props }: any) => (
    <li className="text-[#A3A3A3]" {...props}>
      {children}
    </li>
  ),
  code: ({ children, className, ...props }: any) => {
    const isInline = !className;
    if (isInline) {
      return (
        <code className="bg-[#1A1A1A] px-1.5 py-0.5 rounded text-[13px] font-mono text-[#A78BFA]" {...props}>
          {children}
        </code>
      );
    }
    return (
      <code className={className} {...props}>
        {children}
      </code>
    );
  },
  pre: ({ children, ...props }: any) => (
    <pre className="bg-[#0A0A0A] border border-[#262626] rounded-xl p-5 overflow-x-auto my-6 text-[13px] font-mono text-[#E2E8F0]" {...props}>
      {children}
    </pre>
  ),
  table: ({ children, ...props }: any) => (
    <div className="overflow-x-auto my-6 w-full">
      <table className="w-full border-collapse" {...props}>
        {children}
      </table>
    </div>
  ),
  th: ({ children, ...props }: any) => (
    <th className="bg-[#111111] text-white text-[12px] uppercase tracking-wider px-4 py-2 border-b border-[#262626] text-left" {...props}>
      {children}
    </th>
  ),
  td: ({ children, ...props }: any) => (
    <td className="px-4 py-2 text-[13px] text-[#A3A3A3] border-b border-[#262626]/50 text-left" {...props}>
      {children}
    </td>
  ),
  blockquote: ({ children, ...props }: any) => (
    <blockquote className="bg-[#A78BFA]/5 border-l-4 border-[#A78BFA] pl-4 py-2 rounded-r-lg my-4 text-[13px] text-[#C4B5FD]" {...props}>
      {children}
    </blockquote>
  ),
  strong: ({ children, ...props }: any) => (
    <strong className="text-white font-semibold" {...props}>
      {children}
    </strong>
  ),
  a: ({ children, href, ...props }: any) => {
    if (href && href.startsWith('/')) {
      return (
        <Link href={href} className="text-[#A78BFA] hover:underline" {...props}>
          {children}
        </Link>
      );
    }
    return (
      <a href={href} className="text-[#A78BFA] hover:underline" target="_blank" rel="noopener noreferrer" {...props}>
        {children}
      </a>
    );
  },
};

function preprocessAdmonitions(source: string): string {
  const lines = source.split('\n');
  const result: string[] = [];
  let inAdmonition = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith(':::') && trimmed !== ':::') {
      inAdmonition = true;
      continue;
    } else if (trimmed === ':::') {
      inAdmonition = false;
      continue;
    }

    if (inAdmonition) {
      result.push('> ' + line);
    } else {
      result.push(line);
    }
  }

  return result.join('\n');
}

function splitMdxIntoSections(source: string): string[] {
  const lines = source.split('\n');
  const sections: string[][] = [[]];
  let inCodeBlock = false;

  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      inCodeBlock = !inCodeBlock;
    }

    if (!inCodeBlock && line.startsWith('## ')) {
      sections.push([line]);
    } else {
      sections[sections.length - 1].push(line);
    }
  }

  return sections.map(sec => sec.join('\n')).filter(sec => sec.trim() !== '');
}

interface MdxRendererProps {
  source: string;
}

export default function MdxRenderer({ source }: MdxRendererProps) {
  const preprocessed = preprocessAdmonitions(source);
  const sections = splitMdxIntoSections(preprocessed);

  return (
    <div className="space-y-16">
      {sections.map((section, idx) => (
        <FadeIn key={idx}>
          <MDXRemote
            source={section}
            options={{
              mdxOptions: {
                remarkPlugins: [remarkMath],
                rehypePlugins: [rehypeKatex],
              },
            }}
            components={customComponents}
          />
        </FadeIn>
      ))}
    </div>
  );
}
