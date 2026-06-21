"use client";

import { useState, Fragment } from "react";
import { Check, Copy, Info, Lightbulb, AlertTriangle } from "lucide-react";
import { slugifyHeading } from "@/lib/slugify";

/**
 * Dependency-free renderer for the Paper2Code MDX subset:
 * headings, paragraphs, code blocks, math blocks ($$), callouts (:::),
 * blockquotes, tables, lists, images, and inline formatting.
 *
 * Designed to be swapped for next-mdx-remote + KaTeX later without touching
 * content files — the authoring syntax is standard MDX/markdown.
 */

// ---------------------------------------------------------------------------
// Inline formatting: `code`, **bold**, *italic*, [link](url), $math$
// ---------------------------------------------------------------------------

const INLINE_RE =
  /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\[[^\]]+\]\([^)]+\)|\$[^$\n]+\$)/g;

function renderInline(text: string): React.ReactNode {
  const parts = text.split(INLINE_RE);
  return parts.map((part, i) => {
    if (!part) return null;
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code
          key={i}
          className="px-1.5 py-0.5 rounded bg-[--bg-panel] border border-[--color-border] text-[--accent-light] text-[0.85em] font-mono"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-[--color-text-primary]">
          {renderInline(part.slice(2, -2))}
        </strong>
      );
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={i}>{renderInline(part.slice(1, -1))}</em>;
    }
    if (part.startsWith("[")) {
      const match = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (match) {
        return (
          <a
            key={i}
            href={match[2]}
            className="text-[--accent-cyan] hover:underline underline-offset-2"
          >
            {match[1]}
          </a>
        );
      }
    }
    if (part.startsWith("$") && part.endsWith("$")) {
      return (
        <span
          key={i}
          className="font-mono italic text-[--accent-light] text-[0.9em] whitespace-nowrap"
        >
          {part.slice(1, -1)}
        </span>
      );
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

// ---------------------------------------------------------------------------
// Block parsing
// ---------------------------------------------------------------------------

type Block =
  | { kind: "heading"; level: number; text: string }
  | { kind: "paragraph"; text: string }
  | { kind: "code"; lang: string; code: string }
  | { kind: "math"; tex: string }
  | { kind: "callout"; variant: string; lines: string[] }
  | { kind: "quote"; lines: string[] }
  | { kind: "table"; header: string[]; rows: string[][] }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "image"; alt: string; src: string }
  | { kind: "milestone"; title: string; body: string };

function parseBlocks(body: string): Block[] {
  const lines = body.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "") {
      i++;
      continue;
    }

    // Fenced code block
    if (line.startsWith("```")) {
      const lang = line.slice(3).trim();
      const code: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        code.push(lines[i]);
        i++;
      }
      i++; // closing fence
      blocks.push({ kind: "code", lang, code: code.join("\n") });
      continue;
    }

    // Display math block
    if (line.trim().startsWith("$$")) {
      const inline = line.trim();
      if (inline.length > 2 && inline.endsWith("$$")) {
        blocks.push({ kind: "math", tex: inline.slice(2, -2).trim() });
        i++;
        continue;
      }
      const tex: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("$$")) {
        tex.push(lines[i]);
        i++;
      }
      i++; // closing $$
      blocks.push({ kind: "math", tex: tex.join("\n").trim() });
      continue;
    }

    // Callout  :::note / :::tip / :::warning
    if (line.startsWith(":::")) {
      const variant = line.slice(3).trim() || "note";
      const content: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith(":::")) {
        content.push(lines[i]);
        i++;
      }
      i++; // closing :::
      blocks.push({
        kind: "callout",
        variant,
        lines: content.filter((l) => l.trim() !== ""),
      });
      continue;
    }

    // Heading
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      blocks.push({
        kind: "heading",
        level: headingMatch[1].length,
        text: headingMatch[2],
      });
      i++;
      continue;
    }

    // Milestone
    const milestoneMatch = line.match(/^<Milestone\s+title="([^"]+)"\s*>$/);
    if (milestoneMatch) {
      const title = milestoneMatch[1];
      const innerLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("</Milestone>")) {
        innerLines.push(lines[i]);
        i++;
      }
      i++; // closing tag
      blocks.push({ kind: "milestone", title, body: innerLines.join("\n") });
      continue;
    }

    // Blockquote
    if (line.startsWith(">")) {
      const quote: string[] = [];
      while (i < lines.length && lines[i].startsWith(">")) {
        quote.push(lines[i].replace(/^>\s?/, ""));
        i++;
      }
      blocks.push({ kind: "quote", lines: quote });
      continue;
    }

    // Table
    if (line.trim().startsWith("|")) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i].trim());
        i++;
      }
      const parseRow = (row: string) =>
        row
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
          .map((c) => c.trim());
      const header = parseRow(tableLines[0]);
      const rows = tableLines
        .slice(2) // skip separator row
        .map(parseRow);
      blocks.push({ kind: "table", header, rows });
      continue;
    }

    // Image
    const imageMatch = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (imageMatch) {
      blocks.push({ kind: "image", alt: imageMatch[1], src: imageMatch[2] });
      i++;
      continue;
    }

    // Unordered list
    if (/^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*]\s+/, ""));
        i++;
      }
      blocks.push({ kind: "list", ordered: false, items });
      continue;
    }

    // Ordered list
    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s+/, ""));
        i++;
      }
      blocks.push({ kind: "list", ordered: true, items });
      continue;
    }

    // Paragraph (consume contiguous plain lines)
    const para: string[] = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^(#{1,6}\s|```|\$\$|:::|>|\||[-*]\s|\d+\.\s|!\[)/.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    blocks.push({ kind: "paragraph", text: para.join(" ") });
  }

  return blocks;
}

// ---------------------------------------------------------------------------
// Block components
// ---------------------------------------------------------------------------

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="group relative my-6 rounded-lg border border-[--color-border] bg-[#0A0A12] overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-[--color-border] bg-[--bg-surface]">
        <span className="text-xs font-mono text-[--color-text-tertiary]">
          {lang || "text"}
        </span>
        <button
          onClick={copy}
          className="flex items-center gap-1.5 text-xs text-[--color-text-tertiary] hover:text-[--color-text-primary] transition-colors"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-[#10B981]" /> Copied
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" /> Copy
            </>
          )}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto text-sm leading-relaxed">
        <code className="font-mono text-[#E2E8F0]">{code}</code>
      </pre>
    </div>
  );
}

const CALLOUT_STYLES: Record<
  string,
  { border: string; bg: string; icon: React.ReactNode; label: string }
> = {
  note: {
    border: "border-l-[#06B6D4]",
    bg: "bg-[rgb(6,182,212,0.06)]",
    icon: <Info className="w-4 h-4 text-[#06B6D4]" />,
    label: "Note",
  },
  tip: {
    border: "border-l-[#10B981]",
    bg: "bg-[rgb(16,185,129,0.06)]",
    icon: <Lightbulb className="w-4 h-4 text-[#10B981]" />,
    label: "Tip",
  },
  warning: {
    border: "border-l-[#F59E0B]",
    bg: "bg-[rgb(245,158,11,0.06)]",
    icon: <AlertTriangle className="w-4 h-4 text-[#F59E0B]" />,
    label: "Warning",
  },
};

function Callout({ variant, lines }: { variant: string; lines: string[] }) {
  const style = CALLOUT_STYLES[variant] ?? CALLOUT_STYLES.note;
  return (
    <div
      className={`my-6 rounded-r-lg border-l-4 ${style.border} ${style.bg} px-4 py-3`}
    >
      <div className="flex items-center gap-2 mb-1">
        {style.icon}
        <span className="text-xs font-bold uppercase tracking-wide text-[--color-text-secondary]">
          {style.label}
        </span>
      </div>
      {lines.map((line, i) => (
        <p
          key={i}
          className="text-sm text-[--color-text-secondary] leading-relaxed"
        >
          {renderInline(line)}
        </p>
      ))}
    </div>
  );
}

function MathBlock({ tex }: { tex: string }) {
  return (
    <div className="my-6 px-6 py-4 rounded-lg bg-[--bg-surface] border border-[--color-border] overflow-x-auto text-center">
      <code className="font-mono italic text-[--accent-light] text-base whitespace-pre">
        {tex}
      </code>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

export function MarkdownRenderer({ body }: { body: string }) {
  const blocks = parseBlocks(body);

  return (
    <article className="max-w-none">
      {blocks.map((block, i) => {
        switch (block.kind) {
          case "heading": {
            if (block.level === 1)
              return (
                <h1 key={i} className="text-3xl font-bold mt-2 mb-6">
                  {renderInline(block.text)}
                </h1>
              );
            if (block.level === 2)
              return (
                <h2
                  key={i}
                  id={slugifyHeading(block.text)}
                  className="text-2xl font-bold mt-10 mb-4 pb-2 border-b border-[--color-border] scroll-mt-20"
                >
                  {renderInline(block.text)}
                </h2>
              );
            return (
              <h3 key={i} className="text-lg font-bold mt-8 mb-3">
                {renderInline(block.text)}
              </h3>
            );
          }
          case "paragraph":
            return (
              <p
                key={i}
                className="my-4 text-[--color-text-secondary] leading-relaxed text-[15px]"
              >
                {renderInline(block.text)}
              </p>
            );
          case "code":
            return <CodeBlock key={i} lang={block.lang} code={block.code} />;
          case "math":
            return <MathBlock key={i} tex={block.tex} />;
          case "callout":
            return (
              <Callout key={i} variant={block.variant} lines={block.lines} />
            );
          case "quote":
            return (
              <blockquote
                key={i}
                className="my-6 pl-4 border-l-4 border-[--accent-primary] italic text-[--color-text-secondary]"
              >
                {block.lines.map((line, j) => (
                  <p key={j} className="leading-relaxed">
                    {renderInline(line)}
                  </p>
                ))}
              </blockquote>
            );
          case "table":
            return (
              <div key={i} className="my-6 overflow-x-auto">
                <table className="w-full text-sm border border-[--color-border] rounded-lg overflow-hidden">
                  <thead>
                    <tr className="bg-[--bg-surface]">
                      {block.header.map((cell, j) => (
                        <th
                          key={j}
                          className="px-4 py-2.5 text-left font-semibold text-[--color-text-primary] border-b border-[--color-border]"
                        >
                          {renderInline(cell)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, j) => (
                      <tr
                        key={j}
                        className="border-b border-[--color-border] last:border-b-0 hover:bg-[--bg-surface] transition-colors"
                      >
                        {row.map((cell, k) => (
                          <td
                            key={k}
                            className="px-4 py-2.5 text-[--color-text-secondary]"
                          >
                            {renderInline(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          case "list":
            return block.ordered ? (
              <ol
                key={i}
                className="my-4 ml-6 space-y-2 list-decimal text-[--color-text-secondary]"
              >
                {block.items.map((item, j) => (
                  <li key={j} className="leading-relaxed pl-1">
                    {renderInline(item)}
                  </li>
                ))}
              </ol>
            ) : (
              <ul
                key={i}
                className="my-4 ml-6 space-y-2 list-disc text-[--color-text-secondary]"
              >
                {block.items.map((item, j) => (
                  <li key={j} className="leading-relaxed pl-1">
                    {renderInline(item)}
                  </li>
                ))}
              </ul>
            );
          case "image":
            return (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                key={i}
                src={block.src}
                alt={block.alt}
                className="my-6 rounded-lg border border-[--color-border] max-w-full"
              />
            );
          case "milestone":
            return (
              <div key={i} className="my-10 border border-[--accent-primary] rounded-xl overflow-hidden bg-[#0A0A12] shadow-lg">
                <div className="bg-[rgb(124,58,237,0.15)] px-6 py-4 border-b border-[--accent-primary]">
                  <h3 className="text-xl font-bold text-[--accent-light] flex items-center gap-2">
                    <Check className="w-5 h-5" /> {block.title}
                  </h3>
                </div>
                <div className="p-6">
                  {/* Recursive render for inner MDX */}
                  <MarkdownRenderer body={block.body} />
                </div>
              </div>
            );
          default:
            return null;
        }
      })}
    </article>
  );
}
