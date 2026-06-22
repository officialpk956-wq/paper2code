import { getImplementation, getAllSlugs, getPaper, getArchitecture } from "@/lib/content/loader";
import { notFound } from "next/navigation";
import { Breadcrumb } from "@/components/breadcrumb";
import { MarkdownRenderer } from "@/components/content/markdown-renderer";
import Link from "next/link";
import { ArrowRight, BookOpen, Layers, Target, GraduationCap } from "lucide-react";

interface Props {
  params: Promise<{ slug: string }>;
}

export function generateStaticParams() {
  return getAllSlugs("implementation").map((slug) => ({ slug }));
}

function calculateReadiness(impl: NonNullable<ReturnType<typeof getImplementation>>) {
  const reqPapers = impl.meta.relationships.papers || [];
  const reqArchs = impl.meta.relationships.architectures || [];
  const reqMath = impl.meta.relationships.math || [];
  
  const total = reqPapers.length + reqArchs.length + reqMath.length;
  // For demo purposes, we will mock the user's completed learning to be 2 out of every 3 items.
  // In a real app, this would read from the user's progress state.
  const completed = Math.max(0, total - 1); 
  
  const score = total === 0 ? 100 : Math.round((completed / total) * 100);
  
  return {
    score,
    papers: reqPapers.map(s => ({ slug: s, title: getPaper(s)?.meta.title || s })),
    architectures: reqArchs.map(s => ({ slug: s, title: getArchitecture(s)?.meta.title || s })),
    math: reqMath.map(s => ({ slug: s, title: s })),
  };
}

export default async function ImplementationPage({ params }: Props) {
  const { slug } = await params;
  const impl = getImplementation(slug);
  if (!impl) notFound();

  const readiness = calculateReadiness(impl);

  return (
    <div>
      <div className="border-b border-[--color-border] bg-[--bg-surface]">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Breadcrumb items={[
            { label: "Paper → Code", href: "/paper-to-code" },
            { label: impl.meta.title, current: true }
          ]} />
          
          <div className="mt-4 mb-2 flex items-center gap-3">
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 px-3 py-1 rounded-full flex items-center gap-1">
              <Target className="w-3.5 h-3.5" /> Implementation Journey
            </span>
          </div>

          <h1 className="text-4xl font-bold mb-4 text-[--color-text-primary]">{impl.meta.title}</h1>
          <p className="text-lg text-[--color-text-secondary] mb-8">{impl.meta.description}</p>
          
          <div className="p-6 rounded-xl bg-[--bg-panel] border border-[--color-border]">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold flex items-center gap-2">
                <GraduationCap className="w-5 h-5 text-[--accent-primary]" />
                Readiness Indicator
              </h3>
              <div className="flex items-center gap-3">
                <div className="text-sm text-[--color-text-secondary]">Score:</div>
                <div className="text-2xl font-black text-[--accent-cyan]">{readiness.score}%</div>
              </div>
            </div>
            
            <div className="w-full bg-[--bg-body] rounded-full h-2.5 mb-6 overflow-hidden">
              <div className="bg-[--accent-cyan] h-2.5 rounded-full" style={{ width: `${readiness.score}%` }}></div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {readiness.papers.length > 0 && (
                <div>
                  <div className="text-xs font-bold uppercase tracking-wider text-[--color-text-tertiary] mb-2 flex items-center gap-1.5">
                    <BookOpen className="w-3.5 h-3.5" /> Required Papers
                  </div>
                  <ul className="space-y-1.5">
                    {readiness.papers.map(p => (
                      <li key={p.slug}>
                        <Link href={`/papers/${p.slug}`} className="text-sm text-[--accent-light] hover:underline underline-offset-2 flex items-center gap-1">
                          {p.title} <ArrowRight className="w-3 h-3" />
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {readiness.architectures.length > 0 && (
                <div>
                  <div className="text-xs font-bold uppercase tracking-wider text-[--color-text-tertiary] mb-2 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5" /> Required Architectures
                  </div>
                  <ul className="space-y-1.5">
                    {readiness.architectures.map(a => (
                      <li key={a.slug}>
                        <Link href={`/architectures/${a.slug}`} className="text-sm text-[--accent-light] hover:underline underline-offset-2 flex items-center gap-1">
                          {a.title} <ArrowRight className="w-3 h-3" />
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <MarkdownRenderer body={impl.body} />
      </div>
    </div>
  );
}
