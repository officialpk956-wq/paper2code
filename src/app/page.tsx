import Link from 'next/link';
import { Network, LayoutGrid, Code2, Upload, Zap } from 'lucide-react';
import { ARCHITECTURES } from '@/data/content/architectures';
import { PAPERS } from '@/data/content/papers';
import { CURRICULUM } from '@/data/content/curriculum';
import { SD_SYSTEMS } from '@/data/content/systemDesign';
import { PROBLEMS as DOJO_PROBLEMS } from '@/data/problems';
import { LEGAL } from '@/lib/legal';

const STATS = [
  { value: `${ARCHITECTURES.length}`, label: 'Architectures', href: '/architectures' },
  { value: `${PAPERS.length}`,        label: 'Papers',        href: '/papers' },
  { value: `${CURRICULUM.length}`,    label: 'Learning Domains', href: '/learn' },
  { value: `${SD_SYSTEMS.length}`,    label: 'System Designs', href: '/system-design' },
  { value: `${DOJO_PROBLEMS.length}`, label: 'Dojo Problems', href: '/dojo' },
];

const PAPER_CARDS = [
  { icon: Upload,     title: 'PDF Upload',             desc: 'Drop any arXiv paper and extract structured knowledge.' },
  { icon: Network,    title: 'Knowledge Graph',        desc: 'See how concepts, methods, and citations connect.' },
  { icon: LayoutGrid, title: 'Architecture Blueprint', desc: 'Interactive diagrams for every model in the paper.' },
  { icon: Code2,      title: 'Executable Code',        desc: 'Ready-to-run reference implementations, side by side.' },
];

const PROBLEMS = [
  { num: '#001', title: 'Sigmoid Function',    difficulty: 'Easy',   topics: ['Activation', 'NumPy'],           href: '/dojo/ml-sigmoid' },
  { num: '#010', title: 'Scaled Dot-Product Attention', difficulty: 'Hard', topics: ['Transformers', 'Attention'], href: '/dojo/ml-attention' },
  { num: '#009', title: 'Gradient Descent Step', difficulty: 'Medium', topics: ['Optimization', 'NumPy'],        href: '/dojo/ml-gradient-descent' },
];

const DIFF_COLOR: Record<string, string> = {
  Easy:   'bg-[#4ADE80]/10 text-[#4ADE80] border-[#4ADE80]/20',
  Medium: 'bg-[#FACC15]/10 text-[#FACC15] border-[#FACC15]/20',
  Hard:   'bg-[#F87171]/10 text-[#F87171] border-[#F87171]/20',
};

const FOOTER_COLS = [
  { title: 'Product', links: [{ label: 'Dojo', href: '/dojo' }, { label: 'Papers', href: '/papers' }, { label: 'Learn', href: '/learn' }, { label: 'Labs', href: '/labs' }] },
  { title: 'Company', links: [{ label: 'Pricing', href: '/pricing' }, { label: 'System Design', href: '/system-design' }, { label: 'Architectures', href: '/architectures' }, { label: 'Contact', href: `mailto:${LEGAL.contactEmail}` }] },
  { title: 'Legal',   links: [{ label: 'Privacy', href: '/privacy' }, { label: 'Terms', href: '/terms' }, { label: 'Security', href: '/security' }, { label: 'Cookies', href: '/cookies' }] },
];

export default function HomePage() {
  const topDomains = CURRICULUM.slice(0, 3);

  return (
    <div className="bg-transparent text-white">
      {/* HERO */}
      <section className="relative flex min-h-screen items-center justify-center overflow-hidden">
        <div
          className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
          style={{ width: 700, height: 700, background: 'rgba(167,139,250,0.05)', filter: 'blur(140px)', borderRadius: '9999px' }}
          aria-hidden
        />
        <div className="relative mx-auto flex max-w-5xl flex-col items-center gap-6 px-6 text-center">
          <span className="rounded-full border border-[#A78BFA]/30 bg-[#A78BFA]/5 px-4 py-1.5 text-[13px] text-[#A78BFA]">
            New: AI-powered architecture blueprints →
          </span>
          <h1 className="text-[64px] font-bold leading-[1.05] tracking-tight text-white">
            From Research Papers<br />
            to <span className="text-[#A78BFA]">Running Code</span>
          </h1>
          <p className="mt-2 max-w-[520px] text-[16px] leading-[1.8] text-[#A3A3A3]">
            Upload any ML paper and get coding challenges, architecture diagrams, and guided
            implementations. The fastest way from theory to practice.
          </p>
          <div className="mt-4 flex flex-wrap justify-center gap-4">
            <Link href="/papers"
              className="rounded-full bg-[#A78BFA] px-8 py-3.5 text-[15px] font-bold text-black transition-colors hover:bg-[#C4B5FD]">
              Start Building for Free →
            </Link>
            <Link href="/dojo"
              className="rounded-full border border-[#262626] px-8 py-3.5 text-[15px] text-white transition-colors hover:bg-[#111111]">
              Browse Problems
            </Link>
          </div>
          <div className="mt-16 flex w-full flex-wrap justify-center gap-8 border-y border-[#1A1A1A] py-8 md:gap-16">
            {STATS.map(s => (
              <Link key={s.label} href={s.href} className="text-center group block">
                <div className="text-2xl font-bold text-[#A78BFA] group-hover:brightness-125 transition-all">{s.value}</div>
                <div className="mt-1 text-[11px] uppercase tracking-wider text-[#525252] group-hover:text-[#A3A3A3] transition-colors">{s.label}</div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* PAPERS */}
      <section className="mx-auto max-w-7xl px-6 py-24">
        <span className="inline-block rounded-full border border-[#A78BFA]/20 bg-[#A78BFA]/10 px-3 py-1 text-xs font-semibold text-[#A78BFA]">
          Research Hub
        </span>
        <h2 className="mt-3 text-[34px] font-bold text-white">Upload. Extract. Understand.</h2>
        <p className="mt-2 max-w-lg text-[14px] leading-relaxed text-[#A3A3A3]">
          Turn dense PDFs into interactive knowledge — with graphs, diagrams, and code that runs.
        </p>
        <div className="mt-10 grid grid-cols-2 gap-4 md:grid-cols-4">
          {PAPER_CARDS.map(c => (
            <Link key={c.title} href="/papers"
              className="rounded-xl border border-[#262626] bg-[#111111] p-5 transition-colors hover:border-[#A78BFA]/30 block">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#A78BFA]/12">
                <c.icon size={18} className="text-[#A78BFA]" />
              </div>
              <div className="mt-4 text-sm font-semibold text-white">{c.title}</div>
              <div className="mt-1 text-xs leading-relaxed text-[#A3A3A3]">{c.desc}</div>
            </Link>
          ))}
        </div>
      </section>

      {/* DOJO */}
      <section className="bg-[#0C160F] py-24">
        <div className="mx-auto max-w-7xl px-6">
          <span className="inline-block rounded-full border border-[#A78BFA]/20 bg-[#A78BFA]/10 px-3 py-1 text-xs font-semibold text-[#A78BFA]">
            Practice Dojo
          </span>
          <h2 className="mt-3 text-[34px] font-bold text-white">Code ML from Scratch.</h2>
          <p className="mt-2 max-w-lg text-[14px] leading-relaxed text-[#A3A3A3]">
            Bite-sized problems that build intuition — from sigmoid to full transformers.
          </p>
          <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-3">
            {PROBLEMS.map(p => (
              <Link key={p.num} href={p.href}
                className="rounded-xl border border-[#262626] bg-[#111111] p-5 block transition-colors hover:border-[#A78BFA]/30">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[#525252]">{p.num}</span>
                  <span className={'rounded-full border px-2 py-0.5 text-[10px] font-semibold ' + DIFF_COLOR[p.difficulty]}>
                    {p.difficulty}
                  </span>
                </div>
                <div className="mt-2 text-[15px] font-semibold text-white">{p.title}</div>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {p.topics.map(t => (
                    <span key={t} className="rounded-md bg-[#1A1A1A] px-2 py-0.5 text-[10px] text-[#A3A3A3]">{t}</span>
                  ))}
                </div>
                <div className="mt-4 text-xs font-semibold text-[#A78BFA]">Solve →</div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* LEARN */}
      <section className="mx-auto max-w-7xl px-6 py-24">
        <span className="inline-block rounded-full border border-[#60A5FA]/20 bg-[#60A5FA]/10 px-3 py-1 text-xs font-semibold text-[#60A5FA]">
          Learning Paths
        </span>
        <h2 className="mt-3 text-[34px] font-bold text-white">Master ML from First Principles.</h2>
        <p className="mt-2 max-w-lg text-[14px] leading-relaxed text-[#A3A3A3]">
          Structured tracks that take you from the math to the model.
        </p>
        <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-3">
          {topDomains.map(d => (
            <Link key={d.slug} href={`/learn/${d.slug}`}
              className="rounded-xl border border-[#262626] bg-[#111111] p-5 block transition-colors hover:border-[#60A5FA]/30">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#60A5FA]/12">
                  <Zap size={18} className="text-[#60A5FA]" />
                </div>
                <div>
                  <div className="text-[15px] font-semibold text-white">{d.name}</div>
                  <div className="text-xs text-[#525252]">{d.topics.length} topics</div>
                </div>
              </div>
              <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-[#1A1A1A]">
                <div className="h-full rounded-full bg-[#60A5FA] transition-all" style={{ width: `0%` }} />
              </div>
              <div className="mt-2 text-[11px] text-[#525252]">0% complete</div>
            </Link>
          ))}
        </div>
      </section>

      {/* FOOTER */}
      <footer className="mt-0 border-t border-[#1A1A1A] bg-[#081009] px-12 py-16">
        <div className="mx-auto flex max-w-7xl flex-col justify-between gap-10 md:flex-row">
          <div className="max-w-sm">
            <div className="flex items-center gap-2">
              <span className="inline-block rounded-full" style={{ width: 10, height: 10, background: '#A78BFA' }} />
              <span className="text-[15px] font-bold text-white">paper2code</span>
            </div>
            <p className="mt-2 text-sm text-[#A3A3A3]">Bridge research and practice.</p>
          </div>
          <div className="flex flex-wrap gap-12">
            {FOOTER_COLS.map(col => (
              <div key={col.title} className="flex flex-col gap-2">
                <div className="text-xs font-semibold uppercase tracking-wider text-white">{col.title}</div>
                {col.links.map(l => (
                  <Link key={l.label} href={l.href} className="text-sm text-[#525252] transition-colors hover:text-white">{l.label}</Link>
                ))}
              </div>
            ))}
          </div>
        </div>
        <div className="mx-auto mt-12 flex max-w-7xl items-center justify-between border-t border-[#1A1A1A] pt-6">
          <div className="text-xs text-[#525252]">© 2026 paper2code</div>
          <div className="text-xs text-[#525252]">Privacy · Terms</div>
        </div>
      </footer>
    </div>
  );
}
