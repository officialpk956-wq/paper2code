import Link from 'next/link';
import { Space_Grotesk, JetBrains_Mono } from 'next/font/google';
import { Network, LayoutGrid, Code2, Upload, Zap } from 'lucide-react';
import { ARCHITECTURES } from '@/data/content/architectures';
import { PAPERS } from '@/data/content/papers';
import { CURRICULUM } from '@/data/content/curriculum';
import { SD_SYSTEMS } from '@/data/content/systemDesign';
import { PROBLEMS as DOJO_PROBLEMS } from '@/data/problems';
import { LEGAL } from '@/lib/legal';
import { FadeIn } from '@/components/FadeIn';
import AnimatedCard from '@/components/AnimatedCard';
import { NeuralField } from '@/components/hero/NeuralField';
import { DataTermsField } from '@/components/hero/DataTermsField';

const spaceGrotesk = Space_Grotesk({ subsets: ['latin'], weight: ['500', '600', '700'], variable: '--font-display' });
const jetbrainsMono = JetBrains_Mono({ subsets: ['latin'], weight: ['300', '400', '500'], variable: '--font-mono' });

// Landing-page palette: electric cyan primary, electric violet as its
// gradient/accent partner, dark navy neutrals (matches globals.css :root).
const CYAN = '#00E5FF';
const CYAN_LIGHT = '#66F5FF';
const VIOLET = '#7C5CFF';
const GREEN = '#A3E635';

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

function SectionBadge({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <span
      className="inline-block rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] [font-family:var(--font-display)]"
      style={{ borderColor: `${color}33`, background: `${color}14`, color }}
    >
      {children}
    </span>
  );
}

export default function HomePage() {
  const topDomains = CURRICULUM.slice(0, 3);

  return (
    <div className={`${spaceGrotesk.variable} ${jetbrainsMono.variable} relative bg-transparent text-white`}>
      {/* Page-wide animated background — fixed to the viewport so it stays
          visible behind every section as the page scrolls, not just the hero. */}
      <div className="pointer-events-none fixed inset-0 z-0" aria-hidden>
        {/* DOM fallback glow — always present, sits under the WebGL canvas */}
        <div
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
          style={{ width: 780, height: 780, background: 'radial-gradient(circle, rgba(0,229,255,0.10), rgba(124,92,255,0.05) 55%, transparent 75%)', filter: 'blur(90px)', borderRadius: '9999px' }}
        />
        <div className="absolute inset-0 opacity-45">
          <NeuralField />
        </div>
        <DataTermsField />
      </div>

      <div className="relative z-10">
      {/* HERO */}
      <section className="relative flex min-h-screen items-center justify-center">
        <div className="relative z-10 mx-auto flex max-w-5xl flex-col items-center gap-6 px-6 text-center">
          <Link
            href="/papers"
            className="mb-3 inline-block rounded-full p-[1px] transition-transform hover:scale-105"
            style={{ background: `linear-gradient(90deg, ${CYAN}80, ${VIOLET}80)` }}
          >
            <span className="block rounded-full bg-[#04070F] px-4 py-1.5 text-[13px] [font-family:var(--font-mono)]" style={{ color: CYAN_LIGHT }}>
              New: AI-powered architecture blueprints →
            </span>
          </Link>

          <h1 className="text-[40px] font-bold leading-[1.1] tracking-tight text-white [font-family:var(--font-display)] sm:text-[52px] md:text-[64px] md:leading-[1.05]">
            From Research Papers<br />
            to <span style={{ color: CYAN }}>Running Code</span>
          </h1>

          <p className="mt-2 max-w-[540px] text-[15px] leading-[1.8] text-[#8FA3C4] [font-family:var(--font-mono)]">
            Upload any ML paper and get coding challenges, architecture diagrams, and guided
            implementations. The fastest way from theory to practice.
          </p>

          <div className="mt-4 flex flex-wrap justify-center gap-4">
            <Link
              href="/papers"
              className="glow-cyan rounded-lg px-8 py-3.5 text-[15px] font-bold text-[#04070F] transition-all hover:brightness-110 active:scale-95 [font-family:var(--font-display)]"
              style={{ background: CYAN }}
            >
              Start Building for Free →
            </Link>
            <Link
              href="/dojo"
              className="inline-flex items-center gap-2 rounded-lg border px-8 py-3.5 text-[15px] backdrop-blur-md transition-all hover:text-white active:scale-95 [font-family:var(--font-display)]"
              style={{ borderColor: `${VIOLET}40`, background: 'rgba(15,24,48,0.5)', color: '#D6DCF0' }}
            >
              <Code2 size={16} style={{ color: VIOLET }} className="flex-shrink-0" />
              Browse Problems
            </Link>
          </div>

          <div className="mt-16 flex w-full flex-wrap justify-center gap-8 border-y border-[#1A2744] py-8 md:gap-16">
            {STATS.map((s, idx) => (
              <FadeIn key={s.label} delay={idx * 0.05}>
                <Link href={s.href} className="group block text-center">
                  <div className="text-2xl font-bold transition-all group-hover:brightness-125 [font-family:var(--font-display)]" style={{ color: CYAN }}>
                    {s.value}
                  </div>
                  <div className="mt-1 text-[11px] uppercase tracking-[0.14em] text-[#5C6D8C] transition-colors group-hover:text-[#8FA3C4] [font-family:var(--font-mono)]">
                    {s.label}
                  </div>
                </Link>
              </FadeIn>
            ))}
          </div>
        </div>
      </section>

      {/* PAPERS */}
      <section className="mx-auto max-w-7xl px-6 py-24">
        <SectionBadge color={CYAN}>Research Hub</SectionBadge>
        <h2 className="mt-3 text-[34px] font-bold text-white [font-family:var(--font-display)]">Upload. Extract. Understand.</h2>
        <p className="mt-2 max-w-lg text-[14px] leading-relaxed text-[#8FA3C4] [font-family:var(--font-mono)]">
          Turn dense PDFs into interactive knowledge, featuring graphs, diagrams, and code that runs.
        </p>
        <div className="mt-10 grid grid-cols-2 gap-4 md:grid-cols-4">
          {PAPER_CARDS.map((c, idx) => (
            <FadeIn key={c.title} delay={idx * 0.1}>
              <AnimatedCard>
                <Link
                  href="/papers"
                  className="p2c-glass block h-full rounded-xl p-5 transition-colors hover:border-[#00E5FF]/30"
                >
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl" style={{ background: `${CYAN}1F` }}>
                    <c.icon size={18} style={{ color: CYAN }} />
                  </div>
                  <div className="mt-4 text-sm font-semibold text-white [font-family:var(--font-display)]">{c.title}</div>
                  <div className="mt-1 text-xs leading-relaxed text-[#8FA3C4] [font-family:var(--font-mono)]">{c.desc}</div>
                </Link>
              </AnimatedCard>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* DOJO */}
      <section className="py-24" style={{ background: 'rgba(11,15,30,0.55)' }}>
        <div className="mx-auto max-w-7xl px-6">
          <SectionBadge color={VIOLET}>Practice Dojo</SectionBadge>
          <h2 className="mt-3 text-[34px] font-bold text-white [font-family:var(--font-display)]">Code ML from Scratch.</h2>
          <p className="mt-2 max-w-lg text-[14px] leading-relaxed text-[#8FA3C4] [font-family:var(--font-mono)]">
            Bite-sized problems that build intuition, covering everything from sigmoid to full transformers.
          </p>
          <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-3">
            {PROBLEMS.map((p, idx) => (
              <FadeIn key={p.num} delay={idx * 0.1}>
                <AnimatedCard>
                  <Link href={p.href} className="p2c-glass block h-full rounded-xl p-5 transition-colors hover:border-[#7C5CFF]/30">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-[#5C6D8C] [font-family:var(--font-mono)]">{p.num}</span>
                      <span className={'rounded-full border px-2 py-0.5 text-[10px] font-semibold ' + DIFF_COLOR[p.difficulty]}>
                        {p.difficulty}
                      </span>
                    </div>
                    <div className="mt-2 text-[15px] font-semibold text-white [font-family:var(--font-display)]">{p.title}</div>
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {p.topics.map(t => (
                        <span key={t} className="rounded-md bg-[#141E38] px-2 py-0.5 text-[10px] text-[#8FA3C4] [font-family:var(--font-mono)]">{t}</span>
                      ))}
                    </div>
                    <div className="mt-4 text-xs font-semibold [font-family:var(--font-display)]" style={{ color: VIOLET }}>Solve →</div>
                  </Link>
                </AnimatedCard>
              </FadeIn>
            ))}
          </div>
        </div>
      </section>

      {/* LEARN */}
      <section className="mx-auto max-w-7xl px-6 py-24">
        <SectionBadge color={GREEN}>Learning Paths</SectionBadge>
        <h2 className="mt-3 text-[34px] font-bold text-white [font-family:var(--font-display)]">Master ML from First Principles.</h2>
        <p className="mt-2 max-w-lg text-[14px] leading-relaxed text-[#8FA3C4] [font-family:var(--font-mono)]">
          Structured tracks that take you from the math to the model.
        </p>
        <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-3">
          {topDomains.map((d, idx) => (
            <FadeIn key={d.slug} delay={idx * 0.1}>
              <AnimatedCard>
                <Link href={`/learn/${d.slug}`} className="p2c-glass block h-full rounded-xl p-5 transition-colors hover:border-[#A3E635]/30">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl" style={{ background: `${GREEN}1F` }}>
                      <Zap size={18} style={{ color: GREEN }} />
                    </div>
                    <div>
                      <div className="text-[15px] font-semibold text-white [font-family:var(--font-display)]">{d.name}</div>
                      <div className="text-xs text-[#5C6D8C] [font-family:var(--font-mono)]">{d.topics.length} topics</div>
                    </div>
                  </div>
                  <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-[#141E38]">
                    <div className="h-full rounded-full transition-all" style={{ width: '0%', background: GREEN }} />
                  </div>
                  <div className="mt-2 text-[11px] text-[#5C6D8C] [font-family:var(--font-mono)]">0% complete</div>
                </Link>
              </AnimatedCard>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* FOOTER */}
      <footer className="mt-0 border-t border-[#1A2744] bg-[#060A16]/80 px-12 py-16 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-col justify-between gap-10 md:flex-row">
          <div className="max-w-sm">
            <div className="flex items-center gap-2">
              <span className="inline-block rounded-full" style={{ width: 10, height: 10, background: CYAN }} />
              <span className="text-[15px] font-bold text-white [font-family:var(--font-display)]">paper2code</span>
            </div>
            <p className="mt-2 text-sm text-[#8FA3C4] [font-family:var(--font-mono)]">Bridge research and practice.</p>
          </div>
          <div className="flex flex-wrap gap-12">
            {FOOTER_COLS.map(col => (
              <div key={col.title} className="flex flex-col gap-2">
                <div className="text-xs font-semibold uppercase tracking-[0.14em] text-white [font-family:var(--font-display)]">{col.title}</div>
                {col.links.map(l => (
                  <Link key={l.label} href={l.href} className="text-sm text-[#5C6D8C] transition-colors hover:text-white [font-family:var(--font-mono)]">{l.label}</Link>
                ))}
              </div>
            ))}
          </div>
        </div>
        <div className="mx-auto mt-12 flex max-w-7xl items-center justify-between border-t border-[#1A2744] pt-6">
          <div className="text-xs text-[#5C6D8C] [font-family:var(--font-mono)]">© 2026 paper2code</div>
          <div className="text-xs text-[#5C6D8C] [font-family:var(--font-mono)]">Privacy · Terms</div>
        </div>
      </footer>
      </div>
    </div>
  );
}
