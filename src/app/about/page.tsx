import type { Metadata } from 'next';
import Link from 'next/link';
import { Mail, BookOpen, Code2, Network, Layers, ArrowRight } from 'lucide-react';
import { LEGAL } from '@/lib/legal';

export const metadata: Metadata = {
  title: 'About: paper2code',
  description: 'Who built paper2code, why it exists, and how to get in touch.',
};

const PILLARS = [
  {
    icon: BookOpen,
    color: '#60A5FA',
    bg: '#60A5FA12',
    title: 'Research is hard to operationalise',
    body: 'ML papers are dense, notation-heavy, and rarely include runnable code. Most developers give up before they get a working implementation.',
  },
  {
    icon: Code2,
    color: '#A78BFA',
    bg: '#A78BFA12',
    title: 'The best way to learn is to build',
    body: 'Reading is passive. Writing code that produces the right tensor shapes, loss curves, and outputs cements the intuition nothing else can.',
  },
  {
    icon: Network,
    color: '#4ADE80',
    bg: '#4ADE8012',
    title: 'Context beats isolation',
    body: 'An architecture means more when you can trace its lineage, find the paper it came from, and immediately practise its core operations side-by-side.',
  },
  {
    icon: Layers,
    color: '#FB923C',
    bg: '#FB923C12',
    title: 'Progress needs structure',
    body: 'The Dojo, Learn tracks, and Labs are designed as a deliberate curriculum, not a dump of resources. Each part scaffolds the next.',
  },
];

const TIMELINE = [
  { year: '2024', label: 'First prototype', detail: 'A single-page tool that extracted figures and pseudocode from arXiv PDFs.' },
  { year: 'Early 2025', label: 'Architecture library', detail: 'Added the library of 200+ model blueprints with interactive diagrams for flagship architectures.' },
  { year: 'Mid 2025', label: 'Dojo & Learn', detail: 'Launched coding challenges and structured learning tracks rooted in the CURRICULUM data.' },
  { year: '2026', label: 'paper2code v2', detail: 'Full platform: Papers workspace, Labs mutation engine, AI Tutor, and user accounts.' },
];

export default function AboutPage() {
  return (
    <div className="bg-transparent text-white">

      {/* ── HERO ───────────────────────────────────────────────── */}
      <section className="relative overflow-hidden border-b border-[#1A1A1A]">
        {/* purple glow */}
        <div
          className="pointer-events-none absolute left-1/2 top-0 -translate-x-1/2"
          style={{ width: 600, height: 400, background: 'rgba(167,139,250,0.07)', filter: 'blur(100px)', borderRadius: '9999px' }}
          aria-hidden
        />
        <div className="relative mx-auto max-w-4xl px-6 py-24 text-center">
          <span className="inline-block rounded-full border border-[#A78BFA]/25 bg-[#A78BFA]/8 px-4 py-1.5 text-[12px] font-semibold tracking-wider text-[#A78BFA] uppercase mb-6">
            About the Creator
          </span>
          <h1 className="text-[46px] font-extrabold leading-[1.08] tracking-tight text-white mb-6">
            Built for developers who<br />
            <span className="text-[#A78BFA]">learn by doing.</span>
          </h1>
          <p className="mx-auto max-w-[600px] text-[16px] leading-[1.85] text-[#A3A3A3]">
            paper2code was created to close the gap between a landmark ML paper and the running,
            debuggable code that proves you actually understand it.
          </p>
        </div>
      </section>

      {/* ── WHY ────────────────────────────────────────────────── */}
      <section className="mx-auto max-w-5xl px-6 py-20">
        <div className="mb-4 text-[11px] font-bold uppercase tracking-widest text-[#8A8A8A]">The problem</div>
        <h2 className="text-[28px] font-bold text-white mb-6 leading-tight">
          Why paper2code exists
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {PILLARS.map(({ icon: Icon, color, bg, title, body }) => (
            <div
              key={title}
              className="rounded-xl border border-[#222222] p-6 bg-[#0D0D0D] hover:border-[#2E2E2E] transition-colors"
            >
              <div
                className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl"
                style={{ background: bg }}
              >
                <Icon size={18} style={{ color }} />
              </div>
              <div className="text-[14px] font-semibold text-white mb-2">{title}</div>
              <div className="text-[13px] leading-relaxed text-[#8A8A8A]">{body}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── STORY ──────────────────────────────────────────────── */}
      <section className="border-t border-[#1A1A1A] bg-[#0A0A0A]">
        <div className="mx-auto max-w-5xl px-6 py-20">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-start">
            {/* Text */}
            <div>
              <div className="mb-4 text-[11px] font-bold uppercase tracking-widest text-[#8A8A8A]">The story</div>
              <h2 className="text-[28px] font-bold text-white mb-6 leading-tight">How it started</h2>
              <div className="space-y-4 text-[14px] leading-[1.85] text-[#A3A3A3]">
                <p>
                  I spent months reading the Attention Is All You Need paper, staring at the
                  multi-head attention diagram, and still not having a single passing unit test for
                  my implementation. The gap between the math and the code felt unreasonably wide.
                </p>
                <p>
                  paper2code started as a personal tool: a script that ingested arXiv PDFs and
                  extracted figures, pseudocode, and architecture details into a structured format
                  I could code against. It turned out other developers had the same problem.
                </p>
                <p>
                  Today the platform has grown into a full curriculum: upload any paper, explore
                  its architecture, practice its core operations in the Dojo, trace its lineage
                  through 200+ blueprints, and understand the math through structured Learn tracks.
                </p>
              </div>
            </div>

            {/* Timeline */}
            <div>
              <div className="mb-4 text-[11px] font-bold uppercase tracking-widest text-[#8A8A8A]">Timeline</div>
              <div className="relative">
                {/* vertical line */}
                <div className="absolute left-[15px] top-2 bottom-2 w-px bg-[#262626]" aria-hidden />
                <div className="space-y-7">
                  {TIMELINE.map(({ year, label, detail }) => (
                    <div key={year} className="flex gap-5 relative">
                      <div className="relative z-10 mt-0.5 flex-shrink-0">
                        <div className="w-[30px] h-[30px] rounded-full border border-[#A78BFA]/40 bg-[#A78BFA]/10 flex items-center justify-center">
                          <div className="w-2 h-2 rounded-full bg-[#A78BFA]" />
                        </div>
                      </div>
                      <div>
                        <div className="text-[11px] font-semibold text-[#A78BFA] mb-0.5">{year}</div>
                        <div className="text-[14px] font-semibold text-white mb-1">{label}</div>
                        <div className="text-[13px] leading-relaxed text-[#8A8A8A]">{detail}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── EXPLORE ────────────────────────────────────────────── */}
      <section className="border-t border-[#1A1A1A]">
        <div className="mx-auto max-w-5xl px-6 py-20">
          <div className="mb-4 text-[11px] font-bold uppercase tracking-widest text-[#8A8A8A]">Explore the platform</div>
          <h2 className="text-[28px] font-bold text-white mb-8 leading-tight">Start somewhere</h2>
          <div className="flex flex-wrap gap-3">
            {[
              { label: 'Dojo', href: '/dojo', color: '#FB923C' },
              { label: 'Architecture Library', href: '/architectures', color: '#A78BFA' },
              { label: 'Learn Tracks', href: '/learn', color: '#60A5FA' },
              { label: 'Papers Workspace', href: '/papers', color: '#4ADE80' },
              { label: 'Labs', href: '/labs', color: '#FACC15' },
            ].map(({ label, href, color }) => (
              <Link
                key={label}
                href={href}
                className="inline-flex items-center gap-2 rounded-full border border-[#262626] bg-[#111111] px-5 py-2.5 text-[13px] font-medium text-[#D4D4D4] transition-all hover:text-white hover:border-[#3A3A3A] hover:bg-[#181818]"
              >
                <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: color }} />
                {label}
                <ArrowRight size={12} className="text-[#6A6A6A]" />
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ── CONTACT ────────────────────────────────────────────── */}
      <section className="border-t border-[#1A1A1A] bg-[#0A0A0A]">
        <div className="mx-auto max-w-5xl px-6 py-20">
          <div className="rounded-2xl border border-[#222222] bg-gradient-to-br from-[#A78BFA]/6 to-transparent p-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
            <div>
              <h2 className="text-[22px] font-bold text-white mb-2">Get in touch</h2>
              <p className="text-[14px] leading-relaxed text-[#8A8A8A] max-w-md">
                Questions, bug reports, feature ideas, or just want to say you finally got your
                attention implementation to pass? Email is the best way to reach me.
              </p>
            </div>
            <a
              href={`mailto:${LEGAL.contactEmail}`}
              className="inline-flex items-center gap-2.5 rounded-full bg-[#A78BFA] px-7 py-3 text-[14px] font-bold text-black transition-colors hover:bg-[#C4B5FD] flex-shrink-0 shadow-lg shadow-[#A78BFA]/15"
            >
              <Mail size={16} />
              {LEGAL.contactEmail}
            </a>
          </div>
        </div>
      </section>

    </div>
  );
}
