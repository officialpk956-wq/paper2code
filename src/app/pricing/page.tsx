'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Check } from 'lucide-react';

const FREE_FEATURES = [
  '5 paper uploads / month',
  'Knowledge graph extraction',
  '20 dojo problems',
  'Basic learning paths',
  'Community leaderboard',
];

const PRO_FEATURES = [
  'Unlimited paper uploads',
  'Architecture blueprints',
  'Executable code generation',
  'All 128 dojo problems',
  'AI Tutor (50 sessions/month)',
  'Full learning paths',
  'Priority support',
  'Early access to new labs',
];

export default function PricingPage() {
  const [yearly, setYearly] = useState(false);

  const monthlyPrice = 19;
  const yearlyPrice  = Math.round(monthlyPrice * 12 * 0.75);

  return (
    <div className="min-h-screen bg-transparent text-white">
      {/* HEADER */}
      <div className="text-center px-6 py-20">
        <span className="inline-block rounded-full border border-[#A78BFA]/20 bg-[#A78BFA]/8 px-3 py-1 text-xs font-semibold text-[#A78BFA]">
          Pricing
        </span>
        <h1 className="mt-4 text-[40px] font-bold text-white">Simple, Transparent Pricing</h1>
        <p className="mt-2 text-[15px] text-[#A3A3A3]">Start free. Upgrade when you&apos;re ready to go deep.</p>

        {/* Toggle */}
        <div className="mt-6 flex items-center justify-center gap-3">
          <span className={'text-[13px] ' + (!yearly ? 'text-white' : 'text-[#525252]')}>Monthly</span>
          <button type="button" onClick={() => setYearly(!yearly)}
            className={'relative h-6 w-11 rounded-full transition-colors ' + (yearly ? 'bg-[#A78BFA]' : 'bg-[#262626]')}>
            <div className={'absolute top-1 h-4 w-4 rounded-full bg-white transition-transform ' +
              (yearly ? 'translate-x-5' : 'translate-x-1')} />
          </button>
          <span className={'text-[13px] ' + (yearly ? 'text-white' : 'text-[#525252]')}>
            Yearly <span className="rounded-full bg-[#4ADE80]/15 px-1.5 py-0.5 text-[10px] text-[#4ADE80]">25% off</span>
          </span>
        </div>
      </div>

      {/* CARDS */}
      <div className="mx-auto flex max-w-3xl flex-col gap-5 px-6 pb-24 md:flex-row">
        {/* FREE */}
        <div className="flex flex-1 flex-col rounded-2xl border border-[#262626] bg-[#111111] p-7">
          <div className="text-[13px] font-semibold uppercase tracking-wider text-[#525252]">Free</div>
          <div className="mt-3 flex items-end gap-1">
            <span className="text-[42px] font-bold text-white">$0</span>
            <span className="mb-2 text-[13px] text-[#525252]">/ month</span>
          </div>
          <div className="mt-1 text-[12px] text-[#525252]">No credit card required</div>

          <Link href="/papers"
            className="mt-6 block w-full rounded-xl border border-[#262626] py-3 text-[13px] font-semibold text-white hover:bg-[#111111] transition-colors text-center">
            Get Started Free
          </Link>

          <ul className="mt-6 flex flex-col gap-3">
            {FREE_FEATURES.map(f => (
              <li key={f} className="flex items-start gap-2.5 text-[13px] text-[#A3A3A3]">
                <Check size={14} className="mt-0.5 flex-shrink-0 text-[#4ADE80]" />
                {f}
              </li>
            ))}
          </ul>
        </div>

        {/* PRO */}
        <div className="flex flex-1 flex-col rounded-2xl border border-[#A78BFA]/40 bg-gradient-to-b from-[#0A0A0A] to-[#111111] p-7 relative overflow-hidden">
          <div className="pointer-events-none absolute -right-8 -top-8 h-32 w-32 rounded-full"
            style={{ background: 'rgba(167,139,250,0.08)', filter: 'blur(32px)' }} />
          <div className="flex items-center gap-2">
            <div className="text-[13px] font-semibold uppercase tracking-wider text-[#A78BFA]">Pro</div>
            <span className="rounded-full bg-[#A78BFA]/15 px-2 py-0.5 text-[10px] font-semibold text-[#A78BFA]">Most Popular</span>
          </div>
          <div className="mt-3 flex items-end gap-1">
            <span className="text-[42px] font-bold text-white">
              ${yearly ? Math.round(yearlyPrice / 12) : monthlyPrice}
            </span>
            <span className="mb-2 text-[13px] text-[#525252]">/ month</span>
          </div>
          {yearly && (
            <div className="text-[12px] text-[#525252]">
              Billed ${yearlyPrice}/year · save ${monthlyPrice * 12 - yearlyPrice}
            </div>
          )}
          {!yearly && <div className="text-[12px] text-[#525252]">Billed monthly</div>}

          <Link href="/papers"
            className="mt-6 block w-full rounded-xl bg-[#A78BFA] py-3 text-[13px] font-bold text-black hover:bg-[#C4B5FD] transition-colors text-center">
            Upgrade to Pro →
          </Link>

          <ul className="mt-6 flex flex-col gap-3">
            {PRO_FEATURES.map(f => (
              <li key={f} className="flex items-start gap-2.5 text-[13px] text-white">
                <Check size={14} className="mt-0.5 flex-shrink-0 text-[#A78BFA]" />
                {f}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* FAQ */}
      <div className="mx-auto max-w-xl px-6 pb-24 space-y-3">
        <div className="text-center text-[18px] font-bold text-white mb-6">Frequently Asked Questions</div>
        {[
          { q: 'Can I cancel anytime?', a: 'Yes. Cancel any time and keep access until the end of your billing period.' },
          { q: 'What counts as a paper upload?', a: 'Any PDF you upload to the Research Hub counts. Re-uploads of the same file do not.' },
          { q: 'Is there a student discount?', a: 'Yes — email us with a .edu address for 50% off Pro.' },
          { q: 'Can I try Pro before paying?', a: 'The free tier gives you a solid taste. We\'ll add a trial soon.' },
        ].map(({ q, a }) => (
          <div key={q} className="rounded-xl border border-[#262626] bg-[#111111] p-4">
            <div className="text-[13px] font-semibold text-white">{q}</div>
            <div className="mt-1.5 text-[12px] leading-relaxed text-[#A3A3A3]">{a}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
