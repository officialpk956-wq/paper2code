import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { LEGAL } from '@/lib/legal';

// Shared shell for /privacy, /terms, /security, /cookies. Inherits the global
// TopNavbar + animated background from the root layout; provides consistent
// prose styling, a title, the effective date, and a back link.
export function LegalPageLayout({
  title,
  intro,
  children,
}: {
  title: string;
  intro: string;
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen text-white">
      <div className="mx-auto max-w-3xl px-6 py-16">
        <Link
          href="/"
          className="mb-8 inline-flex items-center gap-1.5 text-[13px] text-[#A3A3A3] transition-colors hover:text-white"
        >
          <ArrowLeft size={15} /> Back to home
        </Link>

        <h1 className="text-[32px] font-bold leading-tight text-white">{title}</h1>
        <p className="mt-2 text-[13px] text-[#525252]">Effective {LEGAL.effectiveDate}</p>
        <p className="mt-6 text-[15px] leading-relaxed text-[#A3A3A3]">{intro}</p>

        <div className="legal-prose mt-10 space-y-10">{children}</div>

        <div className="mt-16 border-t border-[#262626] pt-6 text-[13px] text-[#525252]">
          Questions about this document? Contact us at{' '}
          <a href={`mailto:${LEGAL.contactEmail}`} className="text-[#A78BFA] hover:underline">
            {LEGAL.contactEmail}
          </a>
          .
        </div>
      </div>
    </div>
  );
}

// One titled section of a legal page.
export function LegalSection({ heading, children }: { heading: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="mb-3 text-[18px] font-semibold text-white">{heading}</h2>
      <div className="space-y-3 text-[14px] leading-relaxed text-[#A3A3A3]">{children}</div>
    </section>
  );
}

export default LegalPageLayout;
