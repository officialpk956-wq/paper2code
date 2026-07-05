import type { Metadata } from 'next';
import { LegalPageLayout, LegalSection } from '@/components/LegalPageLayout';
import { LEGAL } from '@/lib/legal';

export const metadata: Metadata = {
  title: 'Cookie Policy — paper2code',
  description: 'How paper2code uses cookies and browser storage.',
};

export default function CookiesPage() {
  return (
    <LegalPageLayout
      title="Cookie Policy"
      intro={`This page explains how ${LEGAL.company} uses cookies and similar browser-storage technologies. We keep this minimal and use no third-party advertising cookies.`}
    >
      <LegalSection heading="What we store">
        <ul className="list-disc space-y-1.5 pl-5">
          <li><span className="text-white">Essential authentication</span> — when you sign in, we store access and refresh tokens in your browser&apos;s local storage so you stay logged in between visits. Without these, the service cannot keep you signed in.</li>
          <li><span className="text-white">Preferences</span> — small values that remember your interface choices and in-progress work (such as unsaved code in an editor).</li>
        </ul>
        <p>These are stored locally in your browser and are sent to our servers only when needed to authenticate your requests.</p>
      </LegalSection>

      <LegalSection heading="Analytics">
        <p>If product analytics are enabled, we may use a privacy-conscious analytics tool to understand aggregate usage (such as which pages are visited) and improve the service. This data is used in aggregate and is not sold. We do not use cross-site advertising trackers.</p>
      </LegalSection>

      <LegalSection heading="Managing storage">
        <p>You can clear this data at any time by signing out, which removes your authentication tokens, or by clearing your browser&apos;s storage for this site through your browser settings. Note that clearing essential storage will sign you out and may reset in-progress work.</p>
      </LegalSection>

      <LegalSection heading="Changes">
        <p>We may update this policy as the service evolves. Material changes will be reflected in the effective date above. For more on how we handle your data overall, see our <a href="/privacy" className="text-[#A78BFA] hover:underline">Privacy Policy</a>.</p>
      </LegalSection>
    </LegalPageLayout>
  );
}
