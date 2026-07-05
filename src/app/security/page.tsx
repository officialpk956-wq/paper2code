import type { Metadata } from 'next';
import { LegalPageLayout, LegalSection } from '@/components/LegalPageLayout';
import { LEGAL } from '@/lib/legal';

export const metadata: Metadata = {
  title: 'Security — paper2code',
  description: 'How paper2code protects your data and how to report a vulnerability.',
};

export default function SecurityPage() {
  return (
    <LegalPageLayout
      title="Security"
      intro={`We take the security of your account and data seriously. This page summarizes the measures we use and how to report a vulnerability responsibly.`}
    >
      <LegalSection heading="How we protect your data">
        <ul className="list-disc space-y-1.5 pl-5">
          <li><span className="text-white">Encryption in transit</span> — all traffic between your browser and our servers is served over HTTPS/TLS.</li>
          <li><span className="text-white">Password protection</span> — passwords are hashed with a strong, salted algorithm (Argon2). We never store or log plaintext passwords.</li>
          <li><span className="text-white">Sandboxed code execution</span> — code you run in the Dojo executes in an isolated sandbox with strict CPU, memory, and time limits, on a network segment that cannot reach our application secrets or database.</li>
          <li><span className="text-white">Access controls</span> — authenticated requests use short-lived tokens, and sensitive actions are checked against ownership and permissions.</li>
          <li><span className="text-white">Rate limiting and abuse protection</span> — sensitive endpoints are rate-limited to reduce brute-force and denial-of-service risk.</li>
          <li><span className="text-white">Reputable infrastructure</span> — we host on established cloud providers with their own physical and network security programs.</li>
        </ul>
      </LegalSection>

      <LegalSection heading="Reporting a vulnerability">
        <p>If you believe you have found a security vulnerability, we want to hear from you. Please email{' '}
          <a href={`mailto:${LEGAL.contactEmail}`} className="text-[#A78BFA] hover:underline">{LEGAL.contactEmail}</a>{' '}
          with:
        </p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>A description of the issue and its potential impact.</li>
          <li>Clear steps to reproduce it.</li>
          <li>Any relevant proof-of-concept, logs, or screenshots.</li>
        </ul>
      </LegalSection>

      <LegalSection heading="Responsible disclosure">
        <p>We ask that you give us a reasonable opportunity to investigate and fix an issue before disclosing it publicly, and that you avoid privacy violations, data destruction, or service disruption while testing. Please only test against your own account. Acting in good faith under these guidelines, we will not pursue action against you for your research.</p>
        <p>We do not currently operate a paid bug-bounty program, but we genuinely appreciate reports and will credit researchers who wish to be acknowledged.</p>
      </LegalSection>
    </LegalPageLayout>
  );
}
