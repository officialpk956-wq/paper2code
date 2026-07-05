import type { Metadata } from 'next';
import { LegalPageLayout, LegalSection } from '@/components/LegalPageLayout';
import { LEGAL, SUBPROCESSORS } from '@/lib/legal';

export const metadata: Metadata = {
  title: 'Privacy Policy — paper2code',
  description: 'How paper2code collects, uses, and protects your information.',
};

export default function PrivacyPage() {
  return (
    <LegalPageLayout
      title="Privacy Policy"
      intro={`This policy explains what information ${LEGAL.company} collects when you use our website and services, how we use it, who we share it with, and the choices you have. By using ${LEGAL.company}, you agree to the practices described here.`}
    >
      <LegalSection heading="1. Information we collect">
        <p>We collect only what we need to run the service:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li><span className="text-white">Account information</span> — your name, email address, and a securely hashed version of your password. We never store your password in plain text.</li>
          <li><span className="text-white">Content you create or upload</span> — PDF papers you upload, code you write and submit in the Dojo, notes, and questions you ask the AI Tutor.</li>
          <li><span className="text-white">Usage and progress data</span> — problems solved, submissions, XP, streaks, and learning progress.</li>
          <li><span className="text-white">Technical data</span> — IP address, browser type, and request metadata, used for security, rate limiting, and debugging.</li>
        </ul>
      </LegalSection>

      <LegalSection heading="2. How we use your information">
        <ul className="list-disc space-y-1.5 pl-5">
          <li>To provide, maintain, and improve the service and your account.</li>
          <li>To run code you submit, generate AI Tutor responses, and track your progress.</li>
          <li>To send you essential emails such as account verification and security notices.</li>
          <li>To protect the service against abuse, fraud, and security threats.</li>
        </ul>
        <p>We do not sell your personal information, and we do not use your uploaded content or code to train our own models.</p>
      </LegalSection>

      <LegalSection heading="3. AI processing">
        <p>When you use the AI Tutor or paper-analysis features, the text of your question and relevant context (such as excerpts of the paper you are studying) is sent to our third-party language-model provider to generate a response. Do not submit confidential or sensitive personal information to these features. AI responses can be inaccurate and should not be relied on as professional advice.</p>
      </LegalSection>

      <LegalSection heading="4. Service providers">
        <p>We rely on trusted providers to operate. Each processes data only as needed to deliver their function:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          {SUBPROCESSORS.map(s => (
            <li key={s.name}><span className="text-white">{s.name}</span> — {s.purpose}.</li>
          ))}
        </ul>
      </LegalSection>

      <LegalSection heading="5. Cookies and local storage">
        <p>We use your browser&apos;s local storage to keep you signed in (authentication tokens) and to remember interface preferences. We do not use third-party advertising cookies. See our <a href="/cookies" className="text-[#A78BFA] hover:underline">Cookie Policy</a> for details.</p>
      </LegalSection>

      <LegalSection heading="6. Data retention">
        <p>We keep your account data for as long as your account is active. Uploaded files and submissions are retained to provide the service and may be removed when you delete them or close your account. Some records may be retained longer where required for security, legal, or accounting reasons.</p>
      </LegalSection>

      <LegalSection heading="7. Your rights">
        <p>You can access and update your account information at any time from your account settings. You may request a copy of your data or ask us to delete your account and associated content by contacting us. We will respond within a reasonable timeframe.</p>
      </LegalSection>

      <LegalSection heading="8. Security">
        <p>We take reasonable technical and organizational measures to protect your data, including encryption in transit, hashed passwords, and sandboxed code execution. No system is perfectly secure; see our <a href="/security" className="text-[#A78BFA] hover:underline">Security page</a> for what we do and how to report a vulnerability.</p>
      </LegalSection>

      <LegalSection heading="9. Children">
        <p>{LEGAL.company} is not directed to children under 13, and we do not knowingly collect personal information from them. If you believe a child has provided us personal information, contact us and we will delete it.</p>
      </LegalSection>

      <LegalSection heading="10. Changes to this policy">
        <p>We may update this policy from time to time. When we make material changes, we will update the effective date above and, where appropriate, notify you. Your continued use of the service after changes take effect constitutes acceptance.</p>
      </LegalSection>
    </LegalPageLayout>
  );
}
