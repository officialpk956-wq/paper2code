import type { Metadata } from 'next';
import { LegalPageLayout, LegalSection } from '@/components/LegalPageLayout';
import { LEGAL } from '@/lib/legal';

export const metadata: Metadata = {
  title: 'Terms of Service — paper2code',
  description: 'The terms that govern your use of paper2code.',
};

export default function TermsPage() {
  return (
    <LegalPageLayout
      title="Terms of Service"
      intro={`These terms govern your access to and use of ${LEGAL.company}. By creating an account or using the service, you agree to these terms. If you do not agree, do not use the service.`}
    >
      <LegalSection heading="1. Eligibility and accounts">
        <p>You must be at least 13 years old to use {LEGAL.company}. You are responsible for the activity on your account and for keeping your credentials secure. Provide accurate information when registering and keep it up to date.</p>
      </LegalSection>

      <LegalSection heading="2. Your content and uploads">
        <p>You retain ownership of the papers, code, and other content you upload or create. By uploading content, you confirm that you have the right to do so and grant us a limited license to store and process it solely to provide the service to you.</p>
        <p>Do not upload material you do not have the rights to, or that infringes the intellectual property or privacy of others. We may remove content and suspend accounts that violate these terms or that are reported and found to be infringing.</p>
      </LegalSection>

      <LegalSection heading="3. Acceptable use">
        <p>When using {LEGAL.company}, you agree not to:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>Attempt to break out of, overload, or misuse the code-execution sandbox, or use it to run malicious, illegal, or resource-abusive code.</li>
          <li>Probe, scan, or test the vulnerability of the service except as permitted under our Security policy&apos;s responsible-disclosure terms.</li>
          <li>Scrape, resell, or redistribute the service or its content without permission.</li>
          <li>Upload unlawful, harmful, hateful, or infringing content, or impersonate others.</li>
          <li>Interfere with other users&apos; use of the service.</li>
        </ul>
      </LegalSection>

      <LegalSection heading="4. AI features">
        <p>The AI Tutor and related features generate content automatically and may be inaccurate, incomplete, or outdated. They are provided for learning and are not a substitute for professional advice. You are responsible for reviewing and verifying any AI-generated output before relying on it.</p>
      </LegalSection>

      <LegalSection heading="5. Intellectual property">
        <p>The service, including its software, design, and original content, is owned by {LEGAL.company} and protected by law. Third-party research papers, names, and trademarks referenced in educational content belong to their respective owners and are used for identification and study purposes.</p>
      </LegalSection>

      <LegalSection heading="6. Plans and payments">
        <p>Some features may require a paid plan. Where paid plans are offered, pricing, billing cycle, and included features are described on our pricing page at the time of purchase. Unless stated otherwise, fees are non-refundable except where required by law. We may change plan pricing or features with reasonable notice.</p>
      </LegalSection>

      <LegalSection heading="7. Disclaimers">
        <p>The service is provided &quot;as is&quot; and &quot;as available,&quot; without warranties of any kind, whether express or implied, including fitness for a particular purpose and non-infringement. We do not warrant that the service will be uninterrupted, error-free, or secure.</p>
      </LegalSection>

      <LegalSection heading="8. Limitation of liability">
        <p>To the maximum extent permitted by law, {LEGAL.company} and its operators will not be liable for any indirect, incidental, special, consequential, or punitive damages, or for any loss of data, profits, or goodwill, arising from your use of the service.</p>
      </LegalSection>

      <LegalSection heading="9. Termination">
        <p>You may stop using the service and delete your account at any time. We may suspend or terminate your access if you violate these terms or if necessary to protect the service or other users. Provisions that by their nature should survive termination will survive.</p>
      </LegalSection>

      <LegalSection heading="10. Changes to these terms">
        <p>We may update these terms from time to time. When we make material changes, we will update the effective date above. Your continued use of the service after changes take effect constitutes acceptance of the revised terms.</p>
      </LegalSection>
    </LegalPageLayout>
  );
}
