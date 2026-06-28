import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Terms of Service | Paper2Code',
  description: 'Terms of Service for Paper2Code',
};

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-[--bg-body] pt-24 pb-16 px-4">
      <div className="max-w-3xl mx-auto bg-[--bg-surface] border border-[--color-border] rounded-xl p-8 md:p-12 shadow-sm">
        <h1 className="text-3xl font-heading font-bold text-[--color-text-primary] mb-2">Terms of Service</h1>
        <p className="text-sm text-[--color-text-tertiary] mb-10">Last updated: June 28, 2026</p>

        <div className="space-y-8 text-[--color-text-secondary] leading-relaxed">
          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">1. Acceptance</h2>
            <p>By using Paper2Code, you agree to these Terms of Service. If you do not agree, please do not use our platform.</p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">2. What Paper2Code Is</h2>
            <p>Paper2Code is an AI-powered research paper learning platform designed to help engineers understand architectures, system design, and ML concepts.</p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">3. Your Account</h2>
            <ul className="list-disc pl-5 space-y-2">
              <li><strong>Age Requirement:</strong> You must be at least 13 years old to use Paper2Code.</li>
              <li><strong>Security:</strong> You are responsible for maintaining the security of your account and password.</li>
              <li><strong>Usage:</strong> One account is permitted per person.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">4. Acceptable Use</h2>
            <ul className="list-disc pl-5 space-y-2">
              <li>Do not upload papers or documents that you do not have the rights to distribute.</li>
              <li>Do not abuse or attempt to circumvent limits on the code execution sandbox (Dojo).</li>
              <li>Do not attempt to scrape, reverse-engineer, or overload the platform infrastructure.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">5. Intellectual Property</h2>
            <ul className="list-disc pl-5 space-y-2">
              <li><strong>Platform:</strong> All platform code, design, and original content belong to Paper2Code.</li>
              <li><strong>User Content:</strong> Papers you upload remain your responsibility to ensure you have the proper rights.</li>
              <li><strong>Generated Output:</strong> Generated code and summaries derived from papers may be used for your personal research and learning.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">6. Limitation of Liability</h2>
            <p>
              Paper2Code is provided "as is" without warranties of any kind, whether express or implied. We are not liable for any damages arising out of your use of the platform.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">7. Termination</h2>
            <p>
              We reserve the right to suspend or terminate accounts that violate these terms, with or without notice.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">8. Contact</h2>
            <p>
              If you have any questions about these Terms, please contact us at:{' '}
              <a href="mailto:legal@paper2code.com" className="text-[--accent-primary] hover:underline">legal@paper2code.com</a>
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
