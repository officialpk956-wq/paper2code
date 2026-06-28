import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Privacy Policy | Paper2Code',
  description: 'Privacy Policy for Paper2Code',
};

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-[--bg-body] pt-24 pb-16 px-4">
      <div className="max-w-3xl mx-auto bg-[--bg-surface] border border-[--color-border] rounded-xl p-8 md:p-12 shadow-sm">
        <h1 className="text-3xl font-heading font-bold text-[--color-text-primary] mb-2">Privacy Policy</h1>
        <p className="text-sm text-[--color-text-tertiary] mb-10">Last updated: June 28, 2026</p>

        <div className="space-y-8 text-[--color-text-secondary] leading-relaxed">
          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">1. What We Collect</h2>
            <ul className="list-disc pl-5 space-y-2">
              <li><strong>Account info:</strong> We collect your name and email when you register.</li>
              <li><strong>Usage data:</strong> Pages visited and features used are tracked via PostHog analytics.</li>
              <li><strong>Paper files:</strong> Research papers you upload for processing.</li>
              <li><strong>Dojo code submissions:</strong> We store your code to track your learning progress.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">2. How We Use It</h2>
            <ul className="list-disc pl-5 space-y-2">
              <li>To provide and improve the platform and your learning experience.</li>
              <li>To send transactional emails (such as verification, password reset, and paper done notifications).</li>
              <li><strong>We never sell your data to third parties.</strong></li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">3. Data Storage</h2>
            <p className="mb-2">
              Your data is stored securely on our servers in our primary regions. Paper files are stored in secure cloud storage (R2/S3).
            </p>
            <p>
              Your submissions and progress data are retained as long as your account remains active.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">4. Your Rights (GDPR)</h2>
            <ul className="list-disc pl-5 space-y-2">
              <li><strong>Access & Correction:</strong> You have the right to access, correct, or delete your personal data.</li>
              <li><strong>Data Portability:</strong> You have the right to request a copy of your data in a portable format.</li>
              <li>To exercise these rights, please contact us at <a href="mailto:privacy@paper2code.com" className="text-[--accent-primary] hover:underline">privacy@paper2code.com</a>.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">5. Cookies</h2>
            <ul className="list-disc pl-5 space-y-2">
              <li><strong>Essential Cookies:</strong> We use essential cookies (and local storage) to manage your authentication (e.g., JWT).</li>
              <li><strong>Analytics Cookies:</strong> We use PostHog for analytics. You can opt out of these non-essential cookies via our cookie banner.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-semibold text-[--color-text-primary] mb-3">6. Contact Us</h2>
            <p>
              If you have any questions about this Privacy Policy, please contact us at:{' '}
              <a href="mailto:privacy@paper2code.com" className="text-[--accent-primary] hover:underline">privacy@paper2code.com</a>
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
