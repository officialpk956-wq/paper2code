// Central config for the legal pages. Edit these three values once and every
// legal page + the footer contact link updates.
//
// TODO(owner): replace CONTACT_EMAIL with your real support address before you
// rely on these documents, and have them reviewed by a lawyer for your
// jurisdiction — they are a solid, honest starting point, not certified advice.

export const LEGAL = {
  company: 'paper2code',
  contactEmail: 'support@paper2code.app',
  effectiveDate: 'July 5, 2026',
} as const;

export const SUBPROCESSORS: { name: string; purpose: string }[] = [
  { name: 'Vercel', purpose: 'Frontend hosting and content delivery' },
  { name: 'Render', purpose: 'Backend API hosting' },
  { name: 'Neon', purpose: 'PostgreSQL database (accounts, progress, uploaded-paper metadata)' },
  { name: 'Upstash', purpose: 'Redis cache and rate-limiting counters' },
  { name: 'Groq', purpose: 'Large-language-model inference for the AI Tutor and paper analysis' },
  { name: 'Cloudflare R2', purpose: 'Object storage for uploaded PDF files' },
  { name: 'Resend', purpose: 'Transactional email (verification, notifications)' },
];
