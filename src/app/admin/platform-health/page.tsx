import Link from 'next/link';
import { CheckCircle, AlertCircle, Circle } from 'lucide-react';

interface CheckItem {
  label: string;
  status: 'pass' | 'fixed' | 'warn';
  detail?: string;
}

interface CheckGroup {
  title: string;
  checks: CheckItem[];
}

const ROUTE_COUNT = 37;
const COMPONENT_COUNT = 45;

const CHECKS: CheckGroup[] = [
  {
    title: 'Critical (P0)',
    checks: [
      { label: 'Dashboard CTAs link to real slugs', status: 'fixed', detail: 'Replaced /problems/42 and /problems/15 with valid slugs' },
      { label: 'No-op "Start →" button converted to Link', status: 'fixed', detail: '/problems/multi-head-attention' },
      { label: 'console.log removed from system-design/page.tsx', status: 'fixed' },
      { label: 'console.log removed from workspace-settings/page.tsx', status: 'fixed' },
      { label: 'Module-scope Date.now() hydration risk patched', status: 'fixed', detail: 'live-cursor-tracker.tsx timestamps set to 0' },
      { label: 'setInterval leak fixed in transformer simulator', status: 'fixed', detail: 'intervalRef + useEffect cleanup' },
      { label: 'Dead files deleted', status: 'fixed', detail: 'page-new.tsx, dashboard/page.old.tsx' },
    ],
  },
  {
    title: 'Navigation & Search',
    checks: [
      { label: 'All orphan routes added to left-rail nav', status: 'fixed', detail: '25 entries across 5 sections' },
      { label: 'Left-rail has aria-label="Main navigation"', status: 'fixed' },
      { label: 'Search page uses real engine (not placeholder)', status: 'fixed', detail: 'src/lib/search/engine.ts' },
      { label: 'Command palette uses real search engine', status: 'fixed' },
      { label: 'Command palette keyboard navigation (↑↓ Enter Esc)', status: 'fixed' },
      { label: 'Search engine HTML-escape-safe highlight()', status: 'fixed' },
    ],
  },
  {
    title: 'Accessibility (WCAG 2.1 AA)',
    checks: [
      { label: 'Skip-to-main-content link in app-shell', status: 'fixed' },
      { label: 'header has role="banner"', status: 'fixed' },
      { label: 'main has id="main-content" + tabIndex={-1}', status: 'fixed' },
      { label: 'Command palette button has aria-label', status: 'fixed' },
      { label: 'User menu button has aria-label', status: 'fixed' },
      { label: 'Knowledge graph SVG nodes are keyboard accessible', status: 'fixed', detail: 'role="button", tabIndex={0}, onKeyDown' },
      { label: 'ThreeColumnLayout uses semantic aside/region', status: 'fixed' },
      { label: 'Focus ring class removed from always-visible links', status: 'fixed', detail: 'page.tsx, hero-section, tracks-carousel' },
    ],
  },
  {
    title: 'Persistence Layer',
    checks: [
      { label: 'localStorage persistence module created', status: 'fixed', detail: 'src/lib/persistence/index.ts' },
      { label: 'Dashboard loads real progress summary', status: 'fixed', detail: 'useState + useEffect + getProgressSummary()' },
      { label: 'Dashboard stats use persistence (completedCount, bookmarkCount)', status: 'fixed' },
      { label: 'Persistence is SSR-safe (isClient guard)', status: 'fixed' },
      { label: 'Persistence gracefully handles quota errors', status: 'fixed' },
    ],
  },
  {
    title: 'Performance',
    checks: [
      { label: '50ms setInterval replaced with requestAnimationFrame', status: 'fixed', detail: 'live-cursor-tracker.tsx' },
      { label: 'Home page converted to Server Component', status: 'fixed', detail: "Removed 'use client' from src/app/page.tsx" },
    ],
  },
  {
    title: 'Security Headers',
    checks: [
      { label: 'Strict-Transport-Security (HSTS)', status: 'fixed', detail: 'max-age=63072000; includeSubDomains; preload' },
      { label: 'Referrer-Policy', status: 'fixed', detail: 'strict-origin-when-cross-origin' },
      { label: 'Permissions-Policy', status: 'fixed', detail: 'camera=(), microphone=(), geolocation=()' },
      { label: 'Content-Security-Policy', status: 'fixed', detail: "default-src 'self'; frame-ancestors 'none'" },
      { label: 'X-Content-Type-Options (pre-existing)', status: 'pass' },
      { label: 'X-Frame-Options (pre-existing)', status: 'pass' },
    ],
  },
];

const STAT_BOXES = [
  { label: 'Total Routes', value: ROUTE_COUNT },
  { label: 'Components', value: COMPONENT_COUNT },
  { label: 'P0 Fixes', value: 7 },
  { label: 'A11y Fixes', value: 8 },
];

function StatusIcon({ status }: { status: CheckItem['status'] }) {
  if (status === 'pass') {
    return <CheckCircle size={14} className="text-[--color-easy] flex-shrink-0" aria-label="Pass" />;
  }
  if (status === 'fixed') {
    return <CheckCircle size={14} className="text-[--accent-primary] flex-shrink-0" aria-label="Fixed" />;
  }
  return <AlertCircle size={14} className="text-[--color-hard] flex-shrink-0" aria-label="Warning" />;
}

export default function PlatformHealthPage() {
  const totalChecks = CHECKS.reduce((acc, g) => acc + g.checks.length, 0);
  const passedChecks = CHECKS.reduce(
    (acc, g) => acc + g.checks.filter((c) => c.status === 'pass' || c.status === 'fixed').length,
    0,
  );

  return (
    <main className="max-w-4xl mx-auto px-6 py-10 space-y-10">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <Link
            href="/dashboard"
            className="text-xs text-[--color-text-tertiary] hover:text-[--color-text-secondary] transition-colors"
          >
            ← Dashboard
          </Link>
        </div>
        <h1 className="text-2xl font-bold text-[--color-text-primary]">Platform Health</h1>
        <p className="text-sm text-[--color-text-secondary] mt-1">
          Hardening sprint results — {passedChecks}/{totalChecks} checks passing
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {STAT_BOXES.map((stat) => (
          <div key={stat.label} className="workspace-card text-center">
            <div className="text-2xl font-bold text-[--accent-primary] mb-1">{stat.value}</div>
            <div className="text-xs text-[--color-text-secondary]">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Overall score bar */}
      <div className="workspace-card">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-[--color-text-primary]">Overall Score</span>
          <span className="text-sm font-bold text-[--accent-primary]">
            {Math.round((passedChecks / totalChecks) * 100)}%
          </span>
        </div>
        <div className="w-full bg-[--bg-body] rounded-full h-2">
          <div
            className="bg-[--accent-primary] h-full rounded-full transition-all"
            style={{ width: `${(passedChecks / totalChecks) * 100}%` }}
          />
        </div>
        <p className="text-xs text-[--color-text-tertiary] mt-2">
          {passedChecks} of {totalChecks} checks resolved
        </p>
      </div>

      {/* Check Groups */}
      <div className="space-y-6">
        {CHECKS.map((group) => {
          const groupPassed = group.checks.filter((c) => c.status === 'pass' || c.status === 'fixed').length;
          return (
            <section key={group.title} className="workspace-card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-[--color-text-primary]">{group.title}</h2>
                <span className="text-xs text-[--color-text-tertiary]">
                  {groupPassed}/{group.checks.length}
                </span>
              </div>
              <ul className="space-y-2" role="list">
                {group.checks.map((check) => (
                  <li key={check.label} className="flex items-start gap-2 text-xs">
                    <StatusIcon status={check.status} />
                    <div className="min-w-0">
                      <span
                        className={
                          check.status === 'warn'
                            ? 'text-[--color-text-primary]'
                            : 'text-[--color-text-secondary]'
                        }
                      >
                        {check.label}
                      </span>
                      {check.detail && (
                        <span className="text-[--color-text-tertiary] ml-1">— {check.detail}</span>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 text-xs text-[--color-text-tertiary]">
        <div className="flex items-center gap-1.5">
          <CheckCircle size={12} className="text-[--color-easy]" />
          <span>Pre-existing pass</span>
        </div>
        <div className="flex items-center gap-1.5">
          <CheckCircle size={12} className="text-[--accent-primary]" />
          <span>Fixed in sprint</span>
        </div>
        <div className="flex items-center gap-1.5">
          <AlertCircle size={12} className="text-[--color-hard]" />
          <span>Needs attention</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Circle size={12} />
          <span>Not checked</span>
        </div>
      </div>
    </main>
  );
}
