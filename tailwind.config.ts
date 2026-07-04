import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        // Canonical theme tokens live in globals.css (:root) — these names
        // resolve to the CSS variables so a future re-theme is a one-file edit.
        brand:        'var(--brand)',
        'brand-hover': 'var(--brand-hover)',
        'brand-2':    'var(--brand-2)',
        bg:           'var(--bg)',
        'bg-elevated': 'var(--bg-elevated)',
        surface:      'var(--surface)',
        'surface-2':  'var(--surface-2)',
        'surface-3':  'var(--surface-3)',
        border:       'var(--border-2)',
        'border-strong': 'var(--border-3)',
        muted:        '#A3A3A3',
        subtle:       'var(--subtle)',
        text:         'var(--text)',
        papers:       'var(--papers)',
        dojo:         'var(--dojo)',
        learn:        'var(--learn)',
        labs:         'var(--labs)',
        arch:         'var(--arch)',
        easy:         'var(--easy)',
        medium:       'var(--medium)',
        hard:         'var(--hard)',
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;
