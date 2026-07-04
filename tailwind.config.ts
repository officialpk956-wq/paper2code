import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        brand:     '#F97316',
        bg:        '#0A0A0A',
        surface:   '#111111',
        'surface-2': '#141414',
        border:    '#1A1A1A',
        muted:     '#A3A3A3',
        subtle:    '#525252',
        text:      '#FAFAFA',
        papers:    '#A78BFA',
        dojo:      '#F97316',
        learn:     '#60A5FA',
        labs:      '#34D399',
        easy:      '#4ADE80',
        medium:    '#FACC15',
        hard:      '#F87171',
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
