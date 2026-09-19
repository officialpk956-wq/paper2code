import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/__tests__/setup.ts'],
    include: ['src/__tests__/**/*.test.{ts,tsx}', 'src/**/*.test.{ts,tsx}'],
    exclude: ['node_modules', '.next'],
    coverage: {
      provider: 'v8',
      // The tree was rebuilt in July 2026; the previous list named 21 paths
      // that no longer existed, so coverage silently measured nothing while
      // a committed three-month-old report kept reporting the old numbers.
      include: [
        'src/lib/**/*.{ts,tsx}',
        'src/components/**/*.{ts,tsx}',
        'src/app/**/*.{ts,tsx}',
        // src/data is static content (prerequisites.ts alone is ~9,900 lines
        // of arrays); counting it as coverable code hides the real number.
      ],
      exclude: [
        'node_modules',
        '.next',
        'src/__tests__/**',
        'src/**/*.test.{ts,tsx}',
        'src/**/*.d.ts',
      ],
      reporter: ['text', 'json', 'html', 'lcov'],
      // Ratchet, not aspiration. The old 70% "passed" because the include
      // list matched no files. Measured on 2026-09-19 against the real tree,
      // code only (static data excluded -- it inflated lines to 33% because
      // array literals execute on import): 16.3% lines, 34.3% functions,
      // 53.9% branches. These floors fail on any regression from that; raise
      // them as tests land, never lower them.
      thresholds: {
        statements: 16,
        branches: 53,
        functions: 34,
        lines: 16,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
