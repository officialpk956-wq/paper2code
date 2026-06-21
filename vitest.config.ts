import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/__tests__/setup.ts'],
    include: ['src/__tests__/**/*.test.{ts,tsx}'],
    exclude: ['node_modules', '.next'],
    coverage: {
      provider: 'v8',
      include: [
        'src/components/labs/**',
        'src/components/block-viz/**',
        'src/components/dojo/**',
        'src/components/learn/LearnHero.tsx',
        'src/components/learn/ContinueLearningCard.tsx',
        'src/components/learn/LearningPaths.tsx',
        'src/components/learn/DomainGrid.tsx',
        'src/components/learn/TrendingTopics.tsx',
        'src/components/learn/Recommendations.tsx',
        'src/components/learn/RecentlyAdded.tsx',
        'src/components/learn/KnowledgeGraphPreview.tsx',
        'src/app/api/labs/**',
        'src/app/api/papers/**/block-hierarchy/**',
        'src/app/api/papers/**/forward-pass/**',
        'src/app/api/dojo/**',
        'src/app/api/learn/**',
        'src/components/domain/**',
        'src/app/api/learn/domain/**',
        'src/components/topic/**',
        'src/app/api/learn/topic/**',
      ],
      exclude: [
        'node_modules',
        '.next',
        'src/__tests__/**',
        // Page-level components not unit-tested (covered by E2E)
        'src/components/block-viz/BlockVizPage.tsx',
        'src/components/dojo/DojoProblemList.tsx',
        'src/components/dojo/SubmissionHistory.tsx',
        'src/components/dojo/TheoryPanel.tsx',
      ],
      reporter: ['text', 'json', 'html', 'lcov'],
      thresholds: {
        statements: 70,
        branches: 70,
        functions: 70,
        lines: 70,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
