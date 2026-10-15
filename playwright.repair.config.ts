import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  testMatch: '**/*.spec.ts',
  workers: 2,
  retries: 0,
  forbidOnly: !!process.env.CI,
  reporter: 'list',
  use: { baseURL: 'http://localhost:3001', ...devices['Desktop Chrome'], screenshot: 'only-on-failure', trace: 'retain-on-failure' },
  webServer: {
    command: 'npm run start -- --port 3001',
    url: 'http://localhost:3001',
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
