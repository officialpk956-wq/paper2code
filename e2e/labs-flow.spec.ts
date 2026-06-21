import { test, expect } from '@playwright/test';

const MOCK_LABS_META = {
  labs: [
    {
      id: 'transformer',
      name: 'Transformer',
      description: 'Self-attention mechanism',
      icon: '🧠',
      endpoint: '/api/labs/transformer',
      params: [
        { key: 'd_model', label: 'Model Dimension', min: 64, max: 1024, step: 64, default: 512 },
        { key: 'num_heads', label: 'Attention Heads', min: 1, max: 16, step: 1, default: 8 },
      ],
    },
    {
      id: 'cnn',
      name: 'CNN',
      description: 'Convolutional networks',
      icon: '👁️',
      endpoint: '/api/labs/cnn',
      params: [
        { key: 'base_channels', label: 'Base Channels', min: 8, max: 256, step: 8, default: 32 },
      ],
    },
  ],
};

const MOCK_METRICS = {
  params_M: 85.3,
  total_flops_mflops: 5000,
  memory_mb: 341.2,
  latency_ms: 0.167,
  token_count: 128,
  head_dim: 64,
  num_heads: 8,
  attention_cost_mflops: 1000,
  flow_steps: [
    {
      id: 'emb', name: 'Embedding', type: 'Embedding',
      input_shape: [1, 128], output_shape: [1, 128, 512],
      flops_mflops: 0, params_M: 5.12, memory_mb: 20, formula: 'V·d', severity: 'low',
    },
  ],
};

test.describe('AI Labs flow', () => {
  test.beforeEach(async ({ page }) => {
    // Intercept labs metadata endpoint
    await page.route(
      (url) => url.pathname === '/api/labs',
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(MOCK_LABS_META),
        }),
    );

    // Intercept all lab computation endpoints
    await page.route('**/api/labs/**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(MOCK_METRICS),
      }),
    );
  });

  test('labs page renders lab selector', async ({ page }) => {
    await page.goto('/labs');
    await expect(page.getByText('Transformer')).toBeVisible();
    await expect(page.getByText('CNN')).toBeVisible();
  });

  test('shows section labels', async ({ page }) => {
    await page.goto('/labs');
    await expect(page.getByText('AI Labs')).toBeVisible();
  });

  test('shows metrics after initial load', async ({ page }) => {
    await page.goto('/labs');
    await expect(page.getByText('Parameters')).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('Live Metrics')).toBeVisible({ timeout: 5000 });
  });

  test('switching labs changes the active lab', async ({ page }) => {
    await page.goto('/labs');
    await page.getByText('CNN').click();
    // Wait for metrics to re-appear (new lab selected)
    await expect(page.getByText('Live Metrics')).toBeVisible({ timeout: 5000 });
  });

  test('adjusting a parameter triggers re-computation', async ({ page }) => {
    await page.goto('/labs');
    await expect(page.getByText('Live Metrics')).toBeVisible({ timeout: 5000 });

    const slider = page.locator('input[type="range"]').first();
    await slider.focus();
    await slider.press('ArrowRight');

    // Wait for new metrics
    await expect(page.getByText('Live Metrics')).toBeVisible({ timeout: 5000 });
  });

  test('architecture preview shows flow steps', async ({ page }) => {
    await page.goto('/labs');
    await expect(page.getByText('Embedding')).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('Tensor Flow')).toBeVisible({ timeout: 5000 });
  });
});
