import { test, expect } from '@playwright/test';

const MOCK_HIERARCHY = {
  paper_id: 'resnet',
  name: 'ResNet-50',
  description: 'Deep Residual Learning for Image Recognition',
  input_shape: [1, 3, 224, 224],
  total_flops_mflops: 4100,
  total_params_M: 25.6,
  stages: [
    {
      id: 'stem',
      name: 'Stem',
      type: 'stem',
      description: 'Initial convolution',
      input_shape: [1, 3, 224, 224],
      output_shape: [1, 64, 112, 112],
      flops_mflops: 236,
      params_M: 0.09,
      blocks: [
        {
          id: 'stem_block',
          name: 'StemBlock',
          type: 'ConvBnRelu',
          input_shape: [1, 3, 224, 224],
          output_shape: [1, 64, 112, 112],
          flops_mflops: 236,
          params_M: 0.09,
          layers: [],
        },
      ],
    },
  ],
};

const MOCK_FORWARD_PASS = {
  architecture: 'resnet',
  steps: [
    {
      step: 0,
      node_id: 'stem_block',
      node_name: 'StemBlock',
      type: 'ConvBnRelu',
      input_shape: [1, 3, 224, 224],
      output_shape: [1, 64, 112, 112],
      flops_mflops: 236,
      params_M: 0.09,
      severity: 'low',
      description: 'Initial convolution block',
    },
  ],
};

test.describe('Block Visualization flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/papers/**/block-hierarchy', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(MOCK_HIERARCHY),
      }),
    );
    await page.route('**/api/papers/**/forward-pass', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(MOCK_FORWARD_PASS),
      }),
    );
  });

  test('block viz page for a valid paper loads without error', async ({ page }) => {
    await page.goto('/block-viz/resnet');
    await expect(page.locator('body')).toBeVisible();
    await expect(page).not.toHaveTitle(/Error/i);
  });

  test('shows hierarchy section with stage names', async ({ page }) => {
    await page.goto('/block-viz/resnet');
    await expect(page.getByText('Stem')).toBeVisible({ timeout: 8000 });
  });

  test('shows block names when stage is expanded', async ({ page }) => {
    await page.goto('/block-viz/resnet');
    // Stages are expanded by default
    await expect(page.getByText('StemBlock')).toBeVisible({ timeout: 8000 });
  });

  test('forward pass player shows step info', async ({ page }) => {
    await page.goto('/block-viz/resnet');
    await expect(page.getByText('StemBlock')).toBeVisible({ timeout: 8000 });
    // Forward pass player should display the first step
    await expect(page.getByText('Step 1 / 1')).toBeVisible({ timeout: 5000 });
  });

  test('clicking a block selects it', async ({ page }) => {
    await page.goto('/block-viz/resnet');
    await expect(page.getByText('StemBlock')).toBeVisible({ timeout: 8000 });
    await page.getByText('StemBlock').click();
    // After selection, the block details panel should show info
    await expect(page.getByText('Initial convolution')).toBeVisible({ timeout: 3000 });
  });
});
