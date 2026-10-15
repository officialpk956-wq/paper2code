import { expect, test } from '@playwright/test';

test('saved ONNX graph renders its actual node and supports search', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('access_token', 'ui-fixture');
    localStorage.setItem('user_profile', JSON.stringify({ name: 'Tester', email: 'test@example.com' }));
  });
  await page.route('**/api/**', route => route.fulfill({ json: {} }));
  await page.route('**/api/model/1', route => process.env.E2E_NEGATIVE_CONTROL === '1'
    ? route.fulfill({ status: 404, json: { detail: 'Graph not found' } })
    : route.fulfill({ json: {
      id: 1, name: 'Regression Relu', format: 'onnx', created_at: '2026-10-15',
      graph_data: {
        nodes: [{ id: 'relu', op_type: 'Relu', label: 'Regression Relu Node', inputs: ['x'], outputs: ['y'],
          input_shapes: { x: [1, 3] }, output_shapes: { y: [1, 3] }, primary_out_shape: [1, 3], params: 0, attrs: {} }],
        edges: [], meta: { total_nodes: 1, total_edges: 0, total_params: 0, total_flops: 3,
          graph_inputs: { x: [1, 3] }, graph_outputs: { y: [1, 3] }, ir_version: 8, opset_version: 17 },
      },
    } }));
  const loaded = page.waitForResponse('**/api/model/1');
  const response = await page.goto('/model-viz?g=1');
  expect(response?.status()).toBe(200);
  expect((await loaded).status()).toBe(200);
  await expect(page.locator('.react-flow__node')).toHaveCount(1);
  await expect(page.locator('.react-flow__node')).toContainText('Relu');
  await expect(page.getByText('3 FLOPs', { exact: true })).toBeVisible();
  await page.getByPlaceholder('Search layers…').fill('relu');
  await expect(page.getByPlaceholder('Search layers…')).toHaveValue('relu');
  await page.locator('.react-flow__node').click();
  await expect(page.getByText('Regression Relu Node').first()).toBeVisible();
});
