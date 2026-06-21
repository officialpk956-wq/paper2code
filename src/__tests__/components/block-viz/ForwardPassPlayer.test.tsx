import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ForwardPassPlayer } from '@/components/block-viz/ForwardPassPlayer';
import type { ForwardPassStep } from '@/components/block-viz/ForwardPassPlayer';

const STEPS: ForwardPassStep[] = [
  {
    step: 0, node_id: 'emb', node_name: 'Embedding', type: 'EmbeddingLayer',
    input_shape: [1, 128], output_shape: [1, 128, 512],
    flops_mflops: 0, params_M: 5.12, severity: 'low', description: 'Token embedding lookup',
  },
  {
    step: 1, node_id: 'attn', node_name: 'MultiHeadAttention', type: 'AttentionLayer',
    input_shape: [1, 128, 512], output_shape: [1, 128, 512],
    flops_mflops: 1000, params_M: 1.05, severity: 'medium', description: 'Self-attention layer',
  },
  {
    step: 2, node_id: 'ffn', node_name: 'FeedForward', type: 'FeedForward',
    input_shape: [1, 128, 512], output_shape: [1, 128, 512],
    flops_mflops: 500, params_M: 0.5, severity: 'low', description: 'Feed-forward network',
  },
];

describe('ForwardPassPlayer', () => {
  it('returns null when steps array is empty', () => {
    const { container } = render(<ForwardPassPlayer steps={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders the first step node name', () => {
    render(<ForwardPassPlayer steps={STEPS} />);
    expect(screen.getByText('Embedding')).toBeInTheDocument();
  });

  it('renders initial step counter', () => {
    render(<ForwardPassPlayer steps={STEPS} />);
    expect(screen.getByText('Step 1 / 3')).toBeInTheDocument();
  });

  it('advances to step 2 on Step forward click', async () => {
    const user = userEvent.setup();
    render(<ForwardPassPlayer steps={STEPS} />);
    await user.click(screen.getByLabelText('Step forward'));
    expect(screen.getByText('MultiHeadAttention')).toBeInTheDocument();
    expect(screen.getByText('Step 2 / 3')).toBeInTheDocument();
  });

  it('retreats on Step back click', async () => {
    const user = userEvent.setup();
    render(<ForwardPassPlayer steps={STEPS} />);
    await user.click(screen.getByLabelText('Step forward'));
    await user.click(screen.getByLabelText('Step forward'));
    await user.click(screen.getByLabelText('Step back'));
    expect(screen.getByText('MultiHeadAttention')).toBeInTheDocument();
    expect(screen.getByText('Step 2 / 3')).toBeInTheDocument();
  });

  it('Restart button goes back to step 1', async () => {
    const user = userEvent.setup();
    render(<ForwardPassPlayer steps={STEPS} />);
    await user.click(screen.getByLabelText('Step forward'));
    await user.click(screen.getByLabelText('Restart'));
    expect(screen.getByText('Step 1 / 3')).toBeInTheDocument();
    expect(screen.getByText('Embedding')).toBeInTheDocument();
  });

  it('slider has correct aria attributes at step 1', () => {
    render(<ForwardPassPlayer steps={STEPS} />);
    const slider = screen.getByRole('slider');
    expect(slider).toHaveAttribute('aria-label', 'Forward pass step');
    expect(slider).toHaveAttribute('aria-valuenow', '0');
    expect(slider).toHaveAttribute('aria-valuemin', '0');
    expect(slider).toHaveAttribute('aria-valuemax', '2');
  });

  it('slider aria-valuenow updates when step advances', async () => {
    const user = userEvent.setup();
    render(<ForwardPassPlayer steps={STEPS} />);
    await user.click(screen.getByLabelText('Step forward'));
    expect(screen.getByRole('slider')).toHaveAttribute('aria-valuenow', '1');
  });

  it('renders all speed buttons', () => {
    render(<ForwardPassPlayer steps={STEPS} />);
    expect(screen.getByText('×0.5')).toBeInTheDocument();
    expect(screen.getByText('×1')).toBeInTheDocument();
    expect(screen.getByText('×2')).toBeInTheDocument();
    expect(screen.getByText('×4')).toBeInTheDocument();
  });

  it('×1 speed button is pressed by default', () => {
    render(<ForwardPassPlayer steps={STEPS} />);
    expect(screen.getByText('×1').closest('button')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('×0.5').closest('button')).toHaveAttribute('aria-pressed', 'false');
  });

  it('speed button updates aria-pressed when clicked', async () => {
    const user = userEvent.setup();
    render(<ForwardPassPlayer steps={STEPS} />);
    await user.click(screen.getByText('×2'));
    expect(screen.getByText('×2').closest('button')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('×1').closest('button')).toHaveAttribute('aria-pressed', 'false');
  });

  it('calls onStepChange when step changes', async () => {
    const user = userEvent.setup();
    const onStepChange = vi.fn();
    render(<ForwardPassPlayer steps={STEPS} onStepChange={onStepChange} />);
    await user.click(screen.getByLabelText('Step forward'));
    expect(onStepChange).toHaveBeenCalledWith(expect.objectContaining({ node_id: 'attn' }));
  });

  it('Step forward button is disabled at last step', async () => {
    const user = userEvent.setup();
    render(<ForwardPassPlayer steps={STEPS} />);
    await user.click(screen.getByLabelText('Step forward'));
    await user.click(screen.getByLabelText('Step forward'));
    expect(screen.getByLabelText('Step forward')).toBeDisabled();
  });

  it('Step back button is disabled at step 1', () => {
    render(<ForwardPassPlayer steps={STEPS} />);
    expect(screen.getByLabelText('Step back')).toBeDisabled();
  });

  it('renders step description', () => {
    render(<ForwardPassPlayer steps={STEPS} />);
    expect(screen.getByText('Token embedding lookup')).toBeInTheDocument();
  });

  it('speed group has accessible label', () => {
    render(<ForwardPassPlayer steps={STEPS} />);
    expect(screen.getByRole('group', { name: 'Playback speed' })).toBeInTheDocument();
  });
});
