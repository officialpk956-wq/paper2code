import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BlockBox } from '@/components/block-viz/BlockBox';
import type { BlockNode } from '@/components/block-viz/BlockBox';

const BLOCK_WITH_LAYERS: BlockNode = {
  id: 'res1',
  name: 'ResidualBlock',
  type: 'Bottleneck',
  input_shape: [1, 64, 56, 56],
  output_shape: [1, 256, 56, 56],
  flops_mflops: 100,
  params_M: 0.5,
  layers: [
    {
      id: 'l1', name: 'Conv2d-1', type: 'Conv2d',
      input_shape: [1, 64, 56, 56], output_shape: [1, 64, 56, 56],
      flops_mflops: 50, params_M: 0.25,
    },
    {
      id: 'l2', name: 'BatchNorm-1', type: 'BatchNorm2d',
      input_shape: [1, 64, 56, 56], output_shape: [1, 64, 56, 56],
      flops_mflops: 5, params_M: 0.01,
    },
  ],
};

const BLOCK_NO_LAYERS: BlockNode = {
  id: 'stem',
  name: 'StemConv',
  type: 'Conv2d',
  input_shape: [1, 3, 224, 224],
  output_shape: [1, 64, 112, 112],
  flops_mflops: 200,
  params_M: 0.1,
  layers: [],
};

describe('BlockBox', () => {
  it('renders block name', () => {
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={vi.fn()} />);
    expect(screen.getByText('ResidualBlock')).toBeInTheDocument();
  });

  it('renders block type', () => {
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={vi.fn()} />);
    expect(screen.getByText('Bottleneck')).toBeInTheDocument();
  });

  it('layers are hidden by default', () => {
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={vi.fn()} />);
    expect(screen.queryByText('Conv2d-1')).not.toBeInTheDocument();
    expect(screen.queryByText('BatchNorm-1')).not.toBeInTheDocument();
  });

  it('expands to show layers when header clicked', async () => {
    const user = userEvent.setup();
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={vi.fn()} />);
    await user.click(screen.getByRole('button'));
    expect(screen.getByText('Conv2d-1')).toBeInTheDocument();
    expect(screen.getByText('BatchNorm-1')).toBeInTheDocument();
  });

  it('collapses layers on second click', async () => {
    const user = userEvent.setup();
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={vi.fn()} />);
    const btn = screen.getByRole('button');
    await user.click(btn);
    await user.click(btn);
    expect(screen.queryByText('Conv2d-1')).not.toBeInTheDocument();
  });

  it('calls onSelect when header clicked', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={onSelect} />);
    await user.click(screen.getByRole('button'));
    expect(onSelect).toHaveBeenCalledWith(BLOCK_WITH_LAYERS);
  });

  it('handles Enter key to expand', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={onSelect} />);
    const btn = screen.getByRole('button');
    btn.focus();
    await user.keyboard('{Enter}');
    expect(onSelect).toHaveBeenCalled();
    expect(screen.getByText('Conv2d-1')).toBeInTheDocument();
  });

  it('handles Space key to expand', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={onSelect} />);
    const btn = screen.getByRole('button');
    btn.focus();
    await user.keyboard(' ');
    expect(onSelect).toHaveBeenCalled();
    expect(screen.getByText('Conv2d-1')).toBeInTheDocument();
  });

  it('aria-expanded is false initially for block with layers', () => {
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={vi.fn()} />);
    expect(screen.getByRole('button')).toHaveAttribute('aria-expanded', 'false');
  });

  it('aria-expanded becomes true when expanded', async () => {
    const user = userEvent.setup();
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={vi.fn()} />);
    await user.click(screen.getByRole('button'));
    expect(screen.getByRole('button')).toHaveAttribute('aria-expanded', 'true');
  });

  it('aria-expanded is absent for block with no layers', () => {
    render(<BlockBox block={BLOCK_NO_LAYERS} isSelected={false} onSelect={vi.fn()} />);
    expect(screen.getByRole('button')).not.toHaveAttribute('aria-expanded');
  });

  it('starts expanded when defaultExpanded=true', () => {
    render(<BlockBox block={BLOCK_WITH_LAYERS} isSelected={false} onSelect={vi.fn()} defaultExpanded />);
    expect(screen.getByText('Conv2d-1')).toBeInTheDocument();
  });
});
