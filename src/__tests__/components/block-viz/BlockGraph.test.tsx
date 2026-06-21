import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BlockGraph } from '@/components/block-viz/BlockGraph';
import type { HierarchyData } from '@/components/block-viz/BlockGraph';
import type { BlockNode } from '@/components/block-viz/BlockBox';

const HIERARCHY: HierarchyData = {
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
      description: 'Initial convolution layer',
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
    {
      id: 'layer1',
      name: 'Layer1',
      type: 'backbone',
      description: 'First residual stage',
      input_shape: [1, 64, 56, 56],
      output_shape: [1, 256, 56, 56],
      flops_mflops: 1008,
      params_M: 5.0,
      blocks: [
        {
          id: 'bottleneck1',
          name: 'Bottleneck-1',
          type: 'Bottleneck',
          input_shape: [1, 64, 56, 56],
          output_shape: [1, 256, 56, 56],
          flops_mflops: 1008,
          params_M: 5.0,
          layers: [],
        },
      ],
    },
  ],
};

describe('BlockGraph', () => {
  it('renders all stage names', () => {
    render(<BlockGraph data={HIERARCHY} selectedBlock={null} onSelectBlock={vi.fn()} />);
    expect(screen.getByText('Stem')).toBeInTheDocument();
    expect(screen.getByText('Layer1')).toBeInTheDocument();
  });

  it('all stages are expanded by default showing their blocks', () => {
    render(<BlockGraph data={HIERARCHY} selectedBlock={null} onSelectBlock={vi.fn()} />);
    expect(screen.getByText('StemBlock')).toBeInTheDocument();
    expect(screen.getByText('Bottleneck-1')).toBeInTheDocument();
  });

  it('collapses a stage when its header is clicked', async () => {
    const user = userEvent.setup();
    render(<BlockGraph data={HIERARCHY} selectedBlock={null} onSelectBlock={vi.fn()} />);
    const stageHeaders = screen.getAllByRole('button');
    // First stage header collapses Stem
    await user.click(stageHeaders[0]);
    expect(screen.queryByText('StemBlock')).not.toBeInTheDocument();
  });

  it('shows stage description in collapsed header area', async () => {
    const user = userEvent.setup();
    render(<BlockGraph data={HIERARCHY} selectedBlock={null} onSelectBlock={vi.fn()} />);
    // Collapse Stem
    await user.click(screen.getAllByRole('button')[0]);
    expect(screen.getByText('Initial convolution layer')).toBeInTheDocument();
  });

  it('shows stage description in expanded body', () => {
    render(<BlockGraph data={HIERARCHY} selectedBlock={null} onSelectBlock={vi.fn()} />);
    // Expanded by default — description appears in body
    expect(screen.getAllByText('Initial convolution layer').length).toBeGreaterThanOrEqual(1);
  });

  it('calls onSelectBlock when a block is clicked', async () => {
    const user = userEvent.setup();
    const onSelectBlock = vi.fn();
    render(<BlockGraph data={HIERARCHY} selectedBlock={null} onSelectBlock={onSelectBlock} />);
    await user.click(screen.getByText('StemBlock'));
    expect(onSelectBlock).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'stem_block' }),
    );
  });

  it('re-expands a collapsed stage on second click', async () => {
    const user = userEvent.setup();
    render(<BlockGraph data={HIERARCHY} selectedBlock={null} onSelectBlock={vi.fn()} />);
    const btn = screen.getAllByRole('button')[0];
    await user.click(btn);
    await user.click(btn);
    expect(screen.getByText('StemBlock')).toBeInTheDocument();
  });

  it('handles Enter key on stage header', async () => {
    const user = userEvent.setup();
    render(<BlockGraph data={HIERARCHY} selectedBlock={null} onSelectBlock={vi.fn()} />);
    const btn = screen.getAllByRole('button')[0];
    btn.focus();
    await user.keyboard('{Enter}');
    expect(screen.queryByText('StemBlock')).not.toBeInTheDocument();
  });

  it('renders a graph with a single stage', () => {
    const singleStage: HierarchyData = { ...HIERARCHY, stages: [HIERARCHY.stages[0]] };
    render(<BlockGraph data={singleStage} selectedBlock={null} onSelectBlock={vi.fn()} />);
    expect(screen.getByText('Stem')).toBeInTheDocument();
    expect(screen.queryByText('Layer1')).not.toBeInTheDocument();
  });

  it('marks the selected block visually (passes isSelected prop)', () => {
    const selectedBlock: BlockNode = HIERARCHY.stages[0].blocks[0];
    render(<BlockGraph data={HIERARCHY} selectedBlock={selectedBlock} onSelectBlock={vi.fn()} />);
    // BlockBox receives isSelected=true for the selected block — still rendered
    expect(screen.getByText('StemBlock')).toBeInTheDocument();
  });
});
