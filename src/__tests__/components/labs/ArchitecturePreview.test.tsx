import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ArchitecturePreview } from '@/components/labs/ArchitecturePreview';

const STEPS = [
  {
    id: 'emb', name: 'Embedding', type: 'EmbeddingLayer',
    input_shape: [1, 128], output_shape: [1, 128, 512],
    flops_mflops: 0, params_M: 5.12, memory_mb: 20,
    formula: 'V·d', severity: 'low',
  },
  {
    id: 'attn', name: 'Attention', type: 'MultiHeadAttention',
    input_shape: [1, 128, 512], output_shape: [1, 128, 512],
    flops_mflops: 1000, params_M: 1.05, memory_mb: 8,
    formula: 'Q·Kᵀ·V', severity: 'medium',
  },
];

describe('ArchitecturePreview', () => {
  it('shows empty state when no steps', () => {
    render(<ArchitecturePreview steps={[]} labName="Transformer" />);
    expect(screen.getByText('Adjust parameters to see the tensor flow')).toBeInTheDocument();
  });

  it('renders all step names', () => {
    render(<ArchitecturePreview steps={STEPS} labName="Transformer" />);
    expect(screen.getByText('Embedding')).toBeInTheDocument();
    expect(screen.getByText('Attention')).toBeInTheDocument();
  });

  it('renders step types', () => {
    render(<ArchitecturePreview steps={STEPS} labName="Transformer" />);
    expect(screen.getByText('MultiHeadAttention')).toBeInTheDocument();
  });

  it('renders the lab name in the header', () => {
    render(<ArchitecturePreview steps={STEPS} labName="Transformer" />);
    expect(screen.getByText('Transformer — Tensor Flow')).toBeInTheDocument();
  });

  it('renders step count in the subheader', () => {
    render(<ArchitecturePreview steps={STEPS} labName="Transformer" />);
    expect(screen.getByText(/2 nodes/)).toBeInTheDocument();
  });

  it('renders formula for each step', () => {
    render(<ArchitecturePreview steps={STEPS} labName="Transformer" />);
    expect(screen.getByText('V·d')).toBeInTheDocument();
    expect(screen.getByText('Q·Kᵀ·V')).toBeInTheDocument();
  });

  it('renders step index numbers', () => {
    render(<ArchitecturePreview steps={STEPS} labName="Transformer" />);
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
  });

  it('renders a single step without crashing', () => {
    render(<ArchitecturePreview steps={[STEPS[0]]} labName="CNN" />);
    expect(screen.getByText(/1 nodes/)).toBeInTheDocument();
    expect(screen.getByText('CNN — Tensor Flow')).toBeInTheDocument();
  });
});
