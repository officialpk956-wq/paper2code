import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LabSelector } from '@/components/labs/LabSelector';

const LABS = [
  { id: 'transformer', name: 'Transformer', description: 'Self-attention mechanism', icon: '🧠' },
  { id: 'cnn', name: 'CNN', description: 'Convolutional networks', icon: '👁️' },
  { id: 'vit', name: 'ViT', description: 'Vision Transformer', icon: '🔲' },
];

describe('LabSelector', () => {
  it('renders all lab names', () => {
    render(<LabSelector labs={LABS} activeLabId="transformer" onSelect={vi.fn()} />);
    expect(screen.getByText('Transformer')).toBeInTheDocument();
    expect(screen.getByText('CNN')).toBeInTheDocument();
    expect(screen.getByText('ViT')).toBeInTheDocument();
  });

  it('renders the section label', () => {
    render(<LabSelector labs={LABS} activeLabId="transformer" onSelect={vi.fn()} />);
    expect(screen.getByText('Labs')).toBeInTheDocument();
  });

  it('renders lab icons', () => {
    render(<LabSelector labs={LABS} activeLabId="transformer" onSelect={vi.fn()} />);
    expect(screen.getByText('🧠')).toBeInTheDocument();
    expect(screen.getByText('👁️')).toBeInTheDocument();
  });

  it('calls onSelect with the lab id when a button is clicked', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<LabSelector labs={LABS} activeLabId="transformer" onSelect={onSelect} />);
    await user.click(screen.getByText('CNN'));
    expect(onSelect).toHaveBeenCalledWith('cnn');
  });

  it('calls onSelect with transformer id when Transformer is clicked', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<LabSelector labs={LABS} activeLabId="cnn" onSelect={onSelect} />);
    await user.click(screen.getByText('Transformer'));
    expect(onSelect).toHaveBeenCalledWith('transformer');
  });

  it('renders buttons for each lab', () => {
    render(<LabSelector labs={LABS} activeLabId="transformer" onSelect={vi.fn()} />);
    expect(screen.getAllByRole('button')).toHaveLength(3);
  });

  it('renders an empty list without crashing', () => {
    render(<LabSelector labs={[]} activeLabId="" onSelect={vi.fn()} />);
    expect(screen.getByText('Labs')).toBeInTheDocument();
    expect(screen.queryAllByRole('button')).toHaveLength(0);
  });

  it('passes description as button title', () => {
    render(<LabSelector labs={LABS} activeLabId="transformer" onSelect={vi.fn()} />);
    expect(screen.getByTitle('Self-attention mechanism')).toBeInTheDocument();
  });
});
