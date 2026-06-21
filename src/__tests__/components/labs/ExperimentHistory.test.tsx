import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ExperimentHistory } from '@/components/labs/ExperimentHistory';
import type { LabExperiment } from '@/components/labs/ExperimentHistory';

function makeExp(id: string, labId: string, overrides?: Partial<LabExperiment>): LabExperiment {
  return {
    id,
    labId,
    labName: labId === 'transformer' ? 'Transformer' : 'CNN',
    params: { d_model: 512 },
    params_M: 85.3,
    total_flops_mflops: 5000,
    memory_mb: 341.0,
    latency_ms: 0.167,
    timestamp: Date.now() - 60_000,
    ...overrides,
  };
}

describe('ExperimentHistory', () => {
  it('shows empty state when no experiments', () => {
    render(<ExperimentHistory history={[]} onClear={vi.fn()} activeLabId="transformer" />);
    expect(screen.getByText('No runs yet for this lab.')).toBeInTheDocument();
  });

  it('shows empty state when experiments exist for other lab only', () => {
    render(
      <ExperimentHistory history={[makeExp('1', 'cnn')]} onClear={vi.fn()} activeLabId="transformer" />,
    );
    expect(screen.getByText('No runs yet for this lab.')).toBeInTheDocument();
  });

  it('renders section label with count when filtered experiments exist', () => {
    const history = [makeExp('1', 'transformer'), makeExp('2', 'cnn')];
    render(<ExperimentHistory history={history} onClear={vi.fn()} activeLabId="transformer" />);
    expect(screen.getByText('History (1)')).toBeInTheDocument();
  });

  it('shows params_M formatted', () => {
    render(
      <ExperimentHistory history={[makeExp('1', 'transformer')]} onClear={vi.fn()} activeLabId="transformer" />,
    );
    expect(screen.getByText('85.30M')).toBeInTheDocument();
  });

  it('shows memory with MB suffix', () => {
    render(
      <ExperimentHistory history={[makeExp('1', 'transformer')]} onClear={vi.fn()} activeLabId="transformer" />,
    );
    expect(screen.getByText(/341\.0 MB/)).toBeInTheDocument();
  });

  it('shows latency in µs for sub-millisecond', () => {
    render(
      <ExperimentHistory history={[makeExp('1', 'transformer', { latency_ms: 0.5 })]} onClear={vi.fn()} activeLabId="transformer" />,
    );
    expect(screen.getByText(/µs/)).toBeInTheDocument();
  });

  it('shows latency in ms for >= 1ms', () => {
    render(
      <ExperimentHistory history={[makeExp('1', 'transformer', { latency_ms: 2.5 })]} onClear={vi.fn()} activeLabId="transformer" />,
    );
    expect(screen.getByText(/2\.50ms/)).toBeInTheDocument();
  });

  it('calls onClear when Clear button is clicked', async () => {
    const user = userEvent.setup();
    const onClear = vi.fn();
    render(
      <ExperimentHistory history={[makeExp('1', 'transformer')]} onClear={onClear} activeLabId="transformer" />,
    );
    await user.click(screen.getByText('Clear'));
    expect(onClear).toHaveBeenCalledOnce();
  });

  it('shows multiple experiments for active lab', () => {
    const history = [makeExp('1', 'transformer'), makeExp('2', 'transformer'), makeExp('3', 'cnn')];
    render(<ExperimentHistory history={history} onClear={vi.fn()} activeLabId="transformer" />);
    expect(screen.getByText('History (2)')).toBeInTheDocument();
  });

  it('renders param config snapshot', () => {
    render(
      <ExperimentHistory history={[makeExp('1', 'transformer')]} onClear={vi.fn()} activeLabId="transformer" />,
    );
    expect(screen.getByText(/d_model=512/)).toBeInTheDocument();
  });
});
