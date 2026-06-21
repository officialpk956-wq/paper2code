import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ParameterControls } from '@/components/labs/ParameterControls';
import type { ParamDef } from '@/components/labs/ParameterControls';

const PARAMS: ParamDef[] = [
  { key: 'd_model', label: 'Model Dimension', min: 64, max: 1024, step: 64, default: 512 },
  { key: 'num_heads', label: 'Attention Heads', min: 1, max: 16, step: 1, default: 8 },
];

describe('ParameterControls', () => {
  it('renders the section label', () => {
    render(<ParameterControls params={PARAMS} values={{}} onChange={vi.fn()} />);
    expect(screen.getByText('Parameters')).toBeInTheDocument();
  });

  it('renders labels for each parameter', () => {
    render(<ParameterControls params={PARAMS} values={{ d_model: 512, num_heads: 8 }} onChange={vi.fn()} />);
    expect(screen.getByText('Model Dimension')).toBeInTheDocument();
    expect(screen.getByText('Attention Heads')).toBeInTheDocument();
  });

  it('labels use htmlFor for accessibility', () => {
    render(<ParameterControls params={PARAMS} values={{ d_model: 512, num_heads: 8 }} onChange={vi.fn()} />);
    const labels = screen.getAllByText(/Model Dimension|Attention Heads/);
    labels.forEach((label) => expect(label.tagName).toBe('LABEL'));
  });

  it('number inputs show current values', () => {
    render(<ParameterControls params={PARAMS} values={{ d_model: 256, num_heads: 4 }} onChange={vi.fn()} />);
    const spinners = screen.getAllByRole('spinbutton');
    expect(spinners[0]).toHaveValue(256);
    expect(spinners[1]).toHaveValue(4);
  });

  it('range sliders have aria-label matching param label', () => {
    render(<ParameterControls params={PARAMS} values={{ d_model: 512, num_heads: 8 }} onChange={vi.fn()} />);
    expect(screen.getByRole('slider', { name: 'Model Dimension' })).toBeInTheDocument();
    expect(screen.getByRole('slider', { name: 'Attention Heads' })).toBeInTheDocument();
  });

  it('range slider has correct aria-valuenow', () => {
    render(<ParameterControls params={PARAMS} values={{ d_model: 512, num_heads: 8 }} onChange={vi.fn()} />);
    const slider = screen.getByRole('slider', { name: 'Model Dimension' });
    expect(slider).toHaveAttribute('aria-valuenow', '512');
  });

  it('range slider has correct aria-valuemin and aria-valuemax', () => {
    render(<ParameterControls params={PARAMS} values={{ d_model: 512, num_heads: 8 }} onChange={vi.fn()} />);
    const slider = screen.getByRole('slider', { name: 'Model Dimension' });
    expect(slider).toHaveAttribute('aria-valuemin', '64');
    expect(slider).toHaveAttribute('aria-valuemax', '1024');
  });

  it('disables all inputs when disabled=true', () => {
    render(<ParameterControls params={PARAMS} values={{ d_model: 512, num_heads: 8 }} onChange={vi.fn()} disabled />);
    screen.getAllByRole('slider').forEach((el) => expect(el).toBeDisabled());
    screen.getAllByRole('spinbutton').forEach((el) => expect(el).toBeDisabled());
  });

  it('calls onChange with correct key and value when range changes', () => {
    const onChange = vi.fn();
    render(<ParameterControls params={PARAMS} values={{ d_model: 512, num_heads: 8 }} onChange={onChange} />);
    const slider = screen.getByRole('slider', { name: 'Model Dimension' });
    fireEvent.change(slider, { target: { value: '576' } });
    expect(onChange).toHaveBeenCalledWith('d_model', expect.any(Number));
  });

  it('calls onChange from number input with valid value', () => {
    const onChange = vi.fn();
    render(<ParameterControls params={PARAMS} values={{ d_model: 512, num_heads: 8 }} onChange={onChange} />);
    const [spinner] = screen.getAllByRole('spinbutton');
    fireEvent.change(spinner, { target: { value: '256' } });
    expect(onChange).toHaveBeenCalledWith('d_model', 256);
  });

  it('renders min/max labels for each parameter', () => {
    render(<ParameterControls params={PARAMS} values={{ d_model: 512, num_heads: 8 }} onChange={vi.fn()} />);
    expect(screen.getByText('64')).toBeInTheDocument();
    expect(screen.getByText('1,024')).toBeInTheDocument();
  });
});
