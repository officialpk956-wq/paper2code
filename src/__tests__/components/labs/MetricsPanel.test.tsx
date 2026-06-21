import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MetricsPanel } from '@/components/labs/MetricsPanel';

describe('MetricsPanel', () => {
  it('shows loading state', () => {
    render(<MetricsPanel metrics={null} loading={true} error={null} labId="transformer" />);
    expect(screen.getByText('Running model…')).toBeInTheDocument();
  });

  it('shows error message', () => {
    render(<MetricsPanel metrics={null} loading={false} error="Connection refused" labId="transformer" />);
    expect(screen.getByText('Connection refused')).toBeInTheDocument();
  });

  it('shows placeholder when no metrics', () => {
    render(<MetricsPanel metrics={null} loading={false} error={null} labId="transformer" />);
    expect(screen.getByText('Adjust parameters to see live metrics.')).toBeInTheDocument();
  });

  it('renders the Live Metrics section label', () => {
    const metrics = { params_M: 85.3, total_flops_mflops: 5000, memory_mb: 341.2, latency_ms: 0.167 };
    render(<MetricsPanel metrics={metrics} loading={false} error={null} labId="transformer" />);
    expect(screen.getByText('Live Metrics')).toBeInTheDocument();
  });

  it('renders common metric rows when metrics provided', () => {
    const metrics = { params_M: 85.3, total_flops_mflops: 5000, memory_mb: 341.2, latency_ms: 0.167 };
    render(<MetricsPanel metrics={metrics} loading={false} error={null} labId="transformer" />);
    expect(screen.getByText('Parameters')).toBeInTheDocument();
    expect(screen.getByText('Total FLOPs')).toBeInTheDocument();
    expect(screen.getByText('Memory')).toBeInTheDocument();
  });

  it('renders transformer-specific metrics', () => {
    const metrics = {
      params_M: 85.3, total_flops_mflops: 5000, memory_mb: 341, latency_ms: 0.1,
      token_count: 128, head_dim: 64, num_heads: 8, attention_cost_mflops: 1000,
    };
    render(<MetricsPanel metrics={metrics} loading={false} error={null} labId="transformer" />);
    expect(screen.getByText('Sequence Tokens')).toBeInTheDocument();
    expect(screen.getByText('Head Dimension')).toBeInTheDocument();
    expect(screen.getByText('Attention Cost')).toBeInTheDocument();
  });

  it('renders ViT-specific token count label', () => {
    const metrics = {
      params_M: 86, total_flops_mflops: 17500, memory_mb: 350, latency_ms: 0.2,
      num_patches: 196, head_dim: 64, num_heads: 12, attention_cost_mflops: 5000,
    };
    render(<MetricsPanel metrics={metrics} loading={false} error={null} labId="vit" />);
    expect(screen.getByText('Token Count')).toBeInTheDocument();
  });

  it('renders CNN receptive field metric', () => {
    const metrics = {
      params_M: 1.2, total_flops_mflops: 100, memory_mb: 10, latency_ms: 0.05,
      receptive_field: 224,
    };
    render(<MetricsPanel metrics={metrics} loading={false} error={null} labId="cnn" />);
    expect(screen.getByText('Receptive Field')).toBeInTheDocument();
  });

  it('renders diffusion-specific metrics', () => {
    const metrics = {
      params_M: 30, total_flops_mflops: 10000, memory_mb: 120, latency_ms: 0.4,
      step_flops_mflops: 100, inference_flops_mflops: 100000,
    };
    render(<MetricsPanel metrics={metrics} loading={false} error={null} labId="diffusion" />);
    expect(screen.getByText('Per-Step FLOPs')).toBeInTheDocument();
    expect(screen.getByText('Full Inference')).toBeInTheDocument();
  });

  it('shows µs unit for sub-millisecond latency', () => {
    const metrics = { params_M: 1, total_flops_mflops: 10, memory_mb: 5, latency_ms: 0.5 };
    render(<MetricsPanel metrics={metrics} loading={false} error={null} labId="cnn" />);
    expect(screen.getByText('µs')).toBeInTheDocument();
  });

  it('shows ms unit for latency >= 1ms', () => {
    const metrics = { params_M: 1, total_flops_mflops: 10, memory_mb: 5, latency_ms: 5.0 };
    render(<MetricsPanel metrics={metrics} loading={false} error={null} labId="cnn" />);
    expect(screen.getByText('ms')).toBeInTheDocument();
  });

  it('does not show transformer metrics for cnn lab', () => {
    const metrics = { params_M: 1.2, total_flops_mflops: 100, memory_mb: 10, latency_ms: 0.05 };
    render(<MetricsPanel metrics={metrics} loading={false} error={null} labId="cnn" />);
    expect(screen.queryByText('Sequence Tokens')).not.toBeInTheDocument();
    expect(screen.queryByText('Attention Cost')).not.toBeInTheDocument();
  });
});
