'use client';

import { SectionLabel } from '@/components/ui/SectionLabel';

interface LabMetrics {
  params_M?: number;
  total_flops_mflops?: number;
  memory_mb?: number;
  latency_ms?: number;
  token_count?: number;
  num_patches?: number;
  head_dim?: number;
  num_heads?: number;
  attention_cost_mflops?: number;
  attention_note?: string;
  step_flops_mflops?: number;
  inference_flops_mflops?: number;
  training_flops_per_iter_mflops?: number;
  training_note?: string;
  receptive_field?: number;
  [key: string]: unknown;
}

interface MetricsPanelProps {
  metrics: LabMetrics | null;
  loading: boolean;
  error: string | null;
  labId: string;
}

function fmt(n: number | undefined, digits = 2): string {
  if (n == null) return '—';
  if (n >= 1000) return `${(n / 1000).toFixed(digits)}G`;
  if (n >= 1) return `${n.toFixed(digits)}`;
  return `${(n * 1000).toFixed(1)}m`;
}

function MetricRow({
  label, value, unit, accent, note,
}: {
  label: string;
  value: string;
  unit?: string;
  accent?: boolean;
  note?: string;
}) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: '2px',
      padding: '10px 12px', borderRadius: '8px',
      background: accent ? 'color-mix(in srgb, var(--accent-primary) 8%, transparent)' : 'var(--bg-card)',
      border: `1px solid ${accent ? 'color-mix(in srgb, var(--accent-primary) 20%, transparent)' : 'var(--color-border)'}`,
    }}>
      <div style={{
        fontSize: '10px', fontWeight: 600, letterSpacing: '0.08em',
        textTransform: 'uppercase', color: 'var(--color-text-muted)',
      }}>
        {label}
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
        <span style={{
          fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono, monospace)',
          color: accent ? 'var(--accent-primary)' : 'var(--color-text-primary)',
        }}>
          {value}
        </span>
        {unit && (
          <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>{unit}</span>
        )}
      </div>
      {note && (
        <div style={{ fontSize: '10px', color: 'var(--color-text-muted)', lineHeight: 1.4 }}>
          {note}
        </div>
      )}
    </div>
  );
}

export function MetricsPanel({ metrics, loading, error, labId }: MetricsPanelProps) {
  if (loading) {
    return (
      <div style={{ padding: '24px', color: 'var(--color-text-muted)', fontSize: '13px', textAlign: 'center' }}>
        <div style={{ marginBottom: '8px', fontSize: '24px' }}>⚙️</div>
        Running model…
      </div>
    );
  }
  if (error) {
    return (
      <div style={{
        padding: '16px', borderRadius: '8px',
        background: 'rgba(var(--color-error-rgb), 0.10)',
        border: '1px solid rgba(var(--color-error-rgb), 0.25)',
        color: 'var(--color-error)', fontSize: '12px',
      }}>
        {error}
      </div>
    );
  }
  if (!metrics) {
    return (
      <div style={{ padding: '24px', color: 'var(--color-text-muted)', fontSize: '13px', textAlign: 'center' }}>
        Adjust parameters to see live metrics.
      </div>
    );
  }

  const rows: { label: string; value: string; unit?: string; accent?: boolean; note?: string }[] = [];

  rows.push({
    label: 'Parameters',
    value: fmt(metrics.params_M),
    unit: 'M params',
    accent: true,
  });
  rows.push({
    label: 'Total FLOPs',
    value: fmt(metrics.total_flops_mflops),
    unit: 'MFLOPs',
  });
  rows.push({
    label: 'Memory',
    value: fmt(metrics.memory_mb),
    unit: 'MB',
  });
  rows.push({
    label: 'Latency (RTX 4090)',
    value: metrics.latency_ms != null
      ? metrics.latency_ms >= 1 ? `${metrics.latency_ms.toFixed(2)}` : `${(metrics.latency_ms * 1000).toFixed(1)}`
      : '—',
    unit: metrics.latency_ms != null && metrics.latency_ms >= 1 ? 'ms' : 'µs',
    note: 'Estimated at 30% GPU efficiency',
  });

  if (labId === 'transformer' || labId === 'vit') {
    const tokens = metrics.token_count ?? metrics.num_patches;
    if (tokens != null) {
      rows.push({ label: labId === 'vit' ? 'Token Count' : 'Sequence Tokens', value: String(tokens), unit: 'tokens' });
    }
    if (metrics.head_dim != null) {
      rows.push({ label: 'Head Dimension', value: String(metrics.head_dim), unit: `× ${metrics.num_heads ?? ''} heads` });
    }
    if (metrics.attention_cost_mflops != null) {
      rows.push({
        label: 'Attention Cost',
        value: fmt(metrics.attention_cost_mflops),
        unit: 'MFLOPs',
        note: metrics.attention_note,
      });
    }
  }

  if (labId === 'cnn' && metrics.receptive_field != null) {
    rows.push({ label: 'Receptive Field', value: `${metrics.receptive_field}×${metrics.receptive_field}`, unit: 'px' });
  }

  if (labId === 'diffusion') {
    if (metrics.step_flops_mflops != null) {
      rows.push({ label: 'Per-Step FLOPs', value: fmt(metrics.step_flops_mflops), unit: 'MFLOPs/step' });
    }
    if (metrics.inference_flops_mflops != null) {
      rows.push({ label: 'Full Inference', value: fmt(metrics.inference_flops_mflops), unit: 'MFLOPs total' });
    }
    if (metrics.training_note) {
      rows.push({ label: 'Scale Note', value: '', note: metrics.training_note });
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <SectionLabel style={{ padding: '0 4px', marginBottom: '4px' }}>
        Live Metrics
      </SectionLabel>
      {rows.map((r, i) => (
        <MetricRow key={i} {...r} />
      ))}
    </div>
  );
}
