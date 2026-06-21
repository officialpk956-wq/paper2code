'use client';

interface FlowStep {
  id: string;
  name: string;
  type: string;
  input_shape: number[];
  output_shape: number[];
  flops_mflops: number;
  params_M: number;
  memory_mb: number;
  formula: string;
  severity: string;
}

interface ArchitecturePreviewProps {
  steps: FlowStep[];
  labName: string;
}

const SEVERITY_COLOR: Record<string, string> = {
  low:    'var(--color-severity-low)',
  medium: 'var(--color-severity-medium)',
  high:   'var(--color-severity-high)',
  info:   'var(--color-text-muted)',
  none:   'var(--color-text-muted)',
};

function shapeStr(shape: number[]): string {
  if (!shape || shape.length === 0) return '—';
  return `[${shape.join(', ')}]`;
}

function ShapeArrow({ from, to }: { from: number[]; to: number[] }) {
  const fromStr = shapeStr(from);
  const toStr = shapeStr(to);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
      <span style={{
        fontFamily: 'var(--font-mono, monospace)', color: 'var(--color-text-muted)',
        background: 'var(--bg-body)', padding: '1px 6px', borderRadius: '4px',
      }}>
        {fromStr}
      </span>
      <span style={{ color: 'var(--color-text-muted)' }}>→</span>
      <span style={{
        fontFamily: 'var(--font-mono, monospace)', color: 'var(--accent-primary)',
        background: 'var(--bg-body)', padding: '1px 6px', borderRadius: '4px',
      }}>
        {toStr}
      </span>
    </div>
  );
}

function FlopsBar({ mflops, maxMflops }: { mflops: number; maxMflops: number }) {
  const pct = maxMflops > 0 ? Math.max(2, (mflops / maxMflops) * 100) : 0;
  const color = mflops >= 1000 ? 'var(--color-severity-high)' : mflops >= 100 ? 'var(--color-severity-medium)' : 'var(--color-severity-low)';
  return (
    <div style={{
      width: '100%', height: '3px', background: 'var(--color-border)',
      borderRadius: '2px', overflow: 'hidden',
    }}>
      <div style={{
        width: `${pct}%`, height: '100%', background: color,
        borderRadius: '2px', transition: 'width 0.3s ease',
      }} />
    </div>
  );
}

export function ArchitecturePreview({ steps, labName }: ArchitecturePreviewProps) {
  if (!steps || steps.length === 0) {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        height: '100%', color: 'var(--color-text-muted)', fontSize: '14px', gap: '12px',
      }}>
        <div style={{ fontSize: '40px' }}>🧪</div>
        <div>Adjust parameters to see the tensor flow</div>
      </div>
    );
  }

  const maxFlops = Math.max(...steps.map((s) => s.flops_mflops), 1);
  const totalFlops = steps.reduce((acc, s) => acc + s.flops_mflops, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0', height: '100%', overflowY: 'auto' }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px 12px',
        borderBottom: '1px solid var(--color-divider)',
        flexShrink: 0,
      }}>
        <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--color-text-primary)' }}>
          {labName} — Tensor Flow
        </div>
        <div style={{ fontSize: '11px', color: 'var(--color-text-muted)', marginTop: '2px' }}>
          {steps.length} nodes · {totalFlops >= 1000
            ? `${(totalFlops / 1000).toFixed(2)} GFLOPs`
            : `${totalFlops.toFixed(1)} MFLOPs`} total
        </div>
      </div>

      {/* Flow nodes */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {steps.map((step, i) => (
          <div key={step.id}>
            {/* Connector line between nodes */}
            {i > 0 && (
              <div style={{
                display: 'flex', justifyContent: 'center', padding: '2px 0',
              }}>
                <div style={{ width: '1px', height: '12px', background: 'var(--color-border)' }} />
              </div>
            )}
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--color-border)',
              borderRadius: '8px',
              padding: '10px 12px',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
            }}>
              {/* Node header */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
                  <span style={{
                    fontSize: '10px', fontWeight: 700, padding: '1px 6px',
                    background: 'var(--bg-active)', borderRadius: '4px',
                    color: 'var(--color-text-muted)', flexShrink: 0,
                  }}>
                    {i + 1}
                  </span>
                  <span style={{
                    fontSize: '13px', fontWeight: 600,
                    color: 'var(--color-text-primary)', overflow: 'hidden',
                    textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {step.name}
                  </span>
                  <span style={{
                    fontSize: '10px', color: 'var(--color-text-muted)',
                    background: 'var(--bg-body)', padding: '1px 5px', borderRadius: '3px',
                    flexShrink: 0,
                  }}>
                    {step.type}
                  </span>
                </div>
                <div style={{
                  fontSize: '11px', fontWeight: 700,
                  fontFamily: 'var(--font-mono, monospace)',
                  color: SEVERITY_COLOR[step.severity] ?? 'var(--color-text-muted)',
                  flexShrink: 0,
                }}>
                  {step.flops_mflops >= 1000
                    ? `${(step.flops_mflops / 1000).toFixed(2)}G`
                    : `${step.flops_mflops.toFixed(1)}M`}
                </div>
              </div>

              {/* Shape flow */}
              <ShapeArrow from={step.input_shape} to={step.output_shape} />

              {/* FLOPs bar */}
              <FlopsBar mflops={step.flops_mflops} maxMflops={maxFlops} />

              {/* Formula */}
              {step.formula && (
                <div style={{
                  fontSize: '10px', color: 'var(--color-text-muted)',
                  fontFamily: 'var(--font-mono, monospace)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {step.formula}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
