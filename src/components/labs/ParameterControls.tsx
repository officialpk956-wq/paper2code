'use client';

import { SectionLabel } from '@/components/ui/SectionLabel';

export interface ParamDef {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

interface ParameterControlsProps {
  params: ParamDef[];
  values: Record<string, number>;
  onChange: (key: string, value: number) => void;
  disabled?: boolean;
}

export function ParameterControls({
  params,
  values,
  onChange,
  disabled = false,
}: ParameterControlsProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <SectionLabel style={{ padding: '0 4px' }}>Parameters</SectionLabel>
      {params.map((p) => {
        const val = values[p.key] ?? p.default;
        const inputId = `param-number-${p.key}`;
        const rangeId = `param-range-${p.key}`;
        return (
          <div key={p.key} style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <label
                htmlFor={inputId}
                style={{
                  fontSize: '12px', fontWeight: 500,
                  color: 'var(--color-text-primary)',
                }}
              >
                {p.label}
              </label>
              <input
                id={inputId}
                type="number"
                value={val}
                min={p.min}
                max={p.max}
                step={p.step}
                disabled={disabled}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  if (!isNaN(v) && v >= p.min && v <= p.max) onChange(p.key, v);
                }}
                style={{
                  width: '72px', fontSize: '12px', fontWeight: 600,
                  textAlign: 'right', background: 'var(--bg-card)',
                  border: '1px solid var(--color-border)', borderRadius: '4px',
                  padding: '2px 6px', color: 'var(--color-text-primary)',
                  fontFamily: 'var(--font-mono, monospace)',
                  opacity: disabled ? 0.5 : 1,
                }}
              />
            </div>
            <div style={{ position: 'relative' }}>
              <input
                id={rangeId}
                type="range"
                aria-label={p.label}
                aria-valuemin={p.min}
                aria-valuemax={p.max}
                aria-valuenow={val}
                min={p.min}
                max={p.max}
                step={p.step}
                value={val}
                disabled={disabled}
                onChange={(e) => onChange(p.key, parseInt(e.target.value, 10))}
                style={{
                  width: '100%', accentColor: 'var(--accent-primary)',
                  opacity: disabled ? 0.4 : 1, cursor: disabled ? 'not-allowed' : 'pointer',
                }}
              />
              <div style={{
                display: 'flex', justifyContent: 'space-between',
                fontSize: '10px', color: 'var(--color-text-muted)', marginTop: '2px',
              }}>
                <span>{p.min.toLocaleString()}</span>
                <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>
                  {val.toLocaleString()}
                </span>
                <span>{p.max.toLocaleString()}</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
