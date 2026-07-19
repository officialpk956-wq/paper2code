'use client';

import { useState } from 'react';

type Props = {
  filename: string;
  onSubmit: (shape: number[]) => void;
  onCancel: () => void;
  disabled?: boolean;
};

/**
 * Shown after the user picks a .pt / .pth file.
 * Collects the input tensor shape (spatial dims, no batch) before we
 * send the file to the E2B sandbox.
 *
 * WHY: torch.fx.symbolic_trace needs a dummy input tensor to run the forward
 * pass for shape inference. We can't guess the shape from the weights alone.
 */
export default function InputShapeForm({ filename, onSubmit, onCancel, disabled }: Props) {
  const [raw, setRaw] = useState('3, 224, 224');
  const [err, setErr] = useState('');

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const parsed = raw
      .split(',')
      .map((s) => parseInt(s.trim(), 10));

    if (parsed.some((n) => isNaN(n) || n <= 0)) {
      setErr('Enter positive integers separated by commas, e.g. 3, 224, 224');
      return;
    }
    if (parsed.length < 1 || parsed.length > 4) {
      setErr('Shape must have 1–4 dimensions');
      return;
    }
    setErr('');
    onSubmit(parsed);
  }

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        marginTop: 24,
        padding: '18px 20px',
        background: '#111',
        border: '1px solid #2a2a2a',
        borderRadius: 12,
        display: 'flex',
        flexDirection: 'column',
        gap: 14,
      }}
    >
      {/* File badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          fontSize: 9,
          fontFamily: 'monospace',
          color: '#f97316',
          background: 'rgba(249,115,22,0.1)',
          border: '1px solid rgba(249,115,22,0.2)',
          padding: '2px 8px',
          borderRadius: 4,
          letterSpacing: '0.06em',
        }}>
          PyTorch
        </span>
        <span style={{ fontSize: 12, color: '#737373', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 260 }}>
          {filename}
        </span>
      </div>

      {/* Shape input */}
      <div>
        <label style={{ fontSize: 11, color: '#737373', display: 'block', marginBottom: 6 }}>
          Input tensor shape{' '}
          <span style={{ color: '#525252' }}>(spatial dims — batch=1 added automatically)</span>
        </label>
        <input
          value={raw}
          onChange={(e) => { setRaw(e.target.value); setErr(''); }}
          disabled={disabled}
          placeholder="3, 224, 224"
          style={{
            width: '100%',
            background: '#0a0a0a',
            border: `1px solid ${err ? 'rgba(239,68,68,0.5)' : '#2a2a2a'}`,
            borderRadius: 6,
            padding: '7px 10px',
            fontSize: 13,
            fontFamily: 'monospace',
            color: '#f5f5f5',
            outline: 'none',
            boxSizing: 'border-box',
          }}
        />
        {err && (
          <p style={{ fontSize: 11, color: '#f87171', marginTop: 5 }}>{err}</p>
        )}
      </div>

      {/* Presets */}
      <div>
        <p style={{ fontSize: 10, color: '#404040', marginBottom: 6, letterSpacing: '0.05em' }}>
          COMMON PRESETS
        </p>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              disabled={disabled}
              onClick={() => { setRaw(p.shape); setErr(''); }}
              style={{
                fontSize: 11,
                color: raw === p.shape ? '#A78BFA' : '#737373',
                background: raw === p.shape ? 'rgba(167,139,250,0.08)' : 'transparent',
                border: `1px solid ${raw === p.shape ? 'rgba(167,139,250,0.3)' : '#2a2a2a'}`,
                borderRadius: 5,
                padding: '3px 10px',
                cursor: 'pointer',
                transition: 'all 0.1s',
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Slowness warning */}
      <p style={{ fontSize: 11, color: '#525252', lineHeight: 1.5 }}>
        <span style={{ color: '#f59e0b' }}>Note:</span> PyTorch parsing runs in an isolated sandbox.
        The first run may take 2–4 min if torch is not pre-installed in the template.
        Subsequent runs are faster.
      </p>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          type="submit"
          disabled={disabled}
          style={{
            flex: 1,
            padding: '8px 0',
            background: disabled ? '#1a1a1a' : '#A78BFA',
            color: disabled ? '#525252' : '#fff',
            border: 'none',
            borderRadius: 8,
            fontSize: 13,
            fontWeight: 600,
            cursor: disabled ? 'default' : 'pointer',
            transition: 'background 0.1s',
          }}
        >
          {disabled ? 'Parsing…' : 'Parse Model'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={disabled}
          style={{
            padding: '8px 16px',
            background: 'none',
            color: '#737373',
            border: '1px solid #2a2a2a',
            borderRadius: 8,
            fontSize: 13,
            cursor: disabled ? 'default' : 'pointer',
          }}
        >
          Cancel
        </button>
      </div>
    </form>
  );
}

const PRESETS = [
  { label: 'ImageNet (3×224×224)', shape: '3, 224, 224' },
  { label: 'Small (3×128×128)',    shape: '3, 128, 128' },
  { label: 'Grayscale (1×28×28)', shape: '1, 28, 28' },
  { label: '1-D (512,)',          shape: '512' },
];
