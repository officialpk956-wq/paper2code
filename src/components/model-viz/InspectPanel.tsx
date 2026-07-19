'use client';

import { X } from 'lucide-react';
import { getOpColor } from '@/lib/model-viz-api';
import type { ParsedNode } from '@/lib/model-viz-api';

type Props = {
  node: ParsedNode;
  onClose: () => void;
};

export default function InspectPanel({ node, onClose }: Props) {
  const color = getOpColor(node.op_type);
  const allOutputShapeEntries = Object.entries(node.output_shapes);

  return (
    <div
      style={{
        width: 300,
        height: '100%',
        background: '#0f0f0f',
        borderLeft: '1px solid #1a1a1a',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 14px',
          borderBottom: '1px solid #1a1a1a',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 3, height: 16, background: color, borderRadius: 2 }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: '#f5f5f5' }}>
            {node.op_type}
          </span>
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: '#525252',
            display: 'flex',
            padding: 2,
          }}
        >
          <X size={14} />
        </button>
      </div>

      {/* Body */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '12px 14px',
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
        }}
      >
        {/* Primary output shape */}
        <Section title="Output Shape">
          {node.primary_out_shape.length > 0 ? (
            <Mono>[{node.primary_out_shape.join(', ')}]</Mono>
          ) : (
            <Gray>Unknown</Gray>
          )}
        </Section>

        {/* All outputs (only if more than one) */}
        {allOutputShapeEntries.length > 1 && (
          <Section title="All Outputs">
            {allOutputShapeEntries.map(([name, shape]) => (
              <Row key={name}>
                <TensorName>{truncate(name, 22)}</TensorName>
                <Mono>[{shape.join(', ')}]</Mono>
              </Row>
            ))}
          </Section>
        )}

        {/* Input shapes */}
        {Object.keys(node.input_shapes).length > 0 && (
          <Section title="Input Shapes">
            {Object.entries(node.input_shapes).map(([name, shape]) => (
              <Row key={name}>
                <TensorName>{truncate(name, 22)}</TensorName>
                <Mono>[{shape.join(', ')}]</Mono>
              </Row>
            ))}
          </Section>
        )}

        {/* Parameters */}
        {node.params > 0 && (
          <Section title="Parameters">
            <Mono>{node.params.toLocaleString()}</Mono>
          </Section>
        )}

        {/* Attributes */}
        {Object.keys(node.attrs).length > 0 && (
          <Section title="Attributes">
            {Object.entries(node.attrs).map(([k, v]) => (
              <Row key={k}>
                <span style={{ fontSize: 11, color: '#737373', fontFamily: 'monospace' }}>{k}</span>
                <Mono style={{ textAlign: 'right', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {truncateAttr(v)}
                </Mono>
              </Row>
            ))}
          </Section>
        )}

        {/* Node name / internal label */}
        {node.label && node.label !== node.op_type && (
          <Section title="Internal Name">
            <span style={{ fontSize: 11, color: '#525252', fontFamily: 'monospace', wordBreak: 'break-all' }}>
              {node.label}
            </span>
          </Section>
        )}
      </div>
    </div>
  );
}

// ── small helpers ──────────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div
        style={{
          fontSize: 9,
          fontWeight: 700,
          color: '#404040',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          marginBottom: 6,
        }}
      >
        {title}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>{children}</div>
    </div>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
      {children}
    </div>
  );
}

function Mono({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <span style={{ fontFamily: 'monospace', fontSize: 11, color: '#a3a3a3', ...style }}>
      {children}
    </span>
  );
}

function TensorName({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        fontFamily: 'monospace',
        fontSize: 10,
        color: '#525252',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </span>
  );
}

function Gray({ children }: { children: React.ReactNode }) {
  return <span style={{ fontSize: 11, color: '#525252' }}>{children}</span>;
}

function truncate(s: string, n: number) {
  return s.length > n ? `…${s.slice(-n)}` : s;
}

function truncateAttr(v: unknown): string {
  const s = JSON.stringify(v);
  if (!s) return '';
  // Arrays like [1,1,2,2] — show as-is if short, else summarize
  if (Array.isArray(v)) {
    if (v.length <= 6) return `[${v.join(', ')}]`;
    return `[${v.slice(0, 4).join(', ')}, … +${v.length - 4}]`;
  }
  return s.length > 32 ? s.slice(0, 30) + '…' : s;
}
