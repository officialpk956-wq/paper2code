'use client';

import { Handle, Position } from '@xyflow/react';
import { getOpColor, formatFlops, severityColor } from '@/lib/model-viz-api';
import type { ParsedNode } from '@/lib/model-viz-api';

type NodeCardProps = {
  data: ParsedNode;
  selected?: boolean;
};

export default function NodeCard({ data, selected }: NodeCardProps) {
  const color = getOpColor(data.op_type);
  const shapeStr =
    data.primary_out_shape.length > 0
      ? `[${data.primary_out_shape.join(', ')}]`
      : '';
  const flopsStr = formatFlops(data.flops);
  // draw a cost dot only when a node is actually expensive — keeps cheap ops clean
  const showCost = data.severity === 'medium' || data.severity === 'high' || data.severity === 'critical';

  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        style={{ opacity: 0, width: 6, height: 6 }}
        isConnectable={false}
      />

      <div
        style={{
          display: 'flex',
          alignItems: 'stretch',
          width: 200,
          minHeight: 64,
          background: '#111111',
          border: `1px solid ${selected ? color : '#2a2a2a'}`,
          borderRadius: 8,
          overflow: 'hidden',
          boxShadow: selected ? `0 0 0 1px ${color}40` : 'none',
          cursor: 'pointer',
          transition: 'border-color 0.1s, box-shadow 0.1s',
        }}
      >
        {/* colored left stripe */}
        <div style={{ width: 4, background: color, flexShrink: 0 }} />

        {/* content */}
        <div
          style={{
            padding: '8px 10px',
            flex: 1,
            minWidth: 0,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            gap: 2,
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 12,
              fontWeight: 600,
              color: '#f5f5f5',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{data.op_type}</span>
            {showCost && (
              <span
                title={`${data.severity} compute cost`}
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: '50%',
                  flexShrink: 0,
                  background: severityColor(data.severity),
                }}
              />
            )}
          </div>

          {shapeStr && (
            <div
              style={{
                fontSize: 10,
                color: '#737373',
                fontFamily: 'monospace',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {shapeStr}
            </div>
          )}

          {(data.params > 0 || flopsStr) && (
            <div style={{ fontSize: 9, color: '#525252', display: 'flex', gap: 8 }}>
              {data.params > 0 && <span>{data.params.toLocaleString()} params</span>}
              {flopsStr && <span style={{ color: severityColor(data.severity) }}>{flopsStr}</span>}
            </div>
          )}
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        style={{ opacity: 0, width: 6, height: 6 }}
        isConnectable={false}
      />
    </>
  );
}
