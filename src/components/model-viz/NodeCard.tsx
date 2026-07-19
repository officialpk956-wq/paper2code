'use client';

import { Handle, Position } from '@xyflow/react';
import { getOpColor } from '@/lib/model-viz-api';
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
              fontSize: 12,
              fontWeight: 600,
              color: '#f5f5f5',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {data.op_type}
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

          {data.params > 0 && (
            <div style={{ fontSize: 9, color: '#525252' }}>
              {data.params.toLocaleString()} params
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
