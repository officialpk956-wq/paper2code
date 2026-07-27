'use client';

import { Handle, Position } from '@xyflow/react';
import { Layers } from 'lucide-react';
import { formatFlops, severityColor } from '@/lib/model-viz-api';
import type { MotifGroup } from '@/lib/model-viz-api';

type Props = { data: { group: MotifGroup }; selected?: boolean };

export default function GroupNode({ data }: Props) {
  const g = data.group;
  const sev = severityColor(g.severity);
  const flopsStr = formatFlops(g.flops);

  return (
    <>
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 6, height: 6 }} isConnectable={false} />

      {/* stacked-card look signals "this is a collapsed group; click to expand" */}
      <div style={{ position: 'relative', width: 200 }}>
        <div style={{ position: 'absolute', inset: 0, transform: 'translate(4px, 4px)', background: '#0d0d0d', border: '1px solid #262626', borderRadius: 8 }} />
        <div
          style={{
            position: 'relative',
            display: 'flex',
            alignItems: 'stretch',
            minHeight: 64,
            background: '#141414',
            border: `1px dashed ${sev}`,
            borderRadius: 8,
            overflow: 'hidden',
            cursor: 'pointer',
          }}
        >
          <div style={{ width: 4, background: sev, flexShrink: 0 }} />
          <div style={{ padding: '8px 10px', flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 3 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600, color: '#f5f5f5' }}>
              <Layers size={12} style={{ color: sev, flexShrink: 0 }} />
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{g.label}</span>
              <span style={{ marginLeft: 'auto', flexShrink: 0, fontSize: 11, fontWeight: 700, color: sev }}>×{g.repeat_count}</span>
            </div>
            <div style={{ fontSize: 9, color: '#525252', display: 'flex', gap: 8 }}>
              <span>{g.node_ids.length} ops</span>
              {g.params > 0 && <span>{g.params.toLocaleString()} params</span>}
              {flopsStr && <span style={{ color: sev }}>{flopsStr}</span>}
            </div>
            <div style={{ fontSize: 8, color: '#404040', letterSpacing: '0.04em' }}>click to expand</div>
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 6, height: 6 }} isConnectable={false} />
    </>
  );
}
