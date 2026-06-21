'use client';

import { SectionLabel } from '@/components/ui/SectionLabel';

interface LabMeta {
  id: string;
  name: string;
  description: string;
  icon: string;
}

interface LabSelectorProps {
  labs: LabMeta[];
  activeLabId: string;
  onSelect: (id: string) => void;
}

export function LabSelector({ labs, activeLabId, onSelect }: LabSelectorProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <SectionLabel style={{ padding: '0 4px', marginBottom: '4px' }}>
        Labs
      </SectionLabel>
      {labs.map((lab) => {
        const active = lab.id === activeLabId;
        return (
          <button
            key={lab.id}
            onClick={() => onSelect(lab.id)}
            title={lab.description}
            style={{
              display: 'flex', alignItems: 'center', gap: '10px',
              padding: '10px 12px', borderRadius: '8px', border: 'none',
              cursor: 'pointer', textAlign: 'left',
              transition: 'background var(--motion-fast), color var(--motion-fast), border-color var(--motion-fast)',
              background: active ? 'var(--bg-active)' : 'transparent',
              borderLeft: `2px solid ${active ? 'var(--accent-primary)' : 'transparent'}`,
              color: active ? 'var(--accent-primary)' : 'var(--color-text-secondary)',
            }}
            onMouseEnter={(e) => {
              if (!active) {
                (e.currentTarget as HTMLButtonElement).style.background = 'var(--bg-hover)';
                (e.currentTarget as HTMLButtonElement).style.color = 'var(--color-text-primary)';
              }
            }}
            onMouseLeave={(e) => {
              if (!active) {
                (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
                (e.currentTarget as HTMLButtonElement).style.color = 'var(--color-text-secondary)';
              }
            }}
          >
            <span style={{ fontSize: '18px', flexShrink: 0 }}>{lab.icon}</span>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: '13px', fontWeight: 600, lineHeight: 1.2 }}>{lab.name}</div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
