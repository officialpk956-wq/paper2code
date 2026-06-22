'use client';

import { useEffect, useState } from 'react';
import { SectionLabel } from '@/components/ui/SectionLabel';

export interface LabExperiment {
  id: string;
  labId: string;
  labName: string;
  params: Record<string, number>;
  params_M: number;
  total_flops_mflops: number;
  memory_mb: number;
  latency_ms: number;
  timestamp: number;
}

const MAX_HISTORY = 20;
const STORAGE_KEY = 'paper2code:lab:experiments';

function loadHistory(): LabExperiment[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(experiments: LabExperiment[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(experiments.slice(0, MAX_HISTORY)));
  } catch { /* localStorage full */ }
}

export function useExperimentHistory() {
  const [history, setHistory] = useState<LabExperiment[]>([]);

  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const addExperiment = (exp: Omit<LabExperiment, 'id' | 'timestamp'>) => {
    const entry: LabExperiment = {
      ...exp,
      id: `${exp.labId}-${Date.now()}`,
      timestamp: Date.now(),
    };
    setHistory((prev) => {
      const next = [entry, ...prev].slice(0, MAX_HISTORY);
      saveHistory(next);
      return next;
    });
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem(STORAGE_KEY);
  };

  return { history, addExperiment, clearHistory };
}

interface ExperimentHistoryProps {
  history: LabExperiment[];
  onClear: () => void;
  activeLabId: string;
}

function relativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.floor(m / 60)}h ago`;
}

export function ExperimentHistory({ history, onClear, activeLabId }: ExperimentHistoryProps) {
  const filtered = history.filter((e) => e.labId === activeLabId);

  if (filtered.length === 0) {
    return (
      <div style={{
        padding: '16px 4px', color: 'var(--color-text-muted)',
        fontSize: '12px', textAlign: 'center',
      }}>
        No runs yet for this lab.
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 4px',
      }}>
        <SectionLabel>History ({filtered.length})</SectionLabel>
        <button
          onClick={onClear}
          style={{
            fontSize: '10px', color: 'var(--color-text-muted)', background: 'none',
            border: 'none', cursor: 'pointer', padding: '2px 4px',
          }}
        >
          Clear
        </button>
      </div>
      <div style={{ maxHeight: '280px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {filtered.map((exp, i) => (
          <div
            key={exp.id}
            style={{
              padding: '8px 10px', borderRadius: '6px',
              background: i === 0 ? 'var(--bg-active)' : 'var(--bg-card)',
              border: `1px solid ${i === 0 ? 'var(--accent-primary)' : 'var(--color-border)'}`,
              fontSize: '11px',
            }}
          >
            <div style={{
              display: 'flex', justifyContent: 'space-between',
              color: 'var(--color-text-muted)', marginBottom: '4px',
            }}>
              <span>#{filtered.length - i}</span>
              <span>{relativeTime(exp.timestamp)}</span>
            </div>
            <div style={{
              display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3px',
              fontFamily: 'var(--font-mono, monospace)',
            }}>
              <span style={{ color: 'var(--color-text-muted)' }}>params:</span>
              <span style={{ color: 'var(--accent-primary)', fontWeight: 700 }}>
                {exp.params_M.toFixed(2)}M
              </span>
              <span style={{ color: 'var(--color-text-muted)' }}>flops:</span>
              <span style={{ color: 'var(--color-text-primary)' }}>
                {exp.total_flops_mflops >= 1000
                  ? `${(exp.total_flops_mflops / 1000).toFixed(2)}G`
                  : `${exp.total_flops_mflops.toFixed(1)}M`}
              </span>
              <span style={{ color: 'var(--color-text-muted)' }}>mem:</span>
              <span style={{ color: 'var(--color-text-primary)' }}>{exp.memory_mb.toFixed(1)} MB</span>
              <span style={{ color: 'var(--color-text-muted)' }}>latency:</span>
              <span style={{ color: 'var(--color-text-primary)' }}>
                {exp.latency_ms >= 1
                  ? `${exp.latency_ms.toFixed(2)}ms`
                  : `${(exp.latency_ms * 1000).toFixed(1)}µs`}
              </span>
            </div>
            {/* Config snapshot */}
            <div style={{
              marginTop: '4px', paddingTop: '4px',
              borderTop: '1px solid var(--color-divider)',
              color: 'var(--color-text-muted)', fontSize: '10px',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {Object.entries(exp.params).map(([k, v]) => `${k}=${v}`).join(' · ')}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
