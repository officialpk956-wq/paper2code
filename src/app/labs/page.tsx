'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Spinner } from '@/components/ui/Spinner';
import { ThreeColumnLayout } from '@/components/layout/three-column-layout';
import { LabSelector } from '@/components/labs/LabSelector';
import { ParameterControls, type ParamDef } from '@/components/labs/ParameterControls';
import { MetricsPanel } from '@/components/labs/MetricsPanel';
import { ArchitecturePreview } from '@/components/labs/ArchitecturePreview';
import { ExperimentHistory, useExperimentHistory } from '@/components/labs/ExperimentHistory';

interface LabMeta {
  id: string;
  name: string;
  description: string;
  icon: string;
  params: ParamDef[];
  endpoint: string;
}

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

interface LabResult {
  params_M?: number;
  total_flops_mflops?: number;
  memory_mb?: number;
  latency_ms?: number;
  flow_steps?: FlowStep[];
  [key: string]: unknown;
}

const DEBOUNCE_MS = 800;

export default function LabsPage() {
  const [labs, setLabs] = useState<LabMeta[]>([]);
  const [activeLabId, setActiveLabId] = useState<string>('transformer');
  const [paramValues, setParamValues] = useState<Record<string, Record<string, number>>>({});
  const [metrics, setMetrics] = useState<LabResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { history, addExperiment, clearHistory } = useExperimentHistory();

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingRef = useRef<Record<string, number> | null>(null);

  // Load labs metadata once
  useEffect(() => {
    fetch('/api/labs')
      .then((r) => r.json())
      .then((data) => {
        if (data.labs) {
          setLabs(data.labs);
          const initValues: Record<string, Record<string, number>> = {};
          for (const lab of data.labs) {
            const defaults: Record<string, number> = {};
            for (const p of lab.params) defaults[p.key] = p.default;
            initValues[lab.id] = defaults;
          }
          setParamValues(initValues);
        }
      })
      .catch(() => { /* metadata failure is non-fatal */ });
  }, []);

  const activeLab = useMemo(() => labs.find((l) => l.id === activeLabId), [labs, activeLabId]);
  const currentParams = useMemo(() => paramValues[activeLabId] ?? {}, [paramValues, activeLabId]);

  const runLab = useCallback(
    async (labId: string, params: Record<string, number>) => {
      const lab = labs.find((l) => l.id === labId);
      if (!lab) return;

      setLoading(true);
      setError(null);

      try {
        const res = await fetch(lab.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          setError(data.error ?? `HTTP ${res.status}`);
          setMetrics(null);
        } else {
          setMetrics(data);
          // Save to experiment history
          addExperiment({
            labId: lab.id,
            labName: lab.name,
            params,
            params_M: data.params_M ?? 0,
            total_flops_mflops: data.total_flops_mflops ?? 0,
            memory_mb: data.memory_mb ?? 0,
            latency_ms: data.latency_ms ?? 0,
          });
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setMetrics(null);
      } finally {
        setLoading(false);
      }
    },
    [labs, addExperiment],
  );

  // Debounced trigger whenever params change
  const scheduleRun = useCallback(
    (labId: string, params: Record<string, number>) => {
      pendingRef.current = params;
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        if (pendingRef.current) runLab(labId, pendingRef.current);
      }, DEBOUNCE_MS);
    },
    [runLab],
  );

  // Auto-run when lab changes and defaults are ready
  useEffect(() => {
    if (!activeLab || Object.keys(currentParams).length === 0) return;
    scheduleRun(activeLabId, currentParams);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeLabId, activeLab]);

  const handleParamChange = useCallback((key: string, value: number) => {
    const next = { ...currentParams, [key]: value };
    setParamValues((prev) => ({ ...prev, [activeLabId]: next }));
    scheduleRun(activeLabId, next);
  }, [currentParams, activeLabId, scheduleRun]);

  const handleLabSelect = useCallback((id: string) => {
    setActiveLabId(id);
    setMetrics(null);
    setError(null);
  }, []);

  // Left panel
  const left = (
    <div style={{
      height: '100%', overflowY: 'auto', padding: '16px 12px',
      display: 'flex', flexDirection: 'column', gap: '24px',
    }}>
      <LabSelector labs={labs} activeLabId={activeLabId} onSelect={handleLabSelect} />

      {activeLab && (
        <ParameterControls
          params={activeLab.params}
          values={currentParams}
          onChange={handleParamChange}
          disabled={loading}
        />
      )}

      <ExperimentHistory
        history={history}
        onClear={clearHistory}
        activeLabId={activeLabId}
      />
    </div>
  );

  // Center panel — tensor flow visualization
  const center = (
    <div style={{ height: '100%', overflowY: 'auto' }}>
      {loading ? (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', height: '100%',
          color: 'var(--color-text-muted)', gap: '16px',
        }}>
          <Spinner size={32} />
          <div style={{ fontSize: '13px' }}>Running model forward pass…</div>
        </div>
      ) : (
        <ArchitecturePreview
          steps={metrics?.flow_steps ?? []}
          labName={activeLab?.name ?? ''}
        />
      )}
    </div>
  );

  // Right panel — metrics
  const right = (
    <div style={{ height: '100%', overflowY: 'auto', padding: '16px 12px' }}>
      <MetricsPanel
        metrics={metrics}
        loading={loading}
        error={error}
        labId={activeLabId}
      />
    </div>
  );

  return (
    <main style={{ height: 'calc(100vh - 56px)', overflow: 'hidden' }}>
      {/* Page header */}
      <div style={{
        padding: '16px 24px 12px',
        borderBottom: '1px solid var(--color-divider)',
        display: 'flex', alignItems: 'center', gap: '12px',
        background: 'var(--bg-body)', flexShrink: 0,
      }}>
        <span style={{ fontSize: '22px' }} aria-hidden="true">🧪</span>
        <div>
          <h1 style={{
            margin: 0, fontSize: '18px', fontWeight: 700,
            color: 'var(--color-text-primary)',
          }}>
            AI Labs
          </h1>
          <p style={{
            margin: 0, fontSize: '12px', color: 'var(--color-text-muted)',
          }}>
            Modify model parameters · observe real tensor shapes · FLOPs from actual PyTorch forward passes
          </p>
        </div>
        {loading && (
          <div style={{
            marginLeft: 'auto', fontSize: '11px', color: 'var(--color-text-muted)',
            display: 'flex', alignItems: 'center', gap: '6px',
          }}>
            <div
              className="animate-pulse"
              style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-primary)' }}
            />
            Computing…
          </div>
        )}
      </div>

      <div style={{ height: 'calc(100% - 68px)' }}>
        <ThreeColumnLayout
          left={left}
          center={center}
          right={right}
          leftWidth="w-72"
          rightWidth="w-80"
          showRight={true}
        />
      </div>
    </main>
  );
}
