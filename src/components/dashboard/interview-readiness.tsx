'use client';

import { motion } from 'framer-motion';

const AXES = [
  { label: 'ML Theory',     value: 72, color: '#8B5CF6' },
  { label: 'Coding',        value: 58, color: '#06B6D4' },
  { label: 'System Design', value: 35, color: '#10B981' },
  { label: 'Behavioral',    value: 80, color: '#F59E0B' },
  { label: 'Research',      value: 45, color: '#EC4899' },
];

/* Pentagon radar chart */
function RadarChart({ axes }: { axes: typeof AXES }) {
  const CX = 80, CY = 80, R = 62;
  const n = axes.length;
  const toXY = (i: number, r: number) => {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2;
    return { x: CX + r * Math.cos(angle), y: CY + r * Math.sin(angle) };
  };

  /* Grid rings */
  const rings = [0.25, 0.5, 0.75, 1];
  /* Axes lines */
  const axisLines = axes.map((_, i) => toXY(i, R));
  /* Data polygon */
  const dataPoints = axes.map((a, i) => toXY(i, R * (a.value / 100)));
  const dataPath = dataPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + 'Z';

  return (
    <svg width="160" height="160" viewBox="0 0 160 160" aria-label="Radar chart" className="flex-shrink-0">
      {/* Grid */}
      {rings.map((r) => {
        const pts = axes.map((_, i) => {
          const p = toXY(i, R * r);
          return `${p.x},${p.y}`;
        }).join(' ');
        return <polygon key={r} points={pts} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="0.8" />;
      })}

      {/* Axis lines */}
      {axisLines.map((p, i) => (
        <line key={i} x1={CX} y1={CY} x2={p.x} y2={p.y} stroke="rgba(255,255,255,0.08)" strokeWidth="0.8" />
      ))}

      {/* Data fill */}
      <motion.polygon
        points={dataPoints.map(p => `${p.x},${p.y}`).join(' ')}
        fill="rgba(139,92,246,0.12)"
        stroke="#8B5CF6"
        strokeWidth="1.5"
        initial={{ opacity: 0, scale: 0.3 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        style={{ transformOrigin: `${CX}px ${CY}px` }}
      />

      {/* Data path as motion */}
      <motion.path d={dataPath} fill="none" stroke="none" />

      {/* Axis labels */}
      {axes.map((a, i) => {
        const p = toXY(i, R + 14);
        return (
          <text key={a.label} x={p.x} y={p.y + 3}
            textAnchor="middle" fontSize="7.5" fill="#64748B" fontFamily="sans-serif" fontWeight="600">
            {a.label}
          </text>
        );
      })}

      {/* Center dot */}
      <circle cx={CX} cy={CY} r="2.5" fill="#8B5CF6" opacity="0.8" />
    </svg>
  );
}

export function InterviewReadiness() {
  const avg = Math.round(AXES.reduce((s, a) => s + a.value, 0) / AXES.length);

  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <div className="px-4 pt-4 pb-3">
        <div className="text-[13px] font-bold text-white">Interview Readiness</div>
        <div className="text-[11px] text-slate-600 mt-0.5">5-axis readiness score</div>
      </div>

      <div className="px-4 pb-4 flex items-center gap-5">
        <RadarChart axes={AXES} />

        <div className="flex-1 space-y-2">
          {AXES.map((a, i) => (
            <motion.div key={a.label}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.07, duration: 0.35 }}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] text-slate-500">{a.label}</span>
                <span className="text-[11px] font-bold" style={{ color: a.color }}>{a.value}%</span>
              </div>
              <div className="h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.07)' }}>
                <motion.div className="h-full rounded-full" style={{ background: a.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${a.value}%` }}
                  transition={{ duration: 0.8, delay: 0.4 + i * 0.08, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
            </motion.div>
          ))}

          {/* Overall score */}
          <div className="pt-2 flex items-center justify-between border-t" style={{ borderColor: 'rgba(255,255,255,0.07)' }}>
            <span className="text-[11px] text-slate-500">Overall</span>
            <span className="text-base font-black" style={{ color: '#A78BFA' }}>{avg}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
