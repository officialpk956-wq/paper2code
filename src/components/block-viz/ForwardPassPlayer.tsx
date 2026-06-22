'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, SkipBack, SkipForward, Rewind } from 'lucide-react';
import { ShapePill } from './ShapePill';
import { FLOPsBadge } from './FLOPsBadge';

export interface ForwardPassStep {
  step: number;
  node_id: string;
  node_name: string;
  type: string;
  input_shape: number[];
  output_shape: number[];
  flops_mflops: number;
  params_M: number;
  severity: string;
  description: string;
}

interface ForwardPassPlayerProps {
  steps: ForwardPassStep[];
  onStepChange?: (step: ForwardPassStep | null) => void;
}

const SPEED_OPTIONS = [
  { label: '×0.5', ms: 2000 },
  { label: '×1', ms: 1000 },
  { label: '×2', ms: 500 },
  { label: '×4', ms: 250 },
];

export const ForwardPassPlayer = React.memo(function ForwardPassPlayer({
  steps,
  onStepChange,
}: ForwardPassPlayerProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedIdx, setSpeedIdx] = useState(1); // ×1 default
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const total = steps.length;
  const step = steps[currentStep] ?? null;

  // Notify parent of active block
  useEffect(() => {
    onStepChange?.(step);
  }, [step, onStepChange]);

  // Playback timer
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentStep((s) => {
          if (s >= total - 1) {
            setIsPlaying(false);
            return s;
          }
          return s + 1;
        });
      }, SPEED_OPTIONS[speedIdx].ms);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, speedIdx, total]);

  const goTo = useCallback((idx: number) => {
    setCurrentStep(Math.max(0, Math.min(total - 1, idx)));
  }, [total]);

  const togglePlay = useCallback(() => {
    if (currentStep >= total - 1) setCurrentStep(0);
    setIsPlaying((p) => !p);
  }, [currentStep, total]);

  if (total === 0) return null;

  const progress = total > 1 ? (currentStep / (total - 1)) * 100 : 100;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      padding: '12px',
      background: 'rgba(255,255,255,0.03)',
      borderRadius: '10px',
      border: '1px solid rgba(255,255,255,0.08)',
    }}>
      {/* Current step info */}
      {step && (
        <div style={{
          padding: '10px',
          borderRadius: '8px',
          background: 'rgba(var(--accent-primary-rgb), 0.08)',
          border: '1px solid rgba(var(--accent-primary-rgb), 0.2)',
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: 6,
          }}>
            <div>
              <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--color-text-primary)' }}>
                {step.node_name}
              </span>
              <span style={{ fontSize: '10px', color: 'var(--text-muted, #6b7280)', marginLeft: 6 }}>
                {step.type}
              </span>
            </div>
            <FLOPsBadge flops={step.flops_mflops} compact />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
            <ShapePill shape={step.input_shape} label="in" compact />
            <span style={{ color: 'var(--text-muted, #6b7280)', fontSize: 11 }}>→</span>
            <ShapePill shape={step.output_shape} label="out" compact />
          </div>

          <p style={{ fontSize: '11px', color: 'var(--text-muted, #9ca3af)', margin: 0, lineHeight: 1.5 }}>
            {step.description}
          </p>
        </div>
      )}

      {/* Progress bar */}
      <div>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '10px',
          color: 'var(--text-muted, #6b7280)',
          marginBottom: 4,
        }}>
          <span>Step {currentStep + 1} / {total}</span>
          <span>{step?.node_id ?? ''}</span>
        </div>
        <div
          role="slider"
          aria-label="Forward pass step"
          aria-valuenow={currentStep}
          aria-valuemin={0}
          aria-valuemax={total - 1}
          aria-valuetext={`Step ${currentStep + 1} of ${total}`}
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft') { e.preventDefault(); goTo(currentStep - 1); }
            if (e.key === 'ArrowRight') { e.preventDefault(); goTo(currentStep + 1); }
          }}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const ratio = (e.clientX - rect.left) / rect.width;
            goTo(Math.round(ratio * (total - 1)));
          }}
          style={{
            width: '100%',
            height: 6,
            borderRadius: 3,
            background: 'rgba(255,255,255,0.1)',
            cursor: 'pointer',
            position: 'relative',
          }}
        >
          <div style={{
            position: 'absolute',
            left: 0,
            top: 0,
            height: '100%',
            width: `${progress}%`,
            borderRadius: 3,
            background: 'var(--accent-primary)',
            transition: 'width 0.15s ease',
          }} />
        </div>
      </div>

      {/* Controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <IconBtn
            onClick={() => { setIsPlaying(false); goTo(0); }}
            title="Restart"
          >
            <Rewind size={14} />
          </IconBtn>
          <IconBtn onClick={() => goTo(currentStep - 1)} title="Step back" disabled={currentStep === 0}>
            <SkipBack size={14} />
          </IconBtn>
          <IconBtn onClick={togglePlay} title={isPlaying ? 'Pause' : 'Play'} primary>
            {isPlaying ? <Pause size={15} /> : <Play size={15} />}
          </IconBtn>
          <IconBtn
            onClick={() => goTo(currentStep + 1)}
            title="Step forward"
            disabled={currentStep >= total - 1}
          >
            <SkipForward size={14} />
          </IconBtn>
        </div>

        {/* Speed selector */}
        <div role="group" aria-label="Playback speed" style={{ display: 'flex', gap: '3px' }}>
          {SPEED_OPTIONS.map((opt, i) => (
            <button
              key={opt.label}
              onClick={() => setSpeedIdx(i)}
              aria-pressed={i === speedIdx}
              style={{
                padding: '2px 7px',
                borderRadius: '4px',
                fontSize: '10px',
                fontWeight: 600,
                border: `1px solid ${i === speedIdx ? 'rgba(var(--accent-primary-rgb), 0.5)' : 'rgba(255,255,255,0.1)'}`,
                background: i === speedIdx ? 'rgba(var(--accent-primary-rgb), 0.15)' : 'transparent',
                color: i === speedIdx ? 'var(--accent-primary-light)' : 'var(--color-text-muted)',
                cursor: 'pointer',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
});

function IconBtn({
  children,
  onClick,
  title,
  disabled = false,
  primary = false,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  disabled?: boolean;
  primary?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={title}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: primary ? 32 : 26,
        height: primary ? 32 : 26,
        borderRadius: '50%',
        border: `1px solid ${primary ? 'rgba(var(--accent-primary-rgb), 0.4)' : 'rgba(255,255,255,0.1)'}`,
        background: primary ? 'rgba(var(--accent-primary-rgb), 0.2)' : 'transparent',
        color: disabled ? 'var(--color-text-muted)' : (primary ? 'var(--accent-primary-light)' : 'var(--color-text-secondary)'),
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.4 : 1,
        transition: 'background 0.1s, color 0.1s',
      }}
    >
      {children}
    </button>
  );
}
