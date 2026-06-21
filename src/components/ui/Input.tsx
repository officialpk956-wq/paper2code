'use client';

import React from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export function Input({ label, error, disabled, id, style, onFocus, onBlur, ...props }: InputProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {label && (
        <label
          htmlFor={id}
          style={{
            fontSize: '10px',
            fontWeight: 700,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: 'var(--color-text-secondary)',
            display: 'block',
          }}
        >
          {label}
        </label>
      )}
      <input
        id={id}
        disabled={disabled}
        style={{
          width: '100%',
          padding: '8px 10px',
          background: 'var(--bg-panel)',
          color: 'var(--color-text-primary)',
          border: `1px solid ${error ? 'var(--color-error)' : 'var(--color-border)'}`,
          borderRadius: 'var(--radius-sm)',
          fontSize: 'var(--text-sm)',
          fontFamily: 'var(--font-body)',
          opacity: disabled ? 'var(--opacity-disabled)' : 1,
          cursor: disabled ? 'not-allowed' : undefined,
          outline: 'none',
          transition: 'border-color var(--motion-fast), box-shadow var(--motion-fast)',
          ...style,
        }}
        onFocus={(e) => {
          e.target.style.borderColor = error ? 'var(--color-error)' : 'rgba(var(--accent-primary-rgb), 0.8)';
          e.target.style.boxShadow = 'var(--focus-ring, 0 0 0 3px rgba(var(--accent-primary-rgb), 0.15))';
          onFocus?.(e);
        }}
        onBlur={(e) => {
          e.target.style.borderColor = error ? 'var(--color-error)' : 'var(--color-border)';
          e.target.style.boxShadow = 'none';
          onBlur?.(e);
        }}
        {...props}
      />
      {error && (
        <span style={{ fontSize: '11px', color: 'var(--color-error)', lineHeight: 1.4 }}>
          {error}
        </span>
      )}
    </div>
  );
}
