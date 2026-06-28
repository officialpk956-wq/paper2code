import React from 'react';

export function Callout({ type = 'info', title, children }: { type?: 'info' | 'warning' | 'tip'; title?: string; children: React.ReactNode }) {
  const styles = {
    info: 'bg-blue-900/20 border-blue-500/50 text-blue-200',
    warning: 'bg-yellow-900/20 border-yellow-500/50 text-yellow-200',
    tip: 'bg-green-900/20 border-green-500/50 text-green-200',
  };

  return (
    <div className={`p-4 rounded-lg border my-4 ${styles[type]}`}>
      {title && <div className="font-bold mb-2">{title}</div>}
      <div className="text-sm">{children}</div>
    </div>
  );
}
