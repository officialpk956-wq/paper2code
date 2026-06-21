'use client';

import { ReactNode } from 'react';

interface ThreeColumnLayoutProps {
  left?: ReactNode;
  center: ReactNode;
  right?: ReactNode;
  leftWidth?: string;
  rightWidth?: string;
  showRight?: boolean;
}

export function ThreeColumnLayout({
  left,
  center,
  right,
  leftWidth = 'w-60',
  rightWidth = 'w-80',
  showRight = true,
}: ThreeColumnLayoutProps) {
  return (
    <div className="flex h-[calc(100vh-56px)] overflow-hidden bg-[--bg-body]">
      {/* Left Panel */}
      {left && (
        <aside
          aria-label="Context panel"
          className={`${leftWidth} border-r border-[--color-divider] overflow-y-auto flex-shrink-0
          hidden md:block bg-[--bg-body]`}
        >
          {left}
        </aside>
      )}

      {/* Center Panel - Primary Content */}
      <div className="flex-1 overflow-y-auto" role="region" aria-label="Main content">
        <div className="h-full">{center}</div>
      </div>

      {/* Right Panel - Inspector */}
      {right && showRight && (
        <aside
          aria-label="Inspector panel"
          className={`${rightWidth} border-l border-[--color-divider] overflow-y-auto flex-shrink-0
          hidden lg:block bg-[--bg-body] sticky top-0 h-[calc(100vh-56px)]`}
        >
          {right}
        </aside>
      )}
    </div>
  );
}
