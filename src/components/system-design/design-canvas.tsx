'use client';

import { Trash2, ZoomIn, ZoomOut } from 'lucide-react';

interface CanvasElement {
  id: string;
  type: string;
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  color: string;
}

const sampleElements: CanvasElement[] = [
  {
    id: '1',
    type: 'load-balancer',
    x: 100,
    y: 50,
    width: 120,
    height: 80,
    label: 'Load Balancer',
    color: 'from-yellow-400 to-orange-400',
  },
  {
    id: '2',
    type: 'api-server',
    x: 100,
    y: 180,
    width: 120,
    height: 80,
    label: 'API Server 1',
    color: 'from-purple-400 to-pink-400',
  },
  {
    id: '3',
    type: 'api-server',
    x: 280,
    y: 180,
    width: 120,
    height: 80,
    label: 'API Server 2',
    color: 'from-purple-400 to-pink-400',
  },
  {
    id: '4',
    type: 'database',
    x: 100,
    y: 340,
    width: 120,
    height: 80,
    label: 'Primary DB',
    color: 'from-blue-400 to-cyan-400',
  },
  {
    id: '5',
    type: 'cache',
    x: 280,
    y: 340,
    width: 120,
    height: 80,
    label: 'Redis Cache',
    color: 'from-green-400 to-emerald-400',
  },
];

interface DesignCanvasProps {
  selectedPattern?: string;
}

export function DesignCanvas({ selectedPattern }: DesignCanvasProps) {
  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Toolbar */}
      <div className="px-6 py-3 border-b border-[--color-border] bg-[--bg-panel] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button className="p-2 rounded hover:bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
            <ZoomIn size={18} />
          </button>
          <button className="p-2 rounded hover:bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
            <ZoomOut size={18} />
          </button>
          <div className="w-px h-6 bg-[--color-border] mx-2" />
          <span className="text-xs text-[--color-text-tertiary]">100%</span>
        </div>

        <div className="text-xs text-[--color-text-secondary]">
          {selectedPattern ? `Ready to add: ${selectedPattern}` : 'Select a pattern to begin'}
        </div>

        <button className="p-2 rounded hover:bg-red-500/20 text-red-400 transition-colors">
          <Trash2 size={18} />
        </button>
      </div>

      {/* Canvas Area */}
      <div className="flex-1 overflow-auto bg-gradient-to-br from-[--bg-body] to-[--bg-surface] relative">
        {/* Grid Background */}
        <svg
          className="absolute inset-0 w-full h-full"
          style={{
            backgroundImage: `
              linear-gradient(0deg, transparent 24%, rgba(124, 58, 237, 0.05) 25%, rgba(124, 58, 237, 0.05) 26%, transparent 27%, transparent 74%, rgba(124, 58, 237, 0.05) 75%, rgba(124, 58, 237, 0.05) 76%, transparent 77%, transparent),
              linear-gradient(90deg, transparent 24%, rgba(124, 58, 237, 0.05) 25%, rgba(124, 58, 237, 0.05) 26%, transparent 27%, transparent 74%, rgba(124, 58, 237, 0.05) 75%, rgba(124, 58, 237, 0.05) 76%, transparent 77%, transparent)
            `,
            backgroundSize: '50px 50px',
          }}
        >
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
              <polygon points="0 0, 10 3, 0 6" fill="#64748B" />
            </marker>
          </defs>

          {/* Connection Lines */}
          <line x1="220" y1="90" x2="220" y2="180" stroke="#64748B" strokeWidth="2" markerEnd="url(#arrowhead)" />
          <line x1="160" y1="260" x2="160" y2="340" stroke="#64748B" strokeWidth="2" markerEnd="url(#arrowhead)" />
          <line x1="340" y1="260" x2="340" y2="340" stroke="#64748B" strokeWidth="2" markerEnd="url(#arrowhead)" />
        </svg>

        {/* Canvas Elements */}
        <div className="relative" style={{ width: '1000px', height: '600px' }}>
          {sampleElements.map((element) => (
            <div
              key={element.id}
              className={`absolute group p-3 rounded-lg border-2 border-[--color-border] bg-gradient-to-br ${element.color} cursor-move
                         hover:border-[--accent-primary] hover:shadow-lg transition-all hover:scale-105`}
              style={{
                left: `${element.x}px`,
                top: `${element.y}px`,
                width: `${element.width}px`,
                height: `${element.height}px`,
                opacity: 0.9,
              }}
            >
              {/* Content */}
              <div className="flex flex-col items-center justify-center h-full">
                <div className="text-sm font-bold text-white mb-1">
                  {element.label.split(' ')[0]}
                </div>
                <div className="text-xs text-white/80 text-center">
                  {element.label.split(' ').slice(1).join(' ')}
                </div>
              </div>

              {/* Hover Actions */}
              <button className="absolute -top-6 -right-6 opacity-0 group-hover:opacity-100 p-1.5 rounded bg-red-500 hover:bg-red-600 text-white transition-all">
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-[--color-border] bg-[--bg-panel] text-xs text-[--color-text-tertiary]">
        {sampleElements.length} components • Drag to move • Right-click for options
      </div>
    </div>
  );
}
