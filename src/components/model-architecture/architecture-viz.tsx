'use client';

interface ArchitectureVizProps {
  selectedLayer?: string;
}

export function ArchitectureViz({ selectedLayer }: ArchitectureVizProps) {
  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h2 className="text-lg font-bold text-[--color-text-primary]">
          Architecture Visualization
        </h2>
        <p className="text-xs text-[--color-text-secondary] mt-1">
          Transformer Model • 4.77M Parameters • 7 Layers
        </p>
      </div>

      {/* Canvas */}
      <div className="flex-1 overflow-auto bg-gradient-to-br from-[--bg-body] to-[--bg-surface] p-6">
        <div className="flex justify-center items-start min-h-full">
          <svg
            viewBox="0 0 1200 800"
            className="w-full max-w-5xl h-auto"
            style={{ minHeight: '600px' }}
          >
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" fill="#64748B" />
              </marker>
              <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style={{ stopColor: '#7C3AED', stopOpacity: 0.3 }} />
                <stop offset="100%" style={{ stopColor: '#7C3AED', stopOpacity: 0.1 }} />
              </linearGradient>
            </defs>

            {/* Title */}
            <text x="600" y="30" textAnchor="middle" fontSize="18" fontWeight="bold" fill="#E2E8F0">
              Transformer Model Architecture
            </text>

            {/* Input Layer */}
            <g>
              <rect x="50" y="80" width="100" height="100" fill="url(#grad1)" stroke="#7C3AED" strokeWidth="2" rx="8" />
              <text x="100" y="120" textAnchor="middle" fontSize="12" fontWeight="bold" fill="#E2E8F0">
                Input
              </text>
              <text x="100" y="140" textAnchor="middle" fontSize="10" fill="#94A3B8">
                [B, 784]
              </text>
              <text x="100" y="155" textAnchor="middle" fontSize="9" fill="#64748B">
                0 params
              </text>
            </g>

            {/* Embedding Layer */}
            <line x1="150" y1="130" x2="220" y2="130" stroke="#64748B" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <g>
              <rect x="220" y="80" width="100" height="100" fill="url(#grad1)" stroke="#7C3AED" strokeWidth="2" rx="8" />
              <text x="270" y="120" textAnchor="middle" fontSize="12" fontWeight="bold" fill="#E2E8F0">
                Embedding
              </text>
              <text x="270" y="140" textAnchor="middle" fontSize="10" fill="#94A3B8">
                [B, 512]
              </text>
              <text x="270" y="155" textAnchor="middle" fontSize="9" fill="#64748B">
                307.2K
              </text>
            </g>

            {/* Transformer Blocks (stacked view) */}
            <line x1="320" y1="130" x2="390" y2="130" stroke="#64748B" strokeWidth="2" markerEnd="url(#arrowhead)" />

            {/* Block 1 */}
            <g>
              <rect
                x="390"
                y="60"
                width="120"
                height="140"
                fill={selectedLayer === 'transformer-1' ? '#7C3AED33' : 'url(#grad1)'}
                stroke={selectedLayer === 'transformer-1' ? '#7C3AED' : '#7C3AED'}
                strokeWidth={selectedLayer === 'transformer-1' ? '3' : '2'}
                rx="8"
              />
              <text x="450" y="110" textAnchor="middle" fontSize="11" fontWeight="bold" fill="#E2E8F0">
                Transformer 1
              </text>
              <text x="450" y="130" textAnchor="middle" fontSize="9" fill="#94A3B8">
                [B, 512]
              </text>
              <text x="450" y="145" textAnchor="middle" fontSize="8" fill="#64748B">
                2.1M params
              </text>
              <text x="450" y="158" textAnchor="middle" fontSize="8" fill="#06B6D4">
                Attn + FFN
              </text>
            </g>

            {/* Block 2 */}
            <line x1="510" y1="130" x2="580" y2="130" stroke="#64748B" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <g>
              <rect
                x="580"
                y="60"
                width="120"
                height="140"
                fill={selectedLayer === 'transformer-2' ? '#7C3AED33' : 'url(#grad1)'}
                stroke={selectedLayer === 'transformer-2' ? '#7C3AED' : '#7C3AED'}
                strokeWidth={selectedLayer === 'transformer-2' ? '3' : '2'}
                rx="8"
              />
              <text x="640" y="110" textAnchor="middle" fontSize="11" fontWeight="bold" fill="#E2E8F0">
                Transformer 2
              </text>
              <text x="640" y="130" textAnchor="middle" fontSize="9" fill="#94A3B8">
                [B, 512]
              </text>
              <text x="640" y="145" textAnchor="middle" fontSize="8" fill="#64748B">
                2.1M params
              </text>
              <text x="640" y="158" textAnchor="middle" fontSize="8" fill="#06B6D4">
                Attn + FFN
              </text>
            </g>

            {/* Pooling */}
            <line x1="700" y1="130" x2="770" y2="130" stroke="#64748B" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <g>
              <rect x="770" y="80" width="100" height="100" fill="url(#grad1)" stroke="#7C3AED" strokeWidth="2" rx="8" />
              <text x="820" y="120" textAnchor="middle" fontSize="12" fontWeight="bold" fill="#E2E8F0">
                Pool
              </text>
              <text x="820" y="140" textAnchor="middle" fontSize="10" fill="#94A3B8">
                [B, 512]
              </text>
              <text x="820" y="155" textAnchor="middle" fontSize="9" fill="#64748B">
                0 params
              </text>
            </g>

            {/* Dense */}
            <line x1="870" y1="130" x2="940" y2="130" stroke="#64748B" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <g>
              <rect x="940" y="80" width="100" height="100" fill="url(#grad1)" stroke="#7C3AED" strokeWidth="2" rx="8" />
              <text x="990" y="120" textAnchor="middle" fontSize="12" fontWeight="bold" fill="#E2E8F0">
                Dense
              </text>
              <text x="990" y="140" textAnchor="middle" fontSize="10" fill="#94A3B8">
                [B, 512]
              </text>
              <text x="990" y="155" textAnchor="middle" fontSize="9" fill="#64748B">
                262.1K
              </text>
            </g>

            {/* Output */}
            <line x1="1040" y1="130" x2="1110" y2="130" stroke="#64748B" strokeWidth="2" markerEnd="url(#arrowhead)" />

            {/* Legend */}
            <g>
              <rect x="50" y="280" width="1100" height="140" fill="none" stroke="#1E293B" strokeWidth="1" strokeDasharray="5,5" rx="8" />
              <text x="60" y="300" fontSize="12" fontWeight="bold" fill="#E2E8F0">
                Layer Types
              </text>

              <circle cx="70" cy="330" r="6" fill="#7C3AED" opacity="0.3" stroke="#7C3AED" strokeWidth="2" />
              <text x="85" y="334" fontSize="11" fill="#E2E8F0">Dense/Transformer</text>

              <circle cx="350" cy="330" r="6" fill="#06B6D4" opacity="0.3" stroke="#06B6D4" strokeWidth="2" />
              <text x="365" y="334" fontSize="11" fill="#E2E8F0">Attention Heads</text>

              <circle cx="700" cy="330" r="6" fill="#10B981" opacity="0.3" stroke="#10B981" strokeWidth="2" />
              <text x="715" y="334" fontSize="11" fill="#E2E8F0">Pooling</text>

              <text x="60" y="365" fontSize="10" fill="#94A3B8">
                Total Parameters: 4.77M
              </text>
              <text x="60" y="380" fontSize="10" fill="#94A3B8">
                FLOPs: 8.2B • Memory (fp32): 18.9MB • Attention: O(n²)
              </text>
              <text x="60" y="395" fontSize="10" fill="#94A3B8">
                Selected: {selectedLayer || 'None'}
              </text>
            </g>
          </svg>
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-[--color-border] bg-[--bg-panel] text-xs text-[--color-text-tertiary]">
        Click on layers to view details • Use layers panel to add/remove components
      </div>
    </div>
  );
}
