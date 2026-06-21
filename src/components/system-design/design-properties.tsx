'use client';

import { Copy, Trash2, Settings } from 'lucide-react';
import { useState } from 'react';

interface ComponentSpec {
  name: string;
  type: string;
  capacity: string;
  latency: string;
  availability: string;
  replication: string;
}

const specs: Record<string, ComponentSpec> = {
  'load-balancer': {
    name: 'Load Balancer',
    type: 'Infrastructure',
    capacity: '10,000 req/s',
    latency: '< 5ms',
    availability: '99.99%',
    replication: 'Active-Active',
  },
  'api-server': {
    name: 'API Server',
    type: 'Compute',
    capacity: '1,000 req/s',
    latency: '< 100ms',
    availability: '99.9%',
    replication: 'Stateless',
  },
  'database': {
    name: 'Database',
    type: 'Storage',
    capacity: '1TB',
    latency: '< 50ms',
    availability: '99.95%',
    replication: 'Master-Slave',
  },
  'cache': {
    name: 'Cache Layer',
    type: 'Storage',
    capacity: '100GB',
    latency: '< 1ms',
    availability: '99.9%',
    replication: 'Distributed',
  },
};

interface DesignPropertiesProps {
  selectedPattern?: string;
}

export function DesignProperties({ selectedPattern = 'api-server' }: DesignPropertiesProps) {
  const [activeTab, setActiveTab] = useState<'specs' | 'scaling' | 'cost'>('specs');
  const spec = specs[selectedPattern] || specs['api-server'];

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-l border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-3">
          Component Properties
        </h3>
        <div className="space-y-2">
          <div>
            <div className="text-xs text-[--color-text-tertiary] mb-1">Component</div>
            <div className="text-sm font-semibold text-[--color-text-primary]">
              {spec.name}
            </div>
          </div>
          <div>
            <div className="text-xs text-[--color-text-tertiary] mb-1">Type</div>
            <div className="inline-block px-2 py-1 rounded text-xs bg-[--accent-primary]/10 text-[--accent-primary] border border-[--accent-primary]/20">
              {spec.type}
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="px-4 pt-4 border-b border-[--color-border] flex gap-2 bg-[--bg-body]">
        {(['specs', 'scaling', 'cost'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-3 py-2 rounded-t text-xs font-medium transition-colors ${
              activeTab === tab
                ? 'bg-[--accent-primary] text-white'
                : 'text-[--color-text-secondary] hover:text-[--color-text-primary]'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'specs' && (
          <div className="space-y-4">
            {/* Capacity */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Capacity
              </div>
              <div className="text-sm font-mono text-[--accent-cyan]">
                {spec.capacity}
              </div>
            </div>

            {/* Latency */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Latency (P99)
              </div>
              <div className="text-sm font-mono text-[--accent-primary]">
                {spec.latency}
              </div>
            </div>

            {/* Availability */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Availability SLA
              </div>
              <div className="text-sm font-mono text-green-400">
                {spec.availability}
              </div>
              <div className="text-xs text-[--color-text-tertiary] mt-1">
                Downtime: {spec.availability === '99.99%' ? '43s/mo' : spec.availability === '99.95%' ? '22m/mo' : '43m/mo'}
              </div>
            </div>

            {/* Replication */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Replication Strategy
              </div>
              <div className="text-sm text-[--color-text-secondary]">
                {spec.replication}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'scaling' && (
          <div className="space-y-4">
            <div className="bg-[--accent-primary]/10 rounded-lg p-3 border border-[--accent-primary]/20">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Horizontal Scaling
              </div>
              <p className="text-xs text-[--color-text-secondary] leading-relaxed">
                Can scale horizontally by adding more instances. Load balancer distributes traffic.
              </p>
            </div>

            <div className="bg-[--accent-cyan]/10 rounded-lg p-3 border border-[--accent-cyan]/20">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Auto-scaling
              </div>
              <p className="text-xs text-[--color-text-secondary] leading-relaxed">
                Recommended: 2-10 instances. Scale up at 80% capacity, down at 20%.
              </p>
            </div>

            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-3">
                Target Metrics
              </div>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-[--color-text-secondary]">CPU Usage</span>
                  <span className="text-[--color-text-primary]">60-70%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[--color-text-secondary]">Memory</span>
                  <span className="text-[--color-text-primary]">50-60%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[--color-text-secondary]">Network</span>
                  <span className="text-[--color-text-primary]">40-50%</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'cost' && (
          <div className="space-y-4">
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Monthly Cost (per instance)
              </div>
              <div className="text-lg font-bold text-[--accent-primary]">
                $150 - $300
              </div>
              <div className="text-xs text-[--color-text-tertiary] mt-1">
                Depends on instance type and region
              </div>
            </div>

            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Cost Optimization
              </div>
              <ul className="text-xs text-[--color-text-secondary] space-y-1">
                <li>• Use spot instances for non-critical workloads</li>
                <li>• Reserved instances for baseline capacity</li>
                <li>• Auto-scaling to handle traffic spikes</li>
              </ul>
            </div>

            <div className="bg-[--accent-cyan]/10 rounded-lg p-3 border border-[--accent-cyan]/20">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Estimated Total (10 instances)
              </div>
              <div className="text-lg font-bold text-[--accent-cyan]">
                $15K - $30K/month
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Actions Footer */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body] flex gap-2">
        <button className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-medium bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          <Copy size={14} />
          Copy
        </button>
        <button className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-medium bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          <Settings size={14} />
          Edit
        </button>
        <button className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-medium bg-red-500/20 hover:bg-red-500/30 text-red-400 transition-colors">
          <Trash2 size={14} />
          Delete
        </button>
      </div>
    </div>
  );
}
