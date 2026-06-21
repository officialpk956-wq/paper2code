'use client';

import { Search, Plus, Zap, Database, Server, GitBranch, Shield, Cloud } from 'lucide-react';
import { useState } from 'react';

interface Pattern {
  id: string;
  name: string;
  category: string;
  description: string;
  icon: React.ReactNode;
  color: string;
}

const patterns: Pattern[] = [
  {
    id: 'load-balancer',
    name: 'Load Balancer',
    category: 'Infrastructure',
    description: 'Distribute traffic across servers',
    icon: <Zap size={20} />,
    color: 'from-yellow-400 to-orange-400',
  },
  {
    id: 'database',
    name: 'Database',
    category: 'Storage',
    description: 'Persistent data storage',
    icon: <Database size={20} />,
    color: 'from-blue-400 to-cyan-400',
  },
  {
    id: 'api-server',
    name: 'API Server',
    category: 'Compute',
    description: 'Backend service',
    icon: <Server size={20} />,
    color: 'from-purple-400 to-pink-400',
  },
  {
    id: 'cache',
    name: 'Cache Layer',
    category: 'Storage',
    description: 'Fast data access',
    icon: <Zap size={20} />,
    color: 'from-green-400 to-emerald-400',
  },
  {
    id: 'message-queue',
    name: 'Message Queue',
    category: 'Communication',
    description: 'Async messaging',
    icon: <GitBranch size={20} />,
    color: 'from-indigo-400 to-purple-400',
  },
  {
    id: 'firewall',
    name: 'Firewall',
    category: 'Security',
    description: 'Access control',
    icon: <Shield size={20} />,
    color: 'from-red-400 to-pink-400',
  },
  {
    id: 'cdn',
    name: 'CDN',
    category: 'Infrastructure',
    description: 'Content delivery',
    icon: <Cloud size={20} />,
    color: 'from-teal-400 to-cyan-400',
  },
  {
    id: 'service-mesh',
    name: 'Service Mesh',
    category: 'Infrastructure',
    description: 'Service communication',
    icon: <GitBranch size={20} />,
    color: 'from-violet-400 to-purple-400',
  },
];

interface PatternLibraryProps {
  onSelectPattern?: (id: string) => void;
  onAddPattern?: (id: string) => void;
}

export function PatternLibrary({ onSelectPattern, onAddPattern }: PatternLibraryProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const categories = [...new Set(patterns.map((p) => p.category))];
  const filtered = patterns.filter((p) => {
    const matchesSearch = p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = !selectedCategory || p.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-3">
          Design Patterns
        </h3>
        <div className="relative mb-3">
          <Search size={16} className="absolute left-2 top-2.5 text-[--color-text-tertiary]" />
          <input
            type="text"
            placeholder="Search patterns..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-2 bg-[--bg-body] border border-[--color-border] rounded text-xs
                     text-[--color-text-primary] placeholder-[--color-text-tertiary]
                     focus:border-[--accent-primary] focus:outline-none transition-colors"
          />
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setSelectedCategory(null)}
            className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
              selectedCategory === null
                ? 'bg-[--accent-primary] text-white'
                : 'bg-[--bg-body] text-[--color-text-secondary] hover:text-[--color-text-primary]'
            }`}
          >
            All
          </button>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                selectedCategory === cat
                  ? 'bg-[--accent-primary] text-white'
                  : 'bg-[--bg-body] text-[--color-text-secondary] hover:text-[--color-text-primary]'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Patterns List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-center">
            <div className="text-xs text-[--color-text-tertiary]">
              No patterns found
            </div>
          </div>
        ) : (
          filtered.map((pattern) => (
            <div
              key={pattern.id}
              onClick={() => onSelectPattern?.(pattern.id)}
              className="group p-3 rounded-lg bg-[--bg-body] border border-[--color-border] hover:border-[--accent-primary] transition-all cursor-pointer"
            >
              {/* Header */}
              <div className="flex items-start gap-3 mb-2">
                <div className={`p-2 rounded bg-gradient-to-br ${pattern.color} text-white`}>
                  {pattern.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-semibold text-[--color-text-primary]">
                    {pattern.name}
                  </div>
                  <div className="text-xs text-[--color-text-tertiary]">
                    {pattern.category}
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onAddPattern?.(pattern.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-[--bg-surface] transition-all"
                >
                  <Plus size={14} className="text-[--accent-primary]" />
                </button>
              </div>

              {/* Description */}
              <p className="text-xs text-[--color-text-secondary] leading-relaxed">
                {pattern.description}
              </p>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body]">
        <p className="text-xs text-[--color-text-tertiary]">
          Drag components to canvas or click + to add
        </p>
      </div>
    </div>
  );
}
