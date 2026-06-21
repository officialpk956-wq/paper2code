'use client';

import { Copy, Star, Eye, Download } from 'lucide-react';
import { useState } from 'react';

interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  author: string;
  uses: number;
  rating: number;
  icon: string;
  layers?: number;
}

const templates: Template[] = [
  {
    id: 'transformer',
    name: 'Transformer Base',
    description: 'Standard transformer architecture with 6 layers and 8 attention heads',
    category: 'Neural Networks',
    author: 'Paper2Code',
    uses: 1247,
    rating: 4.8,
    icon: '⚙️',
    layers: 8,
  },
  {
    id: 'bert',
    name: 'BERT Model',
    description: 'Bidirectional encoder transformer with pre-training setup',
    category: 'Neural Networks',
    author: 'Paper2Code',
    uses: 892,
    rating: 4.7,
    icon: '📖',
    layers: 12,
  },
  {
    id: 'microservices',
    name: 'Microservices Architecture',
    description: 'Scalable microservices with API gateway, load balancer, and databases',
    category: 'System Design',
    author: 'Paper2Code',
    uses: 654,
    rating: 4.9,
    icon: '🏗️',
  },
  {
    id: 'cnn-resnet',
    name: 'ResNet Architecture',
    description: 'Deep residual network with skip connections for image classification',
    category: 'Neural Networks',
    author: 'Community',
    uses: 431,
    rating: 4.6,
    icon: '🖼️',
    layers: 50,
  },
  {
    id: 'api-rest',
    name: 'REST API Documentation',
    description: 'Standard REST API with CRUD operations and authentication',
    category: 'APIs',
    author: 'Paper2Code',
    uses: 721,
    rating: 4.8,
    icon: '🔌',
  },
  {
    id: 'database-schema',
    name: 'E-Commerce Database',
    description: 'Complete schema for e-commerce platform with users, products, orders',
    category: 'Databases',
    author: 'Community',
    uses: 612,
    rating: 4.7,
    icon: '🛍️',
  },
];

interface TemplatesLibraryProps {
  onSelectTemplate?: (id: string) => void;
}

export function TemplatesLibrary({ onSelectTemplate }: TemplatesLibraryProps) {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  const categories = [...new Set(templates.map((t) => t.category))];
  const filtered = templates.filter((t) => {
    const matchesSearch = t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = !selectedCategory || t.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h2 className="text-lg font-bold text-[--color-text-primary] mb-3">
          Templates Library
        </h2>
        <input
          type="text"
          placeholder="Search templates..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-2 bg-[--bg-body] border border-[--color-border] rounded text-xs text-[--color-text-primary] placeholder-[--color-text-tertiary] focus:border-[--accent-primary] focus:outline-none transition-colors"
        />

        {/* Category Filter */}
        <div className="flex flex-wrap gap-2 mt-3">
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

      {/* Templates Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filtered.map((template) => (
            <div
              key={template.id}
              className="group p-4 rounded-lg bg-[--bg-panel] border border-[--color-border] hover:border-[--accent-primary] transition-all cursor-pointer"
              onClick={() => onSelectTemplate?.(template.id)}
            >
              {/* Header */}
              <div className="flex items-start gap-3 mb-3">
                <div className="text-3xl">{template.icon}</div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-bold text-[--color-text-primary]">
                    {template.name}
                  </h3>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="text-xs bg-[--accent-cyan]/20 text-[--accent-cyan] px-1.5 py-0.5 rounded">
                      {template.category}
                    </div>
                    {template.layers && (
                      <div className="text-xs text-[--color-text-tertiary]">
                        {template.layers} layers
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Description */}
              <p className="text-xs text-[--color-text-secondary] leading-relaxed mb-3">
                {template.description}
              </p>

              {/* Stats */}
              <div className="flex items-center justify-between mb-3 text-xs text-[--color-text-tertiary]">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1">
                    <Eye size={12} />
                    {template.uses.toLocaleString()} uses
                  </div>
                  <div className="flex items-center gap-1">
                    <Star size={12} className="text-yellow-400" />
                    {template.rating.toFixed(1)}
                  </div>
                </div>
                <div className="text-xs">by {template.author}</div>
              </div>

              {/* Actions */}
              <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button className="flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded text-xs font-medium bg-[--accent-primary] hover:opacity-90 text-white transition-opacity">
                  <Copy size={12} />
                  Use Template
                </button>
                <button className="px-3 py-2 rounded text-xs font-medium bg-[--bg-body] hover:bg-[--bg-surface] text-[--color-text-secondary]">
                  <Download size={12} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-[--color-border] bg-[--bg-panel] text-xs text-[--color-text-secondary]">
        Showing {filtered.length} of {templates.length} templates
      </div>
    </div>
  );
}
