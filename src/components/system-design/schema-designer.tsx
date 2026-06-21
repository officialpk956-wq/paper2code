'use client';

import { Plus, Trash2, Edit2 } from 'lucide-react';
import { useState } from 'react';

interface TableField {
  id: string;
  name: string;
  type: string;
  nullable: boolean;
  isPrimaryKey: boolean;
  isIndexed: boolean;
}

interface DatabaseTable {
  id: string;
  name: string;
  fields: TableField[];
}

const sampleTables: DatabaseTable[] = [
  {
    id: 'users',
    name: 'users',
    fields: [
      {
        id: 'id',
        name: 'id',
        type: 'BIGINT',
        nullable: false,
        isPrimaryKey: true,
        isIndexed: true,
      },
      {
        id: 'email',
        name: 'email',
        type: 'VARCHAR(255)',
        nullable: false,
        isPrimaryKey: false,
        isIndexed: true,
      },
      {
        id: 'name',
        name: 'name',
        type: 'VARCHAR(255)',
        nullable: true,
        isPrimaryKey: false,
        isIndexed: false,
      },
      {
        id: 'created_at',
        name: 'created_at',
        type: 'TIMESTAMP',
        nullable: false,
        isPrimaryKey: false,
        isIndexed: false,
      },
    ],
  },
  {
    id: 'posts',
    name: 'posts',
    fields: [
      {
        id: 'post_id',
        name: 'id',
        type: 'BIGINT',
        nullable: false,
        isPrimaryKey: true,
        isIndexed: true,
      },
      {
        id: 'post_user_id',
        name: 'user_id',
        type: 'BIGINT',
        nullable: false,
        isPrimaryKey: false,
        isIndexed: true,
      },
      {
        id: 'post_title',
        name: 'title',
        type: 'VARCHAR(500)',
        nullable: false,
        isPrimaryKey: false,
        isIndexed: false,
      },
      {
        id: 'post_created_at',
        name: 'created_at',
        type: 'TIMESTAMP',
        nullable: false,
        isPrimaryKey: false,
        isIndexed: false,
      },
    ],
  },
];

export function SchemaDesigner() {
  const [selectedTable, setSelectedTable] = useState<string>(sampleTables[0].id);
  const table = sampleTables.find((t) => t.id === selectedTable);

  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h2 className="text-lg font-bold text-[--color-text-primary]">
          Database Schema
        </h2>
        <p className="text-xs text-[--color-text-secondary] mt-1">
          PostgreSQL • {sampleTables.length} tables
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden flex">
        {/* Table List */}
        <div className="w-40 border-r border-[--color-border] bg-[--bg-surface] overflow-y-auto">
          {sampleTables.map((t) => (
            <button
              key={t.id}
              onClick={() => setSelectedTable(t.id)}
              className={`w-full px-4 py-3 text-xs font-medium text-left transition-colors border-l-2 ${
                selectedTable === t.id
                  ? 'bg-[--bg-panel] border-[--accent-primary] text-[--accent-primary]'
                  : 'border-transparent text-[--color-text-secondary] hover:bg-[--bg-panel]'
              }`}
            >
              {t.name}
            </button>
          ))}
        </div>

        {/* Table Details */}
        <div className="flex-1 overflow-y-auto p-6">
          {table && (
            <div className="space-y-4">
              {/* Table Info */}
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-bold text-[--color-text-primary]">
                    {table.name}
                  </h3>
                  <button className="p-1 rounded hover:bg-[--bg-body] text-[--color-text-tertiary]">
                    <Edit2 size={14} />
                  </button>
                </div>
                <p className="text-xs text-[--color-text-tertiary]">
                  {table.fields.length} fields
                </p>
              </div>

              {/* Fields */}
              <div className="space-y-2">
                {table.fields.map((field) => (
                  <div
                    key={field.id}
                    className="bg-[--bg-panel] rounded-lg p-3 border border-[--color-border] hover:border-[--accent-primary] transition-colors"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <div className="text-xs font-semibold text-[--color-text-primary] flex items-center gap-2">
                          {field.name}
                          {field.isPrimaryKey && (
                            <span className="px-1.5 py-0.5 rounded text-xs bg-[--accent-primary] text-white">
                              PK
                            </span>
                          )}
                          {field.isIndexed && (
                            <span className="px-1.5 py-0.5 rounded text-xs bg-[--accent-cyan] text-white">
                              IDX
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-[--color-text-tertiary] mt-1 font-mono">
                          {field.type}
                        </div>
                      </div>
                      <button className="p-1 rounded hover:bg-[--bg-body] text-red-400">
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Add Field */}
              <button className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-[--accent-primary] hover:opacity-90 text-white text-xs font-medium transition-opacity">
                <Plus size={14} />
                Add Field
              </button>

              {/* SQL Preview */}
              <div className="bg-[--bg-panel] rounded-lg p-3 border border-[--color-border]">
                <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                  CREATE TABLE
                </div>
                <pre className="text-xs font-mono text-[--color-text-secondary] overflow-x-auto">
{`CREATE TABLE ${table.name} (
  id BIGINT PRIMARY KEY,
  ${table.fields.slice(1, 3).map((f) => `${f.name} ${f.type}`).join(',\n  ')},
  created_at TIMESTAMP
);`}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
