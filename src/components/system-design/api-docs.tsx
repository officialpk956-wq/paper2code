'use client';

import { Copy } from 'lucide-react';
import { useState } from 'react';

interface APIEndpoint {
  id: string;
  method: string;
  path: string;
  description: string;
  requestBody?: string;
  responseBody: string;
  statusCode: number;
}

const endpoints: APIEndpoint[] = [
  {
    id: '1',
    method: 'GET',
    path: '/api/users/:id',
    description: 'Get user by ID',
    responseBody: `{
  "id": 123,
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2024-01-15T10:30:00Z"
}`,
    statusCode: 200,
  },
  {
    id: '2',
    method: 'POST',
    path: '/api/users',
    description: 'Create new user',
    requestBody: `{
  "email": "user@example.com",
  "name": "John Doe"
}`,
    responseBody: `{
  "id": 123,
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2024-01-15T10:30:00Z"
}`,
    statusCode: 201,
  },
  {
    id: '3',
    method: 'PUT',
    path: '/api/users/:id',
    description: 'Update user',
    requestBody: `{
  "name": "Jane Doe"
}`,
    responseBody: `{
  "id": 123,
  "email": "user@example.com",
  "name": "Jane Doe",
  "created_at": "2024-01-15T10:30:00Z"
}`,
    statusCode: 200,
  },
  {
    id: '4',
    method: 'DELETE',
    path: '/api/users/:id',
    description: 'Delete user',
    responseBody: `{
  "message": "User deleted successfully"
}`,
    statusCode: 204,
  },
];

const methodColors: Record<string, string> = {
  GET: 'bg-blue-500',
  POST: 'bg-green-500',
  PUT: 'bg-yellow-500',
  DELETE: 'bg-red-500',
  PATCH: 'bg-purple-500',
};

export function APIDocs() {
  const [selectedEndpoint, setSelectedEndpoint] = useState<string>(endpoints[0].id);
  const endpoint = endpoints.find((e) => e.id === selectedEndpoint);

  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h2 className="text-lg font-bold text-[--color-text-primary]">
          API Documentation
        </h2>
        <p className="text-xs text-[--color-text-secondary] mt-1">
          REST API • {endpoints.length} endpoints
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden flex">
        {/* Endpoint List */}
        <div className="w-56 border-r border-[--color-border] bg-[--bg-surface] overflow-y-auto">
          {endpoints.map((ep) => (
            <button
              key={ep.id}
              onClick={() => setSelectedEndpoint(ep.id)}
              className={`w-full px-4 py-3 text-left transition-colors border-l-2 hover:bg-[--bg-panel] ${
                selectedEndpoint === ep.id
                  ? 'bg-[--bg-panel] border-[--accent-primary]'
                  : 'border-transparent'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className={`px-2 py-0.5 rounded text-xs font-bold text-white ${methodColors[ep.method]}`}>
                  {ep.method}
                </span>
                <span className="text-xs text-[--color-text-secondary] truncate">
                  {ep.path}
                </span>
              </div>
              <p className="text-xs text-[--color-text-tertiary] mt-1">
                {ep.description}
              </p>
            </button>
          ))}
        </div>

        {/* Endpoint Details */}
        {endpoint && (
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Endpoint Header */}
            <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="flex items-center gap-3">
                    <span className={`px-3 py-1.5 rounded text-sm font-bold text-white ${methodColors[endpoint.method]}`}>
                      {endpoint.method}
                    </span>
                    <code className="text-sm font-mono text-[--color-text-primary]">
                      {endpoint.path}
                    </code>
                  </div>
                  <p className="text-sm text-[--color-text-secondary] mt-2">
                    {endpoint.description}
                  </p>
                </div>
                <span className={`px-2 py-1 rounded text-xs font-mono text-white ${
                  endpoint.statusCode >= 200 && endpoint.statusCode < 300
                    ? 'bg-green-500'
                    : endpoint.statusCode >= 400
                    ? 'bg-red-500'
                    : 'bg-gray-500'
                }`}>
                  {endpoint.statusCode}
                </span>
              </div>
            </div>

            {/* Request Body */}
            {endpoint.requestBody && (
              <div className="space-y-2">
                <div className="text-sm font-semibold text-[--color-text-primary]">
                  Request Body
                </div>
                <div className="bg-[--bg-panel] rounded-lg overflow-hidden border border-[--color-border]">
                  <div className="px-4 py-2 bg-[--bg-surface] border-b border-[--color-border] flex justify-between items-center">
                    <span className="text-xs font-semibold text-[--color-text-secondary]">
                      JSON
                    </span>
                    <button className="p-1 hover:bg-[--bg-body] rounded transition-colors">
                      <Copy size={14} className="text-[--color-text-tertiary]" />
                    </button>
                  </div>
                  <pre className="p-4 text-xs font-mono text-[--color-text-primary] overflow-x-auto">
                    {endpoint.requestBody}
                  </pre>
                </div>
              </div>
            )}

            {/* Response Body */}
            <div className="space-y-2">
              <div className="text-sm font-semibold text-[--color-text-primary]">
                Response Body
              </div>
              <div className="bg-[--bg-panel] rounded-lg overflow-hidden border border-[--color-border]">
                <div className="px-4 py-2 bg-[--bg-surface] border-b border-[--color-border] flex justify-between items-center">
                  <span className="text-xs font-semibold text-[--color-text-secondary]">
                    JSON
                  </span>
                  <button className="p-1 hover:bg-[--bg-body] rounded transition-colors">
                    <Copy size={14} className="text-[--color-text-tertiary]" />
                  </button>
                </div>
                <pre className="p-4 text-xs font-mono text-[--color-text-primary] overflow-x-auto">
                  {endpoint.responseBody}
                </pre>
              </div>
            </div>

            {/* Example */}
            <div className="bg-[--accent-primary]/10 rounded-lg p-4 border border-[--accent-primary]/20">
              <div className="text-sm font-semibold text-[--color-text-primary] mb-2 flex items-center gap-2">
                cURL Example
              </div>
              <code className="text-xs font-mono text-[--color-text-secondary]">
                curl -X {endpoint.method} https://api.example.com{endpoint.path}
              </code>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
