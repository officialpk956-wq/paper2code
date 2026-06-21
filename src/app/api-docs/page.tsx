'use client';

import { APIDocs } from '@/components/system-design/api-docs';

export default function APIDocsPage() {
  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      <APIDocs />
    </div>
  );
}
