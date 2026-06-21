'use client';

import { SchemaDesigner } from '@/components/system-design/schema-designer';

export default function SchemaDesignPage() {
  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      <SchemaDesigner />
    </div>
  );
}
