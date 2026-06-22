"use client";

import Link from "next/link";
import { ChevronRight, Home } from "lucide-react";

export interface BreadcrumbItem {
  label: string;
  href?: string;
  current?: boolean;
}

interface BreadcrumbProps {
  items: BreadcrumbItem[];
}

export function Breadcrumb({ items }: BreadcrumbProps) {
  return (
    <nav className="flex items-center gap-1 mb-6" aria-label="Breadcrumb">
      <Link
        href="/"
        className="p-1 hover:bg-[--bg-surface] rounded transition-colors"
        title="Home"
      >
        <Home className="w-4 h-4 text-[--color-text-tertiary] hover:text-[--color-text-secondary]" />
      </Link>

      {items.map((item, index) => (
        <div key={index} className="flex items-center gap-1">
          <ChevronRight className="w-4 h-4 text-[--color-border]" />
          {item.current ? (
            <span className="text-sm text-[--color-text-secondary] px-2 py-1">
              {item.label}
            </span>
          ) : (
            <Link
              href={item.href || "#"}
              className="text-sm text-[--color-text-tertiary] hover:text-[--color-text-secondary] px-2 py-1 rounded hover:bg-[--bg-surface] transition-colors"
            >
              {item.label}
            </Link>
          )}
        </div>
      ))}
    </nav>
  );
}
