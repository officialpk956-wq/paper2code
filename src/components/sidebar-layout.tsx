"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronDown, Menu, X } from "lucide-react";

export interface SidebarItem {
  label: string;
  href: string;
  icon?: React.ReactNode;
  badge?: string;
  children?: SidebarItem[];
  count?: number;
}

interface SidebarLayoutProps {
  title: string;
  description?: string;
  items: SidebarItem[];
  children: React.ReactNode;
}

export function SidebarLayout({
  title,
  description,
  items,
  children,
}: SidebarLayoutProps) {
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [expandedItems, setExpandedItems] = useState<string[]>([]);

  const isActive = (href: string) => pathname === href || pathname.startsWith(href);

  const toggleExpanded = (label: string) => {
    setExpandedItems((prev) =>
      prev.includes(label)
        ? prev.filter((item) => item !== label)
        : [...prev, label]
    );
  };

  const renderItems = (items: SidebarItem[], level = 0) => {
    return items.map((item, idx) => {
      const active = isActive(item.href);
      const hasChildren = item.children && item.children.length > 0;
      const expanded = expandedItems.includes(item.label);

      return (
        <div key={idx}>
          <div className="flex items-center">
            {hasChildren ? (
              <button
                onClick={() => toggleExpanded(item.label)}
                className={`flex-1 flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors group ${
                  active
                    ? "text-[--accent-primary] bg-[rgb(124,58,237,0.1)]"
                    : "text-[--color-text-secondary] hover:text-[--color-text-primary] hover:bg-[--bg-surface]"
                }`}
                style={{ paddingLeft: `${level * 12 + 12}px` }}
              >
                {item.icon && <span className="flex-shrink-0">{item.icon}</span>}
                <span className="flex-1 text-left">{item.label}</span>
                <ChevronDown
                  className={`w-4 h-4 transition-transform ${expanded ? "rotate-180" : ""}`}
                />
              </button>
            ) : (
              <Link
                href={item.href}
                className={`flex-1 flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  active
                    ? "text-[--accent-primary] bg-[rgb(124,58,237,0.1)]"
                    : "text-[--color-text-secondary] hover:text-[--color-text-primary] hover:bg-[--bg-surface]"
                }`}
                style={{ paddingLeft: `${level * 12 + 12}px` }}
                onClick={() => setSidebarOpen(false)}
              >
                {item.icon && <span className="flex-shrink-0">{item.icon}</span>}
                <span className="flex-1">{item.label}</span>
                {item.count !== undefined && (
                  <span className="text-xs font-semibold text-[--color-text-tertiary] bg-[--bg-body] px-2 py-1 rounded">
                    {item.count}
                  </span>
                )}
              </Link>
            )}
          </div>

          {hasChildren && expanded && (
            <div className="mt-1">
              {renderItems(item.children ?? [], level + 1)}
            </div>
          )}
        </div>
      );
    });
  };

  return (
    <div className="flex gap-6">
      {/* Mobile sidebar toggle */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="lg:hidden fixed bottom-6 right-6 z-40 p-3 rounded-lg bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] text-white shadow-lg hover:opacity-90 transition-opacity"
      >
        {sidebarOpen ? (
          <X className="w-5 h-5" />
        ) : (
          <Menu className="w-5 h-5" />
        )}
      </button>

      {/* Sidebar */}
      <aside
        className={`fixed lg:relative left-0 top-0 h-screen lg:h-auto z-30 w-64 bg-[--bg-surface] border-r border-[--color-border] overflow-y-auto pt-16 lg:pt-0 transition-transform lg:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="p-6 border-b border-[--color-border]">
          <h2 className="text-lg font-bold text-[--color-text-primary] mb-1">
            {title}
          </h2>
          {description && (
            <p className="text-xs text-[--color-text-tertiary]">
              {description}
            </p>
          )}
        </div>

        <nav className="p-4 space-y-1">{renderItems(items)}</nav>
      </aside>

      {/* Backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-20 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <main className="flex-1 min-w-0">{children}</main>
    </div>
  );
}
