"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X, Search } from "lucide-react";

const NAV_ITEMS = [
  { label: "Learn", href: "/learn", icon: "📚" },
  { label: "Architectures", href: "/architectures", icon: "🏗" },
  { label: "Papers", href: "/papers", icon: "📄" },
  { label: "System Design", href: "/system-design", icon: "⚙" },
  { label: "Problems", href: "/problems", icon: "⚡" },
  { label: "Roadmaps", href: "/roadmaps", icon: "🗺" },
];

export function MobileNav() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  const isActive = (href: string) => {
    return pathname.startsWith(href);
  };

  return (
    <div className="md:hidden fixed top-0 left-0 right-0 z-50 h-12 bg-[rgb(9,9,15,0.98)] border-b border-[--color-border] backdrop-blur-md">
      <div className="h-full px-4 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 flex-shrink-0">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] flex items-center justify-center text-white font-bold text-sm">
            P
          </div>
        </Link>

        {/* Center: Search */}
        <button className="flex-1 flex items-center gap-2 mx-3 px-3 py-1.5 rounded-md bg-[--bg-surface] border border-[--color-border]">
          <Search className="w-3 h-3 text-[--color-text-tertiary]" />
          <span className="text-xs text-[--color-text-tertiary]">⌘K</span>
        </button>

        {/* Menu Button */}
        <button
          onClick={() => setOpen(!open)}
          className="p-2 rounded-md hover:bg-[--bg-surface] transition-colors"
        >
          {open ? (
            <X className="w-4 h-4" />
          ) : (
            <Menu className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Mobile Menu */}
      {open && (
        <div className="absolute top-12 left-0 right-0 bg-[--bg-surface] border-b border-[--color-border] max-h-[calc(100vh-48px)] overflow-y-auto">
          <div className="py-2">
            {NAV_ITEMS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                className={`flex items-center gap-3 px-4 py-3 text-sm font-medium transition-colors ${
                  isActive(item.href)
                    ? "bg-[rgb(124,58,237,0.1)] text-[--color-text-primary]"
                    : "text-[--color-text-secondary] hover:text-[--color-text-primary] hover:bg-[--bg-panel]"
                }`}
              >
                <span>{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
