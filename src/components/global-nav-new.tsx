"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BookOpen,
  Zap,
  BookMarked,
  Cog,
  Lightbulb,
  Map,
  BarChart3,
  Menu,
  X,
  Search,
  Bell,
  LogOut,
  User,
  Code2,
  GitMerge,
  GitPullRequest,
} from "lucide-react";

const NAV_ITEMS = [
  { label: "Learn", href: "/learn", icon: BookOpen },
  { label: "Architectures", href: "/architectures", icon: Zap },
  { label: "Papers", href: "/papers", icon: BookMarked },
  { label: "System Design", href: "/system-design", icon: Cog },
  { label: "Problems", href: "/problems", icon: Lightbulb },
  { label: "Roadmaps", href: "/roadmaps", icon: Map },
  { label: "Dashboard", href: "/dashboard", icon: BarChart3 },
  { label: "Paper → Code", href: "/paper-to-code", icon: Code2 },
  { label: "Evolution", href: "/evolution", icon: GitMerge },
  { label: "Tensor Trace", href: "/tensor-trace", icon: GitPullRequest },
];

export function GlobalNavNew() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);

  const isActive = (href: string) => pathname.startsWith(href);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-14 bg-[rgb(9,9,15,0.98)] border-b border-[--color-border] backdrop-blur-md">
      <div className="h-full px-4 flex items-center justify-between">
        {/* Logo */}
        <Link
          href="/"
          className="flex items-center gap-2 mr-8 flex-shrink-0 group"
        >
          <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] flex items-center justify-center text-white font-bold text-sm">
            P
          </div>
          <span className="font-heading font-bold text-sm gradient-text hidden sm:block">
            paper2code
          </span>
        </Link>

        {/* Desktop Nav */}
        <div className="hidden lg:flex items-center gap-1 flex-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`group flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
                  active
                    ? "text-[--accent-primary] bg-[rgb(124,58,237,0.1)]"
                    : "text-[--color-text-secondary] hover:text-[--color-text-primary] hover:bg-[--bg-surface]"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden xl:inline">{item.label}</span>
              </Link>
            );
          })}
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-3 ml-4">
          {/* Search */}
          <button className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-md bg-[--bg-surface] border border-[--color-border] text-[--color-text-tertiary] hover:text-[--color-text-secondary] text-xs transition-colors">
            <Search className="w-4 h-4" />
            <span>⌘K</span>
          </button>

          {/* Search icon mobile */}
          <button className="md:hidden p-2 rounded-md hover:bg-[--bg-surface] transition-colors">
            <Search className="w-4 h-4" />
          </button>

          {/* Notifications */}
          <button className="relative p-2 rounded-md hover:bg-[--bg-surface] transition-colors">
            <Bell className="w-4 h-4" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-[--color-hard] rounded-full" />
          </button>

          {/* Profile */}
          <div className="relative hidden sm:block">
            <button
              onClick={() => setProfileOpen(!profileOpen)}
              className="w-8 h-8 rounded-full bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] flex items-center justify-center text-white text-xs font-bold hover:opacity-80 transition-opacity"
            >
              A
            </button>

            {profileOpen && (
              <div className="absolute right-0 mt-2 w-48 bg-[--bg-surface] border border-[--color-border] rounded-lg shadow-lg overflow-hidden animate-in fade-in-0 slide-in-from-top-2 duration-200">
                <div className="px-4 py-3 border-b border-[--color-border]">
                  <p className="text-sm font-semibold">Alice Chen</p>
                  <p className="text-xs text-[--color-text-tertiary]">
                    alice@example.com
                  </p>
                </div>
                <div className="py-2">
                  <a
                    href="#"
                    className="flex items-center gap-3 px-4 py-2 text-sm text-[--color-text-secondary] hover:bg-[--bg-panel] hover:text-[--color-text-primary] transition-colors"
                  >
                    <User className="w-4 h-4" />
                    Profile
                  </a>
                </div>
                <div className="border-t border-[--color-border] py-2">
                  <button className="w-full flex items-center gap-3 px-4 py-2 text-sm text-[--color-hard] hover:bg-[--bg-panel] transition-colors text-left">
                    <LogOut className="w-4 h-4" />
                    Sign out
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Mobile menu */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-2 rounded-md hover:bg-[--bg-surface] transition-colors"
          >
            {mobileMenuOpen ? (
              <X className="w-4 h-4" />
            ) : (
              <Menu className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="absolute top-14 left-0 right-0 bg-[--bg-surface] border-b border-[--color-border] lg:hidden max-h-[calc(100vh-56px)] overflow-y-auto">
          <div className="py-2">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`flex items-center gap-3 px-4 py-3 text-sm font-medium transition-colors ${
                    active
                      ? "bg-[rgb(124,58,237,0.1)] text-[--accent-primary]"
                      : "text-[--color-text-secondary] hover:text-[--color-text-primary] hover:bg-[--bg-panel]"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </nav>
  );
}
