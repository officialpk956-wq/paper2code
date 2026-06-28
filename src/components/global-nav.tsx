"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Search, Bell, Settings, LogOut, User } from "lucide-react";

const NAV_ITEMS = [
  { label: "Learn", href: "/learn" },
  { label: "Architectures", href: "/architectures" },
  { label: "Papers", href: "/papers" },
  { label: "System Design", href: "/system-design" },
  { label: "Problems", href: "/problems" },
  { label: "Roadmaps", href: "/roadmaps" },
];

export function GlobalNav() {
  const pathname = usePathname();
  const router = useRouter();
  const [profileOpen, setProfileOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem("access_token");
    setIsLoggedIn(!!token);
  }, [pathname]); // Re-check on navigation

  const isActive = (href: string) => {
    return pathname.startsWith(href);
  };

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    setIsLoggedIn(false);
    router.push("/login");
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-12 bg-[rgb(9,9,15,0.98)] border-b border-[--color-border] backdrop-blur-md hidden md:block">
      <div className="h-full px-4 flex items-center justify-between">
        {/* Logo */}
        <Link
          href="/"
          className="flex items-center gap-2 mr-8 flex-shrink-0 group"
        >
          <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] flex items-center justify-center text-white font-bold text-sm">
            P
          </div>
          <span className="font-heading font-bold text-sm bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] bg-clip-text text-transparent hidden sm:block">
            paper2code
          </span>
        </Link>

        {/* Nav Items */}
        <div className="flex items-center gap-1 flex-1 overflow-x-auto no-scrollbar">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-item whitespace-nowrap ${
                isActive(item.href) ? "active" : ""
              }`}
            >
              {item.label}
            </Link>
          ))}
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-3 ml-4">
          {/* Search */}
          <button
            onClick={() =>
              window.dispatchEvent(
                new KeyboardEvent("keydown", { key: "k", ctrlKey: true })
              )
            }
            className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-md bg-[--bg-surface] border border-[--color-border] text-[--color-text-tertiary] hover:text-[--color-text-secondary] text-xs transition-colors"
          >
            <Search className="w-4 h-4" />
            <span>⌘K</span>
          </button>

          {/* Search icon (mobile) */}
          <button
            onClick={() =>
              window.dispatchEvent(
                new KeyboardEvent("keydown", { key: "k", ctrlKey: true })
              )
            }
            className="lg:hidden p-2 rounded-md hover:bg-[--bg-surface] transition-colors"
          >
            <Search className="w-4 h-4" />
          </button>

          {!isLoggedIn ? (
            <div className="flex items-center gap-2">
              <Link href="/login" className="px-3 py-1.5 text-sm font-medium text-[--color-text-secondary] hover:text-white transition-colors">
                Log In
              </Link>
              <Link href="/signup" className="px-3 py-1.5 text-sm font-medium bg-[--accent-primary] text-white rounded-md hover:bg-[--accent-light] transition-colors">
                Sign Up
              </Link>
            </div>
          ) : (
            <>
              {/* Dashboard Link */}
              <Link href="/dashboard" className="hidden lg:block px-3 py-1.5 text-sm font-medium text-[--color-text-secondary] hover:text-white transition-colors">
                Dashboard
              </Link>

              {/* Notifications */}
              <button className="relative p-2 rounded-md hover:bg-[--bg-surface] transition-colors">
                <Bell className="w-4 h-4" />
                <span className="absolute top-1 right-1 w-2 h-2 bg-[--color-hard] rounded-full" />
              </button>

              {/* Profile Menu */}
              <div className="relative">
                <button
                  onClick={() => setProfileOpen(!profileOpen)}
                  className="w-8 h-8 rounded-full bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] flex items-center justify-center text-white text-xs font-bold hover:opacity-80 transition-opacity"
                >
                  A
                </button>

                {/* Dropdown Menu */}
                {profileOpen && (
                  <div className="absolute right-0 mt-2 w-48 bg-[--bg-surface] border border-[--color-border] rounded-lg shadow-lg overflow-hidden animate-in fade-in-0 slide-in-from-top-2 duration-200">
                    <div className="px-4 py-3 border-b border-[--color-border]">
                      <p className="text-sm font-semibold text-[--color-text-primary]">
                        User
                      </p>
                      <p className="text-xs text-[--color-text-tertiary]">
                        user@example.com
                      </p>
                    </div>

                    <div className="py-2">
                      <Link
                        href="/dashboard"
                        className="flex items-center gap-3 px-4 py-2 text-sm text-[--color-text-secondary] hover:bg-[--bg-panel] hover:text-[--color-text-primary] transition-colors"
                      >
                        <User className="w-4 h-4" />
                        Dashboard
                      </Link>
                      <Link
                        href="#"
                        className="flex items-center gap-3 px-4 py-2 text-sm text-[--color-text-secondary] hover:bg-[--bg-panel] hover:text-[--color-text-primary] transition-colors"
                      >
                        <Settings className="w-4 h-4" />
                        Settings
                      </Link>
                    </div>

                    <div className="border-t border-[--color-border] py-2">
                      <button 
                        onClick={handleLogout}
                        className="w-full flex items-center gap-3 px-4 py-2 text-sm text-[--color-hard] hover:bg-[--bg-panel] transition-colors text-left"
                      >
                        <LogOut className="w-4 h-4" />
                        Sign out
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
