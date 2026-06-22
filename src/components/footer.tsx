"use client";

import Link from "next/link";
import { Github, Twitter, Mail } from "lucide-react";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-[--color-border] bg-[--bg-surface] mt-20">
      <div className="page-container">
        {/* Content */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 py-12 mb-8">
          {/* Brand */}
          <div className="md:col-span-1">
            <Link href="/" className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] flex items-center justify-center text-white font-bold">
                P
              </div>
              <span className="font-heading font-bold text-sm gradient-text">
                paper2code
              </span>
            </Link>
            <p className="text-sm text-[--color-text-tertiary] leading-relaxed">
              AI Engineering Learning Ecosystem. Master deep learning through
              problems, papers, and system design.
            </p>
          </div>

          {/* Sections */}
          <div>
            <h4 className="font-semibold text-sm mb-4 text-[--color-text-primary]">
              Learn
            </h4>
            <ul className="space-y-2">
              <li>
                <Link
                  href="/learn"
                  className="text-sm text-[--color-text-tertiary] hover:text-[--color-text-secondary] transition-colors"
                >
                  Lessons
                </Link>
              </li>
              <li>
                <Link
                  href="/papers"
                  className="text-sm text-[--color-text-tertiary] hover:text-[--color-text-secondary] transition-colors"
                >
                  Papers
                </Link>
              </li>
              <li>
                <Link
                  href="/roadmaps"
                  className="text-sm text-[--color-text-tertiary] hover:text-[--color-text-secondary] transition-colors"
                >
                  Roadmaps
                </Link>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-semibold text-sm mb-4 text-[--color-text-primary]">
              Resources
            </h4>
            <ul className="space-y-2">
              <li>
                <Link
                  href="/architectures"
                  className="text-sm text-[--color-text-tertiary] hover:text-[--color-text-secondary] transition-colors"
                >
                  Architectures
                </Link>
              </li>
              <li>
                <Link
                  href="/problems"
                  className="text-sm text-[--color-text-tertiary] hover:text-[--color-text-secondary] transition-colors"
                >
                  Problems
                </Link>
              </li>
              <li>
                <Link
                  href="/system-design"
                  className="text-sm text-[--color-text-tertiary] hover:text-[--color-text-secondary] transition-colors"
                >
                  System Design
                </Link>
              </li>
            </ul>
          </div>

          {/* Connect */}
          <div>
            <h4 className="font-semibold text-sm mb-4 text-[--color-text-primary]">
              Connect
            </h4>
            <div className="flex gap-3">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg hover:bg-[--bg-panel] transition-colors text-[--color-text-tertiary] hover:text-[--color-text-secondary]"
              >
                <Github className="w-5 h-5" />
              </a>
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg hover:bg-[--bg-panel] transition-colors text-[--color-text-tertiary] hover:text-[--color-text-secondary]"
              >
                <Twitter className="w-5 h-5" />
              </a>
              <a
                href="mailto:hello@paper2code.dev"
                className="p-2 rounded-lg hover:bg-[--bg-panel] transition-colors text-[--color-text-tertiary] hover:text-[--color-text-secondary]"
              >
                <Mail className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-[--color-border]" />

        {/* Bottom */}
        <div className="py-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-xs text-[--color-text-tertiary]">
            © {currentYear} Paper2Code. All rights reserved.
          </p>
          <div className="flex gap-6 text-xs">
            <a
              href="#"
              className="text-[--color-text-tertiary] hover:text-[--color-text-secondary] transition-colors"
            >
              Privacy
            </a>
            <a
              href="#"
              className="text-[--color-text-tertiary] hover:text-[--color-text-secondary] transition-colors"
            >
              Terms
            </a>
            <a
              href="#"
              className="text-[--color-text-tertiary] hover:text-[--color-text-secondary] transition-colors"
            >
              Status
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
