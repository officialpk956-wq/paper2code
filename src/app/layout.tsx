import type { Metadata, Viewport } from "next";
import { Inter, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/providers";
import { AppShell } from "@/components/layout/app-shell";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

const plusJakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-heading",
});

export const metadata: Metadata = {
  title: "Paper2Code — AI Engineering Learning Ecosystem",
  description:
    "Learn transformers, LLMs, and deep learning through interactive problems, papers, and system design cases.",
  authors: [{ name: "Paper2Code" }],
  keywords: [
    "machine learning",
    "deep learning",
    "transformers",
    "llm",
    "ai engineering",
    "learning platform",
  ],
  icons: {
    icon: "/favicon.ico",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <style>{`
          :root {
            /* Colors - Dark Mode Professional */
            --bg-body: #09090F;
            --bg-surface: #0D0D14;
            --bg-panel: #111827;
            --bg-hover: #1A1F2E;
            --bg-active: #232D3D;

            /* Text Colors */
            --color-text-primary: #E2E8F0;
            --color-text-secondary: #CBD5E1;
            --color-text-tertiary: #94A3B8;
            --color-text-muted: #64748B;

            /* Borders */
            --color-border: #1E293B;
            --color-border-light: #2A2D3A;
            --color-border-focus: #3B4F63;
            --color-divider: #1A1F2E;
            --color-muted: #334155;

            /* Accents */
            --accent-primary: #7C3AED;
            --accent-light: #A78BFA;
            --accent-cyan: #06B6D4;
            --accent-emerald: #10B981;
            --accent-amber: #F59E0B;
            --accent-red: #EF4444;

            /* Status Colors */
            --status-success: #10B981;
            --status-active: #3B82F6;
            --status-warning: #F59E0B;
            --status-error: #EF4444;
            --status-locked: #64748B;

            /* Legacy Colors (kept for compatibility) */
            --color-easy: #10B981;
            --color-medium: #F59E0B;
            --color-hard: #EF4444;
            --color-expert: #EC4899;

            /* Typography */
            --font-heading: "Plus Jakarta Sans", "Inter", sans-serif;
            --font-sans: "Inter", system-ui, sans-serif;
            --font-mono: "JetBrains Mono", monospace;

            /* Text Sizes - Professional Density */
            --text-display: clamp(32px, 5vw, 40px);
            --text-h1: 24px;
            --text-h2: 18px;
            --text-h3: 14px;
            --text-label: 13px;
            --text-base: 12px;
            --text-small: 11px;
            --text-tiny: 10px;
            --text-mono: 11px;

            /* Spacing Tokens */
            --space-1: 4px;
            --space-2: 8px;
            --space-3: 12px;
            --space-4: 16px;
            --space-6: 24px;
            --space-8: 32px;
            --space-12: 48px;

            /* Border Radius */
            --radius-sm: 4px;
            --radius-md: 6px;
            --radius-lg: 8px;
            --radius-xl: 12px;
            --radius-2xl: 16px;

            /* Shadows */
            --shadow-none: none;
            --shadow-hover: 0 2px 6px rgba(0, 0, 0, 0.4);
            --shadow-raised: 0 4px 12px rgba(0, 0, 0, 0.3);
            --shadow-modal: 0 10px 40px rgba(0, 0, 0, 0.5);
            --shadow-glow: 0 0 12px rgba(124, 58, 237, 0.3);

            /* Transitions - Professional Tool Timing */
            --transition-fast: 100ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 300ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slowest: 500ms cubic-bezier(0.34, 1.56, 0.64, 1);
          }

          * {
            border-color: var(--color-border);
          }
        `}</style>
      </head>
      <body
        className={`${inter.variable} ${plusJakarta.variable} bg-[--bg-body] text-[--color-text-primary] font-sans antialiased`}
      >
        <Providers>
          <AppShell>
            {children}
          </AppShell>
        </Providers>
      </body>
    </html>
  );
}
