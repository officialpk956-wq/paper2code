import '@testing-library/jest-dom/vitest';
import { vi, beforeEach, afterEach } from 'vitest';

// Only set up browser-only globals in jsdom environment
if (typeof window !== 'undefined') {
  // Mock localStorage
  const localStorageMock = (() => {
    let store: Record<string, string> = {};
    return {
      getItem: (key: string) => store[key] ?? null,
      setItem: (key: string, value: string) => { store[key] = String(value); },
      removeItem: (key: string) => { delete store[key]; },
      clear: () => { store = {}; },
      get length() { return Object.keys(store).length; },
      key: (index: number) => Object.keys(store)[index] ?? null,
    };
  })();

  Object.defineProperty(window, 'localStorage', {
    value: localStorageMock,
    writable: true,
  });

  // Silence React warning noise from intentional test errors
  const originalError = console.error.bind(console);
  beforeEach(() => {
    console.error = (...args: unknown[]) => {
      const msg = typeof args[0] === 'string' ? args[0] : '';
      if (msg.includes('Warning:') || msg.includes('act(')) return;
      originalError(...args);
    };
    localStorageMock.clear();
  });

  afterEach(() => {
    console.error = originalError;
    vi.clearAllMocks();
  });
}

// Mock next/navigation globally
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), back: vi.fn(), replace: vi.fn() }),
  usePathname: () => '/',
  useSearchParams: () => new URLSearchParams(),
}));

// Mock next/link globally
vi.mock('next/link', () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  default: ({ href, children, ...props }: { href: string; children: unknown; [key: string]: unknown }) => {
    // Use createElement to avoid JSX transform issues in setup file
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const React = require('react') as typeof import('react');
    return React.createElement('a', { href, ...props }, children as React.ReactNode);
  },
}));
