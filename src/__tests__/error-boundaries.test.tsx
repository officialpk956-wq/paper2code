import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import fs from 'fs';
import path from 'path';

vi.mock('@sentry/nextjs', () => ({
  captureException: vi.fn(),
}));
import * as Sentry from '@sentry/nextjs';

import RootError from '@/app/error';
import ProtectedError from '@/app/(protected)/error';
import NotFound from '@/app/not-found';
import GlobalError from '@/app/global-error';

describe('React Error Boundaries & Not-Found Pages (Task P2-2)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('Scenario 1: Root error.tsx renders friendly error message and calls reset() on click', () => {
    const resetMock = vi.fn();
    const testError = new Error('Test crash in route');

    render(<RootError error={testError} reset={resetMock} />);

    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Back to Home/i })).toHaveAttribute('href', '/');

    const tryAgainBtn = screen.getByRole('button', { name: /Try again/i });
    fireEvent.click(tryAgainBtn);
    expect(resetMock).toHaveBeenCalledTimes(1);
    expect(Sentry.captureException).toHaveBeenCalledWith(testError);
  });

  it('Scenario 2: Protected error.tsx renders recovery UI with Dojo link and calls reset()', () => {
    const resetMock = vi.fn();
    const testError = new Error('Protected workspace failure');

    render(<ProtectedError error={testError} reset={resetMock} />);

    expect(screen.getByText(/Workspace Error/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Back to Dojo/i })).toHaveAttribute('href', '/dojo');

    const tryAgainBtn = screen.getByRole('button', { name: /Try again/i });
    fireEvent.click(tryAgainBtn);
    expect(resetMock).toHaveBeenCalledTimes(1);
    expect(Sentry.captureException).toHaveBeenCalledWith(testError);
  });

  it('Scenario 3: not-found.tsx renders 404 page with home link', () => {
    render(<NotFound />);

    expect(screen.getByText(/404 - Page Not Found/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Back to Home/i })).toHaveAttribute('href', '/');
    expect(screen.getByRole('link', { name: /Go to Dojo/i })).toHaveAttribute('href', '/dojo');
  });

  it('Scenario 4: global-error.tsx renders complete html/body layout and calls reset()', () => {
    const resetMock = vi.fn();
    const testError = new Error('Fatal layout crash');

    render(<GlobalError error={testError} reset={resetMock} />);

    expect(screen.getByText(/Application Error/i)).toBeInTheDocument();
    const tryAgainBtn = screen.getByRole('button', { name: /Try again/i });
    fireEvent.click(tryAgainBtn);
    expect(resetMock).toHaveBeenCalledTimes(1);
    expect(Sentry.captureException).toHaveBeenCalledWith(testError);

    // Static check for no app-level dependencies
    const fileContent = fs.readFileSync(path.join(process.cwd(), 'src/app/global-error.tsx'), 'utf-8');
    const importLines = fileContent.split('\n').filter(line => line.startsWith('import'));
    importLines.forEach(line => {
      // Must only import from allowed safe modules
      expect(line).toMatch(/from\s+['"](@sentry\/nextjs|react|next\/.+)['"]/);
    });
  });
});
