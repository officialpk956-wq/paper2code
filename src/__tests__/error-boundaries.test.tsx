import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import RootError from '@/app/error';
import ProtectedError from '@/app/(protected)/error';
import NotFound from '@/app/not-found';
import GlobalError from '@/app/global-error';

describe('React Error Boundaries & Not-Found Pages (Task P2-2)', () => {
  it('Scenario 1: Root error.tsx renders friendly error message and calls reset() on click', () => {
    const resetMock = vi.fn();
    const testError = new Error('Test crash in route');

    render(<RootError error={testError} reset={resetMock} />);

    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Back to Home/i })).toHaveAttribute('href', '/');

    const tryAgainBtn = screen.getByRole('button', { name: /Try again/i });
    fireEvent.click(tryAgainBtn);
    expect(resetMock).toHaveBeenCalledTimes(1);
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
  });
});
