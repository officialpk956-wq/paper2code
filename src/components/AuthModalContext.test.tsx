import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { AuthModalProvider, useAuthModal } from './AuthModalContext';
import { setTokens, clearTokens } from '@/lib/api';

vi.mock('@sentry/nextjs', () => ({
  setUser: vi.fn(),
}));
import * as Sentry from '@sentry/nextjs';

function TestConsumer() {
  const { user, hydrated, signOut, signIn } = useAuthModal();
  return (
    <div>
      <div data-testid="hydrated">{hydrated ? 'yes' : 'no'}</div>
      <div data-testid="user">{user ? JSON.stringify(user) : 'null'}</div>
      <button data-testid="signout-btn" onClick={signOut}>
        Sign Out
      </button>
      <button data-testid="signin-btn" onClick={() => signIn({ id: 50, name: 'Alice', email: 'alice@example.com' })}>
        Sign In
      </button>
    </div>
  );
}

describe('AuthModalContext & Cross-Tab Auth Synchronization', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('Scenario 1: setTokens() dispatches auth-changed and populates user state without remounting', async () => {
    render(
      <AuthModalProvider>
        <TestConsumer />
      </AuthModalProvider>
    );

    expect(screen.getByTestId('user').textContent).toBe('null');

    // Simulate login in another tab or flow: profile saved and setTokens() called
    localStorage.setItem(
      'user_profile',
      JSON.stringify({ id: 42, name: 'Turing', email: 'alan@example.com' })
    );

    act(() => {
      setTokens({ access_token: 'fake_jwt_token', refresh_token: 'fake_refresh' });
    });

    expect(screen.getByTestId('user').textContent).toContain('Turing');
    expect(screen.getByTestId('user').textContent).toContain('alan@example.com');
  });

  it('Scenario 2: clearTokens() clears user state upon logout', async () => {
    localStorage.setItem(
      'user_profile',
      JSON.stringify({ id: 1, name: 'Ada', email: 'ada@example.com' })
    );
    localStorage.setItem('access_token', 'initial_token');

    render(
      <AuthModalProvider>
        <TestConsumer />
      </AuthModalProvider>
    );

    expect(screen.getByTestId('user').textContent).toContain('Ada');

    act(() => {
      clearTokens();
    });

    expect(screen.getByTestId('user').textContent).toBe('null');
    expect(localStorage.getItem('access_token')).toBeNull();
    expect(localStorage.getItem('user_profile')).toBeNull();
  });

  it('Scenario 3: Rapid consecutive auth-changed events do not cause race conditions or state corruption', async () => {
    localStorage.setItem(
      'user_profile',
      JSON.stringify({ id: 99, name: 'VonNeumann', email: 'john@example.com' })
    );
    localStorage.setItem('access_token', 'valid_token');

    render(
      <AuthModalProvider>
        <TestConsumer />
      </AuthModalProvider>
    );

    // Rapid double event dispatch
    act(() => {
      window.dispatchEvent(new Event('auth-changed'));
      window.dispatchEvent(new Event('auth-changed'));
      window.dispatchEvent(new Event('storage'));
    });

    expect(screen.getByTestId('user').textContent).toContain('VonNeumann');
  });

  it('Scenario 4: setTokens() triggers window auth-changed event dispatch', () => {
    const handler = vi.fn();
    window.addEventListener('auth-changed', handler);

    setTokens({ access_token: 'sample_token' });

    expect(handler).toHaveBeenCalledTimes(1);
    window.removeEventListener('auth-changed', handler);
  });

  it('Scenario 5: Call signIn(user) -> assert Sentry.setUser was called with {id, email}', () => {
    render(
      <AuthModalProvider>
        <TestConsumer />
      </AuthModalProvider>
    );
    act(() => {
      screen.getByTestId('signin-btn').click();
    });
    expect(Sentry.setUser).toHaveBeenCalledWith({ id: '50', email: 'alice@example.com' });
  });

  it('Scenario 6: Call signOut() -> assert Sentry.setUser was called with null', () => {
    render(
      <AuthModalProvider>
        <TestConsumer />
      </AuthModalProvider>
    );
    act(() => {
      screen.getByTestId('signout-btn').click();
    });
    expect(Sentry.setUser).toHaveBeenCalledWith(null);
  });

  it('Scenario 7b: Cross-tab logout (storage/auth-changed with no token) clears Sentry user, not just React state', () => {
    localStorage.setItem('access_token', 'valid_token');
    localStorage.setItem('user_profile', JSON.stringify({ id: 7, name: 'Grace', email: 'grace@example.com' }));
    render(
      <AuthModalProvider>
        <TestConsumer />
      </AuthModalProvider>
    );
    expect(screen.getByTestId('user').textContent).toContain('Grace');
    (Sentry.setUser as ReturnType<typeof vi.fn>).mockClear();

    // Simulate a DIFFERENT tab clearing tokens, then this tab's storage listener firing
    act(() => {
      localStorage.removeItem('access_token');
      window.dispatchEvent(new Event('storage'));
    });

    expect(screen.getByTestId('user').textContent).toBe('null');
    expect(Sentry.setUser).toHaveBeenCalledWith(null);
  });

  it('Scenario 7: Hydrate with valid session -> assert Sentry.setUser gets called once hydration resolves', () => {
    localStorage.setItem('access_token', 'valid_token');
    localStorage.setItem('user_profile', JSON.stringify({ id: 44, name: 'Bob', email: 'bob@example.com' }));
    render(
      <AuthModalProvider>
        <TestConsumer />
      </AuthModalProvider>
    );
    expect(screen.getByTestId('user').textContent).toContain('Bob');
    expect(Sentry.setUser).toHaveBeenCalledWith({ id: '44', email: 'bob@example.com' });
  });
});
