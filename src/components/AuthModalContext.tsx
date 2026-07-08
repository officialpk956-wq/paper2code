'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { clearTokens, isLoggedIn } from '@/lib/api';

type Tab = 'signin' | 'signup';
// Matches the backend register response's user object: { id, name, email }.
// id is optional because the login endpoint only returns a token — in that
// case we derive a minimal profile locally.
type User = { id?: number; name: string; email: string };

type AuthModalContextValue = {
  isOpen: boolean;
  tab: Tab;
  open: (tab?: Tab) => void;
  close: () => void;
  setTab: (tab: Tab) => void;
  user: User | null;
  hydrated: boolean;
  signIn: (user: User) => void;
  signOut: () => void;
};

const AuthModalContext = createContext<AuthModalContextValue | null>(null);

export function AuthModalProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const [tab, setTab]       = useState<Tab>('signin');
  const [user, setUser]     = useState<User | null>(null);
  const [hydrated, setHydrated] = useState(false);

  const hydrate = useCallback(() => {
    if (!isLoggedIn()) {
      setUser(null);
      setHydrated(true);
      return;
    }
    try {
      const raw = localStorage.getItem('user_profile');
      if (!raw) {
        setHydrated(true);
        return;
      }
      const p = JSON.parse(raw);
      if (p && typeof p === 'object' && (p.name || p.email)) {
        setUser({
          id: typeof p.id === 'number' ? p.id : undefined,
          name: p.name || String(p.email).split('@')[0],
          email: p.email ?? '',
        });
      }
    } catch {
      setUser(null);
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  useEffect(() => {
    const handleStorage = () => {
      if (!isLoggedIn()) {
        setUser(null);
      } else {
        setUser(prev => {
           if (!prev) hydrate();
           return prev;
        });
      }
    };
    
    window.addEventListener('storage', handleStorage);
    window.addEventListener('auth-changed', handleStorage);
    return () => {
      window.removeEventListener('storage', handleStorage);
      window.removeEventListener('auth-changed', handleStorage);
    };
  }, [hydrate]);

  const open    = useCallback((next?: Tab) => { if (next) setTab(next); setIsOpen(true); }, []);
  const close   = useCallback(() => setIsOpen(false), []);
  const signIn  = useCallback((u: User) => { 
    localStorage.setItem('user_profile', JSON.stringify(u)); 
    setUser(u); 
    setIsOpen(false); 
  }, []);
  const signOut = useCallback(() => {
    clearTokens();
  }, []);

  const value = useMemo(
    () => ({ isOpen, tab, open, close, setTab, user, hydrated, signIn, signOut }),
    [isOpen, tab, open, close, user, hydrated, signIn, signOut],
  );

  return <AuthModalContext.Provider value={value}>{children}</AuthModalContext.Provider>;
}

export function useAuthModal() {
  const ctx = useContext(AuthModalContext);
  if (!ctx) throw new Error('useAuthModal must be used inside AuthModalProvider');
  return ctx;
}
