'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { isLoggedIn } from '@/lib/api';
import { useAuthModal } from './AuthModalContext';

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { open } = useAuthModal();
  const router = useRouter();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (!isLoggedIn()) {
      // Redirect to home and immediately prompt sign up
      router.replace('/');
      open('signup');
    } else {
      setChecked(true);
    }
  }, [open, router]);

  // Don't render children until we've confirmed the user is logged in
  if (!checked) return null;

  return <>{children}</>;
}
