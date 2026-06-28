'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Button } from './ui/Button';

export function CookieBanner() {
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    const consent = localStorage.getItem('cookie_consent');
    if (!consent) {
      setShowBanner(true);
    }
  }, []);

  const handleConsent = (level: 'all' | 'essential') => {
    localStorage.setItem('cookie_consent', level);
    setShowBanner(false);
    
    if (level === 'essential') {
      // Disable PostHog tracking if window.posthog exists
      if (typeof window !== 'undefined' && (window as any).posthog) {
        (window as any).posthog.opt_out_capturing();
      }
    }
  };

  if (!showBanner) return null;

  return (
    <div 
      role="dialog" 
      aria-label="Cookie consent"
      className="fixed bottom-0 left-0 right-0 z-[100] p-4 bg-[#1a1a2e] border-t border-[--color-border] text-white shadow-[0_-4px_20px_rgba(0,0,0,0.3)] animate-in slide-in-from-bottom-full duration-500"
    >
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        <p className="text-sm text-slate-300">
          We use cookies for authentication and analytics. See our{' '}
          <Link href="/privacy" className="text-[--accent-primary] hover:text-[--accent-light] underline underline-offset-2">
            Privacy Policy
          </Link>
          .
        </p>
        <div className="flex items-center gap-3 shrink-0">
          <Button 
            variant="secondary" 
            onClick={() => handleConsent('essential')}
            style={{ borderColor: 'rgba(255,255,255,0.2)', color: '#cbd5e1' }}
          >
            Essential Only
          </Button>
          <Button 
            variant="primary" 
            onClick={() => handleConsent('all')}
          >
            Accept All
          </Button>
        </div>
      </div>
    </div>
  );
}
