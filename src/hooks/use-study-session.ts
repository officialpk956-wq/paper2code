'use client';

/**
 * React hook that manages a study session for the current page visit.
 * Call at the top of any content page to automatically track time spent.
 */

import { useEffect } from 'react';
import {
  startSession,
  endSession,
  recordContentVisit,
  getCurrentSession,
} from '@/lib/study-sessions';

interface UseStudySessionOptions {
  contentType?: string;
  contentSlug?: string;
}

export function useStudySession({
  contentType,
  contentSlug,
}: UseStudySessionOptions = {}): void {
  useEffect(() => {
    // Start a session if none is active
    let session = getCurrentSession();
    if (!session) {
      session = startSession();
    }

    // Record this content as visited
    if (contentType && contentSlug) {
      recordContentVisit(contentType, contentSlug);
    }

    const handleUnload = () => endSession();
    window.addEventListener('beforeunload', handleUnload);

    return () => {
      window.removeEventListener('beforeunload', handleUnload);
    };
  }, [contentType, contentSlug]);
}
