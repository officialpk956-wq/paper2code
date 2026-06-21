"use client";

import { ReactNode, useState, useEffect } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

export function ErrorBoundary({ children, fallback }: ErrorBoundaryProps) {
  const [hasError, setHasError] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      setHasError(true);
      setError(event.error);
    };

    window.addEventListener("error", handleError);
    return () => window.removeEventListener("error", handleError);
  }, []);

  if (hasError) {
    return (
      fallback || (
        <div className="page-container">
          <div className="max-w-md mx-auto py-12">
            <div className="bg-[--bg-surface] border border-[--color-border] rounded-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <AlertCircle className="w-6 h-6 text-[--color-hard]" />
                <h2 className="text-lg font-semibold text-[--color-text-primary]">
                  Something went wrong
                </h2>
              </div>
              <p className="text-sm text-[--color-text-tertiary] mb-6">
                {error?.message || "An unexpected error occurred. Please try again."}
              </p>
              <button
                onClick={() => {
                  setHasError(false);
                  setError(null);
                  window.location.reload();
                }}
                className="btn-primary flex items-center gap-2 text-sm w-full justify-center"
              >
                <RefreshCw className="w-4 h-4" />
                Try again
              </button>
            </div>
          </div>
        </div>
      )
    );
  }

  return children;
}
