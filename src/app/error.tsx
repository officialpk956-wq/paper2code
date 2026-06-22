"use client";

import { useEffect } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="page-container py-20">
      <div className="max-w-md mx-auto">
        <div className="bg-[--bg-surface] border border-[--color-border] rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <AlertCircle className="w-6 h-6 text-[--color-hard]" />
            <h2 className="text-lg font-semibold text-[--color-text-primary]">
              Something went wrong
            </h2>
          </div>
          <p className="text-sm text-[--color-text-tertiary] mb-6">
            {error.message || "An unexpected error occurred."}
          </p>
          <button
            onClick={() => reset()}
            className="btn-primary flex items-center gap-2 justify-center w-full text-sm"
          >
            <RefreshCw className="w-4 h-4" />
            Try again
          </button>
        </div>
      </div>
    </div>
  );
}
