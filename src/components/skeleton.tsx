export function Skeleton({
  className = "",
}: {
  className?: string;
}) {
  return (
    <div
      className={`bg-gradient-to-r from-[--bg-surface] via-[--bg-panel] to-[--bg-surface] animate-shimmer ${className}`}
    />
  );
}

export function PageSkeleton() {
  return (
    <div className="page-container space-y-6">
      {/* Header */}
      <div className="space-y-3">
        <Skeleton className="h-8 w-1/3 rounded-lg" />
        <Skeleton className="h-4 w-2/3 rounded-md" />
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="card p-4">
            <Skeleton className="h-40 w-full rounded-lg mb-4" />
            <Skeleton className="h-4 w-2/3 rounded-md mb-2" />
            <Skeleton className="h-3 w-1/2 rounded-md" />
          </div>
        ))}
      </div>
    </div>
  );
}

export function CardSkeleton() {
  return (
    <div className="card p-4">
      <Skeleton className="h-48 w-full rounded-lg mb-4" />
      <Skeleton className="h-5 w-3/4 rounded-md mb-3" />
      <Skeleton className="h-4 w-full rounded-md mb-2" />
      <Skeleton className="h-4 w-5/6 rounded-md" />
    </div>
  );
}

export function ListSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="card p-4 flex items-center gap-4">
          <Skeleton className="h-12 w-12 rounded-lg flex-shrink-0" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-1/2 rounded-md" />
            <Skeleton className="h-3 w-2/3 rounded-md" />
          </div>
        </div>
      ))}
    </div>
  );
}
