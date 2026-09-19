import PageTransition from '@/components/PageTransition';

// AuthGuard is applied per-page (around each page's interactive content only),
// not here — a client component wrapping every child would force every route
// in this group to fall back to its `!hydrated` loading state during SSR,
// discarding any server-rendered metadata/summary before it reaches the page.
export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <PageTransition>{children}</PageTransition>
  );
}
