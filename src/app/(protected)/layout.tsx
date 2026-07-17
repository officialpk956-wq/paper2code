import PageTransition from '@/components/PageTransition';
import { AuthGuard } from '@/components/AuthGuard';

export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <PageTransition>
      <AuthGuard>{children}</AuthGuard>
    </PageTransition>
  );
}
