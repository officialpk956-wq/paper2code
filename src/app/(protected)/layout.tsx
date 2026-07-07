import PageTransition from '@/components/PageTransition';

export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <PageTransition>{children}</PageTransition>;
}
