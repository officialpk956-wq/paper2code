import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { TopNavbar } from '@/components/TopNavbar';
import { AuthModalProvider } from '@/components/AuthModalContext';
import { AuthModal } from '@/components/AuthModal';
import { AnimatedBackground } from '@/components/AnimatedBackground';
import 'katex/dist/katex.min.css';
import PageTransition from '@/components/PageTransition';

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' });

export const metadata: Metadata = {
  title: 'paper2code — From Papers to Code',
  description: 'Upload ML papers and get coding challenges, architecture diagrams, and guided implementations.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`dark ${inter.variable}`}>
      <body>
        <AnimatedBackground />
        <AuthModalProvider>
          <TopNavbar />
          <PageTransition>
            <div className="page-fade-in">{children}</div>
          </PageTransition>
          <AuthModal />
        </AuthModalProvider>
      </body>
    </html>
  );
}
