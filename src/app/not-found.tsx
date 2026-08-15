import Link from 'next/link';
import { FileQuestion, Home, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-[calc(100dvh-56px)] bg-[#0A0A0A] text-[#EDEDED] flex flex-col items-center justify-center p-6 text-center">
      <div className="w-16 h-16 rounded-2xl bg-[#A78BFA]/10 border border-[#A78BFA]/20 flex items-center justify-center mb-6 text-[#A78BFA]">
        <FileQuestion size={32} />
      </div>

      <h1 className="text-3xl font-bold tracking-tight mb-2">404 - Page Not Found</h1>
      <p className="text-[#A1A1AA] max-w-md mb-8 text-sm leading-relaxed">
        The page or resource you are looking for does not exist, has been removed, or the link may be broken.
      </p>

      <div className="flex flex-wrap items-center justify-center gap-3">
        <Link
          href="/"
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#A78BFA] text-black font-semibold text-sm hover:bg-[#9065FA] transition-colors"
        >
          <Home size={16} />
          Back to Home
        </Link>

        <Link
          href="/dojo"
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#18181B] border border-[#27272A] text-[#EDEDED] font-medium text-sm hover:bg-[#27272A] transition-colors"
        >
          <ArrowLeft size={16} />
          Go to Dojo
        </Link>
      </div>
    </div>
  );
}
