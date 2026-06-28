import React from 'react';
import Image from 'next/image';

export function SVGDiagram({ src, alt, caption }: { src: string; alt?: string; caption?: string }) {
  return (
    <figure className="my-8 flex flex-col items-center">
      <div className="relative w-full max-w-3xl aspect-[16/9] bg-white/5 rounded-xl border border-white/10 p-4 flex items-center justify-center">
        <Image src={src} alt={alt ?? caption ?? ''} fill className="object-contain p-4" unoptimized />
      </div>
      {caption && <figcaption className="mt-3 text-sm text-white/50">{caption}</figcaption>}
    </figure>
  );
}
