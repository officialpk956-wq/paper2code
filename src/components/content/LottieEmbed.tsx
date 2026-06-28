'use client';
import dynamic from 'next/dynamic';

const Player = dynamic(
  () => import('@lottiefiles/react-lottie-player').then(m => m.Player),
  { ssr: false }
);

interface Props {
  src: string;
  height?: number;
  caption?: string;
}

export function LottieEmbed({ src, height = 280, caption }: Props) {
  return (
    <figure className="my-8">
      <div
        className="rounded-xl overflow-hidden bg-white/5 border border-white/10 flex items-center justify-center"
        style={{ height }}
      >
        <Player src={src} autoplay loop style={{ height: height - 32 }} />
      </div>
      {caption && (
        <figcaption className="text-center text-sm text-white/40 mt-2">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
