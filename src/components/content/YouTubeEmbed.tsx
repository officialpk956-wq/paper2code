import React from 'react';

export function YouTubeEmbed({ id, title }: { id: string; title: string }) {
  return (
    <div className="my-8 aspect-video rounded-xl overflow-hidden border border-white/10 shadow-2xl">
      <iframe
        width="100%"
        height="100%"
        src={`https://www.youtube-nocookie.com/embed/${id}`}
        title={title}
        frameBorder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
      ></iframe>
    </div>
  );
}
