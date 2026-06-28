import React from 'react';

export function ObservableEmbed({ url, title }: { url: string; title?: string }) {
  return (
    <div className="my-8 rounded-xl overflow-hidden border border-white/10 shadow-2xl bg-white">
      {title && <div className="bg-white/5 border-b border-white/10 px-4 py-2 text-sm font-semibold text-white/70">{title}</div>}
      <iframe
        width="100%"
        height="500"
        frameBorder="0"
        src={url}
        title={title || "Observable Notebook"}
      ></iframe>
    </div>
  );
}
