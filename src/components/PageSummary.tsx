// Server-rendered, real text that sits above the AuthGuard boundary on every
// protected route — so crawlers, link-preview bots, and the initial HTML
// always have real page-specific content, even before the client-side auth
// check (or its sign-in wall) ever runs.
export function PageSummary({
  eyebrow,
  title,
  description,
}: {
  eyebrow?: string;
  title: string;
  description: string;
}) {
  return (
    <div className="border-b border-[#262626] bg-[#0A0A0A] px-8 py-10">
      {eyebrow && (
        <div className="mb-2 text-[11px] font-bold uppercase tracking-wider text-[#A3A3A3]">{eyebrow}</div>
      )}
      <h1 className="text-[26px] font-bold text-white">{title}</h1>
      <p className="mt-2 max-w-2xl text-[14px] leading-relaxed text-[#A3A3A3]">{description}</p>
    </div>
  );
}

export default PageSummary;
