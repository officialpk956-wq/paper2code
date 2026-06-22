import { SYSTEM_TRACES } from "@/data/system-traces";
import { notFound } from "next/navigation";
import { SystemTraceEngine } from "@/components/system-trace/system-trace-engine";

interface Props {
  params: Promise<{ slug: string }>;
}

export function generateStaticParams() {
  return Object.keys(SYSTEM_TRACES).map((slug) => ({ slug }));
}

export default async function SystemTracePage({ params }: Props) {
  const { slug } = await params;
  const trace = SYSTEM_TRACES[slug];
  if (!trace) notFound();

  return (
    <div className="h-[calc(100vh-56px)] flex flex-col bg-[--bg-body] overflow-hidden">
      <SystemTraceEngine trace={trace} />
    </div>
  );
}
