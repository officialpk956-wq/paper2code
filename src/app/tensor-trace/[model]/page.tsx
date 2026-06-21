import { TRACES } from "@/data/tensor-traces";
import { notFound } from "next/navigation";
import { TraceEngine } from "@/components/tensor-trace/trace-engine";

interface Props {
  params: Promise<{ model: string }>;
}

export function generateStaticParams() {
  return Object.keys(TRACES).map((model) => ({ model }));
}

export default async function TensorTracePage({ params }: Props) {
  const { model } = await params;
  const trace = TRACES[model];
  if (!trace) notFound();

  return (
    <div className="h-[calc(100vh-56px)] flex flex-col bg-[--bg-body] overflow-hidden">
      <TraceEngine trace={trace} />
    </div>
  );
}
