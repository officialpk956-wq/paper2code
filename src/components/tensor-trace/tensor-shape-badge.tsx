import { Box } from "lucide-react";

interface Props {
  shape: string;
  className?: string;
  config?: Record<string, number>;
}

export function TensorShapeBadge({ shape, className = "", config }: Props) {
  // Simple heuristic: if we have a config, try to generate a concrete dimension string
  // "(B, L, d_model)" -> "(2 × 32 × 512)"
  let concreteShape = "";
  if (config) {
    const parts = shape
      .replace(/[()]/g, "")
      .split(",")
      .map((s) => s.trim());
    
    const concreteParts = parts.map(p => {
      if (p === "B") return config.batch_size || "?";
      if (p === "L") return config.seq_len || "?";
      if (p === "d_model") return config.d_model || "?";
      if (p === "heads") return config.num_heads || "?";
      if (p === "d_k") return (config.d_model && config.num_heads) ? config.d_model / config.num_heads : "?";
      if (p === "vocab_size") return 50257; // arbitrary mock for concrete display
      return p;
    });

    concreteShape = `(${concreteParts.join(" × ")})`;
  }

  return (
    <div className={`group relative inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md font-mono text-xs font-semibold bg-[--bg-panel] border border-[--color-border] text-[--color-text-secondary] ${className}`}>
      <Box className="w-3.5 h-3.5 text-[--color-text-tertiary]" />
      <span>{shape}</span>
      
      {/* Tooltip for concrete shape */}
      {concreteShape && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-[--color-text-primary] text-[--bg-body] text-[10px] rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
          Example: {concreteShape}
        </div>
      )}
    </div>
  );
}
