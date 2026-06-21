"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Zap, BarChart3, Calculator } from "lucide-react";

interface ArchitectureInteractiveModulesProps {
  slug: string;
  modules?: ("simulator" | "attention" | "math")[];
}

const MODULE_INFO = {
  simulator: {
    title: "Interactive Simulator",
    description: "Trace your text through the architecture step-by-step",
    icon: Zap,
    color: "from-[--accent-primary] to-[--accent-cyan]",
  },
  attention: {
    title: "Attention Explorer",
    description: "Visualize how tokens attend to each other",
    icon: BarChart3,
    color: "from-[--accent-cyan] to-[rgb(6,182,212)]",
  },
  math: {
    title: "Interactive Math",
    description: "Explore equations with live calculations",
    icon: Calculator,
    color: "from-[rgb(124,58,237)] to-[--accent-primary]",
  },
};

export function ArchitectureInteractiveModules({
  slug,
  modules = ["simulator", "attention", "math"],
}: ArchitectureInteractiveModulesProps) {
  if (slug !== "transformer") return null;

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const item = {
    hidden: { opacity: 0, y: 16 },
    show: { opacity: 1, y: 0 },
  };

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="mt-12 py-8 border-t border-[--color-border]"
    >
      <h2 className="text-2xl font-bold mb-6 text-[--color-text-primary]">
        Interactive Learning Tools
      </h2>

      <div className="grid md:grid-cols-3 gap-4">
        {modules.map((moduleKey) => {
          const interactiveModule =
            MODULE_INFO[moduleKey as keyof typeof MODULE_INFO];
          const Icon = interactiveModule.icon;

          return (
            <motion.div key={moduleKey} variants={item}>
              <Link
                href={`/architectures/${slug}/${moduleKey}`}
                className="group relative h-full"
              >
                <div className="absolute inset-0 bg-gradient-to-r opacity-0 group-hover:opacity-100 rounded-lg blur transition-opacity duration-300"
                     style={{
                       backgroundImage: `linear-gradient(to right, var(${interactiveModule.color.split(" ")[1]}), var(${interactiveModule.color.split(" ")[3]}))`,
                     }} />
                <div className="relative h-full p-6 rounded-lg bg-[--bg-surface] border border-[--color-border] hover:border-[--accent-primary] transition-colors">
                  <div className="flex items-start justify-between mb-3">
                    <Icon className="w-6 h-6 text-[--accent-primary]" />
                    <span className="text-xs font-mono bg-[--bg-body] text-[--accent-cyan] px-2 py-1 rounded">
                      Interactive
                    </span>
                  </div>
                  <h3 className="font-semibold text-[--color-text-primary] mb-2">
                    {interactiveModule.title}
                  </h3>
                  <p className="text-sm text-[--color-text-secondary]">
                    {interactiveModule.description}
                  </p>
                </div>
              </Link>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}
