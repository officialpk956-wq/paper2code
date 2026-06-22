"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Breadcrumb } from "@/components/breadcrumb";
import { PROBLEMS, PROBLEM_CATEGORIES } from "@/data/problems";
import { Clock, BookOpen } from "lucide-react";

export default function ProblemsPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  const filtered =
    selectedCategory === "all"
      ? PROBLEMS
      : PROBLEMS.filter((p) => p.category === selectedCategory);

  return (
    <div className="pb-20">
      <div className="border-b border-[--color-border] bg-[--bg-surface] sticky top-14 z-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Breadcrumb items={[{ label: "Problems", current: true }]} />
          <h1 className="text-3xl font-bold mb-2">AI Engineering Problems</h1>
          <p className="text-[--color-text-secondary]">20 problems across all levels</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-[--color-text-primary] mb-4">Categories</h2>
          <div className="grid md:grid-cols-5 gap-3">
            <button
              onClick={() => setSelectedCategory("all")}
              className={`p-4 rounded-lg border transition-all ${
                selectedCategory === "all"
                  ? "bg-[--accent-primary] text-white"
                  : "bg-[--bg-surface] border-[--color-border]"
              }`}
            >
              <p className="font-semibold">All</p>
              <p className="text-xs">{PROBLEMS.length} problems</p>
            </button>

            {PROBLEM_CATEGORIES.map((cat) => (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.id)}
                className={`p-4 rounded-lg border transition-all text-left ${
                  selectedCategory === cat.id
                    ? `bg-gradient-to-r ${cat.color} text-white`
                    : "bg-[--bg-surface] border-[--color-border]"
                }`}
              >
                <p className="font-semibold text-sm">{cat.name}</p>
              </button>
            ))}
          </div>
        </div>

        <motion.div
          key={selectedCategory}
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          {filtered.map((problem) => (
            <Link key={problem.slug} href={`/problems/${problem.slug}`}>
              <motion.div whileHover={{ y: -4 }} className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border] hover:border-[--accent-primary] transition-all cursor-pointer h-full">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold px-2 py-1 rounded bg-[--bg-body]">{problem.difficulty}</span>
                  <Clock className="w-4 h-4 text-[--color-text-tertiary]" />
                </div>
                <h3 className="font-semibold text-[--color-text-primary] mb-2">{problem.title}</h3>
                <p className="text-sm text-[--color-text-secondary] mb-2 line-clamp-2">{problem.description}</p>
                <p className="text-xs text-[--color-text-tertiary]">{problem.estimatedTime}m</p>
              </motion.div>
            </Link>
          ))}
        </motion.div>
      </div>
    </div>
  );
}
