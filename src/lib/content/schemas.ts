import { z } from "zod";

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

export const CONTENT_TYPES = [
  "architecture",
  "paper",
  "math",
  "system-design",
  "problem",
  "interview",
  "roadmap",
  "implementation",
  "tensor-trace",
] as const;

export const ContentTypeSchema = z.enum(CONTENT_TYPES);
export type ContentType = z.infer<typeof ContentTypeSchema>;

export const DifficultySchema = z.enum(["beginner", "intermediate", "advanced"]);
export type Difficulty = z.infer<typeof DifficultySchema>;

/**
 * Cross-content relationships. Each key holds slugs of related items of that
 * type. The relationship graph (relationships.ts) resolves these against the
 * build-time index and also computes reverse edges automatically.
 */
export const RelationshipsSchema = z
  .object({
    architectures: z.array(z.string()).default([]),
    papers: z.array(z.string()).default([]),
    math: z.array(z.string()).default([]),
    problems: z.array(z.string()).default([]),
    systemDesign: z.array(z.string()).default([]),
    interview: z.array(z.string()).default([]),
    roadmaps: z.array(z.string()).default([]),
    implementations: z.array(z.string()).default([]),
  })
  .partial();

export type Relationships = z.infer<typeof RelationshipsSchema>;

const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export const BaseMetaSchema = z.object({
  slug: z.string().regex(SLUG_PATTERN, "slug must be kebab-case"),
  title: z.string().min(1),
  description: z.string().min(1),
  tags: z.array(z.string()).default([]),
  difficulty: DifficultySchema,
  relationships: RelationshipsSchema.default({}),
});

// ---------------------------------------------------------------------------
// Per-type metadata schemas
// ---------------------------------------------------------------------------

export const ArchitectureMetaSchema = BaseMetaSchema.extend({
  type: z.literal("architecture"),
  year: z.number().int().min(1950).max(2100),
  authors: z.array(z.string()).default([]),
  components: z.array(z.string()).default([]),
  predecessors: z.array(z.string()).default([]),
});

export const PaperMetaSchema = BaseMetaSchema.extend({
  type: z.literal("paper"),
  authors: z.array(z.string()).min(1),
  year: z.number().int().min(1950).max(2100),
  venue: z.string().optional(),
  arxivId: z.string().optional(),
  citations: z.number().int().nonnegative().optional(),
});

export const MathTopicMetaSchema = BaseMetaSchema.extend({
  type: z.literal("math"),
  category: z.enum([
    "linear-algebra",
    "calculus",
    "probability",
    "statistics",
    "optimization",
    "information-theory",
    "other",
  ]),
  keyFormula: z.string().optional(),
});

export const SystemDesignMetaSchema = BaseMetaSchema.extend({
  type: z.literal("system-design"),
  scale: z.string().optional(),
  components: z.array(z.string()).default([]),
  tradeoffs: z.array(z.string()).default([]),
});

export const ProblemMetaSchema = BaseMetaSchema.extend({
  type: z.literal("problem"),
  estimatedMinutes: z.number().int().positive().optional(),
  hints: z.array(z.string()).default([]),
  starterLanguages: z.array(z.string()).default(["python"]),
});

export const InterviewMetaSchema = BaseMetaSchema.extend({
  type: z.literal("interview"),
  questionType: z.enum(["conceptual", "coding", "system-design", "ml-theory"]),
  companies: z.array(z.string()).default([]),
  followUps: z.array(z.string()).default([]),
});

export const RoadmapMetaSchema = BaseMetaSchema.extend({
  type: z.literal("roadmap"),
  icon: z.string().optional(),
  phases: z
    .array(
      z.object({
        id: z.string(),
        name: z.string(),
        nodeSlugs: z.array(z.string()).default([]),
      })
    )
    .default([]),
});

export const ImplementationMetaSchema = BaseMetaSchema.extend({
  type: z.literal("implementation"),
  paperSlug: z.string(),
});

/** Discriminated union over every content type. */
export const ContentMetaSchema = z.discriminatedUnion("type", [
  ArchitectureMetaSchema,
  PaperMetaSchema,
  MathTopicMetaSchema,
  SystemDesignMetaSchema,
  ProblemMetaSchema,
  InterviewMetaSchema,
  RoadmapMetaSchema,
  ImplementationMetaSchema,
]);

export type ArchitectureMeta = z.infer<typeof ArchitectureMetaSchema>;
export type PaperMeta = z.infer<typeof PaperMetaSchema>;
export type MathTopicMeta = z.infer<typeof MathTopicMetaSchema>;
export type SystemDesignMeta = z.infer<typeof SystemDesignMetaSchema>;
export type ProblemMeta = z.infer<typeof ProblemMetaSchema>;
export type InterviewMeta = z.infer<typeof InterviewMetaSchema>;
export type RoadmapMeta = z.infer<typeof RoadmapMetaSchema>;
export type ImplementationMeta = z.infer<typeof ImplementationMetaSchema>;
export type ContentMeta = z.infer<typeof ContentMetaSchema>;

// ---------------------------------------------------------------------------
// Build-time index entry (what generate-content-index.mjs emits)
// ---------------------------------------------------------------------------

export const IndexEntrySchema = z.object({
  slug: z.string(),
  type: ContentTypeSchema,
  title: z.string(),
  description: z.string(),
  tags: z.array(z.string()),
  difficulty: DifficultySchema,
  relationships: RelationshipsSchema,
});

export type IndexEntry = z.infer<typeof IndexEntrySchema>;

export const ContentIndexSchema = z.object({
  generatedAt: z.string(),
  count: z.number().int(),
  entries: z.array(IndexEntrySchema),
});

export type ContentIndex = z.infer<typeof ContentIndexSchema>;
