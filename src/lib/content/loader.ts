import "server-only";
import fs from "fs";
import path from "path";
import {
  ContentMeta,
  ContentMetaSchema,
  ContentType,
  ArchitectureMeta,
  PaperMeta,
  MathTopicMeta,
  SystemDesignMeta,
  ProblemMeta,
  InterviewMeta,
  RoadmapMeta,
  ImplementationMeta,
} from "./schemas";

/**
 * Content lives in Git as plain files:
 *
 *   src/content/<type-dir>/<slug>/meta.json   — metadata, zod-validated
 *   src/content/<type-dir>/<slug>/content.mdx — the lesson body
 *
 * User state (progress, bookmarks, submissions) is NOT stored here — it
 * belongs in the database. This module is read-only at request/build time.
 */

const CONTENT_ROOT = path.join(process.cwd(), "src", "content");

export const TYPE_DIRS: Record<ContentType, string> = {
  architecture: "architectures",
  paper: "papers",
  math: "math",
  "system-design": "system-design",
  problem: "problems",
  interview: "interview",
  roadmap: "roadmaps",
  implementation: "implementations",
  "tensor-trace": "tensor-traces",
};

export interface ContentItem<M extends ContentMeta = ContentMeta> {
  meta: M;
  /** Raw MDX body. Render with <MarkdownRenderer />. */
  body: string;
}

function itemDir(type: ContentType, slug: string): string {
  return path.join(CONTENT_ROOT, TYPE_DIRS[type], slug);
}

/**
 * Load and validate a single content item. Throws with a descriptive error
 * if the metadata fails schema validation — this is intentional so invalid
 * content fails the build instead of shipping silently broken pages.
 */
export function getContentItem<M extends ContentMeta>(
  type: ContentType,
  slug: string
): ContentItem<M> | null {
  const dir = itemDir(type, slug);
  const metaPath = path.join(dir, "meta.json");
  const bodyPath = path.join(dir, "content.mdx");

  if (!fs.existsSync(metaPath)) return null;

  const rawMeta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
  const parsed = ContentMetaSchema.safeParse(rawMeta);

  if (!parsed.success) {
    throw new Error(
      `Invalid content metadata at ${metaPath}:\n${parsed.error.issues
        .map((i) => `  - ${i.path.join(".")}: ${i.message}`)
        .join("\n")}`
    );
  }

  if (parsed.data.type !== type) {
    throw new Error(
      `Type mismatch at ${metaPath}: file declares "${parsed.data.type}" but lives under "${TYPE_DIRS[type]}/"`
    );
  }

  const body = fs.existsSync(bodyPath)
    ? fs.readFileSync(bodyPath, "utf-8")
    : "";

  return { meta: parsed.data as M, body };
}

/** List every slug available for a content type (for generateStaticParams). */
export function getAllSlugs(type: ContentType): string[] {
  const dir = path.join(CONTENT_ROOT, TYPE_DIRS[type]);
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .filter((entry) =>
      fs.existsSync(path.join(dir, entry.name, "meta.json"))
    )
    .map((entry) => entry.name);
}

/** Load every item of one type (metadata + body). */
export function getAllContent<M extends ContentMeta>(
  type: ContentType
): ContentItem<M>[] {
  return getAllSlugs(type)
    .map((slug) => getContentItem<M>(type, slug))
    .filter((item): item is ContentItem<M> => item !== null);
}

// ---------------------------------------------------------------------------
// Typed convenience accessors — the public API the rest of the app uses
// ---------------------------------------------------------------------------

export const getArchitecture = (slug: string) =>
  getContentItem<ArchitectureMeta>("architecture", slug);

export const getPaper = (slug: string) =>
  getContentItem<PaperMeta>("paper", slug);

export const getMathTopic = (slug: string) =>
  getContentItem<MathTopicMeta>("math", slug);

export const getSystemDesign = (slug: string) =>
  getContentItem<SystemDesignMeta>("system-design", slug);

export const getProblem = (slug: string) =>
  getContentItem<ProblemMeta>("problem", slug);

export const getInterviewQuestion = (slug: string) =>
  getContentItem<InterviewMeta>("interview", slug);

export const getRoadmap = (slug: string) =>
  getContentItem<RoadmapMeta>("roadmap", slug);

export const getImplementation = (slug: string) =>
  getContentItem<ImplementationMeta>("implementation", slug);

export const getAllImplementations = () =>
  getAllContent<ImplementationMeta>("implementation");

export const getAllSystemDesigns = () =>
  getAllContent<SystemDesignMeta>("system-design");
