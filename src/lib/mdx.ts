import fs from 'fs';
import path from 'path';

export type MethodologyConceptPart = {
  n: number;
  kind: 'concept';
  title: string;
  prompt: string;
  answer: string;
  hints: string[];
};

export type MethodologyCodePart = {
  n: number;
  kind: 'code';
  title: string;
  prompt: string;
  dojoSlug: string;
};

export type Methodology = {
  paper: string;
  title: string;
  summary: string;
  parts: Array<MethodologyConceptPart | MethodologyCodePart>;
};

export function getMdxContent(category: string, slug: string): string | null {
  const filePath = path.join(process.cwd(), 'src', 'content', category, slug, 'content.mdx');
  try {
    return fs.existsSync(filePath) ? fs.readFileSync(filePath, 'utf-8') : null;
  } catch {
    return null;
  }
}

export function getMethodology(paperSlug: string): Methodology | null {
  const filePath = path.join(
    process.cwd(),
    'src',
    'content',
    'papers',
    paperSlug,
    'methodology.json'
  );
  try {
    return fs.existsSync(filePath)
      ? JSON.parse(fs.readFileSync(filePath, 'utf-8')) as Methodology
      : null;
  } catch {
    return null;
  }
}
