import fs from 'fs';
import path from 'path';

export function getMdxContent(category: string, slug: string): string | null {
  const filePath = path.join(process.cwd(), 'src', 'content', category, slug, 'content.mdx');
  try {
    return fs.existsSync(filePath) ? fs.readFileSync(filePath, 'utf-8') : null;
  } catch {
    return null;
  }
}
