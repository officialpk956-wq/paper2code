/** Shared anchor-id generator — used by the markdown renderer (heading ids)
 *  and the content page TOC so links always match. */
export function slugifyHeading(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}
