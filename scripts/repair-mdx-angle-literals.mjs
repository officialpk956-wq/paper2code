import fs from "node:fs";
import path from "node:path";

const files = [
  "src/content/curriculum/rag-systems/production-rag-at-scale/content.mdx",
];

let replacements = 0;
for (const relativePath of files) {
  const file = path.join(process.cwd(), relativePath);
  const before = fs.readFileSync(file, "utf8");
  replacements += before.split("<1ms").length - 1;
  fs.writeFileSync(file, before.replaceAll("<1ms", "&lt;1ms"));
}

console.log(`Escaped MDX angle-bracket literals: ${replacements}`);
