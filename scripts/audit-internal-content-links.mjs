import fs from "node:fs";
import path from "node:path";

const root = process.cwd();

function read(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function slugsFrom(relativePath) {
  return new Set(
    [...read(relativePath).matchAll(/["']?slug["']?\s*:\s*["']([^"']+)["']/g)].map(
      (match) => match[1],
    ),
  );
}

function walk(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const target = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(target) : [target];
  });
}

const architectures = slugsFrom("src/data/content/architectures.ts");
const systems = slugsFrom("src/data/content/systemDesign.ts");
const papers = slugsFrom("src/data/content/papers.ts");
const physicalPapers = new Set(
  fs.readdirSync(path.join(root, "src/content/papers"), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name),
);

const curriculumRoot = path.join(root, "src/content/curriculum");
const curriculumDomains = new Set(
  fs.readdirSync(curriculumRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name),
);
const curriculumTopics = new Set(
  walk(curriculumRoot)
    .filter((file) => file.endsWith("content.mdx"))
    .map((file) => path.relative(curriculumRoot, path.dirname(file)).replaceAll("\\", "/")),
);

const dojoSlugs = slugsFrom("src/data/problems.ts");

function validateRoute(route) {
  const clean = route.split("#", 1)[0].replace(/\/$/, "");
  const segments = clean.split("/").filter(Boolean);

  if (segments.length === 0) return true;
  if (segments[0] === "architectures" && segments.length === 2) {
    return architectures.has(segments[1]);
  }
  if (segments[0] === "system-design" && segments.length === 2) {
    return systems.has(segments[1]);
  }
  if (segments[0] === "papers" && segments.length === 2) {
    return papers.has(segments[1]) || physicalPapers.has(segments[1]);
  }
  if (segments[0] === "learn" && segments.length === 2) {
    return curriculumDomains.has(segments[1]);
  }
  if (segments[0] === "learn" && segments.length === 3) {
    return curriculumTopics.has(`${segments[1]}/${segments[2]}`);
  }
  if (segments[0] === "dojo" && segments.length === 2) {
    return dojoSlugs.has(segments[1]);
  }

  if (
    segments.length === 1 &&
    new Set(["architectures", "dojo", "learn", "papers", "system-design"]).has(segments[0])
  ) {
    return true;
  }

  // Links to top-level application pages are valid when a matching App Router
  // page exists. Dynamic content routes above are deliberately registry-aware.
  const appPath = path.join(root, "src/app", ...segments);
  return ["page.tsx", "page.ts", "page.jsx", "page.js"].some((name) =>
    fs.existsSync(path.join(appPath, name)),
  );
}

const markdownFiles = walk(path.join(root, "src/content")).filter((file) =>
  /\.(?:md|mdx)$/.test(file),
);
const broken = [];
let checked = 0;

for (const file of markdownFiles) {
  const source = fs.readFileSync(file, "utf8");
  for (const match of source.matchAll(/\[[^\]]*\]\((\/[^)\s]+)\)/g)) {
    checked += 1;
    if (!validateRoute(match[1])) {
      broken.push({
        file: path.relative(root, file).replaceAll("\\", "/"),
        route: match[1],
      });
    }
  }
}

console.log(`Internal Markdown links checked: ${checked}`);
console.log(`Broken internal Markdown links: ${broken.length}`);

if (broken.length > 0) {
  for (const item of broken) console.error(`${item.file}: ${item.route}`);
  process.exitCode = 1;
}
