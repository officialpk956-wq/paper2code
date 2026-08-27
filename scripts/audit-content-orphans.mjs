import fs from "node:fs";
import path from "node:path";

const root = process.cwd();

function parseExport(relativePath, name, shape) {
  const source = fs.readFileSync(path.join(root, relativePath), "utf8");
  const delimiters = shape === "array" ? ["\\[", "\\]"] : ["\\{", "\\}"];
  const expression = new RegExp(
    `export const ${name}[^=]*=\\s*(${delimiters[0]}[\\s\\S]*?${delimiters[1]});`,
  ).exec(source)?.[1];
  if (!expression) throw new Error(`Could not parse ${name}`);
  return Function(`"use strict"; return (${expression});`)();
}

function directories(relativePath) {
  return fs.readdirSync(path.join(root, relativePath), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
}

const architectures = parseExport(
  "src/data/content/architectures.ts",
  "ARCHITECTURES",
  "array",
);
const systems = parseExport("src/data/content/systemDesign.ts", "SD_SYSTEMS", "array");
const papers = parseExport("src/data/content/papers.ts", "PAPERS", "array");
const curriculum = parseExport("src/data/content/curriculum.ts", "CURRICULUM", "array");
const architectureAliases = parseExport(
  "src/data/content/routeAliases.ts",
  "ARCHITECTURE_ROUTE_ALIASES",
  "object",
);
const systemAliases = parseExport(
  "src/data/content/routeAliases.ts",
  "SYSTEM_DESIGN_ROUTE_ALIASES",
  "object",
);
const paperAliases = parseExport(
  "src/data/content/routeAliases.ts",
  "PAPER_ROUTE_ALIASES",
  "object",
);

const categories = [
  {
    name: "architectures",
    physical: directories("src/content/architectures"),
    registered: new Set(architectures.map((item) => item.slug)),
    aliases: architectureAliases,
  },
  {
    name: "curriculum",
    physical: curriculum.flatMap((domain) =>
      directories(`src/content/curriculum/${domain.slug}`).map(
        (topic) => `${domain.slug}/${topic}`,
      ),
    ),
    registered: new Set(
      curriculum.flatMap((domain) =>
        domain.topics.map((topic) => `${domain.slug}/${topic.slug}`),
      ),
    ),
    aliases: {},
  },
  {
    name: "system-design",
    physical: directories("src/content/system-design"),
    registered: new Set(systems.map((item) => item.slug)),
    aliases: systemAliases,
  },
  {
    name: "papers",
    physical: directories("src/content/papers"),
    registered: new Set(papers.map((item) => item.slug)),
    aliases: paperAliases,
  },
];

let failures = 0;
for (const category of categories) {
  const explained = category.physical.filter(
    (slug) => category.registered.has(slug) || Object.hasOwn(category.aliases, slug),
  );
  const unexplained = category.physical.filter((slug) => !explained.includes(slug));
  const invalidTargets = Object.entries(category.aliases).filter(
    ([source, target]) =>
      !category.physical.includes(source) || !category.registered.has(target),
  );

  console.log(
    `${category.name}: ${category.physical.length} physical, ` +
      `${category.registered.size} registered, ${Object.keys(category.aliases).length} aliases, ` +
      `${unexplained.length} unexplained`,
  );

  for (const slug of unexplained) console.error(`  unexplained: ${slug}`);
  for (const [source, target] of invalidTargets) {
    console.error(`  invalid alias: ${source} -> ${target}`);
  }
  failures += unexplained.length + invalidTargets.length;
}

if (failures > 0) process.exitCode = 1;
