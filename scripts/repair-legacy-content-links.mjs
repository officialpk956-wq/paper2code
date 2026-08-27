import fs from "node:fs";
import path from "node:path";

const root = path.join(process.cwd(), "src/content");
const replacements = new Map([
  ["/architectures/transformer/simulator", "/architectures/transformer"],
  ["/architectures/transformer/attention", "/architectures/transformer"],
  ["/math/cross-entropy", "/dojo/ml-cross-entropy"],
  ["/math/softmax", "/dojo/ml-softmax"],
  ["/math/linear-algebra", "/learn/mathematics"],
  ["/math", "/learn/mathematics"],
  ["/problems/attention-calculation", "/dojo/ml-attention"],
  ["/problems/scaled-dot-product-attention", "/dojo/ml-attention"],
  ["/problems/multi-head-attention", "/dojo/dl-multi-head-attention"],
  ["/problems/masked-attention", "/dojo/ml-attention"],
  ["/problems/backpropagation", "/dojo/dl-backprop-single-neuron"],
  ["/problems/gradient-descent", "/dojo/ml-gradient-descent"],
  ["/problems/matrix-multiplication", "/dojo/numpy-dot-product"],
  ["/problems/kv-cache", "/dojo/dl-flash-attention-conceptual"],
  ["/problems/clip-batch-size", "/dojo"],
  ["/learn/reinforcement-learning/offline-reinforcement-learning", "/learn/reinforcement-learning"],
]);

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const target = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(target) : [target];
  });
}

let changedFiles = 0;
let changedLinks = 0;

for (const file of walk(root).filter((item) => /\.(?:md|mdx)$/.test(item))) {
  const before = fs.readFileSync(file, "utf8");
  let after = before;
  for (const [from, to] of replacements) {
    const occurrences = after.split(`](${from})`).length - 1;
    if (occurrences > 0) {
      changedLinks += occurrences;
      after = after.replaceAll(`](${from})`, `](${to})`);
    }
  }
  if (after !== before) {
    fs.writeFileSync(file, after);
    changedFiles += 1;
  }
}

console.log(`Repaired legacy links: ${changedLinks}`);
console.log(`Files changed: ${changedFiles}`);
