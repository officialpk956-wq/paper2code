import { ARCHITECTURES } from '../src/data/content/architectures';
import { PAPERS } from '../src/data/content/papers';
import { CURRICULUM } from '../src/data/content/curriculum';
import { PREREQ_EDGES } from '../src/data/content/prerequisites';
import { SD_SYSTEMS } from '../src/data/content/systemDesign';

let errors = 0;

function assert(condition: boolean, message: string) {
    if (!condition) {
        console.error(`❌ FAIL: ${message}`);
        errors++;
    } else {
        console.log(`✅ PASS: ${message}`);
    }
}

function checkUniqueSlugs(arr: any[], name: string) {
    const slugs = new Set();
    let unique = true;
    for (const item of arr) {
        if (slugs.has(item.slug)) {
            console.error(`❌ Duplicate slug in ${name}: ${item.slug}`);
            unique = false;
            errors++;
        }
        slugs.add(item.slug);
    }
    if (unique) console.log(`✅ PASS: All slugs unique in ${name}`);
}

console.log("--- Verifying Architectures ---");
assert(ARCHITECTURES.length === 214, `ARCHITECTURES.length should be 214, got ${ARCHITECTURES.length}`);
checkUniqueSlugs(ARCHITECTURES, "ARCHITECTURES");

console.log("\\n--- Verifying Papers ---");
assert(PAPERS.length === 191, `PAPERS.length should be 191, got ${PAPERS.length}`);
checkUniqueSlugs(PAPERS, "PAPERS");
const paperRanks = new Set(PAPERS.map(p => p.rank));
let ranksOk = true;
for (let i = 1; i <= 191; i++) {
    if (!paperRanks.has(i)) {
        console.error(`❌ Missing paper rank: ${i}`);
        ranksOk = false;
        errors++;
    }
}
if (ranksOk) console.log("✅ PASS: All paper ranks 1..191 present exactly once");

console.log("\\n--- Verifying Curriculum ---");
assert(CURRICULUM.length === 12, `CURRICULUM.length should be 12, got ${CURRICULUM.length}`);
checkUniqueSlugs(CURRICULUM, "CURRICULUM");

console.log("\\n--- Verifying Prerequisites ---");
assert(PREREQ_EDGES.length >= 1050, `PREREQ_EDGES.length should be >= 1050, got ${PREREQ_EDGES.length}`);

console.log("\\n--- Verifying System Design ---");
assert(SD_SYSTEMS.length === 12, `SD_SYSTEMS.length should be 12, got ${SD_SYSTEMS.length}`);
checkUniqueSlugs(SD_SYSTEMS, "SD_SYSTEMS");
let modulesOk = true;
for (const sys of SD_SYSTEMS) {
    if (sys.modules.length !== 4) {
        console.error(`❌ System "${sys.name}" has ${sys.modules.length} modules, expected 4`);
        modulesOk = false;
        errors++;
    }
}
if (modulesOk) console.log("✅ PASS: All systems have exactly 4 modules");

if (errors > 0) {
    console.error(`\\n❌ FAILED with ${errors} errors.`);
    process.exit(1);
} else {
    console.log("\\n🎉 ALL CHECKS PASSED!");
}
