import * as fs from 'fs';
import * as path from 'path';

// Load data
import { PROBLEMS } from '../../src/data/problems';
import { INTERVIEW_QUESTIONS } from '../../src/data/interview-questions';
import { ROADMAPS } from '../../src/data/roadmaps';
// import { SYSTEM_DESIGN_PATTERNS } from '../../src/data/system-design'; // We will check if it exists later

const outDir = path.join(__dirname, 'json_dump');
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir);
}

fs.writeFileSync(path.join(outDir, 'problems.json'), JSON.stringify(PROBLEMS, null, 2));
fs.writeFileSync(path.join(outDir, 'interview_questions.json'), JSON.stringify(INTERVIEW_QUESTIONS, null, 2));
fs.writeFileSync(path.join(outDir, 'roadmaps.json'), JSON.stringify(ROADMAPS, null, 2));

console.log("Successfully dumped TS data to JSON.");
