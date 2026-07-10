import { PROBLEMS } from '../../src/data/problems';
import { writeFileSync } from 'fs';

writeFileSync(
  'backend/scripts/json_dump/problems_from_ts.json',
  JSON.stringify(PROBLEMS, null, 2),
);
console.log(`exported ${PROBLEMS.length} problems`);
