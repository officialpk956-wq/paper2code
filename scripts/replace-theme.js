const fs = require('fs');
const path = require('path');

const replacements = {
  // Primary green -> Violet
  '#34D399': '#A78BFA',
  '#4ADEA8': '#C4B5FD',
  
  // Dark green backgrounds -> Neutral darks
  '#0E1811': '#0A0A0A',
  '#10231A': '#0A0A0A',
  '#121D16': '#111111',
  '#16241B': '#111111',
  
  // Dark green borders/surfaces -> Neutral borders/surfaces
  '#1B2A20': '#1A1A1A',
  '#1B2C21': '#1A1A1A',
  '#223429': '#262626'
};

function walkDir(dir) {
  let results = [];
  const list = fs.readdirSync(dir);
  list.forEach(file => {
    file = path.join(dir, file);
    const stat = fs.statSync(file);
    if (stat && stat.isDirectory()) {
      results = results.concat(walkDir(file));
    } else {
      if (file.endsWith('.tsx') || file.endsWith('.ts') || file.endsWith('.css')) {
        results.push(file);
      }
    }
  });
  return results;
}

const files = walkDir(path.join(__dirname, '../src'));

let changedFiles = 0;
files.forEach(file => {
  let content = fs.readFileSync(file, 'utf8');
  let originalContent = content;
  
  for (const [green, purple] of Object.entries(replacements)) {
    // Case insensitive replacement just in case
    const regex = new RegExp(green, 'gi');
    content = content.replace(regex, purple);
  }
  
  if (content !== originalContent) {
    fs.writeFileSync(file, content, 'utf8');
    changedFiles++;
    console.log('Updated:', file);
  }
});

console.log(`Replaced green colors with violet/neutral in ${changedFiles} files.`);
