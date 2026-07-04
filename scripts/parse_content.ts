import fs from 'fs';
import path from 'path';

const slugifyCache: Record<string, number> = {};

function makeSlug(name: string, fileContext: string): string {
    const baseSlug = name.toLowerCase()
        .replace(/×/g, 'x').replace(/&/g, 'and')
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
    
    const key = `${fileContext}:${baseSlug}`;
    if (slugifyCache[key] === undefined) {
        slugifyCache[key] = 1;
        return baseSlug;
    } else {
        slugifyCache[key]++;
        return `${baseSlug}-${slugifyCache[key]}`;
    }
}

// 1. Parse Architectures
function parseArchitectures() {
    console.log("Parsing Architectures...");
    const content = fs.readFileSync('207 architectures across 13 categories..txt', 'utf8');
    const lines = content.split('\n');
    let currentCategory = '';
    const architectures = [];
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith('### ')) {
            currentCategory = line.substring(4).trim();
        } else if (line.startsWith('|') && !line.includes('---|') && !line.includes('Name | Year')) {
            const cols = line.split('|').map(s => s.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
            if (cols.length >= 8) {
                const [name, yearStr, authors, difficulty, parent, derivedInto, keyInnovation, industryUsage] = cols;
                architectures.push({
                    slug: makeSlug(name, 'architectures'),
                    name,
                    year: parseInt(yearStr, 10),
                    authors,
                    difficulty: difficulty as 'Beginner'|'Intermediate'|'Advanced'|'Expert',
                    category: currentCategory,
                    parent: parent === '—' ? undefined : parent,
                    derivedInto: derivedInto === '—' ? undefined : derivedInto,
                    keyInnovation,
                    industryUsage
                });
            }
        }
    }
    
    console.log(`Parsed ${architectures.length} architectures.`);
    let out = `export type ArchEntry = {
  slug: string;
  name: string; year: number; authors: string;
  difficulty: 'Beginner'|'Intermediate'|'Advanced'|'Expert';
  category: string;
  parent?: string;
  derivedInto?: string;
  keyInnovation: string; industryUsage: string;
};\n\n`;
    out += `export const ARCHITECTURES: ArchEntry[] = ${JSON.stringify(architectures, null, 2)};\n`;
    fs.writeFileSync('src/data/content/architectures.ts', out);
}

// 2. Parse Papers
function parsePapers() {
    console.log("Parsing Papers...");
    const content = fs.readFileSync('200 Papers.txt', 'utf8');
    const lines = content.split('\n');
    let currentSection = '';
    const papers = [];
    
    let currentPaper: any = null;
    let rankCounter = 0;
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith('## SECTION')) {
            currentSection = line.replace(/^## SECTION [IVXLC]+\s*[-—–]\s*/, '').trim();
        } else if (line.match(/^\*\*#\d+\s*[-—–]\s*(.*?)\*\*$/)) {
            if (currentPaper) papers.push(currentPaper);
            const match = line.match(/^\*\*#\d+\s*[-—–]\s*(.*?)\*\*$/);
            if (match) {
                rankCounter++;
                currentPaper = {
                    slug: makeSlug(match[1], 'papers'),
                    rank: rankCounter,
                    title: match[1],
                    section: currentSection
                };
            }
        } else if (currentPaper && line.startsWith('- **Authors:**')) {
            currentPaper.authors = line.replace('- **Authors:**', '').trim();
        } else if (currentPaper && line.startsWith('- **Year:**')) {
            currentPaper.year = parseInt(line.replace('- **Year:**', '').trim(), 10);
        } else if (currentPaper && line.startsWith('- **Difficulty:**')) {
            currentPaper.difficulty = line.replace('- **Difficulty:**', '').trim();
        } else if (currentPaper && line.startsWith('- **Why Important:**')) {
            currentPaper.whyImportant = line.replace('- **Why Important:**', '').trim();
        } else if (currentPaper && line.startsWith('- **Architectures Introduced:**')) {
            const val = line.replace('- **Architectures Introduced:**', '').trim();
            if (val && val !== '—') currentPaper.architecturesIntroduced = val;
        } else if (currentPaper && line.startsWith('- **Concepts Introduced:**')) {
            const val = line.replace('- **Concepts Introduced:**', '').trim();
            if (val && val !== '—') currentPaper.conceptsIntroduced = val;
        } else if (currentPaper && line.startsWith('- **Industry Impact:**')) {
            const val = line.replace('- **Industry Impact:**', '').trim();
            if (val && val !== '—') currentPaper.industryImpact = val;
        } else if (currentPaper && line.startsWith('- **Built Upon By:**')) {
            const val = line.replace('- **Built Upon By:**', '').trim();
            if (val && val !== '—') currentPaper.builtUponBy = val;
        }
    }
    if (currentPaper) papers.push(currentPaper);
    
    console.log(`Parsed ${papers.length} papers.`);
    let out = `export type PaperEntry = {
  slug: string; rank: number;
  title: string; authors?: string; year?: number;
  difficulty?: string; section: string;
  whyImportant?: string;
  architecturesIntroduced?: string;
  conceptsIntroduced?: string;
  industryImpact?: string; builtUponBy?: string;
};\n\n`;
    out += `export const PAPERS: PaperEntry[] = ${JSON.stringify(papers, null, 2)};\n`;
    fs.writeFileSync('src/data/content/papers.ts', out);
}

// 3. Parse Curriculum
function parseCurriculum() {
    console.log("Parsing Curriculum...");
    const content = fs.readFileSync('full 12-domain curriculum. Domain 11 Advanced + Expert and all of Domain 12.txt', 'utf8');
    const lines = content.split('\n');
    
    const domainsMap = new Map<number, any>();
    const domains: any[] = [];
    let currentDomain: any = null;
    let currentLevel: string = '';
    let currentTopic: any = null;
    let readingWhy = false;
    let totalTopics = 0;
    
    const finishTopic = () => {
        if (currentTopic && currentDomain) {
            if(currentTopic.why) currentTopic.why = currentTopic.why.trim();
            currentDomain.topics.push(currentTopic);
            totalTopics++;
            currentTopic = null;
        }
    };
    
    const setDomain = (num: number, name: string) => {
        finishTopic();
        if (domainsMap.has(num)) {
            currentDomain = domainsMap.get(num);
        } else {
            currentDomain = {
                slug: makeSlug(name, 'curriculum'),
                number: num,
                name: name,
                topics: []
            };
            domainsMap.set(num, currentDomain);
            domains.push(currentDomain);
        }
    };
    
    for (let i = 0; i < lines.length; i++) {
        let rawLine = lines[i];
        let line = rawLine.trim();
        
        if (line.match(/^DOMAIN (\d+)\s*[-—–]\s*(.*)$/)) {
            const match = line.match(/^DOMAIN (\d+)\s*[-—–]\s*(.*)$/);
            if (match) setDomain(parseInt(match[1], 10), match[2].trim());
        } 
        else if (line.match(/^## Domain (\d+)\s*[-—–]\s*(.*)$/i)) {
            const match = line.match(/^## Domain (\d+)\s*[-—–]\s*(.*)$/i);
            if (match) setDomain(parseInt(match[1], 10), match[2].trim());
        }
        else if (line.match(/^├ (BEGINNER|INTERMEDIATE|ADVANCED|EXPERT)$/)) {
            const match = line.match(/^├ (BEGINNER|INTERMEDIATE|ADVANCED|EXPERT)$/);
            if (match) currentLevel = match[1].charAt(0).toUpperCase() + match[1].slice(1).toLowerCase();
        } 
        else if (line.match(/^### (Beginner|Intermediate|Advanced|Expert) Topics$/i)) {
            const match = line.match(/^### (Beginner|Intermediate|Advanced|Expert) Topics$/i);
            if (match) currentLevel = match[1].charAt(0).toUpperCase() + match[1].slice(1).toLowerCase();
        }
        else if (line.match(/^\*\*Topic:\s*(.*?)\*\*$/)) {
            const match = line.match(/^\*\*Topic:\s*(.*?)\*\*$/);
            if (match) {
                finishTopic();
                let title = match[1].trim();
                currentTopic = {
                    slug: makeSlug(title, 'curriculum'),
                    title: title,
                    level: currentLevel,
                    prerequisites: [],
                    studyTime: '',
                    why: '',
                    unlocks: []
                };
                readingWhy = false;
            }
        }
        else if (line.match(/^[├└]\s+([^—:\n]+)(?:\s*[-—–].*)?$/) && !line.includes('Difficulty') && !line.includes('Prerequisites') && !line.includes('Study Time') && !line.includes('Why') && !line.includes('Unlocks') && !line.match(/^[├└]\s+(BEGINNER|INTERMEDIATE|ADVANCED|EXPERT)$/)) {
            finishTopic();
            const match = line.match(/^[├└]\s+(.*)$/);
            if (match) {
                let title = match[1].trim();
                if (title.includes(' — ')) title = title.split(' — ')[0].trim();
                else if (title.includes(' - ')) title = title.split(' - ')[0].trim();
                else if (title.includes('  ')) title = title.split('  ')[0].trim();
                
                currentTopic = {
                    slug: makeSlug(title, 'curriculum'),
                    title: title,
                    level: currentLevel,
                    prerequisites: [],
                    studyTime: '',
                    why: '',
                    unlocks: []
                };
                readingWhy = false;
            }
        } 
        else if (currentTopic && line.includes('Difficulty') && line.includes(':')) {
            readingWhy = false;
        } else if (currentTopic && line.includes('Prerequisites') && line.includes(':')) {
            const val = line.split(':')[1].trim();
            if (val && val !== 'None' && val !== '—') {
                currentTopic.prerequisites = val.split(',').map(s => s.trim());
            }
            readingWhy = false;
        } else if (currentTopic && (line.includes('Study Time') || line.includes('Estimated Study Time')) && line.includes(':')) {
            currentTopic.studyTime = line.split(':')[1].trim();
            readingWhy = false;
        } else if (currentTopic && (line.match(/^│\s+Why\s+:\s+(.*)$/) || line.match(/^- Why It Matters:\s*(.*)$/))) {
            const match = line.match(/^│\s+Why\s+:\s+(.*)$/) || line.match(/^- Why It Matters:\s*(.*)$/);
            if (match) {
                currentTopic.why = match[1].trim();
                readingWhy = true;
            }
        } else if (currentTopic && (line.match(/^│\s+Unlocks\s+:\s+(.*)$/) || line.match(/^- What It Unlocks Next:\s*(.*)$/))) {
            const match = line.match(/^│\s+Unlocks\s+:\s+(.*)$/) || line.match(/^- What It Unlocks Next:\s*(.*)$/);
            if (match) {
                const val = match[1].trim();
                if (val && val !== 'None' && val !== '—') {
                    currentTopic.unlocks = val.split(',').map(s => s.trim());
                }
            }
            readingWhy = false;
        } else if (currentTopic && readingWhy) {
            if (line.startsWith('- ') || line.startsWith('**') || line.startsWith('##') || line === '---') {
                readingWhy = false;
            } else {
                const contentMatch = rawLine.match(/^.*│\s{19}(.*)$/);
                if (contentMatch && contentMatch[1].trim() !== '') {
                    currentTopic.why += ' ' + contentMatch[1].trim();
                } else if (!rawLine.includes('│') && line !== '') {
                     currentTopic.why += ' ' + line;
                } else if (rawLine.includes('│') && line !== '│') {
                     const fallMatch = rawLine.match(/^.*│\s+(.*)$/);
                     if (fallMatch && fallMatch[1].trim() !== '' && !fallMatch[1].includes('Unlocks')) {
                         currentTopic.why += ' ' + fallMatch[1].trim();
                     }
                }
            }
        }
    }
    finishTopic();
    
    console.log(`Parsed ${domains.length} domains, ${totalTopics} topics total.`);
    let out = `// Total topics: ${totalTopics}\n`;
    out += `export type CurriculumTopic = {
  slug: string; title: string;
  level: 'Beginner'|'Intermediate'|'Advanced'|'Expert';
  prerequisites: string[];
  studyTime: string;
  why: string;
  unlocks: string[];
};\n\n`;
    out += `export type CurriculumDomain = {
  slug: string; number: number; name: string;
  topics: CurriculumTopic[];
};\n\n`;
    out += `export const CURRICULUM: CurriculumDomain[] = ${JSON.stringify(domains, null, 2)};\n`;
    fs.writeFileSync('src/data/content/curriculum.ts', out);
}

// 4. Parse Prerequisites
function parsePrerequisites() {
    console.log("Parsing Prerequisites...");
    const content = fs.readFileSync('1,050+ directed prerequisite relationships spanning 9 domains with full cross-domain dependency chains..txt', 'utf8');
    const lines = content.split('\n');
    
    const edges = [];
    let currentDomain = '';
    let currentSection = '';
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith('### ')) {
            const header = line.substring(4).trim();
            if (header.includes(' — ') || header.includes(' - ') || header.includes('  ')) {
                const parts = header.split(/ [-—–] /);
                currentDomain = parts[0].trim();
                currentSection = parts[1].trim();
            } else {
                currentDomain = header;
                currentSection = '';
            }
        } else if (line.includes('->')) {
            const parts = line.split('->');
            if (parts.length === 2) {
                edges.push({
                    from: parts[0].trim(),
                    to: parts[1].trim(),
                    domain: currentDomain,
                    section: currentSection
                });
            }
        }
    }
    
    console.log(`Parsed ${edges.length} prerequisite edges.`);
    let out = `export type PrereqEdge = { from: string; to: string; domain: string; section: string };\n\n`;
    out += `export const PREREQ_EDGES: PrereqEdge[] = ${JSON.stringify(edges, null, 2)};\n`;
    fs.writeFileSync('src/data/content/prerequisites.ts', out);
}

// 5. Parse System Design
function parseSystemDesign() {
    console.log("Parsing System Design...");
    const content = fs.readFileSync('AI System Design Complete Curriculum.txt', 'utf8');
    const lines = content.split('\n');
    
    const systems = [];
    let currentSystem: any = null;
    let currentModule: any = null;
    
    let currentListType = '';
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        if (line.match(/^# SYSTEM \d+\s*[-—–]\s*(.*)$/i) && !line.includes('Already covered as')) {
            if (currentModule && currentSystem) {
                currentSystem.modules.push(currentModule);
                currentModule = null;
            }
            if (currentSystem) systems.push(currentSystem);
            
            const match = line.match(/^# SYSTEM (\d+)\s*[-—–]\s*(.*)$/i);
            if (match) {
                currentSystem = {
                    slug: makeSlug(match[2], 'systemdesign'),
                    number: parseInt(match[1], 10),
                    name: match[2],
                    modules: []
                };
            }
        } else if (line.match(/^## (Beginner|Intermediate|Advanced|Research)$/i)) {
            if (currentModule && currentSystem) {
                currentSystem.modules.push(currentModule);
            }
            currentModule = {
                level: line.substring(3).trim(),
                learningObjectives: [],
                prerequisites: [],
                diagramsNeeded: [],
                caseStudies: [],
                handsOnProjects: [],
                interviewQuestions: []
            };
            currentListType = '';
        } else if (currentModule && line.startsWith('**Learning Objectives**')) {
            currentListType = 'learningObjectives';
        } else if (currentModule && line.startsWith('**Prerequisites**')) {
            currentListType = 'prerequisites';
        } else if (currentModule && line.startsWith('**Architecture Diagrams Needed**')) {
            currentListType = 'diagramsNeeded';
        } else if (currentModule && line.startsWith('**Case Studies Needed**')) {
            currentListType = 'caseStudies';
        } else if (currentModule && line.startsWith('**Hands-On Projects Needed**')) {
            currentListType = 'handsOnProjects';
        } else if (currentModule && line.startsWith('**Interview Questions Needed**')) {
            currentListType = 'interviewQuestions';
        } else if (currentModule && line.startsWith('- ') && currentListType) {
            currentModule[currentListType].push(line.substring(2).trim());
        }
    }
    if (currentModule && currentSystem) {
        currentSystem.modules.push(currentModule);
    }
    if (currentSystem) systems.push(currentSystem);
    
    console.log(`Parsed ${systems.length} system design systems, total modules: ${systems.reduce((acc, sys) => acc + sys.modules.length, 0)}.`);
    
    let out = `export type SDModule = {
  level: 'Beginner'|'Intermediate'|'Advanced'|'Research';
  learningObjectives: string[]; prerequisites: string[];
  diagramsNeeded: string[]; caseStudies: string[];
  handsOnProjects: string[]; interviewQuestions: string[];
};\n\n`;
    out += `export type SDSystem = { slug: string; number: number; name: string; modules: SDModule[] };\n\n`;
    out += `export const SD_SYSTEMS: SDSystem[] = ${JSON.stringify(systems, null, 2)};\n`;
    fs.writeFileSync('src/data/content/systemDesign.ts', out);
}

parseArchitectures();
parsePapers();
parseCurriculum();
parsePrerequisites();
parseSystemDesign();
