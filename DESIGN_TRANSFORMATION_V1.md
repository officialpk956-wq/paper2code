# Paper2Code UI/UX Transformation — Design System & Implementation Plan

**Date:** June 16, 2026  
**Phase:** Premium UI Transformation (UI/UX only, no backend changes)  
**Scope:** Feature-complete foundation → Premium AI Engineering Learning Platform

---

## EXECUTIVE SUMMARY

Transform Paper2Code from a documentation website into a premium AI engineering learning platform comparable to TensorTonic, Linear, and Cursor. This is a **UI/UX transformation only** — no new features, no backend changes, no content changes.

**Key Principle:** Three-column workspace design for all major pages (Architecture, Paper, Paper2Code, System Design).

---

## DESIGN INSPIRATION & BENCHMARKS

### Competitors Analyzed
- **TensorTonic** — Premium dark theme, interactive visualizations, educational flow
- **Linear** — Minimal interface, strong hierarchy, keyboard-first
- **Cursor** — Code-first experience, seamless integrations, premium feel
- **Notion** — Modular workspace, drag-drop, database-driven
- **Raycast** — Command palette, minimal UI, keyboard shortcuts
- **Vercel** — Dashboard clarity, metrics display, premium typography

### Core Design Principles
1. **Dark-first premium experience** (maintain current dark theme)
2. **Three-column workspace** for complex pages
3. **Keyboard-centric interaction** where possible
4. **Interactive > Static** (visualizations over images)
5. **Hierarchy through typography** (clear information density)
6. **Glassmorphism & gradients** for premium feel
7. **Smooth animations** (300-500ms transitions)
8. **Contextual panels** (left nav, center content, right details)

---

## DESIGN SYSTEM REFINEMENTS

### Color System (Current + Refinements)

**Existing Palette (Keep):**
```
--bg-body: #09090F           // Deep background
--bg-surface: #0D0D14        // Card/panel background
--bg-panel: #111827          // Elevated panel
--accent-primary: #7C3AED    // Purple (Transformer)
--accent-cyan: #06B6D4       // Cyan (Modern/Tech)
--accent-light: #A78BFA      // Light purple (hover states)
--color-text-primary: #E2E8F0    // White text
--color-text-secondary: #94A3B8  // Gray text
--color-text-tertiary: #64748B   // Dark gray text
--color-easy: #10B981        // Green
--color-medium: #F59E0B      // Orange
--color-hard: #EF4444        // Red
--color-expert: #EC4899      // Pink
```

**New Accent Additions:**
```
--accent-success: #10B981    // Success state
--accent-warning: #F59E0B    // Warning state
--accent-error: #EF4444      // Error state
--accent-info: #06B6D4       // Info state
--color-border-light: #2A2D3A // Lighter borders for hierarchy
--color-border-focus: #7C3AED // Focused/active border
```

### Typography System (Refined)

**Font Stack:**
- Headings: `Plus Jakarta Sans` (geometric, modern)
- Body: `Inter` (clean, readable)
- Code: `JetBrains Mono` (monospace for code blocks)

**Sizing Hierarchy:**
```
--text-display: 32-40px      // Page titles (clamp based on viewport)
--text-h1: 22px              // Section titles
--text-h2: 16px              // Subsection titles
--text-h3: 13px              // Card titles
--text-base: 12px            // Body text
--text-sm: 11px              // Secondary text
--text-xs: 10px              // Labels, badges
--text-label: 9px            // Tiny labels (uppercase)
```

**Letter Spacing:**
```
Headings: -0.03em (tight)
Normal text: 0em
Labels: +0.05em (loose, uppercase)
```

### Spacing System (Refined)

```
--space-1: 4px    // Micro spacing
--space-2: 8px    // Tight spacing
--space-3: 12px   // Default spacing
--space-4: 16px   // Comfortable spacing
--space-6: 24px   // Section spacing
--space-8: 32px   // Large spacing
```

### Border Radius

```
--radius-sm: 4px      // Buttons, small elements
--radius-md: 8px      // Cards, inputs
--radius-lg: 12px     // Panels, large cards
--radius-xl: 16px     // Large panels
--radius-2xl: 24px    // Hero sections
```

### Transitions & Animations

```
--transition-fast: 100ms ease      // Hover states
--transition-base: 150ms ease      // Normal interactions
--transition-slow: 300ms ease      // Tab switches, major transitions

New additions:
--transition-modal: 500ms cubic-bezier(0.16, 1, 0.3, 1)  // Smooth pop
--transition-slide: 300ms cubic-bezier(0.4, 0, 0.2, 1)   // Slide in/out
```

### Shadow Hierarchy

```
--shadow-sm: 0 1px 2px rgba(0,0,0,0.25)
--shadow-md: 0 4px 6px rgba(0,0,0,0.15)
--shadow-lg: 0 10px 15px rgba(0,0,0,0.1)
--shadow-xl: 0 20px 25px rgba(0,0,0,0.08)
--shadow-glow: 0 0 20px rgba(124,58,237,0.15)
```

---

## GLOBAL LAYOUT ARCHITECTURE

### Desktop Layout (>1200px)

```
┌─────────────────────────────────────────────────────┐
│                    Global Navigation                │ 40px
├──────────┬──────────────────────┬───────────────────┤
│          │                      │                   │
│  Left    │   Center Content     │   Right Context   │
│  Sidebar │   (Main Workspace)   │   Panel (Pinned)  │
│          │                      │                   │
│  280-320 │    600-800px         │   280-320px       │
│          │                      │                   │
└──────────┴──────────────────────┴───────────────────┘
```

### Tablet Layout (768-1199px)

```
┌─────────────────────────────────┐
│    Global Navigation            │
├──────────┬──────────────────────┤
│          │                      │
│ Collapsed│   Center + Right    │
│ Sidebar  │   (Stacked)         │
│          │                      │
└──────────┴──────────────────────┘
```

### Mobile Layout (<768px)

```
┌─────────────────────┐
│  Global Navigation  │
├─────────────────────┤
│  Center Content     │
│  (Full width)       │
│                     │
│  Right Panel        │
│  (Below content)    │
└─────────────────────┘
```

---

## PAGE TRANSFORMATION SPECIFICATIONS

### 1. HOME PAGE (Landing Page Upgrade)

**Current State:** Hero + Features + Tracks (marketing focused)

**Transformation:**
- **New Hero:** Larger, bolder typography with gradient text
- **Call-to-action:** Prominent, gradient buttons
- **Feature Grid:** 2x2 layout with icons, hover effects
- **Tracks Carousel:** Interactive cards with hover state showing details
- **Statistics:** Visual metrics (110+, 50+, 6) with icons
- **Social Proof:** (Optional) Recent achievements or testimonials

**New Elements:**
```
- Aurora background (animated gradients)
- Glassmorphism cards
- Hover elevation effects
- Smooth scroll behavior
- Mobile-first responsive design
```

---

### 2. DASHBOARD (Learning Command Center)

**Current State:** Simple stat cards and progress bars

**New Three-Column Layout:**

**LEFT PANEL (Navigation & Quick Stats):**
- Profile summary (avatar, name, level/streak)
- Current track progress (visual bar)
- Quick actions (Solve Problem, Read Paper, etc.)
- Recent bookmarks (3-5 items)
- Learning path milestones

**CENTER PANEL (Main Content):**
- **Tab 1: Overview**
  - Welcome message with personalized greeting
  - Key stats: Problems solved, papers read, time spent
  - Learning velocity chart (problems/week)
  - Recent activity timeline
  
- **Tab 2: Roadmap**
  - Current learning track visualization
  - Milestone progress (completed/current/upcoming)
  - Estimated time to completion
  - Quick navigation to next lesson
  
- **Tab 3: Recent Activity**
  - Chronological list of last 10 activities
  - Problem submissions (status, time, difficulty)
  - Papers bookmarked
  - Achievements unlocked

**RIGHT PANEL (Context & Suggestions):**
- Next recommended problem (with difficulty)
- Suggested paper to read
- Current learning metrics
- Achievements/badges (completed + locked)
- Time estimate for next milestone

---

### 3. ARCHITECTURE PAGES (Interactive Architecture Workspace)

**New Three-Column Layout:**

**LEFT PANEL (Architecture Navigator):**
- Architecture name + version
- Layer list (expandable)
- Component search
- Quick jump to layers
- Related architectures link

**CENTER PANEL (Interactive Diagram + Theory):**
- **Layer Explorer**
  - Large interactive diagram (SVG/Canvas)
  - Hover to highlight connections
  - Click to expand layer details
  - Animated tensor flow (optional)
  
- **Tab Content:**
  - Description: Architecture overview, historical context
  - Theory: Mathematical foundations, key innovations
  - Components: Layer-by-layer breakdown
  - Comparisons: vs other architectures
  - Timeline: Evolution of the architecture

**RIGHT PANEL (Details + Code):**
- **Pinned Section:**
  - Selected layer details
  - Mathematical formulas (KaTeX)
  - Key metrics/dimensions
  - Related papers link
  
- **Code Section:**
  - PyTorch implementation (syntax-highlighted)
  - TensorFlow alternative
  - Key hyperparameters
  - Usage examples

- **Interview Questions:**
  - Common questions about this layer
  - Suggested answers
  - Related problems

---

### 4. PAPER PAGES (Research Analysis Workspace)

**New Three-Column Layout:**

**LEFT PANEL (Paper Navigation):**
- Paper title + authors + year
- Jump-to-section links
- Related papers
- Key citations count
- Download/Share options

**CENTER PANEL (Paper Content + Analysis):**
- **Sticky Header:**
  - Paper title, authors, year
  - Citation count, DOI link
  - Read/Bookmark buttons
  
- **Tab System:**
  1. **Summary** — One-page executive summary with key takeaways
  2. **Timeline** — Evolution of ideas (pre-paper context)
  3. **Key Concepts** — Annotated definitions of novel concepts
  4. **Figures** — Enhanced figure display with explanations
  5. **Equations** — Mathematical derivations explained step-by-step
  6. **Implementation** — Code walkthrough from the paper
  7. **Impact** — Citations, follow-up papers, real-world applications

**RIGHT PANEL (Context + Learning):**
- **Quick Facts:**
  - Published date, venue, citations
  - Impact score (h-index adjusted)
  - Required background (links to prerequisites)
  
- **Visual Summary:**
  - Key innovation highlighted
  - Architecture diagram (if applicable)
  - Before/after comparison
  
- **Learning Resources:**
  - Related problems to solve
  - Suggested follow-up papers
  - Video explanations (if available)
  - Interview questions about this paper

---

### 5. PAPER-TO-CODE (Three-Column Learning Environment)

**Theory ↔ Code ↔ Tensor Shapes**

**LEFT PANEL (Theory & Concepts):**
- Paper concept explanation
- Key equations (KaTeX rendered)
- Intuitive diagrams
- Constraint notes
- Common pitfalls

**CENTER PANEL (Code Implementation):**
- Code editor (Monaco)
- Function signature template
- Syntax-highlighted examples
- Run button with test harness
- Reset code option

**RIGHT PANEL (Tensor Trace & Shapes):**
- Input/output tensor shapes
- Step-by-step shape flow
- Variable inspector
- Memory usage breakdown
- Test results with detailed output

**Interaction Model:**
- Click on equation → highlights relevant code
- Click on code line → shows tensor shape impact
- Run code → animates tensor flow on right
- Test results show detailed breakdown

---

### 6. TENSOR TRACE (Premium Visual Redesign)

**Current State:** Static shape visualization

**New Features:**
- **Animated Transitions**
  - Tensor creation with pop effect (300ms)
  - Reshape operations with smooth morphing
  - Broadcasting with glow effect
  - Operations with color-coded steps

- **Color Coding**
  - Dimensions: Different colors (H=red, W=green, C=blue)
  - Operations: Color-coded by type (matmul=purple, reshape=cyan)
  - Gradients: Purple glow on important transforms

- **Interactive Inspector**
  - Hover to see tensor details (dtype, storage, requires_grad)
  - Click to expand step details
  - Keyboard navigation (arrow keys for step progression)

- **Step Navigation**
  - Previous/Next buttons with keyboard shortcuts
  - Progress bar showing current step
  - Ability to jump to specific step
  - Slow-motion playback (0.5x, 1x, 2x)

---

### 7. SYSTEM DESIGN (Interactive Architecture Board)

**New Three-Column Layout:**

**LEFT PANEL (System Components):**
- Component list (DB, Cache, Load Balancer, etc.)
- Search/filter components
- Component category grouping
- Drag-to-add to canvas
- Zoom controls

**CENTER PANEL (Interactive Canvas):**
- Drag-drop component placement
- Connection drawing (click-drag between components)
- Component click → edit properties
- Flow visualization (show request paths)
- Real-time layout adjustment

**RIGHT PANEL (Details + Analysis):**
- **Selected Component Details:**
  - Name, type, capacity
  - Technology stack
  - Configuration options
  
- **Request Flow Analysis:**
  - Current request path highlighting
  - Latency breakdown (ms per hop)
  - Bottleneck identification
  
- **Design Considerations:**
  - Trade-offs (consistency vs availability)
  - Scalability notes
  - Common failure modes
  - Interview talking points

---

## COMPONENT HIERARCHY REFINEMENTS

### New Components Required

```
Layout Components:
├── ThreeColumnLayout
│   ├── LeftSidebar
│   ├── CenterContent
│   └── RightPanel
├── StickyHeader
└── ContextPanel

Navigation Components:
├── BreadcrumbNav (enhanced)
├── TabSystem (enhanced with animations)
├── LayerNavigator (for architectures)
└── SectionJump

Content Components:
├── KaTeXRenderer (math formulas)
├── SyntaxHighlighter (code with themes)
├── InteractiveDiagram (canvas-based)
├── TensorShapeInspector
├── EquationExplainer
└── ConceptCard

Interactive Components:
├── DragDropCanvas
├── ConnectionDrawer
├── AnimatedTransition
├── GlowEffect
├── ProgressIndicator
└── KeyboardShortcutGuide

Data Visualization:
├── TensorFlowViz
├── ArchitectureFlowDiagram
├── RequestPathVisualizer
├── TimelineComponent
└── EvolutionGraph
```

### Component Library Expansions

**Buttons:**
- Primary (gradient)
- Secondary (bordered)
- Ghost (minimal)
- Icon (square)
- Floating action (round)

**Cards:**
- Standard card
- Highlighted card (accent border)
- Metric card (with icon)
- Gradient card (accent background)
- Interactive card (hover effects)

**Input Fields:**
- Text input (with icon support)
- Search input (with suggestions)
- Code editor (Monaco integration)
- Textarea (with char counter)

**Badges & Tags:**
- Difficulty badges (Easy/Medium/Hard/Expert)
- Status badges (Active/Completed/Locked)
- Category tags (with colors)
- Skill tags (with progress)

---

## ANIMATION LIBRARY

### Transitions

**Fade In/Out (300ms)**
```css
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
```

**Slide In/Out (300ms)**
```css
@keyframes slideInRight {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}
```

**Scale Pop (250ms cubic-bezier)**
```css
@keyframes pop {
  0% { transform: scale(0.8); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
```

**Glow Effect (2s infinite)**
```css
@keyframes glow {
  0% { box-shadow: 0 0 10px rgba(124,58,237,0.5); }
  50% { box-shadow: 0 0 20px rgba(124,58,237,0.8); }
  100% { box-shadow: 0 0 10px rgba(124,58,237,0.5); }
}
```

---

## INTERACTION PATTERNS

### Keyboard Shortcuts (Global)

```
Cmd/Ctrl + K      → Search/Command Palette
Cmd/Ctrl + /      → Show keyboard shortcuts
Tab               → Navigate elements
Shift + Tab       → Reverse navigate
Enter             → Confirm action
Escape            → Close modal/panel
→                 → Next (step, problem, etc.)
←                 → Previous
```

### Page-Specific Shortcuts

**Architecture Pages:**
```
1-9               → Jump to layer
Shift + →         → Next architecture
Shift + ←         → Previous architecture
```

**Paper2Code:**
```
Ctrl + Enter      → Run code
Ctrl + Shift + Enter → Submit solution
Ctrl + Space      → Code autocomplete
```

**Tensor Trace:**
```
Space             → Play/Pause animation
→                 → Next step
←                 → Previous step
/                 → Toggle slow-motion
```

### Hover States

- Cards: Border color change to accent + slight elevation
- Buttons: Opacity change + color adjustment
- Interactive elements: Cursor change, subtle glow
- Links: Underline appears, slight color change

### Focus States

- Tab navigation: Accent border + glow effect
- Inputs: Accent border + subtle glow
- Interactive elements: Clear focus ring (4px, accent color)

---

## RESPONSIVE BREAKPOINTS

```
Mobile:    < 640px
Tablet:    640px - 1024px
Laptop:    1024px - 1440px
Desktop:   > 1440px
```

**Layout Changes by Breakpoint:**

| Feature | Mobile | Tablet | Laptop | Desktop |
|---------|--------|--------|--------|---------|
| Left sidebar | Hidden | Collapsed | Expanded | Expanded |
| Right panel | Below content | Below content | Right | Right |
| Content width | 100% - 32px | 100% - 48px | 600px | 800px |
| Grid columns | 1 | 2 | 2 | 3-4 |
| Font sizes | -10% | -5% | Base | Base |
| Padding | 16px | 20px | 24px | 32px |

---

## ACCESSIBILITY REQUIREMENTS

1. **Color Contrast:** WCAG AA minimum (4.5:1 for text)
2. **Focus Indicators:** Visible on all interactive elements
3. **Keyboard Navigation:** Full support without mouse
4. **ARIA Labels:** All icons and interactive elements labeled
5. **Semantic HTML:** Proper heading hierarchy, button/link semantics
6. **Reduced Motion:** Respect `prefers-reduced-motion` media query
7. **Text Scaling:** Support up to 200% zoom
8. **Screen Readers:** Compatible with NVDA, JAWS, VoiceOver

---

## PERFORMANCE TARGETS

- **First Contentful Paint (FCP):** < 1.5s
- **Largest Contentful Paint (LCP):** < 2.5s
- **Cumulative Layout Shift (CLS):** < 0.1
- **Time to Interactive (TTI):** < 3.5s
- **Interactive Elements Response:** < 100ms

**Optimization Techniques:**
- Image lazy loading
- Code splitting per page
- CSS containment for panels
- Virtual scrolling for long lists
- Debounced search/filtering

---

## IMPLEMENTATION PRIORITY

### Phase 1 (Week 1-2): Foundation
1. Update design system variables in globals.css
2. Create three-column layout component
3. Implement new global navigation (if needed)
4. Create reusable card, button, badge components

### Phase 2 (Week 2-3): Dashboard & Home
5. Transform home page (hero, features, tracks)
6. Implement new dashboard with three-column layout
7. Add metrics visualization

### Phase 3 (Week 3-4): Architecture Pages
8. Implement interactive architecture workspace
9. Create architecture navigator (left panel)
10. Implement component details (right panel)

### Phase 4 (Week 4-5): Paper Pages
11. Implement research analysis workspace
12. Create paper navigator
13. Implement paper content with tabs
14. Add KaTeX for equations

### Phase 5 (Week 5-6): Advanced Features
15. Implement Paper-to-Code three-column layout
16. Implement Tensor Trace premium visuals
17. Implement System Design interactive board

### Phase 6 (Week 6+): Polish & Optimization
18. Add animations and transitions
19. Keyboard shortcuts implementation
20. Accessibility audit and fixes
21. Performance optimization

---

## DESIGN SYSTEM EXPORT

```css
/* Color Tokens */
--bg-body: #09090F
--bg-surface: #0D0D14
--bg-panel: #111827
--color-border: #1E293B
--color-muted: #334155
--accent-primary: #7C3AED
--accent-light: #A78BFA
--accent-cyan: #06B6D4
--color-text-primary: #E2E8F0
--color-text-secondary: #94A3B8
--color-text-tertiary: #64748B
--color-easy: #10B981
--color-medium: #F59E0B
--color-hard: #EF4444
--color-expert: #EC4899

/* Typography */
--font-sans: Inter
--font-heading: Plus Jakarta Sans
--font-mono: JetBrains Mono

/* Spacing */
--space-1: 4px
--space-2: 8px
--space-3: 12px
--space-4: 16px
--space-6: 24px
--space-8: 32px

/* Radius */
--radius-sm: 4px
--radius-md: 8px
--radius-lg: 12px
--radius-xl: 16px
--radius-2xl: 24px

/* Transitions */
--transition-fast: 100ms ease
--transition-base: 150ms ease
--transition-slow: 300ms ease
```

---

## LOVABLE IMPLEMENTATION STRATEGY

### Approach: Component-First, Staged Rollout

**Step 1: Environment Setup**
- Configure Lovable workspace
- Import current design system
- Create component library scaffold

**Step 2: Component Library**
- Create reusable UI components (Buttons, Cards, Inputs)
- Build layout primitives (ThreeColumnLayout, SidebarLayout)
- Implement design tokens (colors, spacing, typography)

**Step 3: Page Transformations** (Sequential)
1. Home page redesign
2. Dashboard implementation
3. Architecture pages
4. Paper pages
5. Paper-to-Code layout
6. Tensor Trace visuals
7. System Design board

**Step 4: Polish**
- Add animations using Framer Motion
- Implement keyboard shortcuts
- Add accessibility features
- Performance optimization

### Lovable Integration Points

**Phase Management:**
- Use `send_message` to iterate on designs
- Use `plan_mode=true` to discuss approach before building
- Use `get_diff` to review changes
- Maintain version control integration

**Design Assets:**
- Upload design references (screenshots, wireframes)
- Use Figma integration if available
- Export CSS/Tailwind utilities from Lovable

**Connectors to Use:**
- GitHub (for code sync)
- Figma (for design reference)
- Supabase (if backend integration needed later)

---

## SUCCESS METRICS

### Design Metrics
- [ ] All pages implement three-column layout (where applicable)
- [ ] No regressions in existing functionality
- [ ] Mobile-responsive at all breakpoints
- [ ] All keyboard shortcuts working
- [ ] Accessibility audit passing (WCAG AA)

### Performance Metrics
- [ ] FCP < 1.5s
- [ ] LCP < 2.5s
- [ ] CLS < 0.1
- [ ] Bundle size increases < 10%

### User Experience Metrics
- [ ] All animations smooth (60fps)
- [ ] No layout shift on page load
- [ ] Hover states responsive (<100ms)
- [ ] Search/filtering snappy

---

## CONSTRAINTS & GUARDRAILS

⚠️ **CRITICAL CONSTRAINTS:**
1. **No backend changes** — UI/CSS/components only
2. **No new features** — Layout & styling transformation only
3. **No content changes** — Same text, same structure
4. **No breaking changes** — Backward compatible with current URLs and data
5. **Maintain functionality** — All existing interactions must work
6. **Keep brand identity** — Dark theme, purple/cyan accents

---

## NEXT STEPS

1. **Approve design system** — Review colors, typography, spacing
2. **Approve layout specifications** — Review three-column designs
3. **Create Lovable project** — Initialize with design tokens
4. **Build component library** — Reusable UI components
5. **Implement pages sequentially** — Home → Dashboard → Advanced pages
6. **Iterate & polish** — Animations, accessibility, performance

---

## APPENDIX: DETAILED MOCKUP DESCRIPTIONS

### Mockup 1: Home Page (Hero Section)

**Layout:** Full width, centered content
**Hero Section:**
- Background: Animated aurora gradient (purple → cyan)
- Title: "Master AI Engineering" (gradient text) + "Through Problems, Papers & Practice" (white)
- Subtitle: "Learn deep learning and LLMs the hard way..."
- CTA Buttons: Primary (Start Solving) + Secondary (Browse Roadmaps)
- Stats: 110+ Problems, 50+ Papers, 6 Tracks (large text with icons)

### Mockup 2: Dashboard (Overview Tab)

**Left Panel (280px):**
- User profile (avatar, name, "Alex")
- Current track progress bar
- Quick actions (4 buttons)
- Recent bookmarks (5 items)

**Center Panel:**
- Welcome message with personalized greeting
- Key metrics (4 cards: Problems, Papers, Time, Streak)
- Learning velocity chart (line chart, last 7 days)
- Recent activity (timeline, 10 items)

**Right Panel (280px):**
- Next problem recommendation
- Suggested paper
- Current metrics
- 4-5 achievement badges
- Progress to next milestone

### Mockup 3: Architecture Page (Transformer)

**Left Panel:**
- Architecture name "Transformer" with year "2017"
- Layer list (Input Embedding, Multi-Head Attention, etc.)
- Component search input
- Related architectures (3-4 links)

**Center Panel:**
- Large interactive diagram (SVG, clickable)
- Below: Description tab content
- Markdown with images and links
- Theory section with key concepts

**Right Panel:**
- Selected layer details (if clicked)
- Mathematical formulas (KaTeX)
- Key metrics/dimensions
- PyTorch code example (10-15 lines)
- Related problems (2-3 links)

### Mockup 4: Paper Page (Attention Is All You Need)

**Left Panel:**
- "Attention Is All You Need"
- "Vaswani et al., 2017"
- Jump-to-section links
- 84.2K citations badge
- Share/Download buttons

**Center Panel:**
- Sticky header with title, authors, year
- 7 tabs: Summary, Timeline, Concepts, Figures, Equations, Implementation, Impact
- Summary: One-page executive summary with key takeaways
- Rich media (images, equations, code blocks)

**Right Panel:**
- Citation count, venue, DOI
- Impact score
- Required background (links)
- Key innovation diagram
- Related problems (3 links)
- Follow-up papers (3 links)

### Mockup 5: Paper-to-Code (Backpropagation)

**Left Panel (320px):**
- Concept explanation (2-3 paragraphs)
- Key equations (3-5 KaTeX blocks)
- Intuitive diagrams (animated SVG)
- Common pitfalls (red highlighted items)

**Center Panel (500px):**
- Code editor (Monaco, dark theme)
- Function signature visible
- Run button, Reset button
- Example code below (syntax-highlighted)

**Right Panel (280px):**
- Input tensor shapes
- Step-by-step shape flow (vertical layout)
- Variable inspector (table)
- Memory usage (bar chart)
- Test results (4-6 assertions)

### Mockup 6: Tensor Trace (Forward Pass)

**Center (Full-width, single column):**
- Step navigation (← Previous | 5/12 | Next →)
- Playback controls (Play/Pause, Speed selector)
- Large tensor visualization (animated, color-coded)
- Details panel below (dtype, shape, device, requires_grad)

### Mockup 7: System Design (API Rate Limiter)

**Left Panel (280px):**
- Component list (8-10 items)
- Search input
- Category dropdown (Compute, Storage, etc.)
- Zoom in/out controls

**Center Panel:**
- Canvas with 5-6 components (box diagrams)
- Connection lines between components
- Request flow highlighting (animated)
- Component click opens details

**Right Panel (280px):**
- Selected component name + description
- Technology choices
- Configuration options
- Request path latency breakdown
- Bottleneck identification
- Interview talking points

---

## RESOURCES & REFERENCES

**Design System Documentation:**
- Tailwind CSS: https://tailwindcss.com
- Design tokens: https://www.designtokens.org

**Component Libraries:**
- shadcn/ui: https://ui.shadcn.com
- Radix UI: https://radix-ui.com
- Headless UI: https://headlessui.com

**Animation Libraries:**
- Framer Motion: https://www.framer.com/motion
- react-spring: https://react-spring.dev

**Code Editors:**
- Monaco Editor: https://microsoft.github.io/monaco-editor
- Highlight.js: https://highlightjs.org

**Math Rendering:**
- KaTeX: https://katex.org
- MathJax: https://www.mathjax.org

---

**Document Version:** 1.0  
**Last Updated:** June 16, 2026  
**Author:** Design Team  
**Status:** Ready for Implementation
