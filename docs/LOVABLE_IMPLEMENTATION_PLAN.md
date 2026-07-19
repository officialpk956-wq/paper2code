# Paper2Code → Lovable Implementation Plan

**Platform:** Lovable (AI-powered full-stack app builder)  
**Project:** Paper2Code UI/UX Transformation  
**Scope:** 5-6 week phased rollout  
**Deliverable:** Premium AI engineering learning platform

---

## LOVABLE PROJECT STRUCTURE

### Workspace Setup
- **Project Name:** Paper2Code Premium
- **Stack:** Next.js + TypeScript + Tailwind CSS + Shadcn/UI
- **Preview URL:** https://lovable.dev/[project-id]
- **GitHub Integration:** Connected to existing repo

### File Organization
```
/src
  /components
    /layout
      three-column-layout.tsx
      sidebar.tsx
      right-panel.tsx
    /home
      hero-section.tsx
      features-grid.tsx
      tracks-carousel.tsx
    /dashboard
      metrics-cards.tsx
      progress-chart.tsx
      activity-timeline.tsx
    /shared
      gradient-button.tsx
      card-component.tsx
      badge.tsx
  /app
    /(home)/page.tsx          [MODIFIED]
    /dashboard/page.tsx       [MODIFIED]
    /architectures/[slug]/page.tsx [MODIFIED]
    /papers/[slug]/page.tsx   [MODIFIED]
    /paper-to-code/[slug]/page.tsx [MODIFIED]
    /tensor-trace/[slug]/page.tsx [MODIFIED]
    /system-design/[slug]/page.tsx [MODIFIED]
  /styles
    globals.css               [MODIFIED - updated design tokens]
    three-column.css          [NEW]
    animations.css            [NEW]
  /lib
    /ui
      components.ts           [NEW - component exports]
  /public
    /vectors
      architecture-icons.svg  [NEW]
      flow-diagrams.svg       [NEW]
```

---

## PHASE BREAKDOWN

### PHASE 1: FOUNDATION (Week 1)

**Lovable Prompts:**

1. **Initialize Design System**
```
Create a comprehensive design token system in globals.css with:
- Color palette (dark theme, purple + cyan accents)
- Typography system (Plus Jakarta Sans headings, Inter body)
- Spacing tokens (4px, 8px, 12px, 16px, 24px, 32px)
- Border radius variants (sm: 4px, md: 8px, lg: 12px, xl: 16px, 2xl: 24px)
- Transition timing functions (fast: 100ms, base: 150ms, slow: 300ms)

Ensure all CSS variables are defined and exported for use in components.
```

2. **Create Layout Primitives**
```
Build a ThreeColumnLayout component that:
- Accepts leftPanel, centerPanel, rightPanel children
- Responsive: Desktop (L: 280px, R: 280px, C: auto), Tablet (stacked), Mobile (full-width)
- Supports collapsible left sidebar (toggle button, smooth animation)
- Sticky headers on scroll
- Scrollable content areas with proper overflow handling
- Implemented with React hooks for state management

Include variants: two-column, full-width, collapsible-right.
```

3. **Build Core Components**
```
Create shadcn/ui based components:
- GradientButton (primary gradient, secondary border, ghost)
- Card (standard, highlighted, interactive, metric variants)
- Badge (difficulty, status, category, skill levels)
- Tabs (with active underline, smooth fade transitions)
- Input (with icon support, search variant, code highlight)
- Breadcrumb (with navigation, current page highlight)

All components should use design tokens and support dark mode.
```

4. **Global Navigation Update**
```
Enhance GlobalNav component with:
- Logo + brand name on left
- Search input (command palette style, Cmd+K)
- Navigation links (Home, Learn, Dashboard, etc.)
- User profile dropdown (avatar, name, logout)
- Responsive: Hamburger menu on mobile
- Sticky positioning with subtle shadow on scroll

Integrate with existing GlobalNavNew if present.
```

**Deliverables:**
- Design system fully defined in CSS variables
- Three-column layout component working
- 8-10 core components usable
- Global navigation updated and responsive

**Lovable Commands:**
```
send_message: "Create the design system with all CSS variables..."
send_message: "Build ThreeColumnLayout component with full responsiveness..."
send_message: "Create Button, Card, Badge, Input components using shadcn/ui..."
send_message: "Update GlobalNav with search, user menu, responsive design..."
```

---

### PHASE 2: HOME & DASHBOARD (Week 2)

**Lovable Prompts:**

5. **Home Page Redesign**
```
Redesign home page (/src/app/page.tsx) with:
- Hero section: Large gradient title, subtitle, animated aurora background
- CTA buttons: Primary (gradient), Secondary (bordered)
- Features grid: 2x2 layout, hover effects, icon support
- Tracks carousel: 6 cards with gradient backgrounds, hover reveals details
- Stats section: 3 stats (110+, 50+, 6) with large text and icons
- Responsive design: Full-width on mobile, centered on desktop
- Smooth scroll animations (fade-in, slide-up as user scrolls)

Keep all existing content but enhance visual hierarchy and interactivity.
```

6. **Dashboard Three-Column Layout**
```
Transform dashboard (/src/app/dashboard/page.tsx) to three-column layout:

LEFT PANEL (280px):
- User profile section (avatar, name, "Alex")
- Current track progress (visual progress bar, % complete)
- 4 quick action buttons (Solve Problem, Read Paper, etc.)
- Recent bookmarks (5 items with thumbnail/icon, click to navigate)
- Learning milestone progress (vertical steps)

CENTER PANEL:
- Overview tab (default)
  - Personalized welcome message
  - 4 metric cards (Problems Solved, Papers Read, Hours Spent, Streak)
  - Learning velocity chart (line chart, last 7 days)
  - Recent activity timeline (10 items, chronological)
- Other tabs (Roadmap, Analytics) for future expansion

RIGHT PANEL (280px):
- "Up Next" section: Next recommended problem
- Suggested paper to read
- Current learning metrics (speed, accuracy)
- Achievement badges (4-5, some locked)
- Time to next milestone

Implement responsive: Tablet (panels stack), Mobile (full-width stacked).
```

7. **Metrics Visualization**
```
Create metrics display components:
- Metric card: Icon, title, large number, trend indicator (+3 this week)
- Progress bar: Visual progress, percentage text, color-coded (purple accent)
- Chart: Simple line chart showing learning velocity over 7 days
- Timeline: Vertical timeline for recent activities with timestamps
- Badge grid: 2x2 grid of achievement badges with lock state

Use Chart.js or Recharts for visualization. Keep animations smooth (300ms).
```

8. **Responsive Mobile Dashboard**
```
Ensure dashboard works perfectly on mobile:
- Single column layout (left panel hidden, collapsed to mobile nav)
- Full-width content
- Cards stack vertically
- Charts responsive to viewport width
- Touch-friendly button sizes (44px minimum)
- Scroll performance optimized (virtual scrolling for long lists if needed)
```

**Deliverables:**
- Home page visually transformed with premium feel
- Dashboard with three-column layout on desktop
- All metrics and progress visualizations working
- Mobile responsive across all components

**Lovable Commands:**
```
send_message: "Redesign the home page with aurora background, large titles, gradient buttons..."
send_message: "Build dashboard three-column layout with left panel (profile, quick actions)..."
send_message: "Implement metrics cards and learning velocity chart..."
send_message: "Ensure dashboard is fully responsive on mobile devices..."
```

---

### PHASE 3: ARCHITECTURE PAGES (Week 3-4)

**Lovable Prompts:**

9. **Architecture Workspace Layout**
```
Create architecture page three-column layout (/src/app/architectures/[slug]/page.tsx):

LEFT PANEL (280px):
- Architecture name + year (e.g., "Transformer", "2017")
- Searchable layer list (collapsible, click to jump)
- Component search input
- Related architectures (links)
- Difficulty indicator

CENTER PANEL (Main content):
- Sticky header: Architecture title, year, difficulty badge
- Large interactive SVG diagram (clickable layers, hover highlights)
- Tab system (Description, Theory, Components, Comparisons, Timeline)
- Rich content: Markdown with images, equations, code blocks
- Max content width: 800px for readability

RIGHT PANEL (280px):
- Selected layer details (if layer clicked)
- Mathematical formulas (KaTeX, white background, monospace font)
- Key metrics/dimensions (table)
- PyTorch implementation code (10-20 lines, syntax highlighted)
- Related problems (2-3 links with difficulty)
- Next architecture link

Responsive: Tablet (panels stack), Mobile (full-width, layers in accordion).
```

10. **Interactive Architecture Diagram**
```
Implement SVG/Canvas diagram component:
- Visualize neural network layers (Input → Embedding → Attention → FFN → Output)
- Hover state: Highlights layer connections, shows brief tooltip
- Click state: Selects layer, displays details in right panel
- Animated on load: Layers fade in sequentially (100ms between each)
- Color-coded: Different colors for different layer types
- Responsive: Scales to container, mobile-friendly touch targets

Include example layers for Transformer (Multi-Head Attention, Position-wise FFN, etc.)
```

11. **Theory Tab with KaTeX**
```
Create theory content rendering:
- Title: "Attention Mechanism Fundamentals"
- Concept explanation (2-3 paragraphs)
- Key equations rendered with KaTeX:
  - Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
  - Multi-head: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
- Common mistakes section (3-4 items, red styling)
- Visual diagrams (inline SVG)
- Links to related concepts

Ensure equations are large, readable, with proper spacing.
```

12. **Timeline & Comparisons Tabs**
```
Create timeline visualization:
- Year on left, event on right (alternating)
- "2014: Sequence-to-Sequence" → "2017: Transformer" → "2023: Multimodal"
- Each item clickable, shows details

Create comparison table:
- Transformer vs CNN vs RNN
- Rows: Computational Complexity, Training Time, Parallel, Context Window
- Color-code cells for better/worse (green/red)
- Hover to expand details
```

**Deliverables:**
- Architecture pages with three-column layout working
- Interactive SVG diagrams functional
- KaTeX equations rendering correctly
- All tabs (Description, Theory, Components, Timeline, Comparisons) working
- Mobile responsive

**Lovable Commands:**
```
send_message: "Create architecture page three-column layout with interactive diagram..."
send_message: "Build SVG diagram component for transformer layers, make interactive..."
send_message: "Implement KaTeX math rendering for theory tab..."
send_message: "Create timeline and comparison tabs with data visualization..."
send_message: "Ensure all tabs are responsive and touch-friendly on mobile..."
```

---

### PHASE 4: PAPER PAGES (Week 4-5)

**Lovable Prompts:**

13. **Paper Workspace Layout**
```
Create paper page three-column layout (/src/app/papers/[slug]/page.tsx):

LEFT PANEL (280px):
- Paper title + authors + year (e.g., "Attention Is All You Need", "Vaswani et al.", "2017")
- Jump-to-section links (Summary, Timeline, Key Concepts, etc.)
- Related papers (3-5 links)
- Citation count (84.2K)
- Share, Download, Bookmark buttons

CENTER PANEL (Main content):
- Sticky header: Paper title, authors, year, DOI link
- 7-tab system: Summary | Timeline | Key Concepts | Figures | Equations | Implementation | Impact
- Rich content: Long-form text, images, code blocks, equations
- Max content width: 800px for readability
- Proper spacing between sections

RIGHT PANEL (280px):
- Quick facts: Published date, venue, citations, h-index adjusted impact
- Required background (prerequisite links)
- Key innovation highlighted (colored box with description)
- Architecture diagram (if applicable)
- Before/after comparison
- Related problems (3 links with difficulty)
- Follow-up papers (3 links with year)

Responsive: Tablet (panels stack), Mobile (full-width, tabs in accordion).
```

14. **Paper Summary Tab**
```
Create executive summary rendering:
- One-page summary (1000-1500 words)
- Key takeaways section (5-7 bullets)
- Main contribution bullet list
- Impact statement
- Prerequisite concepts (links)
- Quick visual overview (diagram or image)

Style: Clear typography, proper spacing, easy to scan.
```

15. **Paper Equations Tab**
```
Create equation walkthrough:
- Each equation numbered (1, 2, 3, etc.)
- KaTeX rendering for mathematical notation
- Explanation paragraph below each equation
- Step-by-step derivation shown
- Variable definitions table

Example: "Equation 2: Multi-Head Attention"
- Display equation
- "Where:" table explaining Q, K, V, d_k, h
- "This allows the model to..."
```

16. **Paper Timeline Tab**
```
Create paper evolution timeline:
- Vertical timeline showing pre-paper context
- Years on left: 2014, 2015, 2016, 2017 (paper), 2018, 2019
- Events: "Seq2Seq (Sutskever)", "Attention (Bahdanau)", "Transformer (This Paper)"
- Each item shows: year, title, brief description, link to paper if available
- Visual styling: dots on timeline, alternating sides, color-coded by type
```

17. **Paper Implementation Tab**
```
Create code walkthrough:
- Pseudocode or Python implementation
- Syntax highlighted with line numbers
- Key functions highlighted (def attention(), def forward())
- Comments explaining key lines
- Collapsible sections: Import → Model Definition → Forward Pass → Loss
- "Load into Editor" button for Paper-to-Code problems
```

**Deliverables:**
- Paper pages with three-column layout working
- All 7 tabs functional (Summary, Timeline, Concepts, Figures, Equations, Implementation, Impact)
- KaTeX equations rendering in context
- Rich media support (images, code blocks)
- Mobile responsive

**Lovable Commands:**
```
send_message: "Create paper page three-column layout with 7-tab system..."
send_message: "Build executive summary tab with key takeaways section..."
send_message: "Implement equation tab with KaTeX and step-by-step derivations..."
send_message: "Create timeline tab showing paper evolution and related work..."
send_message: "Build implementation tab with code syntax highlighting..."
```

---

### PHASE 5: ADVANCED LAYOUTS (Week 5-6)

**Lovable Prompts:**

18. **Paper-to-Code Three-Column**
```
Create paper-to-code page (/src/app/paper-to-code/[slug]/page.tsx):

LEFT PANEL (320px):
- Concept explanation (2-3 paragraphs)
- Key equations (3-5 KaTeX blocks)
- Intuitive diagrams (animated SVG)
- Common pitfalls section (red styling, 2-3 items)
- Constraint notes

CENTER PANEL (500px):
- Code editor (Monaco Editor integration)
- Function signature visible at top
- Run button (Ctrl+Enter), Reset button
- Example code template visible below
- Test harness integration

RIGHT PANEL (280px):
- Input tensor shapes
- Step-by-step shape flow (vertical list)
- Variable inspector (table: name, shape, dtype)
- Memory usage (bar chart)
- Test results (4-6 assertions, pass/fail)

Interaction:
- Click equation → highlights relevant code line
- Click code line → shows impact on tensor shapes
- Run code → animates tensor flow
- Test results show detailed output

Responsive: Tablet (center panel full width, right panel below), Mobile (single column).
```

19. **Tensor Trace Premium Visuals**
```
Create tensor-trace page (/src/app/tensor-trace/[model]/page.tsx) with:

MAIN CONTENT (Full width):
- Step navigation: "← Previous | Step 5 of 12 | Next →"
- Playback controls: Play/Pause button, Speed selector (0.5x, 1x, 2x)
- Progress indicator: Visual progress bar, current/total steps

VISUALIZATION (Large center area):
- Animated tensor shapes with color-coded dimensions
- Large display: H (red), W (green), C (blue) for CNN example
- Smooth morph transitions when shape changes
- Glow effect on important operations
- Timestamp: "Time: 0.23ms"

DETAILS PANEL (Below visualization):
- Tensor information: dtype, shape, device, requires_grad
- Operation details: MatMul, Reshape, etc.
- Gradual reveal animation (pop effect on load)

FEATURES:
- Keyboard shortcuts: Space (Play/Pause), ← → (Step), / (Toggle slow-motion)
- Hover to expand tensor details
- Click step number to jump
- Memory usage visualization
- Backward pass animation (if applicable)

Animations: 300ms transitions, 60fps, smooth curves.
```

20. **System Design Interactive Board**
```
Create system-design page (/src/app/system-design/[slug]/page.tsx):

LEFT PANEL (280px):
- Component list (8-10 items: DB, Cache, Load Balancer, API Gateway, etc.)
- Search input to filter components
- Category dropdown (Compute, Storage, Networking, etc.)
- Drag-to-add to canvas
- Zoom controls (+ / -)

CENTER PANEL (Interactive Canvas):
- White/dark background canvas area
- Drag-drop component placement (boxes/rectangles)
- Connection drawing: Click component → Click target → Draw line
- Component click opens edit popup
- Remove component button
- Request flow visualization: Highlight path on selection
- Auto-layout options (hierarchical, organic)

RIGHT PANEL (280px):
- SELECTED COMPONENT DETAILS:
  - Name: "PostgreSQL Database"
  - Type: "Data Store"
  - Capacity: "1TB SSD"
  - Technology: "PostgreSQL 14"
  - Config options: Replication, Backup, etc.

- REQUEST FLOW ANALYSIS:
  - Path: API → Cache → DB → Cache → API
  - Latency: 2ms → 0.5ms → 8ms → 0.5ms → 1ms
  - Total: 12ms
  - Bottleneck: Database query (8ms)

- DESIGN CONSIDERATIONS:
  - Trade-offs: Strong consistency vs high availability
  - Scalability: Horizontal vs vertical
  - Common failures: Network partition handling
  - Interview points: "How would you handle cache invalidation?"

INTERACTION:
- Add components via drag-from-left-panel
- Draw connections with click-drag
- Click component to edit properties
- Hover on connection to see latency
- Click "Analyze" to highlight critical path
```

21. **Animations & Interactions**
```
Implement smooth animations:

Fade-in: All content elements (opacity 0 → 1, 300ms)
Slide-up: Sections enter from bottom (translateY: 20px → 0, 300ms)
Scale-pop: Badges, badges, notifications (scale 0.8 → 1, 250ms cubic-bezier)
Glow: Important elements (box-shadow animation, 2s infinite)
Color-shift: Hover states (transition 100ms)

Tensor animations: Morph transitions for shape changes (500ms cubic-bezier)
System design: Connection draw animation (1s stroke animation)

All respect prefers-reduced-motion media query.
```

22. **Keyboard Shortcuts**
```
Implement global shortcuts:
- Cmd/Ctrl + K: Open search/command palette
- Cmd/Ctrl + /: Show keyboard shortcuts help
- Tab: Navigate to next element
- Shift + Tab: Navigate to previous
- Enter: Confirm action
- Escape: Close modal

Page-specific:
- Architecture: 1-9 (jump to layer), Shift+← (prev arch), Shift+→ (next arch)
- Paper2Code: Ctrl+Enter (run), Ctrl+Shift+Enter (submit)
- TensorTrace: Space (play/pause), ← → (steps), / (slow-mo)

Show visual hints (keyboard icon, tooltip on hover).
```

**Deliverables:**
- Paper-to-Code with three-column layout and Monaco editor working
- Tensor Trace with animated visualizations and keyboard controls
- System Design interactive board with drag-drop and flow analysis
- All keyboard shortcuts functional
- Smooth animations throughout

**Lovable Commands:**
```
send_message: "Create paper-to-code three-column layout with Monaco editor..."
send_message: "Implement tensor trace with animated tensor visualization..."
send_message: "Build system design interactive board with drag-drop components..."
send_message: "Add keyboard shortcuts globally and per-page..."
send_message: "Implement smooth animations using Framer Motion or CSS transitions..."
```

---

### PHASE 6: POLISH & OPTIMIZATION (Week 6+)

**Lovable Prompts:**

23. **Accessibility Audit**
```
Ensure WCAG AA compliance:
- Color contrast: All text 4.5:1 minimum
- Focus indicators: Visible on all interactive elements (4px ring, accent color)
- Keyboard navigation: Full support, no mouse required
- ARIA labels: All icons and interactive elements labeled
- Semantic HTML: Proper heading hierarchy (h1 → h2 → h3), button/link semantics
- Screen readers: Test with VoiceOver, NVDA
- Form labels: Associated with inputs
- Alt text: All images have descriptive alt text
- Focus management: Modals trap focus, escape closes them

Test with: axe DevTools, Lighthouse, manual keyboard navigation.
```

24. **Performance Optimization**
```
Optimize for performance targets (FCP < 1.5s, LCP < 2.5s):
- Image lazy loading: Use next/image with loading="lazy"
- Code splitting: Per-page bundles, dynamic imports for heavy components
- CSS containment: Use contain: layout on scrollable panels
- Virtual scrolling: For long lists (activity timeline, problem list)
- Debounced search: 300ms debounce on search inputs
- CSS-in-JS: Consider styled-components or CSS modules for scoped styling

Monitor with: Lighthouse, Web Vitals, Chrome DevTools.
```

25. **Documentation & Handoff**
```
Create developer documentation:
- Component library guide (all components, props, usage examples)
- Design system tokens (colors, spacing, typography, with hex codes)
- Layout patterns (three-column, two-column, full-width)
- Animation library (keyframes, durations, timing functions)
- Keyboard shortcuts reference
- Accessibility checklist
- Performance guidelines

Provide: README, CHANGELOG, CONTRIBUTING.md, component stories (Storybook if available).
```

**Deliverables:**
- Full WCAG AA compliance verified
- Performance metrics met (FCP, LCP, CLS, TTI)
- Complete documentation
- All regressions fixed
- Ready for production deployment

**Lovable Commands:**
```
send_message: "Run accessibility audit, fix WCAG AA issues, verify focus states..."
send_message: "Optimize performance: image loading, code splitting, virtual scrolling..."
send_message: "Generate comprehensive documentation and component guide..."
send_message: "Test all pages on mobile, tablet, desktop - fix responsive issues..."
```

---

## LOVABLE WORKFLOW SUMMARY

### Initial Setup
```
1. Create new Lovable project: "Paper2Code Premium"
2. Connect GitHub repository
3. Set up environment: Node.js, npm, TypeScript
4. Install dependencies: Next.js, Tailwind CSS, shadcn/ui, KaTeX, Monaco
5. Create initial commit with base structure
```

### Weekly Iteration Pattern
```
MONDAY: Plan week (Lovable plan_mode: true)
  - Review this document
  - Identify 3-5 components to build
  - Discuss approach with Claude

TUESDAY-THURSDAY: Implementation (Lovable send_message)
  - Describe component/feature
  - Claude builds and iterates
  - Review changes with get_diff
  - Approve or request changes

FRIDAY: Integration & Testing
  - Test on multiple devices (desktop, tablet, mobile)
  - Check accessibility (keyboard nav, screen reader)
  - Performance audit (Lighthouse)
  - Prepare for next week
```

### Code Review Checklist
- [ ] Matches design system (colors, spacing, typography)
- [ ] Responsive across breakpoints (mobile, tablet, desktop)
- [ ] Keyboard accessible (Tab, Enter, Escape work)
- [ ] No regressions in existing functionality
- [ ] Performance acceptable (no layout shifts)
- [ ] Mobile-friendly touch targets (44px minimum)
- [ ] Animations smooth (60fps)
- [ ] Works in all major browsers (Chrome, Firefox, Safari, Edge)

---

## INTEGRATION WITH EXISTING CODEBASE

### Files to Modify
```
/src/app/globals.css          → Update CSS variables
/src/app/layout.tsx           → Keep, may update nav
/src/app/page.tsx             → Home redesign
/src/app/dashboard/page.tsx   → Dashboard redesign
/src/app/architectures/[slug]/page.tsx → New layout
/src/app/papers/[slug]/page.tsx → New layout
/src/app/paper-to-code/[slug]/page.tsx → New layout
/src/app/tensor-trace/[model]/page.tsx → New layout
/src/app/system-design/[slug]/page.tsx → New layout
```

### Files to Create
```
/src/components/layout/three-column-layout.tsx
/src/components/layout/sidebar.tsx
/src/components/layout/right-panel.tsx
/src/components/shared/gradient-button.tsx
/src/components/shared/card-variants.tsx
/src/components/shared/badge-variants.tsx
/src/components/animations/transition-wrapper.tsx
/src/styles/animations.css
/src/styles/three-column.css
/src/lib/keyboard-shortcuts.ts
/src/lib/animation-config.ts
```

### Backward Compatibility
- All existing routes continue to work
- URL structure unchanged
- Data fetching unchanged
- No backend modifications
- No database changes

### Deployment Strategy
```
1. Build and test locally (npm run dev)
2. Export from Lovable as Next.js project
3. Run full test suite
4. Deploy to staging (Vercel preview)
5. Smoke test all pages
6. Deploy to production with gradual rollout
7. Monitor performance (Google Analytics, Sentry)
```

---

## SUCCESS CRITERIA

### Visual Design
- [ ] All pages implement intended layout (three-column where specified)
- [ ] Design system tokens used consistently
- [ ] Hover/focus states visible and functional
- [ ] Animations smooth and purposeful
- [ ] No visual regressions from current design

### Functionality
- [ ] All existing features work
- [ ] No broken links or missing content
- [ ] Keyboard navigation works fully
- [ ] Search/filtering responsive
- [ ] Forms submit correctly

### Performance
- [ ] FCP < 1.5s
- [ ] LCP < 2.5s
- [ ] CLS < 0.1
- [ ] Bundle size < 10% increase
- [ ] 60fps animations
- [ ] Mobile loading < 3s

### Accessibility
- [ ] WCAG AA compliant
- [ ] Keyboard navigation 100%
- [ ] Screen reader compatible
- [ ] Color contrast verified
- [ ] Focus states clear

### Responsiveness
- [ ] Mobile (< 640px): Single column, responsive text
- [ ] Tablet (640-1024px): Two column where possible
- [ ] Laptop (1024-1440px): Full layout
- [ ] Desktop (> 1440px): Optimal spacing
- [ ] Touch targets minimum 44px

---

## ROLLOUT SCHEDULE

```
WEEK 1:  Foundation (Design tokens, layouts, components)
WEEK 2:  Home & Dashboard (Visual transformation begins)
WEEK 3:  Architecture Pages (Interactive diagrams)
WEEK 4:  Paper Pages (Rich content, KaTeX)
WEEK 5:  Advanced Features (Paper2Code, TensorTrace, System Design)
WEEK 6:  Polish & Optimization (Animations, accessibility, performance)
WEEK 7:  Testing & Deployment (QA, rollout, monitoring)
```

---

## LOVABLE-SPECIFIC NOTES

### Using `send_message` Effectively
- Be specific about requirements
- Include visual references when possible
- Ask for one feature at a time
- Request responsive versions (desktop, tablet, mobile)
- Ask Claude to explain choices before implementation

### Using `plan_mode=true`
- Discuss architecture before building
- Get approval on layout structure
- Clarify complex interactions
- Review component relationships
- Plan responsive breakpoints

### Using `get_diff`
- Review all CSS changes
- Check TypeScript type safety
- Verify HTML structure changes
- Ensure no regressions
- Validate component props

### Performance in Lovable
- Preview is real Next.js, so performance metrics are accurate
- Use Lighthouse in Chrome DevTools to verify metrics
- Test animations on slower devices
- Monitor bundle size growth
- Use React DevTools Profiler to find bottlenecks

---

## FINAL DELIVERABLES

### Code
- ✅ Next.js project with updated pages
- ✅ New components in component library
- ✅ Updated design system
- ✅ Animation library
- ✅ Keyboard shortcuts implemented

### Design Assets
- ✅ Updated design tokens (CSS variables)
- ✅ Component documentation
- ✅ Responsive design specs
- ✅ Animation library reference
- ✅ Accessibility checklist

### Documentation
- ✅ Component API documentation
- ✅ Design system guide
- ✅ Implementation guide
- ✅ Keyboard shortcuts reference
- ✅ Accessibility guide

---

**Ready for Lovable Implementation**

This plan provides clear, actionable prompts for the Lovable AI agent to build each component and page. The phased approach allows for incremental testing and refinement, and the detailed specifications ensure the final product meets the design vision.

**Next Step:** Create Lovable project and begin Phase 1 (Foundation week).
