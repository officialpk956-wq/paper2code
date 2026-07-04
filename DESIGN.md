## BRAND

Name: paper2code
Logo: small orange dot + "paper2code" wordmark
Primary color: #F97316 (amber/orange)
Background: #0A0A0A
Surface: #111111
Border: #1A1A1A
Muted text: #A3A3A3
Primary text: #FAFAFA

Pillar accents:
  Papers   → #A78BFA (violet)
  Dojo     → #F97316 (orange)
  Learn    → #60A5FA (blue)
  Labs     → #34D399 (emerald)

Difficulty colors:
  Easy   → #4ADE80
  Medium → #FACC15
  Hard   → #F87171

Font: Inter (body), JetBrains Mono (code)

---

## GLOBAL: TOP NAVBAR
Height 56px. Full-width. bg #0A0A0A. border-bottom 1px #1A1A1A.

Layout left to right:
  - Orange dot (12px) + "paper2code" (16px bold)
  - Nav links: Dojo | Papers | Learn | Labs | Pricing
    Each: 14px, muted color. Active link: orange color + 2px orange underline + subtle orange bg tint
  - Right side: "Sign In" ghost button + "Get Started" orange filled button
  - Logged-in state: 🔥 5 day streak chip (amber) + ⚡ 1.2k XP chip (violet) + avatar circle (32px)

---

## PAGE 1: HOMEPAGE  /

Full-width marketing page. No sidebar.

### HERO
Full viewport height. Black background.
Soft radial orange glow behind center content (large, very blurred, low opacity).

Center stack:
  - Announcement pill: border orange/30, bg orange/5, text orange
    Text: "🚀  New: AI-powered architecture blueprints →"
  - H1 (72px bold, white, centered, max-width 900px):
    "From Research Papers to Running Code"
  - Subtext (18px, muted, centered, max-width 600px):
    "Upload any ML paper and get coding challenges, architecture diagrams,
     and guided implementations."
  - Two CTA buttons side by side:
    "Start Building →" — solid orange, black text, rounded-full, large
    "Browse Problems" — border only, white text, rounded-full, large
  - Stats bar below buttons (py-8, border-top + border-bottom, subtle):
    1,240+ Papers  |  8,500+ Submissions  |  320+ Learners  |  128 Problems
    Numbers in orange, labels in muted gray

### PAPERS SECTION
py-24. Background: #0A0A0A.
  - Violet pill: "📄 Research Hub"
  - H2: "Upload. Extract. Understand."
  - Body text muted
  - 4-card 2×2 grid with feature cards:
    Each card: dark surface, 1px border, rounded-xl, icon box (violet/15 bg) + title + short desc
    Cards: PDF Upload | Knowledge Graph | Architecture Blueprint | Executable Code

### DOJO SECTION
py-24. Background: #0D0D0D (slightly lighter).
  - Orange pill: "⚔️ Practice Dojo"
  - H2: "Code ML From Scratch."
  - 3 problem preview cards side by side:
    Card: dark surface, border, rounded-xl
    Top-right: difficulty pill (Easy/Medium/Hard in correct color)
    Problem number (muted) + title (white bold)
    Topic tag chips below title
    "Solve →" link bottom-right in orange

### LEARN SECTION
py-24. Background: #0A0A0A.
  - Blue pill: "📚 Learning Paths"
  - H2: "Master the Theory."
  - 3 domain cards (blue accents):
    Each: surface, border, rounded-xl
    Icon box (blue/15) + domain name + topic count
    Thin progress bar at bottom (blue fill, dark track)

### FOOTER
Dark background. Border-top.
Left: logo + tagline
Right: 3 columns of links (Product / Company / Legal)
Bottom: copyright + Privacy · Terms (small muted text)

---

## PAGE 2: DOJO CATALOG  /dojo

Full-height. No sidebar rail. No AppShell.

### LEFT SIDEBAR (260px wide, dark bg, border-right)
Three stacked sections:

  POTD Card (Problem of the Day):
    bg: very dark orange tint (#1C0F00), border: orange/25
    "PROBLEM OF THE DAY" tiny uppercase label in orange
    Problem title
    Row: difficulty pill + countdown timer to midnight

  Progress Card:
    bg surface, border
    "Your Progress" label
    3 mini boxes in a row: Easy count | Medium count | Hard count
    Each box: large number in difficulty color

  Study Plans (3 items):
    Each: rounded card, left border in pillar color, plan name + problem count

  Bottom toggle: "Problems" | "Leaderboard" — two tab buttons

### MAIN AREA — PROBLEMS VIEW

  Topic filter chips row (scrollable horizontally):
    All | Loss Functions | Activation | NLP | Transformers | Attention | Linear Algebra | Optimization | CV | Metrics
    Active chip: solid orange bg, black text
    Inactive chip: surface-2 bg, muted text, 1px border

  Filter bar (below chips):
    Search box (left) + Status dropdown + Difficulty dropdown
    Right side: "128 problems" count in muted text

  Problems table:
    Header: STATUS | # TITLE | TOPICS | DIFFICULTY | ACCEPTANCE
    Each row (52px height):
      STATUS: 10px dot, green if solved
      # TITLE: index muted + title white bold
      TOPICS: 1-2 chip tags
      DIFFICULTY: colored text
      ACCEPTANCE: muted %
    
    Hover state: row bg lightens slightly
    Active/selected row: orange left border, very dark orange row tint, title turns orange

### LEADERBOARD VIEW (when tab is active)
Replaces table:
  "🏆 Leaderboard" title + Weekly / All Time toggle
  Table: Rank | User | Solved | XP | Streak
  Top 3 rows have gold/silver/bronze left accent

---

## PAGE 3: PROBLEM IDE  /dojo/[slug]

Full-height. Minimal chrome at top.

### THIN TOP BAR (52px)
  "← Problems" back link | divider | "#12 · Softmax Function" title | Medium pill
  Right side: "◀ 12/128 ▶" navigation | timer "4:23" in surface box

### SPLIT PANE (fills rest of viewport)

LEFT PANEL (45% width, border-right):
  Tab bar: Description | Theory | Solution 🔒 | Submissions | Notes
  Active tab: orange text + 2px orange bottom border

  Description content area (scrollable, padded):
    Problem title (20px bold)
    Topic tag chips
    Horizontal divider
    Body text (muted, line-height 1.7)
    "Constraints" box (surface, border, rounded, orange "CONSTRAINTS" label)
    Example code block (surface-2, monospace, 12px)

RIGHT PANEL (flex-1):
  Editor toolbar (44px):
    "Python 3.11 ▾" language badge (dark orange tint, orange border)
    Right: "↺ Reset" | "⎘ Copy" buttons (ghost style)
  
  Monaco editor area:
    Very dark bg (#0D0D0D)
    Orange glow on active line (very subtle)
    JetBrains Mono 13px
  
  Action bar (52px, border-top):
    Right-aligned: "▶ Run" ghost button + "Submit" orange button
  
  Console panel (220px, border-top, resizable):
    Drag handle (thin bar, hovering shows orange tint)
    "Testcase" | "Test Result" tabs
    
    Test Result states:
      Loading: 3 pulsing dots + "Running your code..."
      Pass: green banner "✓ Accepted" + output
      Fail: red banner "✗ Wrong Answer" + expected vs actual

---

## PAGE 4: PAPERS HUB  /papers

Full-width. No sidebar rail.

### HEADER BAR (72px)
  "Research Hub" title + subtitle in muted text
  Right side: search input + "All Papers ▾" dropdown

### BODY (side by side layout)

LEFT: UPLOAD ZONE (480px wide)
  Large dashed box (border-2 dashed violet/40, rounded-2xl):
    Center content:
      56px circle icon box (violet/12 bg)
      "Drop your PDF here" heading
      "or click to browse" subtext
      "Browse Files" button (solid violet)
      "PDF up to 50MB" tiny text

RIGHT: PAPERS GRID (fills remaining width)
  "Recent Papers" label
  3-column card grid:
    Each card: surface, border, rounded-xl, hover → violet border tint
      4px color bar at very top (domain color, full width)
      Title (12px semibold, 2-line clamp)
      Authors (muted 10px)
      2 topic tag chips
      Bottom row: status pill + "Open →" violet link

  Status pills:
    Ready: green/12 bg, green text
    Processing: amber/12 bg, amber text
    Queued: muted

---

## PAGE 5: PAPER WORKSPACE  /papers/[id]

Full-height. Minimal header.

### WORKSPACE HEADER (56px)
  "←" back link | doc icon (violet/15 box) | paper title (truncated) | authors + tags | status pill
  Right: Export | Share | ⚙ buttons

### TAB BAR (44px, border-bottom)
  🧠 Summary | 🕸 Knowledge Graph | 🏗 Blueprint | ⚡ Executable | 🎯 Challenges | 💬 AI Tutor
  Active: violet text + 2px violet underline

### KNOWLEDGE GRAPH TAB (most important to show)

Graph area (fills left ~60%):
  Very dark bg. Subtle dot-grid pattern (small dots, very low opacity).
  Circular nodes connected by lines:
    Center node: large, violet glow, paper name
    Concept nodes: blue
    Module nodes: emerald
    Operation nodes: orange
  Top-right: Zoom in / Fit / Zoom out controls (surface, border, stacked)

Inspector sidebar (right ~40%, border-left):
  Default: "Click any node to inspect" centered muted text
  
  Selected state:
    "SELECTED ENTITY" tiny uppercase muted label
    Entity name (18px bold)
    Type pill
    Definition paragraph (muted)
    Formula box (surface, blue border, monospace violet text):
      Formula text
    "Connected To" label + connected node name chips
    Practice CTA box (dark orange tint, orange border, rounded-xl):
      "Practice this concept →" orange text
      Problem name (muted tiny)
      "Solve →" orange button right-aligned
    "All Entities (N)" + breakdown list of entity types

---

## PAGE 6: LEARN DOMAINS  /learn

Full-width. No sidebar.

### PAGE HEADER
  Blue pill: "📚 Learning Paths"
  H2: "Master ML from First Principles"
  Subtext muted

### DOMAIN GRID (3 columns)
  Each domain card (surface, border, rounded-xl, p-6):
    Top row: icon box (40px, domain-color/12 bg) + column (domain name bold + "N topics · N papers" muted)
    Thin progress bar (1.5px height, dark track, domain-color fill)
    Bottom row: "X% complete" in domain color + CTA button
      0% → "Start →" ghost button
      >0% → "Continue →" solid domain-color button

  6 domains:
    Transformer Architecture (blue)
    Optimization & Training (orange)
    Computer Vision (emerald)
    NLP (violet)
    Generative Models (red/pink)
    Reinforcement Learning (yellow)

---

## PAGE 7: TOPIC CHAPTER  /learn/[domain]/[topic]

Full-height. 3-column layout.

### BREADCRUMB BAR (44px)
  Learn / Transformer Architecture / Self-Attention
  Each "/" in muted, active item in blue text

### 3 COLUMNS:

LEFT SIDEBAR (260px, border-right):
  Domain name + "N topics · X% complete" muted
  Thin progress bar (blue fill)
  Divider
  Topic list (each 36px row):
    8px status dot + topic name
    Done: muted text + green dot
    Active: blue text + blue dot + blue/10 bg + blue border
    Upcoming: muted text + empty dot (border only)

MAIN CONTENT (center, max-width 720px, scrollable, padded px-12 py-10):
  "Chapter 3 · Transformer Architecture" tiny blue uppercase label
  Topic title (28px bold)
  Intro paragraph (muted, line-height 1.7)
  Horizontal rule
  
  Content blocks (alternating):
    Section headings (18px bold)
    Body paragraphs (14px muted)
    Formula box (surface, blue border, rounded-xl, monospace blue text)
    Step cards grid (4 columns):
      Each: surface, border, rounded-xl
      Numbered circle (domain-color/15 bg)
      Step title + short description
    Code block (surface-2 bg, rounded-xl, syntax highlighted monospace)
  
  Bottom navigation:
    "← Previous Topic" ghost button | "Next Topic →" solid blue button

RIGHT PANEL (340px, border-left, scrollable):
  Practice CTA box (dark orange tint, orange border):
    "Ready to practice?"
    "Implement this from scratch in the Dojo"
    "⚔️  Solve in Dojo →" orange button (full width)
  
  "KEY CONCEPTS" tiny uppercase muted label
  Concept list: each row (surface, border, rounded), blue dot + concept name
  
  "RELATED PAPERS" tiny uppercase muted label
  Paper mini-cards: violet left accent bar + title + year

---

## PAGE 8: LABS  /labs

Full-width. No sidebar.

### PAGE HEADER
  Emerald pill: "🧪 Interactive Labs"
  H2: "Experiment. Visualize. Understand."
  Subtext muted

### LABS GRID (2 columns)
  Each lab card (surface, border, rounded-xl, p-6, hover → emerald border):
    Preview area (140px height, surface-2 bg, rounded-xl):
      Illustration or abstract animation preview
    Emerald tag chip (tiny, emerald/10 bg)
    Lab title (16px bold)
    Description (13px muted)
    Bottom: "N min" in muted + "Launch →" emerald text

  4 labs:
    Transformer Visualizer
    CNN Feature Maps
    ViT Attention Maps
    Diffusion Process

---

## PAGE 9: PRICING  /pricing

Full-width. Centered content.

### HEADER
  H1: "Simple, Honest Pricing" (centered, 40px bold)
  Subtext: "Start free. Upgrade when you're ready." (muted, centered)
  Monthly/Yearly toggle pill below

### 2-COLUMN PLAN CARDS (max-width 900px, centered)

FREE card (surface, border, rounded-2xl, p-8):
  "Free" label (muted uppercase)
  "$0 / month" (36px bold)
  Divider
  Feature list:
    ✓ 3 papers per month
    ✓ 20 submissions/day
    ✓ Knowledge Graph (basic)
    ✓ Learn paths
    ✗ Solution tab — muted
    ✗ AI Tutor — muted
    ✗ Blueprint + Executable — muted
  "Get Started Free" ghost button (full width)

PRO card (surface, ORANGE border, rounded-2xl, p-8, relative):
  Subtle orange glow behind card (blurred, low opacity)
  "Popular" badge top-right (solid orange, black text, rounded-full)
  "Pro" label (orange uppercase)
  "$19 / month" (36px bold)
  Divider
  Feature list (all ✓, white text):
    ✓ Unlimited papers
    ✓ Unlimited submissions
    ✓ Full Knowledge Graph + Blueprint
    ✓ AI Tutor (unlimited)
    ✓ Solution tab unlocked
    ✓ Executable code export
    ✓ Priority support
  "Upgrade to Pro →" solid orange button (full width)

---

## AUTH MODAL (overlay)
Appears over any page. Never a separate page.

Backdrop: black/60 blur overlay covering full screen.
Modal: max-width 440px, centered, surface bg, border, rounded-2xl, p-8.
X close button top-right.

Two tabs: "Sign In" | "Sign Up" — orange underline active.

Sign In:
  Email input + Password input (full width each, surface-2 bg, border, rounded-lg)
  "Forgot password?" small link (right-aligned, muted)
  "Sign In" solid orange button (full width)
  "or continue with" divider (lines + muted text)
  Google button (surface-2 bg, border, Google logo + "Continue with Google")

Sign Up:
  Username + Email + Password inputs
  "Create Account" orange button
  Same Google button

---

## PROFILE DROPDOWN (on avatar click)
220px panel below avatar, right-aligned. Surface bg, border, rounded-xl, shadow.

Header: avatar (40px) + username bold + email (muted tiny)
Divider
Menu items (36px each, hover: surface-2 bg):
  👤 Profile Stats
  🏆 Achievements
  📊 Submissions
  ⚙️  Settings
Divider
🚪 Sign Out (muted text)

Mini stats strip inside (3 boxes, surface-2 bg):
  Problems Solved | Current Streak | Total XP
  Numbers in orange/pillar colors
