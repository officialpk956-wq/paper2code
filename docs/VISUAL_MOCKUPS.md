# Paper2Code — Visual Mockups & Detailed Specifications

**Purpose:** Detailed visual specifications for each page type to guide Lovable implementation.

---

## MOCKUP 1: HOME PAGE (Landing)

### Layout Structure
```
Full-width, centered content, dark theme
```

### Hero Section (Above Fold)
**Background:** Animated aurora gradient (purple → cyan)
- Radial gradient circles at 20% and 80% positions
- Animation: 8s infinite loop, easing in and out

**Content (Centered Max-Width: 1000px):**

**Badge Pill (Top):**
- Subtle background: rgba(124, 58, 237, 0.1)
- Thin border: 1px rgba(124, 58, 237, 0.25)
- Text: "Now with 110 LeetCode problems"
- Pulsing dot indicator (leading)
- Font: 12px, semibold, letter-spacing +0.05em

**Main Title:**
- "Master AI Engineering"
  - Gradient text: linear-gradient(135deg, #7C3AED, #06B6D4)
  - Font: Plus Jakarta Sans, 56px bold, line-height 1.2
  - Margin-bottom: 12px
- "Through Problems, Papers & Practice"
  - Text: #E2E8F0 (white)
  - Font: 52px, regular weight
  - Line-height: 1.3

**Subtitle:**
- Text: "Learn deep learning and LLMs the hard way. Solve 110+ problems, read 50+ papers, explore architectures interactively, and follow guided roadmaps from beginner to researcher."
- Font: 18px, #94A3B8 (secondary gray)
- Max-width: 700px, centered
- Line-height: 1.6
- Margin-bottom: 32px

**Call-to-Action Buttons (Two-Column Stack on Mobile):**

1. **Primary Button:**
   - Text: "Start Solving Problems" + right arrow icon
   - Background: linear-gradient(90deg, #7C3AED, #06B6D4)
   - Text color: white
   - Padding: 16px 32px
   - Border-radius: 8px
   - Font: 14px, semibold
   - Hover: Slight opacity increase, subtle shadow glow
   - Icon: Lucide ArrowRight (4px margin-left)

2. **Secondary Button:**
   - Text: "Browse Roadmaps"
   - Background: transparent
   - Border: 1px solid #1E293B
   - Text color: #94A3B8
   - Padding: 16px 32px
   - Hover: Border color → #7C3AED, text color → #E2E8F0
   - Border-radius: 8px

**Statistics Section (Below Buttons):**
- Margin-top: 64px
- 3 columns (mobile: 1, tablet: 3)
- Gap: 32px

Each Stat:
```
┌─────────────┐
│ 110+        │  (32px, gradient text, bold)
│ Coding      │  (14px, #94A3B8, line-height 1.6)
│ Problems    │
└─────────────┘
```

---

### Features Section (Below Hero)
**Background:** #09090F (body background)
**Border-top:** 1px #1E293B

**Container:**
- Padding: 96px 32px
- Max-width: 1200px
- Centered

**Heading:**
- Text: "Everything You Need"
- Font: 32px, bold, Plus Jakarta Sans
- Text-align: center
- Margin-bottom: 48px

**Grid Layout:**
- 2 columns (tablet: 1 column, mobile: 1 column)
- Gap: 24px
- Max-width: 800px (centered)

**Feature Cards (4 total):**

Each Card:
```
┌──────────────────────────────────┐
│ 🎯 (Icon, 32px, #7C3AED)         │ 
│                                  │
│ 110+ Coding Problems             │ (16px, bold, #E2E8F0)
│                                  │
│ From linear algebra to LLM       │ (13px, #94A3B8)
│ inference optimization. Solve    │
│ problems that build real         │
│ understanding.                   │
│                                  │
└──────────────────────────────────┘
```

- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 24px
- Hover: Border → #7C3AED, background → #111827
- Icon container: Text-align left, margin-bottom 16px
- Icon hover: scale 1.1 (300ms transition)
- Title: 14px, bold, #E2E8F0
- Description: 13px, #94A3B8, line-height 1.6

---

### Learning Tracks Section (Below Features)
**Background:** #0D0D14
**Border-top:** 1px #1E293B

**Container:**
- Padding: 96px 32px
- Max-width: 1200px
- Centered

**Heading:**
- Text: "Choose Your Path"
- Font: 32px, bold, Plus Jakarta Sans
- Text-align: center
- Margin-bottom: 16px

**Subheading:**
- Text: "From foundational data science to cutting-edge AI research..."
- Font: 14px, #64748B (tertiary)
- Text-align: center
- Max-width: 700px
- Centered
- Margin-bottom: 48px

**Track Cards Grid:**
- 3 columns (tablet: 2, mobile: 1)
- Gap: 16px
- Max-width: 1000px

**Each Track Card (6 total):**

```
┌──────────────────────────────┐
│ 📊 (Emoji, 32px)            │
│ Data Scientist              │ (16px, bold, #E2E8F0)
│ 8 weeks                     │ (13px, #94A3B8)
│                             │
│ [Get started →]             │ (Fade in on hover)
│ (Accent color, 12px)       │
└──────────────────────────────┘
```

- Background: #111827
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 24px
- Overflow: hidden (for gradient overlay)
- Gradient overlay: linear-gradient(135deg, color1, color2) at opacity 0.1
- Hover: Gradient overlay opacity → 0.2, border → #7C3AED
- Transition: 300ms ease

Each track has unique gradient:
1. Data Scientist: #10B981 to #34D399
2. ML Engineer: #8B5CF6 to #A78BFA
3. Deep Learning: #F59E0B to #FCD34D
4. LLM Engineer: #06B6D4 to #67E8F9
5. AI Researcher: #EC4899 to #F9A8D4
6. AI Architect: #EF4444 to #FCA5A5

---

### Final CTA Section
**Background:** #09090F
**Border-top:** 1px #1E293B

**Container:**
- Padding: 96px 32px
- Max-width: 800px
- Centered

**Heading:**
- Text: "Ready to Level Up?"
- Font: 32px, bold
- Text-align: center
- Margin-bottom: 16px

**Description:**
- Font: 14px, #94A3B8
- Text-align: center
- Margin-bottom: 32px
- Line-height: 1.6

**Buttons:**
- Two buttons side-by-side (mobile: stacked)
- Same button styles as hero

---

## MOCKUP 2: DASHBOARD (Learning Command Center)

### Layout Structure
```
┌──────────────────────────────────────────────────┐
│           Global Navigation Bar (40px)           │
├──────────┬──────────────────────┬────────────────┤
│          │                      │                │
│   280px  │     Center Content   │     280px      │
│  LEFT    │      (~600px auto)   │    RIGHT       │
│ SIDEBAR  │                      │    PANEL       │
│          │                      │                │
└──────────┴──────────────────────┴────────────────┘
```

**Desktop:** Full three-column (> 1200px)
**Tablet:** Stack to 2 columns, right panel below
**Mobile:** Single column, left sidebar hidden

---

### LEFT PANEL (280px, Sticky)
**Background:** #0D0D14

**Profile Section (Top 120px):**
```
┌─────────────────────────────────┐
│ ┌─ ─ ─ ─ ─ ┐                   │
│ │  Avatar  │  Alex Chen        │ (14px, bold)
│ │(60x60)   │  Level 7          │ (12px, #94A3B8)
│ └─ ─ ─ ─ ─ ┘                   │
│                                │
│ Current Track:                 │ (11px, caps, muted)
│ Transformer Architecture       │ (13px, bold)
│                                │
│ ████████░░░░░ 35%             │ (Progress bar)
│                                │
└─────────────────────────────────┘
```

- Padding: 20px
- Border-bottom: 1px #1E293B
- Margin-bottom: 20px

**Quick Actions (4 Buttons):**
```
┌─────────────────────────────┐
│ + Solve Problem             │
├─────────────────────────────┤
│ 📚 Read Paper               │
├─────────────────────────────┤
│ 🏗 Explore Architecture     │
├─────────────────────────────┤
│ ⚙ System Design            │
└─────────────────────────────┘
```

- Each button: full-width, 44px height, text-align left
- Background: transparent, hover → #111827
- Border-bottom: 1px #1E293B
- Margin-bottom: 8px
- Font: 13px, #E2E8F0
- Padding: 0 16px
- Icon: 16px, margin-right 12px

**Recent Bookmarks (5 Items):**
- Title: "Recent Bookmarks" (11px, caps, muted)
- Margin-top: 32px
- Margin-bottom: 12px

```
┌──────────────────────────────┐
│ 🔖 Attention Is All You Need │ (13px)
├──────────────────────────────┤
│ 📊 Data Scientist Track      │ (13px)
├──────────────────────────────┤
│ ⚡ Backprop Problems        │ (13px)
├──────────────────────────────┤
│ 🧠 Multi-Head Attention     │ (13px)
├──────────────────────────────┤
│ 🔗 System Design            │ (13px)
└──────────────────────────────┘
```

- Each item: 40px height, clickable, hover → background #111827
- Border-bottom: 1px #1E293B
- Padding: 0 16px
- Font: 12px, #94A3B8

**Learning Milestones:**
- Title: "Your Progress" (11px, caps, muted)
- Margin-top: 32px
- Margin-bottom: 16px

```
Milestone 1: Fundamentals    ✅ 100%
Milestone 2: Attention       ⚡ 35%
Milestone 3: Models          ⏱ 0%
Milestone 4: Training        ⏱ 0%
Milestone 5: Production      ⏱ 0%
```

- Each milestone: 2 lines, 12px font
- Status icon: 16px (green checkmark, blue lightning, gray clock)
- Percentage: right-aligned, #7C3AED for active

---

### CENTER PANEL (Auto-width, 600-800px)

**Header (Sticky):**
```
┌────────────────────────────────────────┐
│ 📍 Dashboard > Home                    │ (Breadcrumb, 11px)
│                                        │
│ Welcome back, Alex! 👋                │ (24px, bold)
│ You're 35% through the Transformer    │ (13px, #94A3B8)
│ Architecture track                    │
└────────────────────────────────────────┘
```

- Padding: 24px
- Background: #0D0D14, sticky on scroll
- Border-bottom: 1px #1E293B

**Tabs (Below Header):**
```
┌──────────────────────────────────┐
│ Overview  | Roadmap  | Analytics │
│━━━━━━━━━━━━━━━━━━━━━━           │ (Accent underline)
│                                  │
└──────────────────────────────────┘
```

- Tab styling: 13px, semibold, #94A3B8
- Active: #7C3AED, bottom border 2px
- Hover: cursor pointer, text color → #E2E8F0
- Transition: 150ms ease
- Padding: 0 20px
- Height: 48px
- Border-bottom: 1px #1E293B (whole tab bar)

**TAB 1: OVERVIEW (Default)**

**Metric Cards (4 in Grid, 2x2):**
```
┌────────────────────────┐ ┌────────────────────────┐
│ 🎯 Problems Solved     │ │ 📚 Papers Read         │
│ 47                     │ │ 12                     │
│ +3 this week           │ │ +1 this week           │
└────────────────────────┘ └────────────────────────┘

┌────────────────────────┐ ┌────────────────────────┐
│ ⏱ Hours Spent         │ │ 🔥 Streak              │
│ 24.5                   │ │ 7 days                 │
│ +8 hours this week     │ │ +1 day                 │
└────────────────────────┘ └────────────────────────┘
```

Each card:
- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 24px
- Icon: 24px, #7C3AED (top-right)
- Title: 12px, #94A3B8, margin-bottom 8px
- Number: 32px, bold, gradient text
- Subtitle: 11px, #64748B

**Learning Velocity Chart:**
- Title: "Your Learning Velocity" (14px, bold)
- Margin: 32px 0 16px 0
- Chart type: Line chart, last 7 days
- Y-axis: 0 to 15 problems
- X-axis: Days (Mon-Sun)
- Line color: #7C3AED
- Fill: gradient, 0.1 opacity
- Hover tooltip: Day + count + time
- Height: 240px
- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 20px

```
15 |     ╱╲
14 |    ╱  ╲    ╱╲
13 |   ╱    ╲  ╱  ╲
12 |  ╱      ╲╱    ╲__
   |_____________________
   Mon Tue Wed Thu Fri
```

**Recent Activity Timeline:**
- Title: "Recent Activity" (14px, bold)
- Margin: 32px 0 16px 0
- Shows 10 most recent activities

```
[Icon] Activity Description          [Time]
       [metadata in smaller font]

[Icon] Solved "Matrix Multiplication"        Today 2:45pm
       Easy · Data Science Track · +10 XP

[Icon] Bookmarked "Attention Is All You Need" Yesterday 9:20am
       Vaswani et al. · +5 XP

[Icon] Completed Roadmap: Fundamentals       Jun 14, 10:00am
       Data Scientist Track · Milestone 1 · +50 XP
```

Each activity:
- Padding: 16px
- Border-left: 4px
- Border color: varies by type (problem: #7C3AED, paper: #06B6D4, milestone: #10B981)
- Background: transparent, hover → #111827
- Icon: 20px, left
- Title: 13px, bold, #E2E8F0
- Metadata: 11px, #94A3B8
- Time: right-aligned, 11px, #64748B
- Border-bottom: 1px #1E293B (between items)

---

### RIGHT PANEL (280px, Sticky)
**Background:** #0D0D14

**Up Next Section:**
```
┌────────────────────────────────┐
│ 📍 Up Next                     │ (12px, caps, muted)
│                                │
│ Matrix Transpose Operation     │ (14px, bold)
│ Medium · Linear Algebra        │ (12px, #94A3B8)
│ Estimated: 15 min             │ (11px, #64748B)
│                                │
│ [Start Problem →]              │ (Button, gradient)
└────────────────────────────────┘
```

- Background: #111827
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 20px
- Margin-bottom: 24px

**Suggested Paper:**
```
┌────────────────────────────────┐
│ 📄 Suggested Reading           │ (12px, caps, muted)
│                                │
│ Attention Is All You Need      │ (14px, bold)
│ Vaswani et al. · 2017         │ (11px, #94A3B8)
│ 84.2K citations               │ (11px, #7C3AED)
│                                │
│ [Read Paper →]                 │ (Button)
└────────────────────────────────┘
```

Same styling as "Up Next"

**Current Metrics:**
```
┌────────────────────────────────┐
│ 📊 Learning Metrics            │
│                                │
│ Accuracy: 92%                  │ (Bar: 92% filled, green)
│ Speed: 85%                     │ (Bar: 85% filled, orange)
│ Consistency: 78%               │ (Bar: 78% filled, cyan)
│                                │
│ Time to Milestone: 5 days      │ (12px, bold)
└────────────────────────────────┘
```

- Background: #111827
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 20px
- Margin-bottom: 24px
- Each metric: 12px label, 32px bar below

**Achievements:**
```
┌────────────────────────────────┐
│ 🏆 Achievements (2/8)          │
│                                │
│ ┌──┐ ┌──┐                      │
│ │⚡│ │🔥│  [locked]   [locked]   │
│ └──┘ └──┘                      │
│ Streak  Solver                │ (Unlocked, 12px)
│                                │
│ [locked] [locked] [locked]     │
│ [locked] [locked] [locked]     │
└────────────────────────────────┘
```

- Background: #111827
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 20px
- Grid: 4 columns of badges
- Each badge: 60px square, centered, border-radius 12px
- Unlocked: Icon 28px, background subtle color, text below
- Locked: Grayed out, lock icon
- Hover: Show tooltip with unlock condition

---

## MOCKUP 3: ARCHITECTURE PAGE (Transformer)

### Layout Structure
Same as Dashboard three-column, with architectural content

---

### LEFT PANEL (280px)
**Architecture Header:**
```
┌────────────────────────────────┐
│ Transformer                    │ (18px, bold, #E2E8F0)
│ 2017 · Vaswani et al.         │ (12px, #94A3B8)
│                                │
│ Difficulty: Hard              │ (Badge, red)
└────────────────────────────────┘
```

- Padding: 16px
- Border-bottom: 1px #1E293B
- Margin-bottom: 16px

**Search Layers:**
- Input: 40px height, "Search layers..." placeholder
- Icon: Magnifying glass (16px)
- Border: 1px #1E293B
- Background: #111827
- Focus: Border → #7C3AED
- Margin-bottom: 16px

**Layer List (Expandable):**
```
├─ Input Embedding
│  ├─ Token Embedding      ← Click to select
│  └─ Positional Encoding  ← Highlights connections
├─ Multi-Head Attention (Expandable)
│  ├─ Query Linear
│  ├─ Key Linear
│  ├─ Value Linear
│  └─ Output Linear
├─ Feed Forward Network
│  ├─ Dense 1
│  └─ Dense 2
└─ Output Layer
```

- Font: 12px
- Height: 32px per item
- Padding: 8px 12px
- Border-left: 3px transparent
- Hover: Background #111827, border-left #7C3AED
- Active: Border-left #7C3AED, text → #7C3AED, background #111827
- Expandable: Click arrow to expand/collapse
- Tree indent: 16px per level

**Related Architectures:**
- Title: "Related Architectures" (11px, caps, muted)
- Margin-top: 32px

```
├─ Vision Transformer (ViT)
├─ BERT
├─ GPT
└─ Diffusion Transformer
```

- Same styling as layer list
- All links (no tree structure)

---

### CENTER PANEL

**Sticky Header:**
```
┌────────────────────────────────────┐
│ Transformer: The Architecture      │ (22px, bold)
│ That Changed Everything            │
│                                    │
│ [★ Bookmark]  [Share]  [Favorite]  │ (Icons + text)
└────────────────────────────────────┘
```

- Padding: 24px
- Background: Gradient (135deg, #7C3AED to #06B6D4) at 0.05 opacity, full width
- Sticky on scroll
- Border-bottom: 1px #1E293B

**Interactive Diagram (SVG):**
```
Large visual showing Transformer architecture:

Input Tokens
     ↓
[Embedding Layer]
     ↓
[Position Encoding]
     ↓
[Encoder Block] ⟷ [Decoder Block]  ← Hover highlights connections
     ↓                    ↓
[Multi-Head]        [Multi-Head]
Attention           Attention
     ↓                    ↓
[Feed Forward]      [Feed Forward]
     ↓                    ↓
[Output Probabilities]
```

- Height: 400px
- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 32px
- Boxes: Color-coded (#7C3AED for encoder, #06B6D4 for decoder)
- Connections: Animated arrows, glow on hover
- Click box: Selects layer, updates right panel
- Hover: Tooltips with layer details

**Tab System:**
```
┌──────────────────────────────────────┐
│ Description | Theory | Components   │
│             | Comparison | Timeline │
│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │ (Underline)
└──────────────────────────────────────┘
```

**TAB 1: DESCRIPTION**

Content:
```
# Transformer: Attention Is All You Need

## Overview
The Transformer architecture represents a fundamental shift in how we 
approach sequence modeling. Instead of relying on recurrence or convolution, 
it uses attention mechanisms to process sequences in parallel.

## Key Innovation
Self-attention allows each position in the input to attend to all other 
positions, enabling the model to capture long-range dependencies 
efficiently.

[Large diagram showing components]

## Architecture Components
- Multi-Head Attention
- Feed Forward Networks
- Position Embeddings
- Layer Normalization

[Links to related sections]
```

- Font: 13px Inter, line-height 1.8
- Headings: Plus Jakarta Sans, bold
- h1: 22px, margin-bottom 16px
- h2: 16px, margin-bottom 12px
- p: 13px, margin-bottom 16px
- Links: #7C3AED, underline on hover
- Code blocks: Monospace, dark background, syntax highlighting

**TAB 2: THEORY**

```
# Mathematical Foundations

## Attention Mechanism

The core of the Transformer is the scaled dot-product attention:

Attention(Q, K, V) = softmax(QK^T / √(d_k))V

Where:
- Q: Query matrix
- K: Key matrix  
- V: Value matrix
- d_k: Dimension of keys

[SVG visualization of attention computation]

## Multi-Head Attention

Instead of a single attention function, the model uses h different linear 
projections of the queries, keys, and values:

MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O

Where each head learns different representations.

## Common Mistakes

❌ Thinking attention is matrix multiplication
- Attention requires normalization via softmax

❌ Using full sequence length for attention
- Can use masking to prevent attending to future positions

❌ Forgetting positional encoding
- Transformer needs position information to understand order
```

- KaTeX rendering for equations
- Equations: Centered, larger font, white text
- Inline code: Monospace, gray background
- Bullet lists: 12px, #E2E8F0, margin-left 20px
- Common Mistakes: Red styling, ❌ emoji
- SVG diagrams: Large, colorful, interactive

---

### RIGHT PANEL (280px)

**Selected Layer Details (If layer clicked):**
```
┌────────────────────────────────┐
│ 🎯 Multi-Head Attention        │ (14px, bold)
│ encoder · layer 1              │ (12px, muted)
│                                │
│ Input Shapes:                  │
│ Q: [batch, seq_len, d_model]  │ (11px, mono)
│ K: [batch, seq_len, d_model]  │
│ V: [batch, seq_len, d_model]  │
│                                │
│ Output Shape:                  │
│ [batch, seq_len, d_model]     │
└────────────────────────────────┘
```

- Background: #111827
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 16px
- Margin-bottom: 16px

**Mathematical Formula:**
```
┌────────────────────────────────┐
│ Formula                        │
│                                │
│ MultiHead(Q,K,V)              │
│ = Concat(h₁,...,hₕ)W^O       │
│                                │
│ Where:                         │
│ h_i = Attention(QW_i^Q,       │
│                KW_i^K, VW_i^V) │
└────────────────────────────────┘
```

- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 16px
- Margin-bottom: 16px
- Font: 11px monospace
- White text on dark background
- Equations centered

**PyTorch Implementation:**
```
┌────────────────────────────────┐
│ PyTorch Code                   │
│                                │
│ class MultiHeadAttention:      │
│   def __init__(self, d_model): │
│     self.num_heads = 8         │
│     self.d_k = d_model // 8    │
│                                │
│   def forward(self, Q, K, V):  │
│     # Reshape for multi-head   │
│     Q = self.to_multi_head(Q)  │
│     # Compute attention        │
│     scores = Q @ K.T / √d_k   │
│     attn = softmax(scores)     │
│     return attn @ V            │
└────────────────────────────────┘
```

- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 8px
- Padding: 12px
- Font: 10px monospace
- Syntax highlighting: keywords (#A78BFA), strings (#10B981), etc.
- Line numbers: #64748B, 8px
- "Load into Editor" button below (14px, secondary button)

**Related Problems:**
```
┌────────────────────────────────┐
│ 📌 Related Problems            │
│                                │
│ 1. Implement Attention         │
│    Medium · 12 min             │ (Link)
│                                │
│ 2. Multi-Head Attention        │
│    Hard · 25 min               │ (Link)
│                                │
│ 3. Masked Attention            │
│    Hard · 30 min               │ (Link)
└────────────────────────────────┘
```

- Background: #111827
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 16px
- Problem item: 36px height, clickable
- Difficulty: Badge (green, orange, red)
- Time estimate: 11px, muted

---

## MOCKUP 4: PAPER PAGE (Attention Is All You Need)

### Similar structure to Architecture page but with different tab content

---

### CENTER PANEL TABS

**TAB 1: SUMMARY**

```
# Executive Summary: Attention Is All You Need

## Key Takeaway
The Transformer architecture replaces recurrence with pure attention, 
enabling parallel processing of sequences and achieving state-of-the-art 
results across multiple domains.

## Main Contributions
- ✅ Pure attention-based architecture (no recurrence)
- ✅ Parallel sequence processing
- ✅ State-of-the-art translation quality
- ✅ Foundation for modern LLMs (BERT, GPT, T5)

## Technical Innovation
Position-wise feed-forward networks + multi-head attention + positional 
encoding = transformer block (repeatable unit)

## Impact
This paper fundamentally changed how we approach NLP and serves as the 
foundation for all modern language models.

[Before-After comparison diagram]

## Prerequisite Knowledge
- ✓ Recurrent Neural Networks
- ✓ Sequence-to-Sequence Models  
- ✓ Basic Linear Algebra
- ✓ Attention Mechanisms
```

**TAB 2: TIMELINE**

Shows evolution of ideas leading to transformer:

```
2014: Seq2Seq with Attention
      Sutskever et al.
      ↓
2015: Attention Is All You Need (Vaswani et al.)
      [HIGHLIGHTED - THIS PAPER]
      ↓
2018: BERT: Pre-training with Masked LM
      Devlin et al.
      ↓
2019: GPT-2 / Megatron
      OpenAI / NVIDIA
```

Each item clickable, showing details in hoverable popover

**TAB 3: EQUATIONS**

Shows all key equations with explanations:

```
# Key Equations

## 1: Scaled Dot-Product Attention

Attention(Q, K, V) = softmax(QK^T / √(d_k))V

**Explanation:**
This equation is the core of the Transformer. It computes attention 
weights by computing the dot product between query and key, scaling 
by √(d_k) to prevent the gradients from vanishing, then applying 
softmax to get attention weights. Finally, multiply by the values.

**Variables:**
- Q: Query matrix (batch × seq_len × d_k)
- K: Key matrix (batch × seq_len × d_k)
- V: Value matrix (batch × seq_len × d_v)
- d_k: Dimension of keys (typically d_model / h)

**Why scaling?** Without scaling by √(d_k), the dot products can become 
very large, making softmax concentration in just a few positions.

---

## 2: Multi-Head Attention

MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O

Where: head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

**Explanation:**
Instead of using a single attention function, multi-head attention 
projects the queries, keys, and values h times with different linear 
transformations, runs attention in parallel, concatenates, and projects 
the output again.

**Benefit:** Allows the model to attend to multiple representation 
subspaces.
```

All equations large, readable, with supporting text

---

### RIGHT PANEL

**Quick Facts:**
```
┌────────────────────────────────┐
│ 📊 Publication Details         │
│                                │
│ Published: Jun 2017            │
│ Venue: NeurIPS 2017           │
│ DOI: 10.5555/3295222.3295349  │
│ Citations: 84.2K              │
│ h-index Impact: 9/10          │ (Purple bar)
│                                │
│ Authors: 8                     │
│ Affiliation: Google            │
└────────────────────────────────┘
```

**Prerequisite Knowledge:**
```
┌────────────────────────────────┐
│ 📚 Prerequisite Knowledge      │
│                                │
│ [Seq2Seq Models] (Link)       │
│ [Attention Mechanisms] (Link)  │
│ [Neural Networks] (Link)       │
│ [Linear Algebra] (Link)        │
└────────────────────────────────┘
```

**Key Innovation Highlight:**
```
┌────────────────────────────────┐
│ 💡 Key Innovation              │
│                                │
│ Removing recurrence and using  │
│ pure attention for sequence    │
│ processing enables:            │
│ • Parallel computation         │
│ • Long-range dependencies      │
│ • Transfer learning (BERT,GPT) │
│                                │
│ This foundation enables modern │
│ LLMs and foundation models.    │
└────────────────────────────────┘
```

**Related Problems:**
```
3 problems linked:
- Implement Attention (Medium)
- Transformer Block (Hard)
- Positional Encoding (Medium)
```

**Follow-Up Papers:**
```
Top 3:
- BERT (Devlin et al., 2018) - 15K citations
- GPT-2 (Radford et al., 2019) - 10K citations
- T5 (Raffel et al., 2019) - 8K citations
```

---

## MOCKUP 5: PAPER-TO-CODE (3-Column Learning Environment)

### Layout: Three-Column Code Learning

```
┌──────────────────────────────────────────────────┐
│           Global Navigation Bar (40px)           │
├──────────┬──────────────────────┬────────────────┤
│          │                      │                │
│  Theory  │   Code Editor &      │  Tensor Shapes │
│  &       │   Run Results        │  & Inspector   │
│  Concept │                      │                │
│          │                      │                │
│ 320px    │  (Auto)              │   280px        │
└──────────┴──────────────────────┴────────────────┘
```

**Desktop:** Full three-column (> 1440px)
**Tablet:** Stack to 2 columns
**Mobile:** Single column stack

---

### LEFT PANEL (320px) - THEORY

```
# Backpropagation: Learning Through Error

## Concept
Backpropagation is the algorithm that allows neural networks to learn 
from data by computing gradients of the loss with respect to the network 
parameters.

## Key Insight
By computing gradients in reverse (from output to input), we can efficiently 
calculate how each parameter should change to reduce the error.

## Key Equations

∂L/∂w = ∂L/∂z · ∂z/∂w     [Chain Rule]

Where:
- L: Loss function
- z: Pre-activation (w·x + b)  
- w: Weight parameter

## Mathematical Intuition

The chain rule allows us to decompose complex gradients:

[Animated SVG showing chain rule propagation backwards]

∂L/∂w₁ = ∂L/∂ŷ · ∂ŷ/∂h · ∂h/∂w₁

## Common Mistakes

❌ Computing gradients forward instead of backward
- Backprop must flow from loss to parameters

❌ Not scaling gradients by learning rate
- Small gradients die out (vanishing gradients)

❌ Forgetting to zero gradients between batches
- Gradients accumulate, causing divergence
```

Content:
- Font: 12px Inter, line-height 1.8
- Headings: Plus Jakarta Sans, bold
- Equations: KaTeX, large readable
- Numbered lists: 12px, margin-left 20px
- Code blocks: Monospace, gray background
- SVG diagrams: Centered, interactive on hover

**Scroll area:** Max-height: 100vh - 80px, with scrollbar

---

### CENTER PANEL - CODE EDITOR & RESULTS

**Editor Header:**
```
┌────────────────────────────────┐
│ Implement backpropagation      │
│                                │
│ def backward_pass(...):        │
│     # Your code here           │
│                                │
│ [Reset] [Run Tests] [Submit]   │
└────────────────────────────────┘
```

Header styling:
- Padding: 16px
- Background: Gradient (purple → cyan) at 0.05 opacity
- Border-bottom: 1px #1E293B
- Function name: 13px, monospace

**Monaco Editor:**
- Height: 50% of center panel
- Language: Python
- Theme: Dark (VS Code dark)
- Font: JetBrains Mono, 12px
- Line numbers: Enabled
- Word wrap: Enabled
- Syntax highlighting: Full support

```python
def backward_pass(forward_result, loss_grad):
    """
    Compute gradients for parameters
    
    Args:
        forward_result: Output from forward pass
        loss_grad: Gradient of loss w.r.t output
    
    Returns:
        dict: Gradients for each parameter
    """
    grads = {}
    
    # Starting gradient
    dz = loss_grad  # Gradient w.r.t output activation
    
    # Backprop through output layer
    dW_out = dz.T @ forward_result['h']  # [out_dim, hidden_dim]
    db_out = dz.sum(axis=0, keepdims=True)  # [1, out_dim]
    
    # Backprop through hidden layer
    dh = dz @ forward_result['W_out'].T  # [batch, hidden_dim]
    dh_relu = dh * (forward_result['z'] > 0)  # ReLU gradient
    
    dW_hidden = dh_relu.T @ forward_result['X']  # [hidden_dim, in_dim]
    db_hidden = dh_relu.sum(axis=0, keepdims=True)  # [1, hidden_dim]
    
    grads['W_out'] = dW_out
    grads['b_out'] = db_out
    grads['W_hidden'] = dW_hidden
    grads['b_hidden'] = db_hidden
    
    return grads
```

Buttons:
- Reset: Secondary button (bordered), clears editor, restores template
- Run Tests: Primary button (gradient), executes code, shows results below
- Submit: Primary button (gradient), submits for grading

Button styling:
- 40px height, 14px font, semibold
- Padding: 8px 16px
- Margin: 8px 4px
- Hover: Opacity change, shadow glow

**Test Results Section:**
```
┌────────────────────────────────┐
│ ✅ Tests Passed: 4/4           │
│                                │
│ ✓ test_gradient_shape          │ (12px, green)
│   Expected: (10, 5), Got: (10,5) 
│   Time: 0.23ms                 │ (11px, muted)
│                                │
│ ✓ test_gradient_values         │
│   Relative error < 1e-5        │
│   Time: 0.45ms                 │
│                                │
│ ✓ test_numerical_gradient      │
│   Gradients within tolerance   │
│   Time: 2.34ms                 │
│                                │
│ ✓ test_backprop_chain_rule     │
│   Chain rule correctly applied │
│   Time: 0.12ms                 │
│                                │
│ Total Time: 3.14ms            │
└────────────────────────────────┘
```

Results styling:
- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 8px
- Padding: 16px
- Height: 50% of center panel, scrollable
- Header: 14px, bold, green checkmark
- Each test: 12px, green checkmark, test name bold
- Details: 11px, #94A3B8
- Time: right-aligned, 11px

Passes: Green checkmark, border-left green
Failures: Red X, border-left red

---

### RIGHT PANEL (280px) - TENSOR SHAPES & INSPECTOR

**Tensor Flow Diagram:**
```
┌────────────────────────────────┐
│ Forward → Backward Gradient    │ (12px, caps, muted)
│ Flow                           │
│                                │
│ Input (batch=4, features=5)   │ (12px, mono)
│      ↓                         │
│ [Linear] → (4, 10)            │
│      ↓                         │
│ [ReLU]                        │
│      ↓                         │
│ [Linear] → (4, 3)             │ (Output shape)
│      ↓                         │
│ Loss (scalar)                  │
│                                │
│ ← ← ← ← BACKWARD ← ← ← ←      │ (Annotated)
│                                │
│ dL/dW_out: (3, 10)            │
│ dL/dW_hidden: (10, 5)         │
│ dL/dX: (4, 5)                 │
│                                │
└────────────────────────────────┘
```

Styling:
- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 16px
- Monospace font: 11px
- Arrows: Accent color
- Shapes: Colored badges (green for input, purple for hidden, cyan for output)

**Variable Inspector Table:**
```
┌────────────────────────────────┐
│ 🔍 Variables                   │ (12px, caps, muted)
│                                │
│ Name        │ Shape   │ Dtype  │ (Header row)
│─────────────┼─────────┼────────┤
│ X           │ 4×5     │ float32│ (12px, monospace)
│ W_hidden    │ 5×10    │ float32│
│ b_hidden    │ 1×10    │ float32│
│ z_hidden    │ 4×10    │ float32│
│ h           │ 4×10    │ float32│ (Highlighted)
│ W_out       │ 10×3    │ float32│
│ b_out       │ 1×3     │ float32│
│ y_pred      │ 4×3     │ float32│
│ loss        │ scalar  │ float32│
│ dL/dW_out   │ 3×10    │ float32│
│ dL/dW_hidden│ 10×5    │ float32│
└────────────────────────────────┘
```

Styling:
- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 8px
- Padding: 12px
- Header row: Bold, muted color, border-bottom
- Cell height: 28px
- Monospace font: 11px
- Active/highlighted row: Background #111827
- Hover: Show details popup

**Memory Usage Breakdown:**
```
┌────────────────────────────────┐
│ 💾 Memory Usage                │
│                                │
│ Forward Tensors  ███░░░░ 45%  │ (Bar chart)
│ Weight Matrices  █████░░░ 52%  │
│ Gradients        ██░░░░░░  3%  │
│                                │
│ Total: 2.3 MB                  │ (14px, bold)
└────────────────────────────────┘
```

Styling:
- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 8px
- Padding: 16px
- Each row: 28px height
- Label: 12px, #E2E8F0
- Bar: Gradient color (purple-ish)
- Percentage: 12px, right-aligned

---

## MOCKUP 6: TENSOR TRACE (Premium Visual)

### Full-Width Single Column

**Navigation & Controls (Top 120px):**

```
┌────────────────────────────────────────────────┐
│ Step 5 / 12    [← Previous] [Play] [Next →]   │ (Center aligned)
│                                                │
│ Speed: [0.5x] [1x] [2x]   [Export]           │
│                                                │
│ ██████████░░░░░░░░░░░░░░  42% (Progress bar)  │
│ Forward Pass → ReLU Activation                 │ (Description)
└────────────────────────────────────────────────┘
```

Styling:
- Height: 120px
- Padding: 16px 32px
- Background: #0D0D14
- Border-bottom: 1px #1E293B
- Sticky on scroll

Navigation:
- Previous/Next buttons: 40px height, icon + text
- Play button: Circle (44px), primary color, pause when playing
- Speed buttons: 32px height, toggle group
- Progress bar: 100% width, 8px height, accent color

**Main Visualization Area (Large Center):**

```
Large animated tensor shape visualization:

Input (Batch of Images):
████████████████  8
████████████████  8
████████████████  8
████████████████  8  [Height=28, Width=28, Channels=3]

            ↓ ReLU Activation

Output (After ReLU):
████████████████  8
████████████████  8
████████████████  8
████████████████  8

[Dimensions highlighted with colors]
H (red)  W (green)  C (blue)
```

Visualization:
- Height: 60% of viewport (after nav)
- Background: Large canvas/SVG area (#0D0D14)
- Tensor representation: Colored 3D-ish boxes with dimension labels
- Animation: Smooth morph transitions (500ms) when shape changes
- Glow effect: Accent color glow around active tensor
- Hover: Show detailed information popup

**Details Panel (Below Visualization):**

```
┌────────────────────────────────────────────────┐
│ Tensor Information                             │
│                                                │
│ Name: x_relu        Dtype: float32             │ (12px)
│ Shape: [8, 28, 28, 3]  Device: GPU:0          │
│ Requires Grad: True    Storage: 18.8 KB       │ (12px, muted)
│                                                │
│ Operation: relu(x) = max(0, x)                │ (13px)
│ Time: 0.23 ms  Flops: 188.8 K                 │ (12px)
│                                                │
│ Gradient Info:                                 │
│ Input Grad: [8, 28, 28, 3]  [Download]       │
│ Weight Grad: [3, 3, 3, 64]  [Download]       │
└────────────────────────────────────────────────┘
```

Styling:
- Padding: 24px 32px
- Background: #111827
- Border-top: 1px #1E293B
- Grid layout: 2 columns on desktop, 1 on mobile
- Font: 12px monospace for shapes, 13px for operation

**Keyboard Shortcuts Overlay (If / pressed):**

```
Keyboard Shortcuts

Space        Play / Pause animation
→            Next step
←            Previous step
/            Toggle slow-motion (0.5x)
G            Toggle gradient visualization
M            Toggle memory breakdown
?            Show this help
```

Overlay:
- Position: Center modal
- Background: #0D0D14 with 0.95 opacity
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 32px
- Close: Escape key

---

## MOCKUP 7: SYSTEM DESIGN (Interactive Board)

### Three-Column Layout

---

### LEFT PANEL (280px) - COMPONENT LIBRARY

```
┌────────────────────────────────┐
│ 🔍 Search Components           │ (Input field)
│                                │
│ Category: [All ▼]             │ (Dropdown)
│                                │
│ [Zoom: - • +]                 │ (Controls)
│                                │
│ Compute                        │ (Category)
│ ├─ API Gateway   (Drag)       │
│ ├─ Load Balancer (Drag)       │ (12px, clickable)
│ ├─ Cache        (Drag)        │
│ └─ Web Server   (Drag)        │
│                                │
│ Storage                        │ (Category)
│ ├─ PostgreSQL   (Drag)        │
│ ├─ Redis        (Drag)        │
│ ├─ S3 Bucket    (Drag)        │
│ └─ MongoDB      (Drag)        │
│                                │
│ Messaging                      │ (Category)
│ ├─ Message Queue (Drag)       │
│ └─ Pub/Sub       (Drag)       │
└────────────────────────────────┘
```

Styling:
- Background: #0D0D14
- Border-right: 1px #1E293B
- Category titles: 11px, caps, muted, margin-top 24px first
- Items: 32px height, clickable, drag cursor
- Hover: Background #111827, border-left accent color
- Drag indicator: Icon on right (≡ or →)

---

### CENTER PANEL - INTERACTIVE CANVAS

```
Visual canvas showing components and connections:

         [API Gateway]
              ↓
    [Load Balancer]
         ↙    ↘
    [Cache]  [Web Server]
         ↓         ↓
    [PostgreSQL]   [Redis]
         ↓         ↓
    [S3 Storage]   [Message Queue]
```

Canvas:
- Height: 100% viewport height
- Background: Light grid pattern (#0D0D14 with subtle grid)
- Pan: Click+drag to move around
- Zoom: Scroll wheel or zoom buttons
- Each component: Rectangle box (color-coded by type)
- Connections: Lines between components, arrows showing direction
- Click component: Selects it, shows details in right panel
- Drag from component: Draws connection line
- Right-click: Delete component or edit properties

Components:
- Size: ~80px × 60px
- Color: Category color (compute: purple, storage: cyan, etc.)
- Icon: Type icon (database, server, etc.)
- Label: Component name (12px)
- Border: 1px accent color, thicker when selected

Connections:
- Line: 2px, accent color
- Animated flow: 4px arrow moving along line when active
- Hover: Show latency tooltip

---

### RIGHT PANEL (280px) - ANALYSIS & DESIGN

**Selected Component Details:**

```
┌────────────────────────────────┐
│ PostgreSQL Database            │ (14px, bold)
│ Data Store                     │ (12px, muted)
│                                │
│ Configuration:                 │
│ • Replicas: 3                 │ (12px, bullets)
│ • Shards: 5                   │
│ • Backup: Daily               │
│ • Engine: PostgreSQL 14       │
│                                │
│ Capacity:                      │
│ • Storage: 1TB SSD             │
│ • QPS: 10K reads/sec          │
│ • Connections: 500             │
│                                │
│ [Edit Config] [Delete]        │ (Buttons)
└────────────────────────────────┘
```

Styling:
- Background: #111827
- Border: 1px #1E293B
- Border-radius: 12px
- Padding: 16px
- Margin-bottom: 16px

**Request Flow Analysis:**

```
┌────────────────────────────────┐
│ 📊 Request Path Latency        │
│                                │
│ Total: 34ms                    │ (14px, bold, accent)
│                                │
│ Client → API Gateway:  2ms │ █   │ (4px bars)
│ API Gateway → LB:      1ms │     │
│ LB → Web Server:       1ms │     │
│ Server → Cache:        0.5ms │   │
│ Cache Miss → DB:       8ms │ █████  │ Bottleneck!
│ Query Processing:     12ms │ ███████ │
│ Result → Cache:       1ms │     │
│ Cache → Server:      0.5ms │    │
│ Server → LB:         1ms │     │
│ LB → Client:         2ms │ █   │
│                                │
└────────────────────────────────┘
```

Styling:
- Background: #0D0D14
- Border: 1px #1E293B
- Border-radius: 8px
- Padding: 12px
- Each row: 24px
- Service name: 12px, left-aligned
- Latency: 12px, right-aligned
- Bar: Colored gradient (green → orange → red based on duration)
- Bottleneck: Red color, labeled "Bottleneck!"

**Design Considerations:**

```
┌────────────────────────────────┐
│ 💡 Design Points               │
│                                │
│ 🔄 Consistency vs Availability │
│    → Chose eventual consistency │
│    → Replication delay: ~500ms │
│                                │
│ 📈 Scalability                 │
│    → Horizontal sharding       │
│    → 5 shards, grow as needed  │
│                                │
│ 🛡️ Resilience                  │
│    → 3 replicas for HA        │
│    → Auto-failover enabled     │
│                                │
│ ❓ Interview Question           │
│    "How would you handle a     │
│    network partition between   │
│    the cache and database?"    │
│                                │
└────────────────────────────────┘
```

Styling:
- Background: #111827
- Border: 1px #1E293B
- Border-radius: 8px
- Padding: 16px
- Emoji indicators: Large (20px)
- Bold topics: 12px, bold
- Details: 11px, #94A3B8

---

## RESPONSIVE BEHAVIORS

### Tablet (768px - 1024px)
- Left panel: Collapse to hamburger menu
- Right panel: Move below center content
- Three-column becomes two-column + stacked
- Panels scroll independently

### Mobile (< 768px)
- All panels stack vertically
- Full width content
- Left panel: Hamburger menu
- Right panel: Collapsible section below
- Cards and text scale down
- Touch-friendly sizes (44px minimum)
- Horizontal scrolling for tables/charts

---

## COLOR REFERENCE FOR COMPONENTS

- **Compute:** #7C3AED (purple)
- **Storage:** #06B6D4 (cyan)
- **Networking:** #10B981 (green)
- **Messaging:** #F59E0B (orange)
- **Security:** #EF4444 (red)
- **Monitoring:** #EC4899 (pink)

---

**All mockups ready for Lovable implementation.**

