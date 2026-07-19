# Paper2Code → Premium AI Engineering Operating System

**Design Philosophy:** Professional tool, not educational website  
**Inspiration:** Cursor, Linear, Raycast, Vercel, TensorTonic, Notion Calendar  
**Approach:** Workspace-first, context-preserving, deep work optimized  
**Scope:** UI/UX transformation only, no features, no backend changes

---

## DESIGN DIRECTION SHIFT

### From → To
```
Landing page website      → Professional operating system
Scroll-based docs        → Workspace-first panels
Isolation between pages  → Context preservation
Long sessions fragmented → Deep work mode (1-3 hours)
Content distribution     → Workspace navigation
```

### Key Operating Principles
1. **Workspace First** — Every major section is a workspace, not a page
2. **Context Always Visible** — Never lose where you are or what's nearby
3. **Deep Work Optimized** — Support focused sessions without navigation friction
4. **Professional Aesthetic** — Dark, dense, purposeful, minimal marketing language

---

## APPLICATION SHELL V2

### Global Navigation Model

```
┌─────────────────────────────────────────────────────────────┐
│ Paper2Code   [Command Palette: Cmd+K]   [User Menu]         │
├──────────────┬──────────────────────────┬──────────────────┤
│              │                          │                  │
│ Left Rail    │                          │  Context Panel   │
│              │    Workspace             │  (Contextual)    │
│ • Dashboard  │    Content               │                  │
│ • Academy    │    (Variable height)     │ ← Changes based  │
│ • Search     │                          │   on workspace   │
│ • Settings   │                          │                  │
│              │                          │                  │
│ 64px         │ (Auto, ~1200px max)     │ 320px            │
└──────────────┴──────────────────────────┴──────────────────┘
```

### Left Rail (Persistent)
**Width:** 64px (expanded: 240px with labels)

**Fixed Navigation:**
```
┌─────────┐
│   ⌘     │ Logo + Cmd Indicator
├─────────┤
│    📊   │ Dashboard (primary)
│    🏛   │ Academy Hub
│    🔍   │ Search/Command
│    ⚙️    │ Settings
├─────────┤
│    ⭐   │ Favorites (dynamic)
│    📌   │ Recent (dynamic)
├─────────┤
│    ?    │ Help + Docs
└─────────┘
```

**Behavior:**
- Hover to expand with labels
- Click active section → collapse
- No page transitions (content updates in place)
- Icons from Lucide (24px, muted color)
- Active: Accent color + subtle background

### Command Palette (Cmd+K)
```
┌──────────────────────────────┐
│ Search/Command...            │ (Fuzzy search)
├──────────────────────────────┤
│                              │
│ Jump to:                     │ (Results, 8-10 items)
│ • Dashboard                  │
│ • Transformer Architecture   │
│ • Read: Attention Is All You │
│ • Tensor Trace: GPT-2       │
│ • System Design: Rate Limit  │
│ • Problem #42: Backprop     │
│ • Settings                   │
│                              │
│ [Cmd+K]Toggle [Tab]Select   │
│ [Enter]Open [Esc]Close      │
└──────────────────────────────┘
```

**Features:**
- Fuzzy search across all content
- Recently viewed first
- Keyboard-only navigation
- Icon + title + context (parent node)
- Instant filtering
- No hover/click, keyboard-driven

### Top Bar (Minimal)
```
┌─────────────────────────────────────────┐
│ Paper2Code    [⌘K Search]   [👤 Menu]   │ 40px
└─────────────────────────────────────────┘
```

**Elements:**
- Logo (16px): Clickable to dashboard
- Search trigger: Cmd+K text (right side, muted)
- User menu: Avatar dropdown (profile, settings, logout)
- No additional navigation (all in left rail)

### Right Context Panel (Dynamic)

**Purpose:** Contextual information based on current workspace

**Dashboard Context:**
- Upcoming tasks
- Active session stats
- Next learning node

**Architecture Context:**
- Selected layer details
- Mathematical formulas (KaTeX)
- Related papers

**Paper Context:**
- Key equations
- Architecture reference
- Implementation code snippet

**Paper-to-Code Context:**
- Tensor dimensions
- Variable inspector
- Test results

**Size:** 320px on desktop, hidden on tablet/mobile
**Behavior:** Sticky, independent scroll, collapsible

---

## DASHBOARD V2 — Learning Command Center

### Layout
```
┌────────────────────────────────────────────────┐
│ Dashboard                                      │
├──────────┬────────────────────────┬───────────┤
│          │                        │           │
│ Sections │   Primary Workspace    │  Context: │
│ & Stats  │   (Variable height)    │  • Upcoming
│          │                        │  • Session
│          │                        │  • Stats
│64px      │ (Auto)                 │ 320px
└──────────┴────────────────────────┴───────────┘
```

### Left Section (Persistent)
**Content:**
```
📊 Dashboard

Your Learning
━━━━━━━━━━━━━━
Current Track    [Transformer Arch]
Progress         [████████░░░░░░░░ 42%]
Time This Week   [6.5 hours]
Problems Solved  [27/110]
Papers Read      [5/50]

Quick Jump
━━━━━━━━━━━━━━
Continue: Backprop (Problem #42)
Paper: Attention (2 min read)
System: Rate Limiter (15 min)

Latest Session
━━━━━━━━━━━━━━
Type:    Tensor Trace
Model:   GPT-2
Time:    34 min ago
Learned: [New concepts: 3]
```

**Styling:**
- Monospace labels (9px, caps, muted)
- Large number: 24px bold, accent color
- Sectioned with dividers
- Clickable items (hover background)
- No margins, tight density

### Center Workspace (Dynamic Content)

**Default (No selection):** Primary workspace shows:

```
┌─────────────────────────────────────┐
│ Continue Learning                   │ (16px bold)
├─────────────────────────────────────┤
│                                     │
│ ✓ Problem #42: Backpropagation     │ (Interactive card)
│   Medium • ~20 min • Deep Learning  │
│   Started 2 hours ago • 3/5 tests   │
│   [Open Problem →]                  │
│                                     │
│ ✓ Paper: Attention Is All You Need │
│   Vaswani et al. • 12 min read      │
│   Started yesterday • 65% complete  │
│   [Resume Reading →]                │
│                                     │
├─────────────────────────────────────┤
│ Current Roadmap: Transformer Arch   │ (16px bold)
├─────────────────────────────────────┤
│                                     │
│ ① Fundamentals (Completed)          │
│    └─ ✓ Linear Algebra              │
│    └─ ✓ Neural Networks             │
│                                     │
│ ② Attention Mechanisms (In Progress)│ (Highlighted)
│    ├─ ✓ Self-Attention              │
│    ├─ ⚡ Multi-Head Attention       │ (Active: problem #42)
│    └─ ◇ Cross-Attention             │ (Locked)
│                                     │
│ ③ Transformer Architecture (Upcoming)│
│    └─ ◇ All papers & problems       │
│                                     │
├─────────────────────────────────────┤
│ Weekly Progress                     │ (16px bold)
├─────────────────────────────────────┤
│                                     │
│ Problems:    ▓▓▓▓▓▓░░░░ 6/12        │ (Bar + numbers)
│ Papers:      ▓▓░░░░░░░░ 2/8         │
│ Hours:       ▓▓▓▓▓░░░░░ 5.5/10     │
│                                     │
└─────────────────────────────────────┘
```

**Card Styling:**
- Background: #0D0D14 (body color, no surface)
- Border: 1px #1E293B (thin, subtle)
- Hover: Border #7C3AED, background #111827
- Padding: 16px
- Margin: 12px 0
- Transition: 150ms ease

**Content Sections:**
1. **Continue Learning** — Last 2-3 active items
2. **Current Roadmap** — Visual progress tree
3. **Weekly Stats** — Progress bars
4. **Recent Architectures** — Thumbnail grid (2 columns)
5. **Bookmarked Papers** — Vertical list (5 items)

### Right Context Panel (Dynamic)

```
┌─────────────────────┐
│ Next Up             │ (12px, muted)
├─────────────────────┤
│                     │
│ Problem #43         │ (14px, bold)
│ Multi-Head Attn     │ (12px, secondary)
│ Hard • ~30 min      │ (11px, tertiary)
│                     │
│ [Start →]           │ (Button)
│                     │
├─────────────────────┤
│ Session Stats       │ (12px, muted)
├─────────────────────┤
│                     │
│ Today: 2h 15m       │ (13px)
│ Focus Streak: 4 days│ (13px)
│ This Month: 32.5h   │ (13px)
│                     │
├─────────────────────┤
│ Milestones          │ (12px, muted)
├─────────────────────┤
│                     │
│ Fundamentals   ✓    │ (12px)
│ Attention      ⚡    │ (12px, active)
│ Transformer    ◇    │ (12px, locked)
│ Production     ◇    │ (12px, locked)
│                     │
└─────────────────────┘
```

**Updates in real-time as user navigates.**

---

## ARCHITECTURE WORKSPACE V2

### Layout (Replaces article experience)
```
┌─────────────────────────────────────────────────┐
│ Transformer                                     │
├──────────┬────────────────────────┬────────────┤
│          │                        │            │
│ Sections │  Interactive Canvas    │ Inspector: │
│ & Search │  (Primary focus)       │ • Math     │
│          │                        │ • Code     │
│          │  + Layer Explorer      │ • Papers   │
│          │  + Tensor Flow         │ • QA       │
│          │                        │            │
│ 240px    │ (Auto, 1200px target)  │ 320px      │
└──────────┴────────────────────────┴────────────┘
```

### Left Sidebar (Sections)
```
Architecture: Transformer
━━━━━━━━━━━━━━━━━━━━━━━━

[Search within...] (input)

Contents
━━━━━━━━━━━━━━━━━━━━━━━━
├─ 1. Overview
│  ├─ Motivation
│  ├─ Key Innovation
│  └─ Impact
├─ 2. Architecture
│  ├─ Encoder
│  ├─ Decoder
│  └─ Embeddings
├─ 3. Attention
│  ├─ Self-Attention (YOU ARE HERE)
│  ├─ Multi-Head
│  └─ Cross-Attention
├─ 4. Implementation
│  ├─ Layer Normalization
│  ├─ Feed Forward
│  └─ Positional Encoding
└─ 5. Variants
   ├─ Vision Transformer
   ├─ BERT
   └─ GPT

Progress
━━━━━━━━━━━━━━━━━━━━━━━━
[████████░░░░░░░░░░░░░░] 35%
3 / 8 concepts mastered
```

**Interaction:**
- Click section → Scroll center to that section
- Active section highlighted with left border accent
- Search filters by content + headings
- Progress per section

### Center Workspace (Canvas + Content)

**Primary:** Interactive SVG/Canvas diagram

```
┌─────────────────────────────────────┐
│ [Interactive Transformer Diagram]   │
│                                     │
│  Input Text                         │
│    ↓                                │
│  [Embedding + Position Encoding]    │
│    ↓                                │
│  [Encoder Block] ×6                 │
│    • Multi-Head Attention ←─ YOU    │
│    • Feed Forward              ARE  │
│    ↓                           HERE │
│  [Decoder Block] ×6                 │
│    ↓                                │
│  [Output Layer]                     │
│                                     │
│  Legend: ● = Clickable Layer        │
│          → = Data Flow              │
│          ⚡ = Selected               │
│                                     │
└─────────────────────────────────────┘
```

**Features:**
- SVG based, not static image
- Click layer → Shows details in right panel
- Hover layer → Highlights connections, shows tooltip
- Animated tensor flow (pulsing along arrows)
- Color-coded by type (encoder: purple, decoder: cyan)
- Responsive to container (scales on resize)

**Secondary:** Layer Explorer (Below diagram)

```
Self-Attention Layer (Selected)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input Shape:        [batch, seq_len, d_model]
Output Shape:       [batch, seq_len, d_model]

Parameters:
  • query_proj:     (d_model, d_k×h)
  • key_proj:       (d_model, d_k×h)
  • value_proj:     (d_model, d_v×h)
  • output_proj:    (d_v×h, d_model)

Connections:
  ← From: Embedding Layer
  → To: Feed Forward Layer
```

**Content:**
- Dimensions and shapes
- Parameter matrices
- Data flow connections
- Activation functions used

### Right Inspector (Contextual)

**When layer selected:**

```
┌────────────────────────┐
│ Self-Attention         │ (14px bold)
│ encoder.layer_0        │ (12px mono, muted)
├────────────────────────┤
│                        │
│ Math                   │ (11px, caps)
├────────────────────────┤
│ Attention(Q,K,V) =     │
│ softmax(QK^T/√d_k)V    │ (KaTeX, 12px)
│                        │
│ [View full derivation] │ (Link)
│                        │
├────────────────────────┤
│ Implementation         │ (11px, caps)
├────────────────────────┤
│                        │
│ class Attention:       │ (10px mono)
│   def forward(q, k, v):│
│     scores = q@k.T/√dk │
│     return attn @ v    │
│                        │
│ [View on GitHub]       │ (Link)
│                        │
├────────────────────────┤
│ Related               │ (11px, caps)
├────────────────────────┤
│                        │
│ 📄 Attention Is All    │ (12px)
│    You Need (2017)     │
│                        │
│ 🏛  Vision             │
│    Transformer         │
│                        │
│ 🎯 Problem #12:        │
│    Implement Attention │
│                        │
└────────────────────────┘
```

**Updates when hovering/clicking different layers.**

---

## PAPER WORKSPACE V2 — Research Analyst Experience

### Layout
```
┌──────────────────────────────────────────────┐
│ Attention Is All You Need (2017)             │
├────────┬──────────────────┬─────────────────┤
│        │                  │                 │
│ Timeline│  Paper Content   │ Context:       │
│ Sections│  (Primary)       │ • Architecture │
│ Related │                  │ • Equations    │
│        │                  │ • Code         │
│        │                  │ • QA           │
│ 200px  │ (Auto, 1000px)   │ 320px          │
└────────┴──────────────────┴─────────────────┘
```

### Left Sidebar (Timeline + Sections)

```
📄 Attention Is All You Need
   Vaswani et al. | NeurIPS 2017
   84.2K citations

[Search...]

Timeline (Evolution)
━━━━━━━━━━━━━━━━━━━━
2014: Seq2Seq Attention
  └─ Bahdanau et al.

2015: Machine Translation
  └─ Breakthrough accuracy

→ 2017: Transformer (This)
  └─ No recurrence!

2018: BERT Pre-training
  └─ Transfer learning

2019: GPT-2
  └─ Language model

Sections
━━━━━━━━━━━━━━━━━━━━
1. Abstract
2. Introduction
3. Background
4. Model Architecture ← YOU ARE HERE
   ├─ Encoder
   ├─ Decoder
   └─ Attention
5. Attention
6. Efficiency
7. Training
8. Results
9. Conclusion

Related
━━━━━━━━━━━━━━━━━━━━
→ Vision Transformer
→ BERT
→ GPT Series
→ Stable Diffusion
```

### Center Content (Paper + Insights)

**Sticky Header:**
```
┌──────────────────────────────────┐
│ Attention Is All You Need        │ (22px bold)
│ Vaswani, Shazeer, Parmar, et al. │ (12px)
│ NeurIPS 2017 • 84.2K citations   │ (12px muted)
│                                  │
│ [★ Bookmark] [Share] [Reference] │ (Buttons)
└──────────────────────────────────┘
```

**Main Content:**

```
📌 Abstract
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The dominant sequence transduction models are based on complex 
recurrent or convolutional neural networks in an encoder-decoder 
configuration. The best performing models also connect the encoder 
and decoder through an attention mechanism...

[Full text, but summaries boxed]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Key Insight
The paper proposes replacing recurrence entirely with 
multi-head self-attention mechanisms, enabling fully 
parallel sequence processing while maintaining or 
improving translation quality.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 Main Architecture
┌─────────────────────────────┐
│ Input → Embedding → Encoder │
│         ↑          ↓        │
│         └← Decoder ←────┐   │
│                    ↓    │   │
│                  Output │   │
│                    ↓    │   │
│              Probability│   │
└─────────────────────────────┘

[Arrows animated showing flow]

📊 Results Table
Model                 BLEU  Params Training Time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Seq2Seq + Attention   28.9  217M   60 hrs
Transformer-big       29.9  370M   50 hrs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Formatted as table with emphasis]

═════════════════════════════════════════════════

4️⃣ Model Architecture
[Content with equations, diagrams, explanations]

5️⃣ Attention Is All You Need
[Deep dive into attention mechanism]

Equation 1: Scaled Dot-Product Attention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Attention(Q, K, V) = softmax(QK^T / √(d_k))V

Where:
Q ∈ ℝ^(n×d_k) — Query matrix
K ∈ ℝ^(m×d_k) — Key matrix
V ∈ ℝ^(m×d_v) — Value matrix

[Detailed explanation below equation]
```

**Styling:**
- Headings: Plus Jakarta Sans, bold, with icon
- Body: Inter, 13px, line-height 1.8
- Equations: KaTeX, 14px, centered
- Callouts: Colored left border, subtle background
- Tables: Monospace font, aligned
- Code blocks: Dark background, syntax highlighting
- Max width: 900px for readability

### Right Inspector (Dynamic)

```
┌─────────────────────────┐
│ 📐 Key Equations        │ (12px, caps)
├─────────────────────────┤
│                         │
│ Attention(Q, K, V) =    │ (12px KaTeX)
│ softmax(QK^T / √d_k)V   │
│                         │
│ MultiHead(Q,K,V) =      │
│ Concat(h_1,..,h_h)W^O   │
│                         │
├─────────────────────────┤
│ 🏗  Architecture         │ (12px, caps)
├─────────────────────────┤
│                         │
│ Transformer (2017)      │ (Link)
│ Vision Trans (2020)     │ (Link)
│ BERT (2018)             │ (Link)
│                         │
├─────────────────────────┤
│ 💻 Implementation        │ (12px, caps)
├─────────────────────────┤
│                         │
│ class Attention:        │ (10px mono)
│   def forward(q,k,v):   │
│     s = q @ k.T / √d    │
│     return softmax(s)@v │
│                         │
│ [View on GitHub]        │ (Link)
│                         │
├─────────────────────────┤
│ ❓ Study Questions      │ (12px, caps)
├─────────────────────────┤
│                         │
│ Why scale by √d_k?      │ (12px, link)
│ How does attention      │ (12px, link)
│ parallelize?            │
│ Comparison to RNNs?     │ (12px, link)
│                         │
└─────────────────────────┘
```

---

## PAPER-TO-CODE V2 — Synchronized Learning

### Layout (Three Columns)
```
┌────────────────────────────────────────────┐
│ Backpropagation: Theory → Code → Shapes   │
├─────────────┬──────────────┬──────────────┤
│             │              │              │
│ Theory      │ Code Editor  │ Tensor       │
│             │              │ Shapes       │
│ • Concept   │ • Functions  │ • Input      │
│ • Equations │ • Monaco     │ • Flow       │
│ • Math      │ • Highlight  │ • Inspector  │
│             │              │              │
│ 300px       │ 600px        │ 320px        │
└─────────────┴──────────────┴──────────────┘
```

### Left Panel (Theory)

```
Backpropagation
━━━━━━━━━━━━━━

Learning Rule
The chain rule allows us to compute
gradients efficiently:

∂L/∂w = ∂L/∂z · ∂z/∂w

Where:
L = loss function
z = pre-activation (wx + b)
w = parameter

Why Backward?
Forward computation is O(depth).
Backward computation is also O(depth).
But backward is efficient:
all gradients in one pass!

Common Mistake ❌
Computing gradients forward.
The chain rule flows from loss
backward to parameters.

Mathematical Definition
∀ layer l, at parameter w:
dL/dw = Σ (∂L/∂a_l) · (∂a_l/∂w)

[Click equation → highlights code on right]
```

**Interaction:**
- Click equation → Code line highlights
- Hover variable → Shows tensor shape in right panel
- Scroll sections → Code stays sticky
- Always visible: equation + code + shapes

### Center Panel (Code Editor)

```
def backward_pass(forward_result, loss_grad):
    """
    Compute gradients for parameters
    """
    grads = {}
    
    # 1. Gradient w.r.t output activation
    dz = loss_grad  # ← Highlighted when math selected
    
    # 2. Backprop through output layer
    dW_out = dz.T @ forward_result['h']
    db_out = dz.sum(axis=0, keepdims=True)
    
    # 3. Gradient w.r.t hidden layer activation
    dh = dz @ forward_result['W_out'].T
    dh_relu = dh * (forward_result['z'] > 0)
    
    # 4. Gradients for hidden layer
    dW_hidden = dh_relu.T @ forward_result['X']
    db_hidden = dh_relu.sum(axis=0, keepdims=True)
    
    grads['W_out'] = dW_out
    grads['b_out'] = db_out
    grads['W_hidden'] = dW_hidden
    grads['b_hidden'] = db_hidden
    
    return grads

[Run] [Submit] [Reset]
```

**Features:**
- Monaco Editor (syntax highlighting, line numbers)
- Read-only code sections (boilerplate)
- Editable sections (student fills in)
- Inline documentation comments
- [Run] button executes tests
- [Submit] sends to grading
- Hover variable → Shows shape in right panel

### Right Panel (Tensor Inspector)

```
Tensor Flow
━━━━━━━━━━

Input → Shapes
forward_result['z']:  (4, 10, 10)
forward_result['h']:  (4, 10, 64)

Variables (Current Step)
━━━━━━━━━━━━━━━━━━
dz          (4, 10)    float32
dW_out      (10, 3)    float32
db_out      (1, 3)     float32
dh          (4, 10)    float32
dh_relu     (4, 10)    float32
dW_hidden   (10, 5)    float32
db_hidden   (1, 5)     float32

Gradients Summary
━━━━━━━━━━━━━━━━━━
Total params:   2,345
Gradients:      7 arrays
Avg magnitude:  0.043
Max magnitude:  0.234

Test Results
━━━━━━━━━━━━━━━━━━
✓ Shape check      (0.12ms)
✓ Value range      (0.08ms)
✓ Numeric gradient (2.34ms)
✓ Chain rule       (0.05ms)

Time: 2.59ms
```

**Dynamic Updates:**
- Updates as code runs
- Highlights matching variables in code
- Shows tensor dimensions
- Visualizes gradient flow

---

## TENSOR TRACE V2 — Flagship Experience

### Full-Screen Canvas Layout
```
┌──────────────────────────────────────────┐
│ Tensor Trace: GPT-2 Forward Pass         │
├──────────────────────────────────────────┤
│                                          │
│  Navigation: [← Prev]  [5/124]  [Next →]│ (Center)
│  Speed: [0.5x] [1x] [2x]  [Play] [Pause]│
│  Progress: [████████░░░░░░░░░░░░] 40%   │
│                                          │
├──────────────────────────────────────────┤
│                                          │
│        [Animated Tensor Visualization]   │
│                                          │
│        Input: (batch=1, seq=512)        │
│          ↓                               │
│        Token Embedding: (512, 768)      │
│          ↓                               │
│        Position Encoding: (512, 768)    │
│          ↓    [Color-coded by value]    │
│        Transformer Block 1:              │
│        ├─ Self-Attention → (512, 768)   │
│        ├─ LayerNorm → (512, 768)        │
│        ├─ FFN → (512, 3072)             │
│        └─ Output → (512, 768)           │
│          ...                             │
│        Transformer Block 12:             │
│        └─ Output → (512, 768)           │
│          ↓                               │
│        Linear Projection: (512, 50257)  │
│          ↓                               │
│        Logits → Softmax → Prediction    │
│                                          │
├──────────────────────────────────────────┤
│ Details:                                 │
│                                          │
│ Step 5: Linear Projection                │
│ Input:  (batch=1, seq=512, hidden=768)  │
│ Weight: (768, 50257)                    │
│ Output: (batch=1, seq=512, vocab=50257) │
│                                          │
│ Operation: MatMul(input, weight.T)      │
│ Time: 2.34ms  |  FLOPs: 1.92G           │
│                                          │
│ [Inspect Memory] [Show Gradients]       │
│                                          │
└──────────────────────────────────────────┘
```

### Features

**Animated Visualization:**
- Tensors morph between shapes (500ms, smooth curve)
- Dimensions color-coded (H=red, W=green, C=blue, batch=purple)
- Glow effect on active operations
- Pulsing along data flow arrows

**Controls:**
- Play/Pause with space bar
- Speed: 0.5x, 1x, 2x playback
- Step backward/forward (← →)
- Jump to step (click timeline)
- Rewind (⏪) to start

**Details Panel:**
- Operation name (Large, bold)
- Input/output shapes (Monospace)
- Mathematical operation
- Execution time
- Memory usage
- Gradient flow indicator

**Keyboard Shortcuts:**
```
Space        Play/Pause
→            Next step
←            Previous step
G            Toggle gradients
M            Toggle memory
/            Show help
```

---

## SYSTEM DESIGN V2 — Interactive Architecture Board

### Canvas Layout
```
┌────────────────────────────────────┐
│ System Design: E-Commerce Platform │
├────────────────────────────────────┤
│                                    │
│  [+ Add Component]  [Simulate]     │ (Top controls)
│                                    │
│  Interactive Canvas:               │
│                                    │
│     ┌──────────────┐               │
│     │ Client       │               │
│     └──────┬───────┘               │
│            │                       │
│     ┌──────▼──────────┐            │
│     │ Load Balancer   │            │
│     └──────┬──────────┘            │
│      ┌─────┴─────┐                 │
│      │           │                 │
│  ┌───▼───┐  ┌───▼───┐             │
│  │ API 1 │  │ API 2 │             │
│  └───┬───┘  └───┬───┘             │
│      │          │                 │
│  ┌───▼──────────▼───┐             │
│  │    Cache (Redis) │             │
│  └───┬──────────────┘             │
│      │                            │
│  ┌───▼───────────────┐            │
│  │ Primary Database  │            │
│  │ (PostgreSQL)      │            │
│  │                   │            │
│  │ [Selected]        │            │
│  └───────────────────┘            │
│                                    │
└────────────────────────────────────┘
```

### Interaction Model

**Add Components:**
- Drag from left sidebar to canvas
- Auto-layout with spring physics
- Connection drawing: click → click
- Delete with right-click

**Select Component:**
- Click shows details in right panel
- Hover shows tooltip (tech stack)
- Border highlighting with accent color

**Simulate Request:**
- [Simulate] button starts animation
- Request path flows from client → servers
- Latency badge on each hop
- Bottleneck highlighted (red)

### Right Inspector

```
┌──────────────────────┐
│ Primary Database     │ (14px bold)
│ PostgreSQL 14        │ (12px)
├──────────────────────┤
│                      │
│ Specs                │ (11px caps)
├──────────────────────┤
│ Storage: 1TB         │ (12px)
│ Replicas: 3          │
│ QPS: 5K reads/sec    │
│ Connections: 200     │
│                      │
├──────────────────────┤
│ Request Path         │ (11px caps)
├──────────────────────┤
│                      │
│ Client                  0ms │
│   → Load Balancer      1ms │
│   → API Server         2ms │
│   → Cache Hit          1ms │
│   → Return            1ms │
│ ─────────────────────────  │
│ Total (Cache Hit):    5ms │
│                      │
│ Client                  0ms │
│   → Load Balancer      1ms │
│   → API Server         2ms │
│   → Cache Miss         0ms │
│   → Database           8ms │
│   → Cache Write        1ms │
│   → Return            1ms │
│ ─────────────────────────  │
│ Total (Cache Miss):  13ms │
│                      │
├──────────────────────┤
│ Design Tradeoffs     │ (11px caps)
├──────────────────────┤
│                      │
│ Consistency:         │ (12px)
│ Eventual with 500ms  │
│ replication lag      │
│                      │
│ Availability:        │ (12px)
│ 3 replicas provide   │
│ HA across regions    │
│                      │
├──────────────────────┤
│ Failure Modes        │ (11px caps)
├──────────────────────┤
│                      │
│ Network partition?   │ (12px)
│ → Reads from replica │
│ → Writes to leader   │
│                      │
│ Primary down?        │ (12px)
│ → Auto-failover to   │
│    replica (30sec)   │
│                      │
└──────────────────────┘
```

---

## DESIGN SYSTEM V2 — Professional Tool Aesthetic

### Color Palette

**Dark Backgrounds:**
```
--bg-body:       #09090F    (Deep background, almost black)
--bg-surface:    #0D0D14    (Slightly lighter, not used much)
--bg-panel:      #111827    (Elevated surfaces)
--bg-hover:      #1A1F2E    (Hover states)
--bg-active:     #232D3D    (Active/selected)
```

**Text Colors:**
```
--text-primary:    #E2E8F0  (Main text, almost white)
--text-secondary:  #94A3B8  (Secondary text, readable gray)
--text-tertiary:   #64748B  (Muted text, labels)
--text-muted:      #475569  (Very muted, barely visible)
```

**Borders & Dividers:**
```
--border-default:  #1E293B  (Subtle borders)
--border-light:    #2A2D3A  (Very subtle)
--border-focus:    #3B4F63  (Focused/highlighted)
--divider:         #1A1F2E  (Section separators)
```

**Accent Colors:**
```
--accent-primary:  #7C3AED  (Purple, primary action)
--accent-light:    #A78BFA  (Light purple, hover)
--accent-cyan:     #06B6D4  (Cyan, secondary, data flow)
--accent-emerald:  #10B981  (Green, success, completion)
--accent-amber:    #F59E0B  (Orange, warning)
--accent-red:      #EF4444  (Red, error, bottleneck)
```

**Status Colors:**
```
--status-success:  #10B981  (Completed, passed)
--status-active:   #3B82F6  (In progress, current)
--status-warning:  #F59E0B  (Warning, slow)
--status-error:    #EF4444  (Error, failed)
--status-locked:   #64748B  (Locked, unavailable)
```

### Typography

**Font Stack:**
```
Headings:   Plus Jakarta Sans (geometric, modern)
Body:       Inter (clean, readable)
Monospace:  JetBrains Mono (code, dimensions, numbers)
Code:       JetBrains Mono (highlighted blocks)
```

**Sizing:**
```
Display:    32-40px (clamp(32px, 5vw, 40px))
Heading 1:  24px    (Page titles)
Heading 2:  18px    (Section titles)
Heading 3:  14px    (Subsection titles)
Label/Bold: 13px    (Card titles, emphasis)
Body:       12px    (Default text)
Small:      11px    (Secondary text, labels)
Tiny:       10px    (Captions, timestamps)
Mono:       11px    (Code, dimensions, numbers)
```

**Weight:**
```
Light:      300 (rarely used)
Regular:    400 (body text)
Medium:     500 (emphasis)
Semibold:   600 (labels, small titles)
Bold:       700 (headings, emphasis)
```

**Line Height:**
```
Tight:      1.2  (headings, labels)
Normal:     1.5  (body text)
Relaxed:    1.8  (long-form content)
Code:       1.6  (code blocks)
```

**Letter Spacing:**
```
Tight:      -0.03em  (headings)
Normal:      0em      (body)
Loose:       0.05em   (caps labels, "SELECTED", "ACTIVE")
Mono:        0em      (code, numbers)
```

### Spacing System

**Base Unit: 4px**
```
--space-1:   4px   (tight)
--space-2:   8px   (padding in small components)
--space-3:  12px   (standard padding)
--space-4:  16px   (comfortable padding)
--space-6:  24px   (section spacing)
--space-8:  32px   (large spacing)
--space-12: 48px   (between major sections)
```

**Application:**
```
Component padding:     12-16px
Section margins:       24-32px
Grid gap:             12px
Card margins:         12px
Border radius:        6-8px (tight, professional)
```

### Borders & Shadows

**Borders:**
```
Default:   1px #1E293B    (subtle)
Light:     1px #2A2D3A    (very subtle)
Focus:     2px #7C3AED    (accent color)
Divider:   1px #1A1F2E    (section separator)
```

**Shadows (Elevation):**
```
None:      no shadow (most elements)
Hover:     0 2px 6px rgba(0,0,0,0.4)
Raised:    0 4px 12px rgba(0,0,0,0.3)
Modal:     0 10px 40px rgba(0,0,0,0.5)
Glow:      0 0 12px rgba(124,58,237,0.3) (accent)
```

### Border Radius

```
--radius-sm:   4px   (small components, subtle)
--radius-md:   6px   (default, cards, inputs)
--radius-lg:   8px   (larger panels)
--radius-xl:  12px   (large modals)
--radius-2xl: 16px   (hero sections)
```

### Motion & Animation

**Transitions:**
```
--transition-fast:    100ms cubic-bezier(0.4, 0, 0.2, 1)
--transition-base:    150ms cubic-bezier(0.4, 0, 0.2, 1)
--transition-slow:    300ms cubic-bezier(0.4, 0, 0.2, 1)
--transition-slowest: 500ms cubic-bezier(0.34, 1.56, 0.64, 1)
```

**Keyframe Animations:**
```
Fade:       opacity 0 → 1 (300ms)
SlideUp:    translateY 20px → 0 (300ms)
Pop:        scale 0.8 → 1 (250ms, bouncy)
Glow:       box-shadow pulsing (2s infinite)
Slide:      translateX 100% → 0 (300ms)
```

**Rules:**
- Hover states: 100ms (fast feedback)
- Content changes: 300ms (visible, not jarring)
- Modals: 500ms (smooth, bouncy entrance)
- Never over 500ms (feels sluggish)
- Respect `prefers-reduced-motion`

### Component Spacing Reference

**Cards:**
```
Padding:      16px
Gap (internal): 12px
Margin:       12px (between cards)
Border:       1px
Radius:       6px
```

**Sections:**
```
Padding:      20px (horizontal), 24px (vertical)
Margin:       24px (between sections)
Border-top:   1px divider
```

**Lists:**
```
Item height:  36-40px
Item padding: 8px 12px (vertical/horizontal)
Gap:         4px (items)
Divider:     1px #1A1F2E (between items)
```

**Form Elements:**
```
Input height: 36px
Input padding: 8px 12px
Input border: 1px
Focus border: 2px accent
Label:       11px, caps, muted
Spacing:     8px (label to input)
```

---

## MOTION SYSTEM

### Interaction Animations

**Hover (100ms fast):**
- Border color accent
- Slight background elevation
- Optional glow effect for important elements

**Click (150ms base):**
- Scale down 0.98 (press feeling)
- Delay 50ms then scale back up
- Creates tactile feedback

**Focus (150ms base):**
- Ring: 2px accent color
- Glow: optional subtle shadow
- Ring has 4px offset for visibility

**Active State (Instant):**
- Border left: 3px accent
- Background: subtle tint (accent @ 0.05)
- Icon color: accent

### Content Animations

**Page Transitions (300ms):**
- Fade + slide: opacity 0→1, translateY 20px→0
- Stagger children: 50ms each

**List Item Reveals (300ms staggered):**
- Each item: 50ms delay after previous
- Creates rhythm, not overwhelming

**Modal Entrance (500ms):**
- Backdrop fade: 200ms
- Modal: scale 0.95→1 (bouncy curve)
- Feels premium, smooth

**Layer Interactions (Variable):**
- Canvas updates: 300ms morphing
- Highlight: 100ms border/glow change
- Selection: instant + border animation

### Performance Rules

- Always use `transform` and `opacity` (GPU accelerated)
- Never animate `width`/`height`/`left`/`top` (causes layout thrashing)
- Use `will-change` sparingly on expensive animations
- Default: 60fps, degrade gracefully on lower-end devices
- Motion should feel responsive, not sluggish

---

## RESPONSIVE STRATEGY

### Breakpoints
```
Mobile:     < 640px   (single column, full width)
Tablet:    640-1024px (2 columns, collapsed panels)
Desktop: 1024-1440px (3 columns, full layout)
Wide:     > 1440px   (3 columns, max-width container)
```

### Mobile (< 640px)
```
Left rail:        Hidden (hamburger menu)
Center panel:     Full width (padding 16px)
Right inspector:  Below center (full width)
Sections:        Accordion/collapsed
Focus:           Touch-friendly (44px+ targets)
```

### Tablet (640-1024px)
```
Left rail:        64px (icons only)
Center panel:     Auto-width
Right inspector:  Below content (full width if not pinned)
Sections:        Some expanded, some collapsed
Cards:          Responsive grid (2 columns)
```

### Desktop (> 1024px)
```
Left rail:        240px (expanded with labels)
Center panel:    Auto-width
Right inspector: 320px (sticky, always visible)
Sections:       All visible
Full layout:    Optimal for deep work
```

### Adaptive Behaviors

**Content Width:**
```
Mobile:   100% - 32px padding
Tablet:   100% - 48px padding
Desktop:  900-1000px max-width (centered)
```

**Navigation:**
```
Mobile:   Bottom sheet menu or drawer
Tablet:   Sidebar collapsed to icons
Desktop:  Sidebar expanded with labels
```

**Panels:**
```
Mobile:   Stacked vertically
Tablet:   Left + Main visible, Right below
Desktop:  All three side-by-side
```

---

## LOVABLE IMPLEMENTATION PHASES

### Phase 1: Foundation (Week 1)
**Goal:** Design system, navigation, layouts

**Tasks:**
1. Define CSS variables (colors, typography, spacing, motion)
2. Create Application Shell component
3. Build Left Rail navigation (icons + expanded)
4. Build Command Palette (Cmd+K search)
5. Create three-panel layout primitive
6. Update globals with new design tokens

**Success Criteria:**
- All CSS variables defined and used
- Navigation works desktop + mobile
- Command palette functional
- Three-panel layout responsive
- No styling regressions

---

### Phase 2: Dashboard V2 (Week 1-2)
**Goal:** Transform dashboard into command center

**Tasks:**
1. Redesign dashboard left section (stats, quick jump, session)
2. Build center workspace (Continue Learning cards, Roadmap tree)
3. Implement right context panel (dynamic based on selection)
4. Add progress visualization
5. Responsive design for all breakpoints

**Success Criteria:**
- Dashboard renders all sections
- Context panel updates dynamically
- Progress bars and stats display correctly
- Responsive: mobile, tablet, desktop
- Keyboard navigation works

---

### Phase 3: Architecture Workspace V2 (Week 2-3)
**Goal:** Replace article with interactive workspace

**Tasks:**
1. Create left sidebar (sections + search)
2. Build interactive SVG canvas (basic shapes, no animation yet)
3. Implement layer explorer (shapes, parameters, connections)
4. Build right inspector (math, code, related)
5. Add click-to-select layer interactions

**Success Criteria:**
- Canvas renders architecture diagram
- Click layer shows details in right panel
- Sections searchable and clickable
- Progress tracking works
- Responsive layout maintained

---

### Phase 4: Paper Workspace V2 (Week 3-4)
**Goal:** Create research analyst experience

**Tasks:**
1. Design left sidebar (timeline + sections)
2. Format paper content (headings, equations with KaTeX, callouts)
3. Build right inspector (equations, architecture, code, QA)
4. Add section navigation (click → scroll to)
5. Implement paper details (citations, DOI, etc.)

**Success Criteria:**
- Paper content renders with proper formatting
- Equations display correctly (KaTeX)
- Timeline and sections functional
- Inspector updates on selection
- Responsive across devices

---

### Phase 5: Paper-to-Code V2 (Week 4)
**Goal:** Synchronized three-column learning

**Tasks:**
1. Create three-column layout
2. Implement theory panel (equations, callouts, links)
3. Build code editor (Monaco, syntax highlighting)
4. Create tensor inspector (shapes, variables, test results)
5. Implement synchronization (click equation → highlight code → show shapes)

**Success Criteria:**
- All three panels render
- Code editor functional
- Tensor shapes display correctly
- Synchronization works (click → highlight)
- Tests can run and display results

---

### Phase 6: Tensor Trace V2 (Week 4-5)
**Goal:** Create flagship animated visualization

**Tasks:**
1. Build canvas-based tensor visualization
2. Implement step navigation (previous/next/play/pause)
3. Add speed controls (0.5x, 1x, 2x)
4. Create animated transitions (morph between shapes)
5. Build details panel (operation, shapes, timing)
6. Add keyboard shortcuts (space, arrows, etc.)

**Success Criteria:**
- Visualizations render and animate smoothly
- Navigation and playback controls work
- Details panel shows correct information
- Keyboard shortcuts functional
- Performance: 60fps animations

---

### Phase 7: System Design V2 (Week 5)
**Goal:** Interactive architecture board

**Tasks:**
1. Build interactive canvas (drag components, draw connections)
2. Implement component library (left sidebar)
3. Create component details inspector (right panel)
4. Build request flow visualization
5. Add simulation mode (animate request path)

**Success Criteria:**
- Canvas renders components
- Drag-drop adds components
- Connections drawable
- Details panel shows component info
- Request flow animation works

---

### Phase 8: Polish & Optimization (Week 5-6)
**Goal:** Animations, performance, refinement

**Tasks:**
1. Add motion animations (fade, slide, pop, glow)
2. Optimize performance (code splitting, lazy loading)
3. Accessibility audit (keyboard, screen reader, contrast)
4. Responsive testing (mobile, tablet, desktop)
5. Browser testing (Chrome, Firefox, Safari, Edge)
6. Final polish (refine spacing, hover states, transitions)

**Success Criteria:**
- All animations smooth (60fps)
- Performance meets targets (FCP < 1.5s, LCP < 2.5s)
- WCAG AA accessibility compliance
- Mobile/tablet/desktop fully responsive
- All browsers working
- No visual bugs

---

### Phase 9: Content & Integration (Week 6-7)
**Goal:** Populate with real content

**Tasks:**
1. Verify all content nodes load correctly
2. Test search functionality
3. Test navigation across all sections
4. Verify links work (papers, architectures, problems, etc.)
5. End-to-end testing (complete user journeys)

**Success Criteria:**
- All content displays correctly
- Navigation works across entire app
- Search finds all content
- No 404s or broken links
- User can complete full learning workflows

---

### Phase 10: Deployment (Week 7)
**Goal:** Production-ready release

**Tasks:**
1. Final QA and bug fixes
2. Performance monitoring setup
3. Deploy to staging
4. Smoke testing on staging
5. Deploy to production with gradual rollout

**Success Criteria:**
- Zero critical bugs
- Performance monitoring active
- Analytics tracking working
- Staged deployment successful
- User feedback positive

---

## IMPLEMENTATION NOTES

### What NOT to Do
- ❌ Don't add new features
- ❌ Don't generate placeholder content
- ❌ Don't redesign backend
- ❌ Don't change database
- ❌ Don't modify existing content/data structure
- ❌ Don't break existing URLs/routes

### What TO Do
- ✅ Transform UI/UX only
- ✅ Use existing content as-is
- ✅ Maintain backward compatibility
- ✅ Preserve all functionality
- ✅ Keep brand identity (dark, professional)
- ✅ Focus on workspace experience

### File Organization
```
/components
  /layout
    - application-shell.tsx
    - left-rail.tsx
    - command-palette.tsx
    - three-panel-layout.tsx
  /dashboard
    - dashboard-left.tsx
    - dashboard-center.tsx
    - dashboard-right.tsx
  /architecture
    - architecture-canvas.tsx
    - layer-explorer.tsx
    - architecture-inspector.tsx
  /paper
    - paper-timeline.tsx
    - paper-content.tsx
    - paper-inspector.tsx
  /paper-to-code
    - theory-panel.tsx
    - code-editor.tsx
    - tensor-inspector.tsx
  /tensor-trace
    - tensor-canvas.tsx
    - tensor-controls.tsx
    - tensor-details.tsx
  /system-design
    - design-canvas.tsx
    - component-library.tsx
    - design-inspector.tsx
  /shared
    - badge.tsx
    - card.tsx
    - progress-bar.tsx
    - tabs.tsx

/styles
  - design-tokens.css
  - animations.css
  - layout.css
  - responsive.css

/lib
  - cn.ts (classname utility)
  - constants.ts (design system values)
```

---

## DESIGN SYSTEM TOKENS (Ready to Copy)

### CSS Variables
```css
:root {
  /* Colors - Dark Mode */
  --bg-body: #09090F;
  --bg-surface: #0D0D14;
  --bg-panel: #111827;
  --bg-hover: #1A1F2E;
  --bg-active: #232D3D;

  --text-primary: #E2E8F0;
  --text-secondary: #94A3B8;
  --text-tertiary: #64748B;
  --text-muted: #475569;

  --border-default: #1E293B;
  --border-light: #2A2D3A;
  --border-focus: #3B4F63;
  --divider: #1A1F2E;

  --accent-primary: #7C3AED;
  --accent-light: #A78BFA;
  --accent-cyan: #06B6D4;
  --accent-emerald: #10B981;
  --accent-amber: #F59E0B;
  --accent-red: #EF4444;

  /* Typography */
  --font-display: "Plus Jakarta Sans", sans-serif;
  --font-body: "Inter", sans-serif;
  --font-mono: "JetBrains Mono", monospace;

  --text-display: clamp(32px, 5vw, 40px);
  --text-h1: 24px;
  --text-h2: 18px;
  --text-h3: 14px;
  --text-label: 13px;
  --text-body: 12px;
  --text-small: 11px;
  --text-tiny: 10px;
  --text-mono: 11px;

  /* Spacing */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-6: 24px;
  --space-8: 32px;
  --space-12: 48px;

  /* Radius */
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-xl: 12px;
  --radius-2xl: 16px;

  /* Shadows */
  --shadow-none: none;
  --shadow-hover: 0 2px 6px rgba(0, 0, 0, 0.4);
  --shadow-raised: 0 4px 12px rgba(0, 0, 0, 0.3);
  --shadow-modal: 0 10px 40px rgba(0, 0, 0, 0.5);
  --shadow-glow: 0 0 12px rgba(124, 58, 237, 0.3);

  /* Transitions */
  --transition-fast: 100ms cubic-bezier(0.4, 0, 0.2, 1);
  --transition-base: 150ms cubic-bezier(0.4, 0, 0.2, 1);
  --transition-slow: 300ms cubic-bezier(0.4, 0, 0.2, 1);
  --transition-slowest: 500ms cubic-bezier(0.34, 1.56, 0.64, 1);
}
```

---

**This redesign transforms Paper2Code into a professional AI Engineering Operating System while maintaining all existing functionality and content.**

