# Paper2Code — Application Shell

Complete Next.js 15 App Router shell for the Paper2Code AI Engineering Learning Ecosystem.

## 📁 Project Structure

```
src/
├── app/
│   ├── layout.tsx              # Root layout with global providers
│   ├── globals.css             # Global styles + design system
│   ├── page.tsx                # Home/landing page
│   ├── learn/page.tsx
│   ├── architectures/page.tsx
│   ├── papers/page.tsx
│   ├── math/page.tsx
│   ├── system-design/page.tsx
│   ├── problems/page.tsx
│   ├── interview/page.tsx
│   ├── playground/page.tsx
│   ├── roadmaps/page.tsx
│   ├── not-found.tsx           # 404 page
│   └── error.tsx               # Error boundary
└── components/
    ├── providers.tsx           # Theme + client providers
    ├── global-nav.tsx          # Desktop navigation bar
    ├── mobile-nav.tsx          # Mobile navigation drawer
    ├── footer.tsx              # Global footer
    ├── command-palette.tsx     # CMD+K command palette
    ├── empty-state.tsx         # Reusable empty state
    ├── skeleton.tsx            # Loading skeletons
    └── error-boundary.tsx      # Error boundary wrapper
```

## 🎨 Design System

### CSS Custom Properties (CSS Variables)

All colors, typography, spacing, and animations are defined as CSS variables in `globals.css`:

```css
--bg-body: #09090F          /* Main background */
--bg-surface: #0D0D14       /* Card/surface background */
--bg-panel: #111827         /* Secondary surface */
--color-border: #1E293B     /* Border color */
--accent-primary: #7C3AED   /* Violet brand color */
--accent-cyan: #06B6D4      /* Cyan accent */
--text-primary: #E2E8F0     /* Primary text */
--text-secondary: #94A3B8   /* Secondary text */
--text-tertiary: #64748B    /* Tertiary text */
```

### Typography Scale

- `--text-display`: 28-36px (clamp) - Hero headlines
- `--text-h1`: 22px - Page titles
- `--text-h2`: 16px - Section headers
- `--text-h3`: 13px - Card titles
- `--text-base`: 12px - Body text
- `--text-sm`: 11px - Captions
- `--text-xs`: 10px - Labels
- `--font-heading`: Plus Jakarta Sans
- `--font-sans`: Inter
- `--font-mono`: JetBrains Mono

### Component Classes

Reusable utility classes defined in `globals.css`:

- `.btn-primary` - Primary CTA button
- `.btn-secondary` - Secondary button
- `.btn-ghost` - Borderless button
- `.card` - Card container
- `.input-field` - Form input
- `.badge-easy`, `.badge-medium`, `.badge-hard`, `.badge-expert` - Difficulty badges
- `.page-container` - Max-width page wrapper
- `.section-header` - Section title + description
- `.glass-effect` - Frosted glass effect
- `.gradient-text` - Gradient text effect
- `.nav-item` - Navigation item

## 🧭 Navigation Structure

### Global Navigation (Desktop)
- Sticky top bar with 9 main sections
- Search bar (⌘K)
- User profile menu with dropdown
- Notifications bell
- Search + Settings shortcuts

### Mobile Navigation
- Collapsible hamburger menu
- Full-screen navigation drawer
- Quick search button
- Responsive on screens < 768px

### Routes

```
/                    Home/Landing
/learn               Learning modules
/architectures       Architecture explorer
/papers              Research papers
/math                Mathematics section
/system-design       System design cases
/problems            Coding problems
/interview           Interview prep
/playground          Interactive tools
/roadmaps            Learning tracks
```

## 🎯 Key Components

### GlobalNav (`components/global-nav.tsx`)
- Sticky navigation bar
- 9 nav items with active state
- Search button triggering command palette
- Profile dropdown with sign-out
- Responsive hiding on mobile

### MobileNav (`components/mobile-nav.tsx`)
- Hamburger menu toggle
- Slide-out navigation drawer
- Quick search integration
- Icon-based nav items for mobile
- Hidden on desktop (md breakpoint)

### CommandPalette (`components/command-palette.tsx`)
- Keyboard shortcut: ⌘K (Cmd+K or Ctrl+K)
- Filterable command list
- Grouped by category
- Navigation + actions
- Escape to close

### Footer (`components/footer.tsx`)
- Branded section links
- Social links (GitHub, Twitter, Email)
- Privacy / Terms / Status links
- Responsive 4-column layout

### Skeletons (`components/skeleton.tsx`)
- Shimmer animation
- PageSkeleton - Full page placeholder
- CardSkeleton - Card placeholder
- ListSkeleton - List item placeholder

### ErrorBoundary (`components/error-boundary.tsx`)
- Catches client-side errors
- Refresh/retry button
- Graceful error display

### EmptyState (`components/empty-state.tsx`)
- Icon + title + description
- Optional action button
- Reusable across sections

## 🎬 Getting Started

### Installation

```bash
npm install
# or
yarn install
# or
pnpm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

### Type Check

```bash
npm run type-check
```

### Formatting

```bash
npm run format
npm run format:check
```

## 🎨 Color Palette

### Semantic Colors
- **Easy**: `#10B981` (Emerald)
- **Medium**: `#F59E0B` (Amber)
- **Hard**: `#EF4444` (Red)
- **Expert**: `#EC4899` (Pink)
- **Roadmap**: `#F97316` (Orange)

### Brand Gradient
- From: `#7C3AED` (Violet)
- To: `#06B6D4` (Cyan)

## 📱 Responsive Breakpoints

Using Tailwind's default breakpoints:
- `sm`: 640px
- `md`: 768px (Desktop nav shows, mobile nav hides)
- `lg`: 1024px
- `xl`: 1280px
- `2xl`: 1536px

**Desktop Nav** shown at `md` and up.
**Mobile Nav** shown at `md` and below.

## 🎭 Animations

### Built-in Animations
- `animate-in`, `fade-in-0`, `slide-in-from-top-2` - Page transitions
- `animate-shimmer` - Loading skeleton effect
- `animate-aurora` - Aurora background gradient
- `transition-colors`, `transition-opacity` - Hover effects

### Motion Durations
- `--transition-fast`: 100ms
- `--transition-base`: 150ms (default)
- `--transition-slow`: 300ms

## 🔒 Type Safety

Full TypeScript strict mode enabled:
- No implicit `any`
- Strict null checks
- All functions typed

## 🛠️ Tech Stack

- **Framework**: Next.js 15 App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS 3
- **UI Icons**: Lucide React
- **Animations**: Framer Motion
- **Theme**: next-themes
- **Linting**: ESLint
- **Formatting**: Prettier

## 📋 Next Steps

1. **Install dependencies**: `npm install`
2. **Start dev server**: `npm run dev`
3. **Implement section pages** in each route (replace PageSkeleton)
4. **Add page-specific components** under `src/components/[section]/`
5. **Implement content fetching** with TanStack Query
6. **Build feature-specific layouts** (e.g., split panes for Problems)
7. **Add animations** with Framer Motion

## 🎓 Example: Adding a Feature Page

Create `src/app/problems/[id]/page.tsx`:

```typescript
import { PageSkeleton } from "@/components/skeleton";

export default function ProblemPage({ params }: { params: { id: string } }) {
  return (
    <div className="page-container">
      <div className="section-header">
        <h1>Problem #{params.id}</h1>
        <p>Scaled Dot-Product Attention</p>
      </div>

      {/* Replace with actual problem content */}
      <PageSkeleton />
    </div>
  );
}
```

## 🚀 Performance

- **Fast nav**: Sticky positioning, no re-renders
- **Code splitting**: Each route lazy-loaded
- **Image optimization**: Next.js Image component ready
- **CSS**: Utility-first, minimal bundle size
- **JS**: ~50KB gzipped (with dependencies)

## 📄 License

MIT
