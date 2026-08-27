# Build-in-Public Posts — Paper2Code

Working log for the PM-narrative content push. Every post below is either **READY**
(every factual claim is either personal reasoning or a number verified against the
live codebase/DB) or **ON HOLD** (needs real production usage data we don't have yet
— see `docs/BUILD_IN_PUBLIC_METRICS_TODO.md`... actually not created; tracked inline below).

Rule for this whole log: no invented percentages, no fabricated funnels, no "a user told
me" quotes that didn't happen. If a number appears in a post, it was queried, not guessed.

---

## READY — Post 1: Why I built this (Week 1, thesis)

**Format:** short caption + the 4-slide carousel (`post1_deck.pptx`). The slides carry
the detail (causal-chain problem framing, JTBD, comparison table, the loop, the
build/buy/skip call on the tutor, the real coverage-gap numbers, the north-star metric
+ its supporting-metrics tree + P0/P1 roadmap) — the caption just has to earn the swipe.

> Most ML learning tools measure whether you watched the video. Not whether you can
> build the thing.
>
> I've been building Paper2Code around one bet: a learning product should measure
> capability, not comprehension. The core loop — upload a paper → practice against it
> in a versioned Dojo → get unstuck with a tutor grounded in that paper, not a generic
> chatbot.
>
> Where it actually stands today, no padding: 49 Dojo problems, 91 architecture
> write-ups in the reference library — and 0% of those problems are expert-tier. Only
> 8 of the 49 are even linked to a reference page yet. Real gaps, reported honestly,
> not smoothed over for the post.
>
> The metric I'm sequencing everything against: the % of users who go from reading a
> paper to submitting a related problem solution in the same week. I don't have that
> number yet — the feature that produces it doesn't exist yet either. That's exactly
> why it's next on the roadmap, not something further out.
>
> Swipe through for the full breakdown: the problem, the loop, where it stands today,
> and the north star.
>
> PM angle: define your north star before you have the pipeline to measure it. It's
> what tells you what to build next.
>
> #BuildInPublic #ProductManagement

**Grounded in:** real problem count (49), real category breakdown (queried from
`problems` table, 2026-08-18), real architecture content count (`src/content/architectures`,
91 dirs). No usage/retention claims — those don't exist yet.

---

## READY — Post 2: What I'm not building (Week 1-2, scope)

**Format:** caption below + 4-slide carousel (`post2_deck.pptx`). Voice matches the
elevated/unusual-vocabulary style of the Post 1 caption as actually published (see
`post1_deck.pdf` the user shared back) — thesaurus-forward, ↳/• structural bullets,
"Product takeaway:" close instead of "PM angle:".

> Every roadmap draft for Paper2Code sprouts the same appendage: follows, comments,
> a discussion thread bolted under each problem. I keep amputating it.
>
> I've been guarding this product against a single trap: mistaking borrowed
> retention machinery for genuine product logic.
>
> The filter I run every feature candidate through:
> ↳ Does this reinforce the primary circuit — learn, practice, measure, improve?
> ↳ Or does it siphon attention into a rival loop wearing a friendly disguise?
> ↳ Comment threads and follow graphs are retention machinery lifted wholesale from
>   social platforms. They optimize for minutes spent reacting to strangers, not
>   minutes spent building competence.
>
> What survives the cut, and what doesn't:
> • Leaderboard — retained. A byproduct of the practice circuit, earned through
>   solved problems, never manufactured for its own sake.
> • Comments/follows — severed. A substitute activity dressed up as a byproduct.
>
> Not a permanent verdict. A scoped-out decision with a named reversal trigger,
> not a vague "maybe someday."
>
> The signal that flips this call: hard evidence that users are churning
> specifically because they have no outlet to discuss a stuck problem. That
> telemetry doesn't exist yet. Until it does, this stays a hypothesis I refuse
> to build for on a hunch.
>
> Flip through the attached breakdown for the full audit: the filter, what
> survived, what got amputated, and the exact condition that reverses this call.
>
> Product takeaway: scope discipline isn't "no" forever. It's "no, and here is
> the precise evidence that flips it to yes."
>
> #BuildInPublic #ProductManagement

**Grounded in:** actual codebase state (no follow/comment models exist — verified,
not assumed) + product reasoning. No fabricated A/B test numbers — the earlier draft
of this post claimed a "14% lower retention" test that never ran; removed. The
"telemetry doesn't exist yet" line matches the confirmed PostHog gap from Post 1
(API key unset, events silently no-op'ing).

---

## READY — Post 3: The gap I'm closing next (Week 2, roadmap)

> **The gap:** Paper2Code runs two libraries that don't talk to each other yet.
> - **Reference library:** 91 architecture write-ups, 25 paper deep-dives, 9
>   implementation walkthroughs, 12 system-design case studies.
> - **Practice library:** 49 Dojo problems — 13 easy, 26 medium, 10 hard, **0
>   expert-tier**.
>
> Only 8 of those 49 problems have a paired reference page. Read about Vision
> Transformers, and there's no path from that page to a problem that exercises it.
> Verified against the actual content directories, not eyeballed.
>
> **How I'm prioritizing the fix** (impact vs. effort, cheapest first):
> 1. **Map existing problems → reference pages** (low effort, closes the gap for
>    content that already exists on both sides — no new content needed).
> 2. **Add expert-tier problems** (medium effort, gives advanced users a ceiling
>    instead of a wall at "medium").
> 3. **Auto-suggest problems on paper upload** (higher effort — needs the
>    knowledge-graph tags to drive a recommendation, plus new click-tracking that
>    doesn't exist yet — but it's the one that actually closes the loop from Post 1).
>
> **What I'm explicitly not doing:** posting a fake "click-through went from 0% to
> 85%" number for a feature that isn't built. I'll post the real number once it
> ships and has a week of traffic — good or bad.
>
> **PM angle:** sequence roadmap items by what's cheapest to ship *given what
> already exists*, not by what sounds most impressive to announce.

**Grounded in:** live query of `problems` table (difficulty split), directory count
of `src/content/architectures|papers|implementations|system-design`, and manual check
of `src/content/problems` (8 entries) against the 49-problem Dojo catalog. The 0%→85%
CTR claim from the original doc was explicitly removed — no click tracking exists yet
for this feature (confirmed: no analytics-event table in `backend/models.py`).

---

## READY — Post 4: The bug where the product lied about its own confidence (Week 2, honesty/postmortem)

> Found a bug this week that worried me more than a crash would have.
>
> The paper-to-code pipeline reconstructs an architecture from a paper and scores how
> confident it is in the result. Turns out that score doesn't check what I assumed it
> checked. It has real templates for some architectures — Transformer, ResNet, and
> others — but not for YOLO, MobileNet, LeNet, or DeepLabV3+. When a paper introduces
> one of those, the pipeline quietly falls back to a generic 3-block placeholder that
> has nothing to do with the real network. The confidence score never finds out. It
> still reports it as ~90% confident.
>
> A crash tells you something's wrong. A wrong answer delivered at 90% confidence
> teaches you to trust something you shouldn't have.
>
> The fix isn't complicated: tie the confidence score to whether a real template
> actually matched, not just whether the architecture's name got detected. Until
> that ships, a generic-template fallback needs to look visibly different in the UI
> from a real match — not wear the same badge.
>
> PM angle: a tool that looks equally confident whether it's right or wrong is worse
> than one that visibly fails. Users can work around "I don't know." They can't work
> around "I'm sure" when you're the one who's wrong.

**Grounded in:** reproduced directly against the live code — a paper tagged `YOLO`
returned 3 generic components at a 90% confidence score. Confirmed by reading
`architecture_reconstruction_service.py` (template registry has no entry for
YOLO/MobileNet/LeNet/DeepLabV3+) and running the reconstruction path, not inferred.

---

## READY — Post 5: Why the Dojo's code runner is down right now (Week 2, honest ops update)

> The Dojo's "submit code" button is broken in production right now, and I'm not
> hiding that.
>
> I self-host Piston, the engine that safely runs submitted code. It won't start on
> Render's free tier. Piston needs to create a folder to isolate and run code safely
> — the free tier gives you a read-only filesystem, so it crashes on startup. It also
> needs deeper container permissions to sandbox code properly, and free hosting
> blocks that for security reasons.
>
> The real fix is a paid tier with a persistent disk, or different hosting entirely.
> As a student building this solo with no funding, that's a real cost decision, not
> a five-minute patch — and I'd rather say that plainly than quietly leave a broken
> button and hope no one notices.
>
> PM angle: not every gap gets closed with more engineering time. Some gaps get
> closed with money you don't have yet, and pretending otherwise just burns the
> engineering time you do have on the wrong problem.

**Grounded in:** actual Render deployment logs — `mkdir: cannot create directory
'isolate/': Read-only file system` on container startup. Root cause (read-only
filesystem + stripped container capabilities on free-tier PaaS) confirmed from the
real error, not guessed.

---

## ON HOLD — needs real production data before drafting

These stay unwritten until there's real usage to point to. Production DB is on Render;
this session only has access to a local dev/fixture SQLite DB (12 seeded test users,
2 real submissions) — not campaign-worthy, and using it would just be fabrication with
extra steps.

| Post | Needs | Blocked on |
|---|---|---|
| Activation funnel (signup → solve → 2nd submission → streak) | Real signup/submission volume | Prod DB access or organic traffic |
| Retention playbook (streaks vs. churn) | Cohort retention by `User.streak` over time | Same |
| "Why I cut/kept the allowlist" | Waitlist conversion numbers | Real waitlist data |
| Feature prioritization (Dojo vs Labs vs Tutor engagement) | Per-feature usage/retention | PostHog events (currently no-op — `POSTHOG_API_KEY` unset) |
| Paper→problem bridge lift | Before/after retention by activation path | Feature doesn't exist yet (see Post 3 above) + no event tracking |
| Acceptance-rate calibration | Real submission volume per problem | Currently 2 submissions total in dev DB; `acceptance_rate` column exists but is meaningless at this volume |
| Learner win / social proof | An actual learner milestone | Real users, real usage |
| Postmortem ("shipped X, users ignored it") | Real CTR before/after a fix | Same tracking gap as above |

To unblock this whole column: either (a) get Render Postgres read access so real
(if small) numbers can be queried, or (b) set `POSTHOG_API_KEY` and instrument
dojo-submission / streak / CTR events, then wait for real traffic to accumulate.

---

## Queued material — real bugs found, not yet turned into a post

Found during the pipeline/auth audit (2026-08-25), not used in Posts 4-5. Save for
Week 3+ so we're not dumping every finding into one week.

| Finding | Angle it could support |
|---|---|
| Retry on paper ingestion isn't idempotent — a failed-then-retried upload can leave an orphaned duplicate `Paper` row | "Reliability isn't a feature, it's what happens when the happy path fails" |
| Signup/reset-password hash Argon2 inline inside `async def` routes, so concurrent signups serialize on the event loop instead of running in parallel | Pairs well with a real concurrency/latency number once there's signup traffic to measure it against |
| Local/no-R2 deployments leak temp PDFs on successful ingestion (cleanup only runs on failure) | Minor — probably a footnote in a larger "what I fixed this week" round-up, not its own post |
| Shape-inference failures in the graph compiler are silently swallowed (`except Exception: pass`, no logging) | Same round-up post as above |
