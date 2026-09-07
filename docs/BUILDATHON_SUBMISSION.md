# paper2code — Razorpay AI Buildathon submission brief

## Track

**Open Track.**

## One-line pitch

paper2code turns a machine-learning research paper into an evidence-backed,
runnable PyTorch implementation, and clearly shows what was cited, inferred,
or could not be verified.

## Problem

Reproducing an ML paper is slow and unreliable. A reader has to find the
implementation-relevant sections, resolve terminology, infer missing details,
write code, and discover broken assumptions only after running it. Existing
chat-based code generation often hides its uncertainty and gives no source for
an architectural claim.

## What is working today

1. A user uploads a PDF and accepts the paper-processing terms.
2. The backend preserves page-aware text chunks and retrieves implementation
   context with hybrid dense + BM25 + architecture-family ranking.
3. Extraction produces a structured architecture specification.
4. Each field is marked `cited`, `inferred`, or `default`. A cited field must
   have an LLM-proposed quote that is verified against a real persisted paper
   chunk; invented quotes are not accepted.
5. Known architecture families generate self-contained PyTorch source.
6. Code is validated through import, construction, and a real forward pass;
   failures receive at most two repair attempts after the first generation.
7. The workspace exposes generated source, validation result, and page-level
   extraction evidence.

## Five-minute demo script

Use a supported paper with a prepared, legal PDF: **ResNet-50**, **U-Net**, or
**Vision Transformer**. Do not make a universal claim during the demo.

### 0:00–0:35 — Problem

"A paper can say it uses residual bottleneck blocks, but reproducing the exact
architecture normally means manually searching pages, translating the design
into code, and debugging it. paper2code makes that process inspectable."

### 0:35–1:20 — Upload and retrieval

Upload the PDF. Show the asynchronous stages. Explain that the system keeps
page/section chunks, ranks implementation-relevant evidence with dense search,
BM25, and architecture-aware relationships, then passes only focused context
to extraction.

### 1:20–2:25 — Evidence, not magic

Open **Knowledge Graph → Extraction evidence**. Show one field with its value,
"Cited (page N)", and supporting quote. Then show one `Inferred` field and say
it is deliberately not presented as paper fact.

### 2:25–3:35 — Code and verification

Open **Executable**. Show the self-contained model source, entrypoint, input
shape, output shape, and the successful forward-pass validation. Run it once
in the sandbox if the environment is warm.

### 3:35–4:20 — Reliability boundary

Show either a saved `needs_review` result or state this plainly:
"Unsupported or ambiguous papers are not force-marked successful. The system
keeps the diagnostic and stops after a bounded repair budget."

### 4:20–5:00 — Value

"The outcome is not merely generated code. It is runnable code plus an audit
trail of what the paper supports, what the system inferred, and what failed."

## Metrics to record before recording the final video

Record measured values only; leave a metric out if it has not been measured.

| Metric | How to measure | Target for the submission |
| --- | --- | --- |
| Supported-paper execution rate | Successful forward-pass validations / attempted supported papers | 3/3 curated demos |
| Evidence coverage | Cited fields / all extracted fields, reported beside inferred/default | Show the real percentage for each demo |
| Honest uncertainty | Inferred + default fields displayed rather than fabricated citations | 100% of fields have a status |
| Repair transparency | Attempts and final diagnostic retained for a failure case | One visible example |
| Latency | Upload submission to completed task | Record locally for each demo; do not promise a fixed SLA |

## Truthful scope statement

paper2code currently targets supported architecture families and makes
uncertainty visible. It does **not** guarantee a paper-faithful implementation
for arbitrary research papers, and scanned/OCR-heavy PDFs remain a known weak
case.

## Submission package checklist

- [ ] Public repository with secrets removed and a clear local setup section.
- [ ] A 60-second README quick-start for one supported PDF.
- [ ] Three legally usable demo PDFs and a recorded result for each.
- [ ] Five-minute uncut screen recording following the script above.
- [ ] Architecture diagram showing PDF → hybrid retrieval → evidence → code → validation/repair.
- [ ] One `needs_review` example to demonstrate honest failure handling.
- [ ] Exact metrics from the table above; no invented benchmark claims.

## September 5 priority order

1. Stabilize the three curated supported-paper demos.
2. Record evidence coverage, execution result, and latency for each demo.
3. Capture the five-minute video.
4. Run type-check, frontend tests, backend non-live tests, and production build.
5. Remove secrets, publish only after the user reviews the diff, then submit.
