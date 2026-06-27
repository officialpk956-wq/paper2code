# Research Pipeline Audit
**Agent 7 — Research Pipeline Auditor**

## Pipeline Correctness
- **PDF Ingestion**: High risk of memory bloat. A 50MB PDF parsed in memory will consume ~500MB of RAM during extraction.
- **Determinism**: Zero. The pipelines rely on LLM outputs to generate schemas (`generate_code_ready_schema.py`) without strict JSON-schema enforcement or validation retry loops (e.g., using Instructor/Pydantic validation loops).
- **Edge Cases**: Papers with non-standard dual-column layouts or heavy mathematical formulas (LaTeX) will break the current regex/splitters.

## Risk Classification
- **[P1] Hallucinated Architectures**: The LLM may hallucinate tensor shapes. There is no rigorous mathematical validation step confirming that the generated shapes conform to the paper's equations.
