# PERFORMANCE COUNCIL REPORT
**Compiled by Agents P1-P7**

## [P6] Async Processing Auditor Findings
- **Evidence**: `core/paper_to_code_generator.py` executes synchronous HTTP requests to the OpenAI API inside FastAPI route handlers.
- **Risk**: P0. Since Uvicorn uses a limited thread pool for synchronous endpoints, 40 concurrent LLM requests will permanently stall the web server, making the entire platform unresponsive.
- **Fix Recommendation**: Offload all AI generation tasks to a Celery worker pool backed by Redis. (Effort: 2 weeks)
