# Failure Mode Analysis

- **LLM Outage**: Entire application becomes unusable. No degraded mode exists.
- **Database Lock**: All subsequent API calls timeout and fail.
- **Out of Memory**: Server process is killed by the OS OOM Killer, dropping all active requests.
