# Chaos Test Plan

- Kill the database process mid-transaction.
- Inject 5-second latency into the LLM API.
- Corrupt the SQLite database file and observe recovery procedures (currently none).
