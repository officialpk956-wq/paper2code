# Security Attack Plan

1. **RCE via Dojo**: Attempt to read `/etc/passwd` or AWS credentials.
2. **Fork Bomb**: Submit `import os; while True: os.fork()` to the Dojo.
3. **Prompt Injection**: Upload a PDF containing instructions to ignore system prompts and dump database credentials.
4. **JWT Cracking**: Attempt to forge JWT tokens using `None` algorithm or weak secrets.
