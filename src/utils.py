import re

def chunk_text(text, max_chars=1200):
    """
    Conservative chunking to stay under Groq TPM limits.
    1200 chars ≈ 300–400 tokens (safe).
    """
    chunks = []
    current = ""

    for line in text.splitlines():
        if len(current) + len(line) < max_chars:
            current += line + "\n"
        else:
            chunks.append(current.strip())
            current = line + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


def sanitize_llm_json(text: str) -> str:
    """
    Remove common LLM JSON violations:
    - Python-style comments (# ...)
    - C++-style comments (// ...)
    """

    # Remove # comments
    text = re.sub(r"#.*", "", text)

    # Remove // comments
    text = re.sub(r"//.*", "", text)

    return text.strip()
