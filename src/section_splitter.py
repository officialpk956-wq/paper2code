# src/section_splitter.py

import json
from pathlib import Path
from collections import defaultdict

from src.llm_client import classify_section
from src.utils import chunk_text


TEXT_DIR = Path("outputs/texts")
OUT_DIR = Path("outputs/sections")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_parse_llm_output(raw_output: str, fallback_text: str):
    """
    Safely parse LLM output.
    Handles:
    - dict JSON
    - list of dicts
    - malformed / extra text
    """
    try:
        parsed = json.loads(raw_output)

        # Case 1: Proper dict
        if isinstance(parsed, dict):
            return (
                parsed.get("section", "other"),
                parsed.get("content", fallback_text),
            )

        # Case 2: List of dicts
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return (
                parsed[0].get("section", "other"),
                parsed[0].get("content", fallback_text),
            )

    except Exception:
        pass

    # Fallback
    return "other", fallback_text


def process_file(txt_path: Path):
    print(f"Processing paper: {txt_path.name}")

    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)

    section_store = defaultdict(list)

    for i, chunk in enumerate(chunks, start=1):
        print(f"  Chunk {i}/{len(chunks)}")

        try:
            raw_result = classify_section(chunk)
            section, content = safe_parse_llm_output(raw_result, chunk)
        except Exception as e:
            print("  ⚠️ LLM failure, falling back to 'other':", e)
            section, content = "other", chunk

        section_store[section].append(content)

    # Merge chunks per section
    final_sections = {
        section: "\n\n".join(contents).strip()
        for section, contents in section_store.items()
    }

    out_file = OUT_DIR / f"{txt_path.stem}.json"
    out_file.write_text(
        json.dumps(final_sections, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"  Saved → {out_file}")


def main():
    txt_files = sorted(TEXT_DIR.glob("*.txt"))

    if not txt_files:
        print("No text files found in outputs/texts/")
        return

    for txt in txt_files:
        process_file(txt)

    print("Section splitting complete.")


if __name__ == "__main__":
    main()
