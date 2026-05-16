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


def process_text(text: str) -> dict:
    """
    Process raw text string through section classifier.
    Returns {section_name: merged_content}.
    No file I/O — accepts string input for PDF extraction (F4).

    Args:
        text: Raw text string to process

    Returns:
        Dict mapping section names to merged content
    """
    chunks = chunk_text(text)
    section_store = defaultdict(list)

    for chunk in chunks:
        try:
            raw_result = classify_section(chunk)
            section, content = safe_parse_llm_output(raw_result, chunk)
        except Exception:
            section, content = "other", chunk

        section_store[section].append(content)

    # Merge chunks per section
    return {
        section: "\n\n".join(contents).strip()
        for section, contents in section_store.items()
    }


def process_file(txt_path: Path):
    """
    Process file-based text through section classifier.
    Refactored to delegate to process_text().
    """
    print(f"Processing paper: {txt_path.name}")

    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    final_sections = process_text(text)

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
