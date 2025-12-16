import os
from pathlib import Path

PDF_DIR = Path(r"I:\papper2code\data\pdfs")
OUT_DIR = Path(r"I:\papper2code\outputs\texts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_with_pdfplumber(path):
    try:
        import pdfplumber
        text = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text.append(t)
        return "\n".join(text)
    except Exception:
        return None

def extract_with_fitz(path):
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    except Exception:
        return None

for pdf in sorted(PDF_DIR.glob("*.pdf")):
    print("Processing:", pdf.name)
    text = extract_with_pdfplumber(str(pdf))
    if not text:
        print("  pdfplumber failed, trying PyMuPDF...")
        text = extract_with_fitz(str(pdf))

    if not text:
        print("  ERROR: Failed to extract text from", pdf.name)
        continue

    out_file = OUT_DIR / f"{pdf.stem}.txt"
    out_file.write_text(text, encoding="utf-8")
    print("  Saved →", out_file)

print("Extraction complete.")
