#!/usr/bin/env python3
import pdfplumber
import re
from pathlib import Path

DATA_DIR = Path("data")

# ------------------------------------------------------------------
# Auto-detect PDF
# ------------------------------------------------------------------
pdf_files = list(DATA_DIR.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError("❌ No PDF found in data/ folder")

pdf_path = pdf_files[0]
txt_path = DATA_DIR / f"{pdf_path.stem}_clean.txt"

print(f"📘 Using PDF: {pdf_path.name}")

# ------------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
      - Fix hyphenation across line breaks
      - Remove line breaks inside paragraphs
      - Preserve paragraph boundaries
      - Remove page numbers / numeric-only lines
      - Normalize spacing
    """

    # Fix hyphenated line breaks FIRST
    text = re.sub(r'-\n+', '', text)

    # Mark paragraph breaks
    text = re.sub(r'\n{2,}', '\u2029', text)

    # Remove single newlines (inside paragraphs)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Restore paragraph breaks
    text = text.replace('\u2029', '\n\n')

    # Remove numeric-only lines (page numbers)
    text = '\n'.join(
        line for line in text.splitlines()
        if not line.strip().isdigit()
    )

    # Collapse extra spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()

# ------------------------------------------------------------------
# PDF extraction
# ------------------------------------------------------------------
all_text = []

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text:
            all_text.append(text)

raw_text = "\n\n".join(all_text)
cleaned_text = clean_text(raw_text)

# ------------------------------------------------------------------
# Save output
# ------------------------------------------------------------------
txt_path.write_text(cleaned_text, encoding="utf-8")

print(f"✅ Cleaned text saved → {txt_path}")
print(f"📄 Characters: {len(cleaned_text):,}")