#!/usr/bin/env python3
"""
pdf_to_txt.py

Robust PDF → clean text extractor for the Thought-to-Structure NLP pipeline.

Handles:
  - Multiple PDFs in the data/ folder (processes all, not just the first)
  - Scanned PDFs with no text layer → OCR fallback via pytesseract
  - Multi-column layouts → column-aware extraction via pdfplumber bounding boxes
  - Tables → extracted as plain text rows, not garbled linear output
  - PDF metadata → title, author, page count saved alongside the text
  - TOC / cover / header noise → filtered by heuristic line classifier
  - Hyphenation, page numbers, URLs → cleaned automatically

Dependencies:
  pip install pdfplumber pytesseract Pillow
  sudo apt install tesseract-ocr   # or brew install tesseract on macOS
"""

import json
import re
import sys
import warnings
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

# Optional: OCR for scanned pages
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    warnings.warn(
        "pytesseract / Pillow not installed — scanned pages will be skipped. "
        "Install with: pip install pytesseract Pillow && sudo apt install tesseract-ocr"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR        = Path("data")
SKIP_FIRST_PAGE = True          # skip cover / TOC page
OCR_DPI         = 300           # resolution for rasterising scanned pages
MIN_TEXT_CHARS  = 20            # fewer chars than this → page is probably scanned
# Column detection: if the page has two horizontal bands of text separated by a
# gap wider than this fraction of page width, treat it as two-column.
COLUMN_GAP_RATIO = 0.10


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
def extract_metadata(pdf: pdfplumber.PDF, pdf_path: Path) -> dict:
    """
    Pull title, author, page count and other info from PDF metadata.
    Falls back to filename-derived title if metadata is absent.
    """
    raw_meta = pdf.metadata or {}

    def _clean(val):
        if not val:
            return None
        # Strip BOM / encoding artifacts common in older PDFs
        return str(val).strip().lstrip("\ufeff").strip() or None

    title  = _clean(raw_meta.get("Title"))  or pdf_path.stem.replace("_", " ").title()
    author = _clean(raw_meta.get("Author")) or "Unknown"

    return {
        "filename":   pdf_path.name,
        "title":      title,
        "author":     author,
        "page_count": len(pdf.pages),
        "subject":    _clean(raw_meta.get("Subject")),
        "creator":    _clean(raw_meta.get("Creator")),
    }


# ---------------------------------------------------------------------------
# Column-aware text extraction
# ---------------------------------------------------------------------------
def _extract_columns(page: Page) -> str:
    """
    Detect and handle multi-column layouts.

    Strategy:
      1. Extract all word bounding boxes from the page.
      2. Find horizontal gaps in the x-distribution wider than COLUMN_GAP_RATIO
         of page width — these are column dividers.
      3. Sort words into columns by their x position, then within each column
         by y (top-to-bottom) to reconstruct reading order.
      4. Fall back to standard extract_text() if no columns detected.
    """
    words = page.extract_words()
    if not words:
        return ""

    page_width = page.width
    gap_threshold = page_width * COLUMN_GAP_RATIO

    # Find x-midpoints of all words
    x_mids = sorted(w["x0"] for w in words)

    # Detect gaps between consecutive x positions
    column_splits = [0.0]
    for i in range(1, len(x_mids)):
        if x_mids[i] - x_mids[i - 1] > gap_threshold:
            split_x = (x_mids[i - 1] + x_mids[i]) / 2
            # Only record a split if it's meaningfully different from the last one
            if split_x - column_splits[-1] > gap_threshold:
                column_splits.append(split_x)
    column_splits.append(page_width + 1)

    if len(column_splits) <= 2:
        # Single column — use standard extraction (better whitespace handling)
        return page.extract_text() or ""

    # Multi-column: assign each word to its column bucket
    num_cols = len(column_splits) - 1
    columns: list[list[dict]] = [[] for _ in range(num_cols)]

    for word in words:
        x_mid = (word["x0"] + word["x1"]) / 2
        for col_idx in range(num_cols):
            if column_splits[col_idx] <= x_mid < column_splits[col_idx + 1]:
                columns[col_idx].append(word)
                break

    # Sort each column top-to-bottom, left-to-right within the same line
    col_texts = []
    for col in columns:
        if not col:
            continue
        sorted_words = sorted(col, key=lambda w: (round(w["top"] / 5) * 5, w["x0"]))
        # Group into lines by proximity of top coordinate
        lines, current_line, last_top = [], [], None
        for word in sorted_words:
            if last_top is None or abs(word["top"] - last_top) < 8:
                current_line.append(word["text"])
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word["text"]]
            last_top = word["top"]
        if current_line:
            lines.append(" ".join(current_line))
        col_texts.append("\n".join(lines))

    return "\n\n".join(col_texts)


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------
def _extract_tables(page: Page) -> list[str]:
    """
    Extract tables from a page as plain-text rows.

    pdfplumber detects table regions automatically. Each cell is joined with
    a tab, each row with a newline, and an empty line separates tables so
    Phase 1 treats them as separate paragraphs.
    """
    tables = page.extract_tables()
    if not tables:
        return []

    result = []
    for table in tables:
        rows = []
        for row in table:
            # Replace None cells with empty string
            cells = [str(cell).strip() if cell else "" for cell in row]
            rows.append("\t".join(cells))
        result.append("\n".join(rows))
    return result


# ---------------------------------------------------------------------------
# OCR fallback for scanned pages
# ---------------------------------------------------------------------------
def _ocr_page(page: Page) -> str:
    """
    Rasterise a page and run Tesseract OCR on it.
    Only called when pdfplumber extracts fewer than MIN_TEXT_CHARS characters,
    which indicates a scanned/image-only page with no text layer.
    """
    if not OCR_AVAILABLE:
        warnings.warn(f"Page appears scanned but pytesseract is not installed — skipping.")
        return ""

    try:
        img = page.to_image(resolution=OCR_DPI).original
        text = pytesseract.image_to_string(img, lang="eng")
        return text
    except Exception as e:
        warnings.warn(f"OCR failed on page: {e}")
        return ""


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
      - Fix hyphenation across line breaks
      - Remove line breaks inside paragraphs (preserve paragraph boundaries)
      - Remove page numbers, URLs, TOC lines, all-caps header artifacts
      - Normalize spacing and Unicode
    """
    # Normalize unicode whitespace
    text = text.replace("\u00a0", " ").replace("\u2028", "\n")

    # Fix hyphenated line breaks FIRST (before any newline processing)
    text = re.sub(r'-\n+', '', text)

    # Mark paragraph breaks (2+ newlines)
    text = re.sub(r'\n{2,}', '\u2029', text)

    # Remove single newlines within paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Restore paragraph breaks
    text = text.replace('\u2029', '\n\n')

    def _is_noise_line(line: str) -> bool:
        """Return True if this line is PDF noise that should be removed."""
        stripped = line.strip()
        if not stripped:
            return False
        # Page numbers (digits only, optionally with spaces)
        if re.fullmatch(r'[\d\s]+', stripped):
            return True
        # URLs and web headers
        if re.match(r'https?://|www\.', stripped):
            return True
        # TOC artifact: spaced-out letters ("T C ABLE OF ONTENTS")
        if re.search(r'\bOF\s+ONTENTS\b|\bTABLE\s+OF\s+CONTENTS\b', stripped, re.IGNORECASE):
            return True
        # Running headers/footers: short all-caps lines (≤6 words, no punctuation)
        words = stripped.split()
        if 1 <= len(words) <= 6:
            if all(w.isupper() or w.isdigit() for w in words):
                if not re.search(r'[.!?,;:\"\']', stripped):
                    return True
        # TOC-style lines: 4+ title-cased words, no punctuation, no lowercase connectors
        if len(words) >= 4:
            title_or_caps   = sum(1 for w in words if w.istitle() or w.isupper())
            has_punctuation = bool(re.search(r'[.!?,;:\"\']', stripped))
            lowercase_connectors = sum(1 for w in words if w.islower() and len(w) > 2)
            if (title_or_caps / len(words) >= 0.85
                    and not has_punctuation
                    and lowercase_connectors == 0):
                return True
        return False

    # Filter noise lines
    text = '\n'.join(line for line in text.splitlines() if not _is_noise_line(line))

    # Collapse multiple spaces / tabs
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Remove lines that are just punctuation or a single character
    text = '\n'.join(
        line for line in text.splitlines()
        if len(line.strip()) > 1 or line.strip().isalpha()
    )

    return text.strip()


# ---------------------------------------------------------------------------
# Single PDF processor
# ---------------------------------------------------------------------------
def process_pdf(pdf_path: Path) -> tuple[str, dict]:
    """
    Extract and clean all text from a single PDF file.

    Returns:
        cleaned_text : str   — full cleaned text ready for Phase 1
        metadata     : dict  — title, author, page_count, etc.
    """
    print(f"\n📘 Processing: {pdf_path.name}")
    all_text_parts = []
    scanned_pages  = []
    table_parts    = []

    with pdfplumber.open(pdf_path) as pdf:
        metadata   = extract_metadata(pdf, pdf_path)
        total      = len(pdf.pages)

        print(f"   Title  : {metadata['title']}")
        print(f"   Author : {metadata['author']}")
        print(f"   Pages  : {total}")

        for i, page in enumerate(pdf.pages, start=1):
            # Skip first page (cover / TOC)
            if SKIP_FIRST_PAGE and i == 1:
                print(f"   ⏭️  Page {i:>3} — skipped (cover/TOC)")
                continue

            # Extract tables first (before text, so their regions are noted)
            page_tables = _extract_tables(page)
            if page_tables:
                table_parts.extend(page_tables)
                print(f"   📊 Page {i:>3} — {len(page_tables)} table(s) extracted")

            # Extract text with column awareness
            text = _extract_columns(page)

            # OCR fallback for scanned pages
            if len(text.strip()) < MIN_TEXT_CHARS:
                print(f"   🔍 Page {i:>3} — appears scanned, attempting OCR…")
                text = _ocr_page(page)
                if text.strip():
                    scanned_pages.append(i)
                    print(f"   ✅ Page {i:>3} — OCR recovered {len(text):,} chars")
                else:
                    print(f"   ⚠️  Page {i:>3} — OCR returned nothing, skipping")
                    continue
            else:
                print(f"   ✅ Page {i:>3} — {len(text):,} chars")

            all_text_parts.append(text)

    # Combine body text and table text
    raw_text     = "\n\n".join(all_text_parts)
    raw_with_tables = raw_text
    if table_parts:
        raw_with_tables += "\n\n" + "\n\n".join(table_parts)

    cleaned = clean_text(raw_with_tables)

    metadata["scanned_pages"] = scanned_pages
    metadata["extracted_pages"] = len(all_text_parts)

    return cleaned, metadata


# ---------------------------------------------------------------------------
# Main — process ALL PDFs in data/
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"❌ No PDF files found in '{DATA_DIR}/' folder.")

    print(f"📂 Found {len(pdf_files)} PDF(s) in {DATA_DIR}/")

    processed, failed = [], []

    for pdf_path in pdf_files:
        try:
            cleaned_text, metadata = process_pdf(pdf_path)

            # Save cleaned text
            txt_path = DATA_DIR / f"{pdf_path.stem}_clean.txt"
            txt_path.write_text(cleaned_text, encoding="utf-8")

            # Save metadata as JSON alongside the text
            meta_path = DATA_DIR / f"{pdf_path.stem}_meta.json"
            meta_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print(f"\n✅ Saved → {txt_path}  ({len(cleaned_text):,} chars)")
            print(f"📋 Saved → {meta_path}")

            if metadata["scanned_pages"]:
                print(f"🔍 OCR used on pages: {metadata['scanned_pages']}")

            processed.append(pdf_path.name)

        except Exception as e:
            print(f"\n❌ Failed to process {pdf_path.name}: {e}", file=sys.stderr)
            failed.append(pdf_path.name)

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Processed : {len(processed)} file(s)")
    if failed:
        print(f"❌ Failed    : {len(failed)} file(s): {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()