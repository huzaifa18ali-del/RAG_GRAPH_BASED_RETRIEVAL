#!/usr/bin/env python3
"""
phase1_data_prep.py

Phase 1 — Data Preprocessing:
- Reads cleaned text files from "raw_texts/"
- Splits paragraphs into semantically meaningful sentences using spaCy
- Uses dependency parsing to split complex sentences into sub-clauses
- Filters out fragments shorter than MIN_WORDS (avoids noisy embeddings)
- Outputs JSON with paragraph_id, sentence, source_file, and parent

Note on coreference resolution:
  neuralcoref (spaCy 2.x only) is NOT compatible with spaCy 3.x.
  For pronoun resolution on spaCy 3.x use:
    pip install spacy-experimental
    python -m spacy download en_coreference_web_trf
"""

import os
import json
import spacy

from config_loader import load_config

_cfg = load_config()

RAW_FOLDER      = _cfg.paths.data_dir
OUTPUT_FILE     = _cfg.paths.full("output_sentences")
RAW_FILE_SUFFIX = _cfg.paths.raw_file_suffix
MIN_WORDS       = _cfg.phase1.min_words

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def split_sentence_by_clause(sent):
    """
    Splits a spaCy Span into meaningful sub-clauses using dependency parsing.

    Strategy:
      - Find tokens whose dependency label is 'conj' or 'advcl' AND whose
        POS is VERB — these are genuine clause boundaries.
      - Assign every token to the clause whose root's subtree contains it.
        Tokens not covered by any split root belong to the main ROOT clause.
      - Reconstruct each clause from its tokens preserving original spacing.
      - Falls back to the full sentence if no split points are found.

    Args:
        sent: spaCy Span (one element of doc.sents).

    Returns:
        List of clause strings. Always at least one element.
    """
    split_roots = set()
    root_token  = None

    for token in sent:
        if token.dep_ == "ROOT":
            root_token = token
        elif token.dep_ in ("conj", "advcl") and token.pos_ == "VERB":
            split_roots.add(token.i)

    if not split_roots or root_token is None:
        return [sent.text.strip()]

    # Precompute subtree index sets for each split root ONCE (not per token).
    # This is the O(N²) fix — the previous version recomputed sent.root.subtree
    # inside the inner loop on every token iteration, doing redundant work.
    sr_subtrees: dict[int, set[int]] = {
        sr: {t.i for t in sent.doc[sr].subtree}
        for sr in split_roots
    }

    clause_groups: dict[int, list] = {root_token.i: []}
    for sr in split_roots:
        clause_groups[sr] = []

    for token in sent:
        assigned = False
        for sr, subtree in sr_subtrees.items():
            if token.i in subtree:
                clause_groups[sr].append(token)
                assigned = True
                break
        if not assigned:
            clause_groups[root_token.i].append(token)

    clauses = []
    for root_i in sorted(clause_groups.keys()):
        tokens = clause_groups[root_i]
        if not tokens:
            continue
        text = "".join(t.text_with_ws for t in tokens).strip()
        text = text.lstrip(" ,;")
        if text:
            clauses.append(text)

    return clauses if clauses else [sent.text.strip()]

def process_raw_texts(
    input_folder: str = RAW_FOLDER,
    output_file:  str = OUTPUT_FILE,
    file_suffix:  str = RAW_FILE_SUFFIX,
):
    results        = []
    paragraph_id   = 0   # global counter — carries across files intentionally
    skipped_short  = 0

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created '{input_folder}' folder. Add .txt files and run again.")
        return

    text_files = sorted(
        f for f in os.listdir(input_folder)
        if f.endswith(file_suffix)
    )

    if not text_files:
        print(f"No *{file_suffix} files found in '{input_folder}/'. "
              f"Run pdf_to_txt.py first.")
        return

    for filename in text_files:
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for paragraph in paragraphs:
            paragraph_id += 1
            normalized_paragraph = " ".join(paragraph.split())
            doc = nlp(normalized_paragraph)

            for sent in doc.sents:
                clauses = split_sentence_by_clause(sent)
                for clause in clauses:
                    if not clause:
                        continue
                    # Drop fragments shorter than MIN_WORDS — these are noise
                    # (conjunctions, isolated names, partial phrases) that produce
                    # low-quality embeddings and inflate the noise cluster.
                    if len(clause.split()) < MIN_WORDS:
                        skipped_short += 1
                        continue
                    results.append({
                        "paragraph_id": paragraph_id,
                        "sentence":     clause,
                        "source_file":  filename,   # traceability across multiple files
                        "parent":       None,
                    })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed   : {len(text_files)} file(s)")
    print(f"Paragraphs  : {paragraph_id}")
    print(f"Sentences   : {len(results)}  (kept)")
    print(f"Skipped     : {skipped_short}  (< {MIN_WORDS} words)")
    print(f"Output      : {output_file}")

if __name__ == "__main__":
    process_raw_texts()

def main():
    """Entry point called by run_pipeline.py."""
    process_raw_texts()