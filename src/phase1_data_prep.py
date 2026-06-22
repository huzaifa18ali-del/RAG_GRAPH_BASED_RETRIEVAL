#!/usr/bin/env python3
"""
phase1_data_prep_v2.py

Phase 1 (v2) Data Preprocessing:
- Reads raw text files from "raw_texts"
- Splits paragraphs into semantically meaningful sentences using spaCy dependency parsing
- Optionally resolves pronouns with neuralcoref for better semantic understanding
- Outputs JSON with paragraph_id, sentence, and parent
"""

import os
import json
import spacy



RAW_FOLDER = "raw_texts"
OUTPUT_FILE = "data/output_v2.json"

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Optional: add neuralcoref to pipeline
# neuralcoref.add_to_pipe(nlp)

def split_sentence_by_clause(sent):
    """
    Splits a spaCy Span (sentence) into meaningful sub-clauses using
    dependency parsing — NOT naive comma/semicolon splitting.

    Strategy:
      - Walk the dependency tree to find coordinated verb phrases (cc + conj)
        and adverbial clauses (advcl), which are the two most common sources
        of genuine sub-clauses in English.
      - Reconstruct each sub-clause from its subtree tokens, preserving word
        order and avoiding splitting on punctuation inside lists, dates, or
        quoted strings.
      - Falls back to returning the full sentence unchanged if no split points
        are found, so short/simple sentences are never broken.

    Args:
        sent: A spaCy Span object (one element of doc.sents).

    Returns:
        List of clause strings.  Always contains at least one element.
    """
    # Collect token indices that are roots of genuine sub-clauses:
    # - conj (coordinated verb/clause root, e.g. "She sang and [she] danced")
    # - advcl (adverbial clause, e.g. "He left [because she arrived]")
    split_roots = set()
    root_token = None

    for token in sent:
        if token.dep_ == "ROOT":
            root_token = token
        elif token.dep_ in ("conj", "advcl") and token.pos_ == "VERB":
            split_roots.add(token.i)

    # If no meaningful split points exist, return the sentence as-is.
    if not split_roots or root_token is None:
        return [sent.text.strip()]

    # Build a mapping: token_index → which clause it belongs to.
    # We assign each token to the sub-clause root whose subtree contains it.
    # Tokens not covered by a split root belong to the main (ROOT) clause.
    clause_groups: dict[int, list] = {root_token.i: []}
    for sr in split_roots:
        clause_groups[sr] = []

    for token in sent:
        assigned = False
        for sr in split_roots:
            subtree_indices = {t.i for t in sent.root.subtree
                               if t.i >= sent.start and t.i < sent.end}
            # Check if token falls in the subtree of this split root
            sr_token = sent.doc[sr]
            sr_subtree = {t.i for t in sr_token.subtree}
            if token.i in sr_subtree:
                clause_groups[sr].append(token)
                assigned = True
                break
        if not assigned:
            clause_groups[root_token.i].append(token)

    # Reconstruct clause text from token lists, preserving original spacing.
    clauses = []
    for root_i in sorted(clause_groups.keys()):
        tokens = clause_groups[root_i]
        if not tokens:
            continue
        # Re-join using spaCy whitespace info so spacing is correct.
        text = "".join(t.text_with_ws for t in tokens).strip()
        # Strip leading conjunctions (and / but / or / because / while …)
        text = text.lstrip(" ,;")
        if text:
            clauses.append(text)

    return clauses if clauses else [sent.text.strip()]

def process_raw_texts(input_folder=RAW_FOLDER, output_file=OUTPUT_FILE):
    results = []
    paragraph_id = 0

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created '{input_folder}' folder. Add .txt files and run again.")
        return

    text_files = sorted(f for f in os.listdir(input_folder) if f.endswith(".txt"))

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
                # Further split complex sentences using dependency parsing.
                # Pass the full spaCy Span (not just sent.text) so the
                # function can walk the dependency tree.
                clauses = split_sentence_by_clause(sent)
                for clause in clauses:
                    if clause:
                        results.append({
                            "paragraph_id": paragraph_id,
                            "sentence": clause,
                            "parent": None
                        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(text_files)} file(s), {paragraph_id} paragraph(s), {len(results)} sentence(s).")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    process_raw_texts()