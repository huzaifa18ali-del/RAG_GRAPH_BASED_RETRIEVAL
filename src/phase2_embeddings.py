#!/usr/bin/env python3
"""
phase2_embeddings.py

Phase 2 of the NLP pipeline: Generate sentence embeddings.

Loads preprocessed sentences from a JSON file, optionally chunks long
sentences, generates normalized dense vector embeddings using a
sentence-transformers model, and saves the resulting embeddings as a
NumPy array alongside a manifest JSON for downstream traceability.
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from config_loader import load_config

_cfg = load_config()

# --- Configuration ---
INPUT_PATH    = _cfg.paths.full("output_sentences")
OUTPUT_PATH   = _cfg.paths.full("output_embeddings")
MANIFEST_PATH = _cfg.paths.full("output_manifest")

MODEL_NAME   = _cfg.phase2.model_name
BATCH_SIZE   = _cfg.phase2.batch_size
MAX_TOKENS   = _cfg.phase2.max_tokens
CHUNK_STRIDE = _cfg.phase2.chunk_stride


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _word_chunks(text: str, max_words: int, stride: int) -> list[str]:
    """
    Split *text* into overlapping word-level chunks.

    Using a sliding window with *stride* overlap ensures that semantic
    meaning near chunk boundaries is captured by at least one chunk.

    Args:
        text:      Raw sentence / paragraph text.
        max_words: Maximum number of words per chunk.
        stride:    Number of words to step forward per chunk (overlap = max_words - stride).

    Returns:
        List of chunk strings.  Returns [text] unchanged if len(words) <= max_words.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride  # slide forward

    return chunks


def deduplicate_records(records: list[dict]) -> list[dict]:
    """
    Remove records with identical sentence text, keeping the first occurrence.

    Duplicate sentences (repeated headers, dialogue tags like '"Yes." "Yes."')
    produce identical embeddings which become perfect nearest-neighbors in Phase 3,
    inflating edge weights and creating false clusters around repeated fragments.

    Returns deduplicated list and prints how many were removed.
    """
    seen:    set[str]  = set()
    unique:  list[dict] = []
    removed: int        = 0

    for rec in records:
        text = rec["sentence"].strip()
        if text in seen:
            removed += 1
            continue
        seen.add(text)
        unique.append(rec)

    if removed:
        print(f"  Deduplication removed {removed} duplicate sentence(s).")
    return unique


def chunk_records(
    records: list[dict],
    max_tokens: int = MAX_TOKENS,
    stride: int = CHUNK_STRIDE,
) -> tuple[list[str], list[dict]]:
    """
    Expand sentence records into (possibly multiple) chunk strings.

    Each chunk inherits its parent record's metadata so downstream phases
    can still trace every embedding back to its originating paragraph.

    Args:
        records:    Sentence records from the preprocessed JSON.
        max_tokens: Word count threshold above which chunking is triggered.
        stride:     Sliding-window step size (in words).

    Returns:
        texts:    Flat list of chunk strings ready for encoding.
        manifest: Parallel list of dicts with traceability metadata.
    """
    texts, manifest = [], []

    for rec_idx, rec in enumerate(records):
        sentence    = rec["sentence"]
        para_id     = rec.get("paragraph_id", None)
        parent      = rec.get("parent", None)
        source_file = rec.get("source_file", None)

        chunks = _word_chunks(sentence, max_tokens, stride)

        for chunk_idx, chunk in enumerate(chunks):
            texts.append(chunk)
            manifest.append(
                {
                    "paragraph_id":   para_id,
                    "source_file":    source_file,
                    "parent":         parent,
                    "sentence_index": rec_idx,    # index into output_v2.json — O(1) lookup
                    "chunk_index":    chunk_idx,  # which chunk of this sentence (0 if no split)
                    "total_chunks":   len(chunks),
                    # chunk_text intentionally omitted — retrieve via:
                    #   records[manifest[i]["sentence_index"]]["sentence"]
                    # This avoids doubling manifest size for large corpora.
                }
            )

    return texts, manifest


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def generate_embeddings(
    sentences: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate L2-normalized sentence embeddings.

    Normalization converts raw embeddings to unit vectors so that
    cosine_similarity(a, b) == dot(a, b) — enabling fast matrix ops
    (e.g. ``embeddings @ embeddings.T``) in downstream clustering / search.

    Args:
        sentences:  Texts to encode.
        model_name: HuggingFace sentence-transformers model identifier.
        batch_size: Sentences per forward pass (lower if GPU OOM).
        normalize:  If True, L2-normalize each embedding vector.

    Returns:
        float32 NumPy array of shape (N, embedding_dim).
    """
    print(f"Loading model : {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(sentences)} chunks  (batch_size={batch_size}, normalize={normalize})…")
    embeddings: np.ndarray = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,   # ← unit vectors for cosine similarity
    )

    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_records(filepath: str) -> list[dict]:
    """Load preprocessed sentence records from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    return data


def save_embeddings(embeddings: np.ndarray, filepath: str) -> None:
    """Persist the embeddings array to *filepath* (.npy)."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    np.save(filepath, embeddings)
    print(f"Embeddings saved  → {filepath}  {embeddings.shape}")


def save_manifest(manifest: list[dict], filepath: str) -> None:
    """
    Persist the chunk manifest to *filepath* (.json).

    Each manifest entry maps an embedding index to its source sentence via
    `sentence_index` (an index into output_v2.json). To recover the original
    text in any downstream phase:

        records  = json.load(open("data/output_v2.json"))
        sentence = records[entry["sentence_index"]]["sentence"]

    chunk_text is NOT stored — use the above pattern instead. This keeps the
    manifest small regardless of corpus size.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"Manifest  saved   → {filepath}  ({len(manifest)} entries)")


def resolve_chunk_text(manifest_entry: dict, records: list[dict]) -> str:
    """
    Recover the original sentence text for a manifest entry.

    Use this in Phase 3, 4, 5 anywhere you previously accessed
    manifest_entry["chunk_text"] — which no longer exists.

    Args:
        manifest_entry : one entry from embeddings_manifest.json
        records        : the loaded output_v2.json list

    Returns:
        The original sentence string.
    """
    idx = manifest_entry.get("sentence_index")
    if idx is None or idx >= len(records):
        return ""
    return records[idx]["sentence"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Phase 2 entry point: chunk → embed → save."""

    # 1. Load
    print(f"Loading records from : {INPUT_PATH}")
    records = load_records(INPUT_PATH)
    print(f"  {len(records)} sentence records loaded.")

    # 2. Deduplicate
    records = deduplicate_records(records)
    print(f"  {len(records)} unique sentences after deduplication.")

    # 3. Chunk long sentences
    texts, manifest = chunk_records(records)
    n_chunked = sum(1 for m in manifest if m["total_chunks"] > 1)
    print(f"  {len(texts)} chunks total  ({n_chunked} originated from multi-chunk sentences)")

    if not texts:
        print("No text to encode. Exiting.")
        return

    # 4. Embed (normalized)
    embeddings = generate_embeddings(texts)

    # 5. Summary
    print("\n--- Summary ---")
    print(f"  Chunks encoded    : {len(texts)}")
    print(f"  Embedding dim     : {embeddings.shape[1]}")
    print(f"  Shape             : {embeddings.shape}")
    print(f"  Dtype             : {embeddings.dtype}")
    print(f"  Norms (first 3)   : {np.linalg.norm(embeddings[:3], axis=1)}")

    # 6. Persist
    save_embeddings(embeddings, OUTPUT_PATH)
    save_manifest(manifest, MANIFEST_PATH)


if __name__ == "__main__":
    main()