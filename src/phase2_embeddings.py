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


# --- Configuration ---
INPUT_PATH    = os.path.join("data", "output_v2.json")
OUTPUT_PATH   = os.path.join("data", "embeddings.npy")
MANIFEST_PATH = os.path.join("data", "embeddings_manifest.json")

MODEL_NAME   = "all-mpnet-base-v2"   # Upgraded from all-MiniLM-L6-v2
BATCH_SIZE   = 32                     # mpnet is heavier; reduce if OOM
MAX_TOKENS   = 384                    # Chunk threshold (words ≈ tokens for English)
CHUNK_STRIDE = 64                     # Overlap between consecutive chunks (words)


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

    for rec in records:
        sentence    = rec["sentence"]
        para_id     = rec.get("paragraph_id", None)
        parent      = rec.get("parent", None)

        chunks = _word_chunks(sentence, max_tokens, stride)

        for chunk_idx, chunk in enumerate(chunks):
            texts.append(chunk)
            manifest.append(
                {
                    "paragraph_id": para_id,
                    "parent":       parent,
                    "chunk_index":  chunk_idx,
                    "total_chunks": len(chunks),
                    "chunk_text":   chunk,
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

    The manifest lets downstream phases map embedding index → original
    paragraph / sentence, essential when one sentence expands to N chunks.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"Manifest  saved   → {filepath}  ({len(manifest)} entries)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Phase 2 entry point: chunk → embed → save."""

    # 1. Load
    print(f"Loading records from : {INPUT_PATH}")
    records = load_records(INPUT_PATH)
    print(f"  {len(records)} sentence records loaded.")

    # 2. Chunk long sentences
    texts, manifest = chunk_records(records)
    n_chunked = sum(1 for m in manifest if m["total_chunks"] > 1)
    print(f"  {len(texts)} chunks total  ({n_chunked} originated from multi-chunk sentences)")

    if not texts:
        print("No text to encode. Exiting.")
        return

    # 3. Embed (normalized)
    embeddings = generate_embeddings(texts)

    # 4. Summary
    print("\n--- Summary ---")
    print(f"  Chunks encoded    : {len(texts)}")
    print(f"  Embedding dim     : {embeddings.shape[1]}")
    print(f"  Shape             : {embeddings.shape}")
    print(f"  Dtype             : {embeddings.dtype}")
    print(f"  Norms (first 3)   : {np.linalg.norm(embeddings[:3], axis=1)}")  # should all ≈ 1.0

    # 5. Persist
    save_embeddings(embeddings, OUTPUT_PATH)
    save_manifest(manifest, MANIFEST_PATH)


if __name__ == "__main__":
    main()