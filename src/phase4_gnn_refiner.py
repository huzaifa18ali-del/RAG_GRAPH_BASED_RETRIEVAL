#!/usr/bin/env python3
"""
phase4_gnn_refiner.py

Phase 4 of the NLP pipeline: Cluster-aware Structured Summarization.

Loads the idea graph (Phase 3), cluster summary, and optional embeddings
manifest, then produces a structured summary ordered by cluster importance.

Per cluster:
  - Centroid sentence becomes the topic heading
  - Top-N nodes ranked by intra-cluster centrality fill the paragraph
  - Cross-cluster bridge sentences are flagged separately
  - Short sentences are optionally merged via spaCy dependency parsing

Outputs:
  - data/summary.json        : machine-readable structured summary
  - data/summary.txt         : human-readable plain-text outline
  - Console                  : live progress + diagnostics
"""

from __future__ import annotations

import json
import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Optional

import networkx as nx

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    warnings.warn("spaCy / en_core_web_sm not available — sentence merging disabled.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IDEA_GRAPH_PATH    = os.path.join("data", "idea_graph.json")
CLUSTER_PATH       = os.path.join("data", "cluster_summary.json")
MANIFEST_PATH      = os.path.join("data", "embeddings_manifest.json")
OUTPUT_JSON_PATH   = os.path.join("data", "summary.json")
OUTPUT_TEXT_PATH   = os.path.join("data", "summary.txt")

TOP_N_PER_CLUSTER  = 4       # Max body sentences per cluster (excl. centroid)
MAX_BRIDGES        = 2       # Cross-cluster bridge sentences per cluster
MIN_SENT_WORDS     = 6       # Sentences shorter than this are merge candidates
MAX_LINE_WIDTH     = 120     # Wrap width for plain-text output
PAGERANK_DAMPING   = 0.85
INCLUDE_NOISE      = True    # Whether to append the noise cluster at the end


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(filepath: str, label: str) -> Any:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{label} not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"Loaded {label:<22} ← {filepath}")
    return data


def _save_json(obj: Any, filepath: str, label: str = "JSON") -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    print(f"Saved  {label:<22} → {filepath}")


def _save_text(text: str, filepath: str, label: str = "Text") -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Saved  {label:<22} → {filepath}")


# ---------------------------------------------------------------------------
# Graph construction (PageRank as secondary importance signal)
# ---------------------------------------------------------------------------

def build_networkx_graph(idea_graph: list[dict]) -> nx.DiGraph:
    """
    Build a weighted directed graph from the idea graph.

    Node attrs : sentence, paragraph_id, cluster_id, threshold_used
    Edge attrs : weight (cosine similarity), cross_cluster (bool)
    """
    G = nx.DiGraph()

    for node in idea_graph:
        G.add_node(
            node["sentence_id"],
            sentence=node["sentence"],
            paragraph_id=node.get("paragraph_id"),
            cluster_id=node.get("cluster_id", -1),
            threshold_used=node.get("threshold_used", 0.0),
            chunk_index=node.get("chunk_index", 0),
            total_chunks=node.get("total_chunks", 1),
        )

    edge_count = 0
    for node in idea_graph:
        src = node["sentence_id"]
        for nb in node.get("neighbors", []):
            dst, sim, cross = nb["sentence_id"], nb["similarity"], nb.get("cross_cluster", False)
            if G.has_edge(src, dst):
                if sim > G[src][dst]["weight"]:
                    G[src][dst]["weight"] = sim
            else:
                G.add_edge(src, dst, weight=sim, cross_cluster=cross)
                edge_count += 1

    print(f"Graph: {G.number_of_nodes()} nodes, {edge_count} edges")
    return G


def compute_pagerank(G: nx.DiGraph) -> dict[int, float]:
    """Weighted PageRank — secondary importance signal used to break ties."""
    if G.number_of_nodes() == 0:
        return {}
    return nx.pagerank(G, alpha=PAGERANK_DAMPING, weight="weight", max_iter=300, tol=1e-9)


# ---------------------------------------------------------------------------
# Sentence merging (spaCy, optional)
# ---------------------------------------------------------------------------

def _subject_verb_chain(sent_text: str) -> Optional[str]:
    """
    Extract a minimal 'subject + verb (+ object)' chain from a sentence.

    Returns the chain as a string, or None if spaCy can't parse it cleanly.
    This is used to build a readable merged sentence head.
    """
    if not SPACY_AVAILABLE:
        return None
    doc = _NLP(sent_text)
    tokens = []
    for tok in doc:
        if tok.dep_ in {"nsubj", "nsubjpass", "ROOT", "dobj", "attr", "prep"}:
            tokens.append(tok.text)
    return " ".join(tokens) if len(tokens) >= 2 else None


def merge_short_sentences(sentences: list[str], min_words: int = MIN_SENT_WORDS) -> list[str]:
    """
    Merge consecutive short sentences into the preceding longer one.

    Strategy:
      - Iterate through the sentence list.
      - If a sentence has fewer than *min_words* words, append it to the
        previous sentence (joined with "; ").
      - If it's the first sentence and short, it becomes the seed and the
        next sentence is appended to it.

    This avoids choppy single-clause entries in the final paragraph without
    requiring any neural generation step.

    Args:
        sentences: Ordered list of sentence strings.
        min_words: Word count below which a sentence is a merge candidate.

    Returns:
        Reduced list with short sentences absorbed into neighbours.
    """
    if not sentences:
        return sentences

    merged: list[str] = []
    buffer = ""

    for sent in sentences:
        word_count = len(sent.split())
        if buffer:
            if word_count < min_words:
                # Both buffer and current are short — keep joining
                buffer = buffer.rstrip(".") + "; " + sent.lstrip()
            else:
                merged.append(buffer)
                buffer = sent
        else:
            if word_count < min_words:
                buffer = sent   # hold and try to merge with next
            else:
                merged.append(sent)

    if buffer:
        if merged:
            merged[-1] = merged[-1].rstrip(".") + "; " + buffer.lstrip()
        else:
            merged.append(buffer)

    return merged


# ---------------------------------------------------------------------------
# Cluster ordering & sentence selection
# ---------------------------------------------------------------------------

def order_clusters(cluster_summary: list[dict]) -> tuple[list[dict], Optional[dict]]:
    """
    Sort clusters by size descending.  Separate noise cluster (-1) if present.

    Returns:
        (real_clusters, noise_cluster_or_None)
    """
    noise = None
    real  = []
    for cs in cluster_summary:
        if cs.get("is_noise") or cs.get("cluster_id") == -1:
            noise = cs
        else:
            real.append(cs)

    real.sort(key=lambda c: c["size"], reverse=True)
    return real, noise


def select_cluster_sentences(
    cluster: dict,
    idea_graph_index: dict[int, dict],
    pagerank: dict[int, float],
    used_sids: set[int],
    top_n: int = TOP_N_PER_CLUSTER,
    max_bridges: int = MAX_BRIDGES,
    pagerank_rank_map: dict[int, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Pick body sentences and bridge sentences for one cluster.

    Selection pipeline
    ------------------
    1. Gather all nodes belonging to this cluster from the graph index.
    2. Score each node as:
           score = 0.6 * centrality_rank + 0.4 * pagerank_rank
       (rank-based fusion avoids scale mismatches between the two signals)
    3. Take the top-N scoring nodes that haven't appeared in a prior cluster.
    4. Among the neighbors of selected nodes, collect cross-cluster edges as
       bridge candidates (deduplicated, sorted by similarity desc).

    Args:
        cluster:          Cluster summary entry from cluster_summary.json.
        idea_graph_index: {sentence_id → node dict} look-up table.
        pagerank:         PageRank scores for all nodes.
        used_sids:        Sentence IDs already committed to earlier clusters.
        top_n:            Max body sentences to return.
        max_bridges:      Max bridge sentences to return.

    Returns:
        (body_sentences, bridge_sentences) — each is a list of enriched dicts.
    """
    cid          = cluster["cluster_id"]
    top_nodes    = cluster.get("top_nodes", [])   # pre-ranked by intra-cluster centrality
    centroid_sid = cluster.get("centroid_sentence_id")

    # --- Build centrality rank map (lower index = higher centrality) ---
    centrality_rank = {
        entry["sentence_id"]: rank
        for rank, entry in enumerate(top_nodes)
    }
    max_rank = max(len(top_nodes) - 1, 1)

    # --- Score & filter ---
    scored: list[tuple[float, dict]] = []
    for entry in top_nodes:
        sid = entry["sentence_id"]
        if sid == centroid_sid or sid in used_sids:
            continue
        node = idea_graph_index.get(sid)
        if node is None:
            continue

        c_rank  = centrality_rank.get(sid, max_rank)
        # Use the precomputed rank map to avoid re-sorting on every node.
        pr_rank = pagerank_rank_map.get(sid, len(pagerank)) if pagerank_rank_map else 0
        pr_max  = max(len(pagerank) - 1, 1)

        # Normalise both ranks to [0, 1] then fuse (lower score = more important)
        c_norm  = c_rank  / max_rank
        pr_norm = pr_rank / pr_max
        score   = 0.6 * c_norm + 0.4 * pr_norm    # weighted fusion

        scored.append((
            score,
            {
                "sentence_id": sid,
                "sentence":    node["sentence"],
                "cluster_id":  cid,
                "centrality":  entry.get("centrality", 0.0),
                "pagerank":    round(pagerank.get(sid, 0.0), 8),
                "paragraph_id": node.get("paragraph_id"),
            }
        ))

    scored.sort(key=lambda x: x[0])    # ascending — lower score = more important
    body_entries  = [e for _, e in scored[:top_n]]
    selected_sids = {e["sentence_id"] for e in body_entries}
    if centroid_sid is not None:
        selected_sids.add(centroid_sid)

    # --- Bridge sentences: cross-cluster neighbors of selected nodes ---
    bridge_candidates: dict[int, dict] = {}
    for sid in selected_sids:
        node = idea_graph_index.get(sid)
        if node is None:
            continue
        for nb in node.get("neighbors", []):
            if not nb.get("cross_cluster"):
                continue
            nb_sid = nb["sentence_id"]
            if nb_sid in used_sids or nb_sid in selected_sids:
                continue
            nb_node = idea_graph_index.get(nb_sid)
            if nb_node is None:
                continue
            if nb_sid not in bridge_candidates or nb["similarity"] > bridge_candidates[nb_sid]["similarity"]:
                bridge_candidates[nb_sid] = {
                    "sentence_id":    nb_sid,
                    "sentence":       nb_node["sentence"],
                    "cluster_id":     nb_node.get("cluster_id", -1),
                    "similarity":     nb["similarity"],
                    "source_sid":     sid,
                }

    bridges = sorted(bridge_candidates.values(), key=lambda x: x["similarity"], reverse=True)
    bridges = bridges[:max_bridges]

    return body_entries, bridges


def _build_pagerank_rank_map(pagerank: dict[int, float]) -> dict[int, int]:
    """
    Pre-compute a {sentence_id → rank} map from the pagerank scores (0 = highest).

    This must be called ONCE and the result passed around, rather than calling
    the old _pagerank_rank() per node — which re-sorted the entire dict on
    every single invocation, giving O(N² log N) total work for N nodes.
    """
    sorted_sids = sorted(pagerank, key=lambda k: pagerank[k], reverse=True)
    return {s: i for i, s in enumerate(sorted_sids)}


# ---------------------------------------------------------------------------
# Paragraph assembly
# ---------------------------------------------------------------------------

def assemble_paragraph(
    centroid_sentence: str,
    body_entries: list[dict],
    merge_short: bool = True,
) -> str:
    """
    Combine centroid + body sentences into a single readable paragraph.

    The centroid always leads.  Body sentences are ordered by their original
    sentence_id to restore natural reading order.  Optionally, short
    sentences are merged to avoid choppy rhythm.

    Args:
        centroid_sentence: The cluster's most central sentence (topic setter).
        body_entries:      Scored body sentence dicts.
        merge_short:       Whether to apply short-sentence merging.

    Returns:
        A single paragraph string.
    """
    # Restore reading order for body
    body_ordered = sorted(body_entries, key=lambda e: e["sentence_id"])
    sentences    = [centroid_sentence] + [e["sentence"] for e in body_ordered]

    if merge_short and SPACY_AVAILABLE:
        sentences = merge_short_sentences(sentences)

    # Ensure every sentence ends with punctuation
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s and s[-1] not in ".!?":
            s += "."
        cleaned.append(s)

    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# Plain-text rendering
# ---------------------------------------------------------------------------

def render_plain_text(structured_summary: list[dict], line_width: int = MAX_LINE_WIDTH) -> str:
    """
    Render the structured summary as a plain-text outline.

    Format per cluster
    ------------------
    ## [Cluster N]  (size=K)
    TOPIC: <centroid sentence>

    <paragraph>

    BRIDGING IDEAS:
      → [Cluster X] "bridge sentence"  (similarity=0.82)

    Args:
        structured_summary: Output of build_structured_summary().
        line_width:         Hard-wrap column width.

    Returns:
        Full plain-text string ready to write to file or print.
    """
    wrapper = textwrap.TextWrapper(width=line_width, initial_indent="  ", subsequent_indent="  ")
    lines: list[str] = []
    sep = "─" * line_width

    lines.append("STRUCTURED SUMMARY")
    lines.append("=" * line_width)

    for entry in structured_summary:
        cid     = entry["cluster_id"]
        size    = entry["size"]
        noise   = " [NOISE]" if entry.get("is_noise") else ""
        heading = f"## Cluster {cid}{noise}  (size={size})"
        lines.append("")
        lines.append(heading)
        lines.append(sep)

        # Topic heading
        lines.append(f"TOPIC:  {entry['centroid_sentence']}")
        lines.append("")

        # Paragraph
        para = entry.get("paragraph", "")
        if para:
            lines.append(wrapper.fill(para))
        lines.append("")

        # Bullet subpoints (top nodes)
        if entry.get("body_sentences"):
            lines.append("  Key points:")
            for b in entry["body_sentences"]:
                bullet = f"  • [sid={b['sentence_id']}  c={b['centrality']:.4f}]  {b['sentence']}"
                lines.append(
                    textwrap.fill(
                        bullet,
                        width=line_width,
                        initial_indent="",
                        subsequent_indent="      ",
                    )
                )
            lines.append("")

        # Bridge sentences
        if entry.get("bridges"):
            lines.append("  Bridging ideas:")
            for br in entry["bridges"]:
                bridge_line = (
                    f"  → [Cluster {br['cluster_id']}] "
                    f"\"{br['sentence']}\"  "
                    f"(sim={br['similarity']:.4f})"
                )
                lines.append(
                    textwrap.fill(
                        bridge_line,
                        width=line_width,
                        initial_indent="",
                        subsequent_indent="       ",
                    )
                )
            lines.append("")

    lines.append("=" * line_width)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Master summary builder
# ---------------------------------------------------------------------------

def build_structured_summary(
    cluster_summary:  list[dict],
    idea_graph:       list[dict],
    pagerank:         dict[int, float],
    include_noise:    bool = INCLUDE_NOISE,
    top_n:            int  = TOP_N_PER_CLUSTER,
    max_bridges:      int  = MAX_BRIDGES,
) -> list[dict]:
    """
    Orchestrate the full summarization pass over all clusters.

    Processing order
    ----------------
    1. Sort real clusters by size desc; optionally append noise cluster.
    2. For each cluster:
       a. Pick centroid sentence (always first).
       b. Select top-N body sentences via centrality + PageRank fusion.
       c. Identify cross-cluster bridge sentences.
       d. Assemble paragraph (with optional short-sentence merging).
    3. Track used sentence IDs globally to prevent repetition.

    Args:
        cluster_summary:  Loaded cluster_summary.json.
        idea_graph:       Loaded idea_graph.json (list of node dicts).
        pagerank:         PageRank scores {sentence_id: score}.
        include_noise:    Append the noise cluster (-1) if present.
        top_n:            Max body sentences per cluster.
        max_bridges:      Max bridge sentences per cluster.

    Returns:
        List of cluster summary dicts enriched with paragraph + bridge info.
    """
    # Build O(1) look-up index
    graph_index: dict[int, dict] = {node["sentence_id"]: node for node in idea_graph}

    # Precompute PageRank rank map ONCE — avoids O(N² log N) re-sorting per cluster.
    pr_rank_map = _build_pagerank_rank_map(pagerank)

    ordered_clusters, noise_cluster = order_clusters(cluster_summary)
    if include_noise and noise_cluster:
        ordered_clusters.append(noise_cluster)

    used_sids: set[int] = set()
    structured: list[dict] = []

    for cluster in ordered_clusters:
        cid          = cluster["cluster_id"]
        centroid_sid = cluster.get("centroid_sentence_id")
        centroid_txt = cluster.get("centroid_sentence", "")

        # Centroid is always "used" — prevent it appearing as a body sentence later
        if centroid_sid is not None:
            used_sids.add(centroid_sid)

        body_entries, bridges = select_cluster_sentences(
            cluster, graph_index, pagerank, used_sids,
            top_n=top_n, max_bridges=max_bridges,
            pagerank_rank_map=pr_rank_map,
        )

        # Mark body + bridge sids as used
        used_sids.update(e["sentence_id"] for e in body_entries)

        paragraph = assemble_paragraph(centroid_txt, body_entries)

        structured.append({
            "cluster_id":         cid,
            "is_noise":           cluster.get("is_noise", False),
            "size":               cluster["size"],
            "centroid_sentence_id": centroid_sid,
            "centroid_sentence":  centroid_txt,
            "centroid_centrality": cluster.get("centroid_centrality", 0.0),
            "paragraph":          paragraph,
            "body_sentences":     body_entries,
            "bridges":            bridges,
        })

    return structured


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(G: nx.DiGraph, pagerank: dict[int, float]) -> None:
    values    = list(pagerank.values())
    in_deg    = [d for _, d in G.in_degree()]
    top5      = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]

    print()
    print("─" * 60)
    print("GRAPH DIAGNOSTICS")
    print("─" * 60)
    print(f"  Nodes            : {G.number_of_nodes()}")
    print(f"  Edges            : {G.number_of_edges()}")
    print(f"  Weak components  : {nx.number_weakly_connected_components(G)}")
    print(f"  Avg in-degree    : {sum(in_deg)/max(len(in_deg),1):.2f}")
    print(f"  PageRank min/max : {min(values):.6f} / {max(values):.6f}")
    print()
    print("  Top-5 by PageRank:")
    for sid, score in top5:
        txt = G.nodes[sid]["sentence"]
        print(f"    [{sid:>4}] {score:.8f}  \"{txt[:68]}{'…' if len(txt)>68 else ''}\"")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Phase 4 entry point: cluster-aware structured summarization."""

    # 1 — Load all inputs
    idea_graph      = _load_json(IDEA_GRAPH_PATH, "idea_graph.json")
    cluster_summary = _load_json(CLUSTER_PATH,    "cluster_summary.json")
    manifest        = None
    if os.path.isfile(MANIFEST_PATH):
        manifest    = _load_json(MANIFEST_PATH,   "embeddings_manifest.json")
        # manifest is available for future chunk-traceability features;
        # Phase 4 currently operates on the idea graph directly and does not
        # need per-chunk metadata, so it is loaded but not passed further.

    # 2 — Build NetworkX graph + PageRank
    G        = build_networkx_graph(idea_graph)
    pagerank = compute_pagerank(G)
    print_diagnostics(G, pagerank)

    # 3 — Build structured summary
    structured_summary = build_structured_summary(
        cluster_summary,
        idea_graph,
        pagerank,
        include_noise=INCLUDE_NOISE,
        top_n=TOP_N_PER_CLUSTER,
        max_bridges=MAX_BRIDGES,
    )

    # 4 — Render + save plain text
    plain_text = render_plain_text(structured_summary)
    print("\n" + plain_text)
    _save_text(plain_text, OUTPUT_TEXT_PATH, label="Plain-text summary")

    # 5 — Save structured JSON
    _save_json(structured_summary, OUTPUT_JSON_PATH, label="Structured JSON summary")

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()