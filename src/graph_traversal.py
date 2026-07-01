#!/usr/bin/env python3
"""
graph_traversal.py

Phase 2 (Intelligence Upgrade) — Milestone 1: dynamic, query-time graph
traversal over the idea graph using Personalized PageRank (PPR).

Replaces static, pre-computed cluster membership as the retrieval mechanism.
Given a query embedding, this module:

  1. Searches the FAISS HNSW index built over a task's embeddings to find
     the top-N nearest clauses as seed nodes.
  2. Builds a personalization vector concentrated on those seed nodes
     (normalized to sum to 1.0).
  3. Runs networkx.pagerank() over the idea graph using that personalization
     vector and the existing per-edge `weight` (cosine similarity) attribute.
  4. Returns the top-M clauses by resulting PPR score.

Falls back to plain HNSW nearest-neighbor retrieval (no graph walk) if the
graph has no edges at all, or if PageRank's power iteration fails to
converge — both are real, expected conditions on small or sparse corpora,
not bugs, so they are handled rather than raised.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from networkx.exception import PowerIterationFailedConvergence
from sklearn.preprocessing import normalize

log = logging.getLogger("graph_traversal")
if not log.handlers:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

# ---------------------------------------------------------------------------
# Reuse phase3's HNSW index builder — one source of truth for how the index
# is constructed (same M / efConstruction semantics, same inner-product /
# L2-normalisation contract) instead of a second, drifting implementation.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from phase3_idea_graph import build_hnsw_index  # noqa: E402


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RankedClause:
    sentence_id:  int
    sentence:     str
    score:        float          # PPR score, or raw cosine similarity on fallback
    is_seed:      bool
    cluster_id:   Optional[int] = None
    paragraph_id: Optional[int] = None
    expansion_source: Optional[int] = None   # sentence_id of the heading-like clause
                                              # that pulled this one in via
                                              # expand_heading_context(); None for
                                              # clauses that PPR itself ranked


@dataclass
class GraphIndex:
    """
    Everything needed to answer queries against one ingested document's
    artifacts: the FAISS HNSW index over its embeddings, the networkx graph
    built from its idea_graph.json edges, and a sentence_id -> node lookup.

    One instance is built per task (per data_dir) and is safe to cache and
    reuse across multiple queries against the same document — building it
    is the expensive part; searching/walking it is cheap.
    """
    data_dir:        str
    embeddings:      np.ndarray
    faiss_index:     "faiss.IndexHNSWFlat"
    nx_graph:        nx.DiGraph
    nodes_by_id:     dict[int, dict]
    source_mtime:    float                 # mtime of idea_graph.json at build time,
                                            # used by callers to detect staleness
    built_at:        float = field(default_factory=time.time)


class GraphIndexError(Exception):
    """Raised when a task's graph/embedding artifacts can't be loaded or built."""


# ---------------------------------------------------------------------------
# Loading + index construction
# ---------------------------------------------------------------------------

def _load_json(path: Path, label: str) -> list[dict]:
    if not path.exists():
        raise GraphIndexError(f"{label} not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise GraphIndexError(f"{label} is not valid JSON: {exc}") from exc


def _load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise GraphIndexError(f"embeddings.npy not found: {path}")
    try:
        return np.load(path).astype(np.float32)
    except (OSError, ValueError) as exc:
        raise GraphIndexError(f"Failed to load embeddings from {path}: {exc}") from exc


def _build_networkx_graph(idea_graph: list[dict], min_edge_weight: float = 0.0) -> nx.DiGraph:
    """
    Build a weighted directed graph from idea_graph.json nodes.

    Mirrors phase4_gnn_refiner.build_networkx_graph's schema exactly (node
    attrs: sentence, paragraph_id, cluster_id; edge attr: weight) so the
    two graph-consuming phases (static summarization, dynamic PPR retrieval)
    stay structurally consistent with each other.
    """
    G = nx.DiGraph()

    for node in idea_graph:
        G.add_node(
            node["sentence_id"],
            sentence=node["sentence"],
            paragraph_id=node.get("paragraph_id"),
            cluster_id=node.get("cluster_id", -1),
        )

    for node in idea_graph:
        src = node["sentence_id"]
        for nb in node.get("neighbors", []):
            dst = nb["sentence_id"]
            # Phase 3 now writes two kinds of edges into "neighbors": genuine
            # cosine-similarity edges (edge_type "similarity") and fixed-weight
            # document-order edges added so PPR can "keep reading" contiguous
            # text (edge_type "sequential"; "both" when a pair earned both).
            # Both kinds store their weight under the same "similarity" key,
            # so no branching is needed here — they feed the same `weight`
            # edge attribute below, which is exactly what nx.pagerank() reads
            # (weight="weight" in personalized_pagerank_search). Sequential
            # edges are deliberately weighted low (default 0.3, below the
            # similarity floor) so they nudge the walk forward without
            # overpowering a real semantic match.
            weight = float(nb["similarity"])
            if weight < min_edge_weight:
                continue
            if G.has_edge(src, dst):
                if weight > G[src][dst]["weight"]:
                    G[src][dst]["weight"] = weight
            else:
                G.add_edge(src, dst, weight=weight, edge_type=nb.get("edge_type", "similarity"))

    return G


def build_graph_index(
    data_dir: str | Path,
    embeddings_filename: str = "embeddings.npy",
    idea_graph_filename: str = "idea_graph.json",
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    min_edge_weight: float = 0.0,
) -> GraphIndex:
    """
    Load a task's embeddings + idea graph and build the FAISS HNSW index and
    networkx graph needed for query-time PPR traversal.

    Args:
        data_dir:              Task-isolated data directory (see main.py).
        embeddings_filename:   Relative filename of the Phase 2 embeddings array.
        idea_graph_filename:   Relative filename of the Phase 3 idea graph JSON.
        hnsw_m, hnsw_ef_construction: passed through to phase3's build_hnsw_index,
                                should match the values used at ingestion time.
        min_edge_weight:       Optional floor applied when wiring the traversal
                                graph (0.0 = trust the edges Phase 3 already produced).

    Returns:
        A populated GraphIndex.

    Raises:
        GraphIndexError: if required artifacts are missing, malformed, or the
                          index fails to build.
    """
    data_dir = Path(data_dir)
    embeddings_path = data_dir / embeddings_filename
    idea_graph_path = data_dir / idea_graph_filename

    embeddings = _load_embeddings(embeddings_path)
    idea_graph = _load_json(idea_graph_path, "idea_graph.json")

    if len(idea_graph) != len(embeddings):
        raise GraphIndexError(
            f"idea_graph.json has {len(idea_graph)} nodes but embeddings.npy has "
            f"{len(embeddings)} rows — artifacts are out of sync. Re-run ingestion."
        )

    try:
        faiss_index = build_hnsw_index(embeddings, m=hnsw_m, ef_construction=hnsw_ef_construction)
    except RuntimeError as exc:
        raise GraphIndexError(f"Failed to build FAISS HNSW index: {exc}") from exc

    nx_graph = _build_networkx_graph(idea_graph, min_edge_weight=min_edge_weight)
    nodes_by_id = {node["sentence_id"]: node for node in idea_graph}

    log.info(
        "GraphIndex built for %s: %d nodes, %d edges, embeddings dim=%d",
        data_dir, nx_graph.number_of_nodes(), nx_graph.number_of_edges(), embeddings.shape[1],
    )

    return GraphIndex(
        data_dir=str(data_dir),
        embeddings=embeddings,
        faiss_index=faiss_index,
        nx_graph=nx_graph,
        nodes_by_id=nodes_by_id,
        source_mtime=idea_graph_path.stat().st_mtime,
    )


# ---------------------------------------------------------------------------
# Seed node retrieval (HNSW)
# ---------------------------------------------------------------------------

def _find_seed_nodes(
    query_vector: np.ndarray,
    graph_index: GraphIndex,
    seed_top_n: int,
    ef_search: int,
) -> list[tuple[int, float]]:
    """
    Search the HNSW index for the top_n nearest clauses to the query vector.

    Returns:
        List of (sentence_id, cosine_similarity) tuples, best first.

    Raises:
        GraphIndexError: if the FAISS search itself fails.
    """
    n = graph_index.embeddings.shape[0]
    top_n = min(seed_top_n, n)

    graph_index.faiss_index.hnsw.efSearch = max(ef_search, top_n)

    try:
        vec = normalize(query_vector.reshape(1, -1).astype(np.float32), norm="l2")
        similarities, indices = graph_index.faiss_index.search(vec, top_n)
    except Exception as exc:
        raise GraphIndexError(f"FAISS seed search failed: {exc}") from exc

    seeds = [
        (int(idx), float(sim))
        for sim, idx in zip(similarities[0], indices[0])
        if idx >= 0
    ]
    return seeds


# ---------------------------------------------------------------------------
# Personalized PageRank traversal
# ---------------------------------------------------------------------------

def personalized_pagerank_search(
    query_vector: np.ndarray,
    graph_index: GraphIndex,
    seed_top_n: int = 8,
    result_top_m: int = 6,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1.0e-6,
    ef_search: int = 64,
) -> tuple[list[RankedClause], bool]:
    """
    Dynamic query-time graph surf: HNSW seed retrieval -> Personalized
    PageRank walk over the idea graph -> top-M ranked clauses.

    Args:
        query_vector: Raw (not necessarily normalized) query embedding, same
                       dimensionality as the corpus embeddings.
        graph_index:   Pre-built GraphIndex for the target document.
        seed_top_n:    Number of HNSW nearest neighbors to seed the walk from.
        result_top_m:  Number of top-ranked clauses to return.
        alpha:         PageRank damping factor.
        max_iter, tol: PageRank power-iteration controls.
        ef_search:     HNSW candidate list size for the seed search.

    Returns:
        (ranked_clauses, used_fallback)
        ranked_clauses: top-M RankedClause objects, best first.
        used_fallback:  True if PPR could not run (disconnected graph / no
                         convergence) and plain HNSW similarity ranking was
                         used instead.

    Raises:
        GraphIndexError: if even the fallback seed search fails — this is
                          the only condition under which no results can be
                          produced at all.
    """
    seeds = _find_seed_nodes(query_vector, graph_index, seed_top_n, ef_search)

    if not seeds:
        raise GraphIndexError("HNSW seed search returned no candidates — empty corpus?")

    def _fallback() -> list[RankedClause]:
        """Standard local seed-node retrieval: rank purely by cosine similarity."""
        ranked = []
        for sid, sim in sorted(seeds, key=lambda x: x[1], reverse=True)[:result_top_m]:
            node = graph_index.nodes_by_id.get(sid, {})
            ranked.append(RankedClause(
                sentence_id=sid,
                sentence=node.get("sentence", ""),
                score=sim,
                is_seed=True,
                cluster_id=node.get("cluster_id"),
                paragraph_id=node.get("paragraph_id"),
            ))
        return ranked

    # Graceful fallback #1: graph has no edges at all (e.g. every node was
    # isolated by Phase 3's similarity floor, or a degenerate 1-2 node corpus).
    # PageRank is mathematically well-defined here but topologically useless —
    # it can never redistribute rank beyond the seed set, so skip straight
    # to plain similarity ranking instead of pretending a graph walk happened.
    if graph_index.nx_graph.number_of_edges() == 0:
        log.warning(
            "GraphIndex for %s has zero edges — falling back to seed-only retrieval.",
            graph_index.data_dir,
        )
        return _fallback(), True

    # Build the personalization vector: weight strictly on seed nodes,
    # normalized to sum to 1.0. Seeds with higher cosine similarity get
    # proportionally more of the restart mass.
    seed_weights = {sid: max(sim, 0.0) for sid, sim in seeds}
    total_weight = sum(seed_weights.values())

    if total_weight <= 0.0:
        # All seed similarities were <= 0 (degenerate / near-orthogonal query).
        # Fall back rather than dividing by zero or feeding pagerank a
        # meaningless uniform-looking vector that isn't actually seeded.
        log.warning(
            "All seed similarities <= 0 for query against %s — falling back.",
            graph_index.data_dir,
        )
        return _fallback(), True

    personalization = {
        sid: weight / total_weight
        for sid, weight in seed_weights.items()
        if sid in graph_index.nx_graph
    }

    if not personalization:
        # Seed nodes from HNSW aren't present in the graph (shouldn't happen
        # if artifacts are in sync, but data corruption / partial writes are
        # a real production failure mode worth guarding against explicitly).
        log.warning(
            "None of the HNSW seed nodes exist in the idea graph for %s — falling back.",
            graph_index.data_dir,
        )
        return _fallback(), True

    try:
        scores = nx.pagerank(
            graph_index.nx_graph,
            alpha=alpha,
            personalization=personalization,
            weight="weight",
            max_iter=max_iter,
            tol=tol,
        )
    except PowerIterationFailedConvergence as exc:
        log.warning(
            "PageRank failed to converge for %s (%s) — falling back to seed-only retrieval.",
            graph_index.data_dir, exc,
        )
        return _fallback(), True
    except nx.NetworkXError as exc:
        # Any other networkx-level failure (malformed graph, etc.) — same
        # graceful degradation rather than a 500 for the caller.
        log.warning(
            "PageRank raised a NetworkXError for %s (%s) — falling back.",
            graph_index.data_dir, exc,
        )
        return _fallback(), True

    seed_ids = set(personalization.keys())
    ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:result_top_m]

    ranked_clauses = [
        RankedClause(
            sentence_id=sid,
            sentence=graph_index.nodes_by_id.get(sid, {}).get("sentence", ""),
            score=float(score),
            is_seed=sid in seed_ids,
            cluster_id=graph_index.nodes_by_id.get(sid, {}).get("cluster_id"),
            paragraph_id=graph_index.nodes_by_id.get(sid, {}).get("paragraph_id"),
        )
        for sid, score in ranked_ids
    ]

    return ranked_clauses, False


# ---------------------------------------------------------------------------
# Rule-based heading-context expansion
#
# PPR is a global ranking: a moderate-weight sequential edge from a heading
# to the very next clause can still be outcompeted by higher-similarity
# edges elsewhere in the graph, so a heading-like clause can surface as a
# top result while the body text that immediately follows it (the actual
# content a human would obviously "keep reading" into) never makes the cut.
# This pass runs AFTER personalized_pagerank_search() and deterministically
# pulls in the next few clauses for any top-ranked clause that looks like a
# heading/title/introductory fragment, regardless of that clause's own PPR
# score. It is intentionally simple regex/rule-based — no ML, no LLM call —
# so it stays fast, deterministic, and cheap to reason about; false
# positives are acceptable since expansion only adds grounded context.
# ---------------------------------------------------------------------------

# A run of 2+ consecutive ALL-CAPS words at the very start of the clause,
# e.g. "NASTENKA'S HISTORY" in 'NASTENKA'S HISTORY "Half my story you know
# already..."'. Apostrophes/periods are allowed inside a caps word (O'BRIEN,
# U.S.).
_LEADING_CAPS_RE = re.compile(r"^((?:[A-Z][A-Z'’.]*\s+){1,6}[A-Z][A-Z'’.]*)\b")

# Three-or-more dot runs (with or without spaces, i.e. "..." or ". . .")
# or a unicode ellipsis character, anchored at the end of the clause.
_TRAILING_ELLIPSIS_RE = re.compile(r"(?:\.\s*){3,}$")


def _is_heading_like(sentence: str, min_words_for_heading: int) -> bool:
    """
    Deterministic, regex-based heuristic for "this clause looks like a
    heading, title, or introductory fragment that likely continues into
    the next clause(s)".

    Matches any of:
      1. Short clause (<= min_words_for_heading words) that is mostly
         ALL-CAPS, or mostly Title-Cased — a standalone heading pattern.
      2. A leading run of 2+ ALL-CAPS words at the start of the clause,
         even if the full clause (heading + quoted continuation) is longer
         than min_words_for_heading — covers "SECTION TITLE... actual
         quoted text starts here" in one clause.
      3. The clause trails off in an ellipsis ("..." / ". . ." / "…"),
         suggesting it's cut short and continues elsewhere.
      4. The clause contains a colon with only a handful of words after
         it (e.g. "TITLE: " or "Chapter Two: The Storm"), suggesting it
         introduces content rather than completing a thought.

    Deliberately over-inclusive: a false positive here just means an extra,
    still-grounded clause gets pulled into context, which the synthesizer's
    strict grounding instructions make cheap.
    """
    text = sentence.strip()
    if not text:
        return False

    words = text.split()
    word_count = len(words)

    # --- Check 1: short clause, mostly ALL-CAPS or mostly Title Case ---
    if word_count <= min_words_for_heading:
        letters = [c for c in text if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio >= 0.7:
                return True
        if word_count >= 2:
            capitalized = sum(1 for w in words if w[:1].isalpha() and w[0].isupper())
            if capitalized / word_count >= 0.7:
                return True

    # --- Check 2: leading ALL-CAPS run, regardless of overall clause length ---
    leading = _LEADING_CAPS_RE.match(text)
    if leading:
        prefix_letters = [c for c in leading.group(1) if c.isalpha()]
        if len(prefix_letters) >= 4 and all(c.isupper() for c in prefix_letters):
            return True

    # --- Check 3: trails off in an ellipsis ---
    if text.endswith("…") or _TRAILING_ELLIPSIS_RE.search(text):
        return True

    # --- Check 4: colon followed by very little additional text ---
    if ":" in text:
        after_colon = text.rsplit(":", 1)[1].strip()
        after_words = after_colon.split()
        if len(after_words) <= 5:
            return True

    return False


def _following_clause_ids(
    heading_id: int,
    heading_paragraph: Optional[int],
    nodes_by_id: dict[int, dict],
    hard_cap: int,
) -> list[int]:
    """
    Find up to `hard_cap` sentence_ids that come after `heading_id` in the
    source document, in document order, preferring paragraph-scoped
    grouping and falling back to raw sentence_id adjacency when
    paragraph_id is missing or unreliable.

    This returns the full candidate window up to the hard cap — it does
    NOT apply heading-boundary stopping itself. Boundary detection needs
    to inspect each candidate's sentence text against the heading heuristic
    in document order, which is `_expand_from_heading`'s job; this function
    is purely "what's the ordered pool of candidates to consider."

    "Unreliable" covers the observed real-world failure mode where certain
    PDF extraction outputs leave paragraph_id constant (or near-constant)
    across the entire document — in that case grouping by paragraph_id
    would effectively mean "everything after the heading in the whole
    corpus," which isn't a useful "next few clauses" signal, so raw
    sentence_id proximity is used instead.
    """
    def _raw_adjacent() -> list[int]:
        return [cid for cid in (heading_id + k for k in range(1, hard_cap + 1)) if cid in nodes_by_id]

    if heading_paragraph is None or not nodes_by_id:
        return _raw_adjacent()

    total_nodes = len(nodes_by_id)
    same_paragraph_ids = [
        sid for sid, node in nodes_by_id.items()
        if node.get("paragraph_id") == heading_paragraph and sid > heading_id
    ]

    # Guard: if this "paragraph" covers ~the whole corpus, paragraph_id isn't
    # a meaningful grouping signal here — fall back to raw adjacency instead
    # of silently returning up to `hard_cap` clauses from anywhere in the doc.
    if total_nodes > 5:
        in_paragraph = sum(
            1 for node in nodes_by_id.values() if node.get("paragraph_id") == heading_paragraph
        )
        if in_paragraph >= total_nodes * 0.9:
            return _raw_adjacent()

    if not same_paragraph_ids:
        return _raw_adjacent()

    same_paragraph_ids.sort()
    return same_paragraph_ids[:hard_cap]


def _expand_from_heading(
    heading_id: int,
    heading_paragraph: Optional[int],
    nodes_by_id: dict[int, dict],
    max_expansion_per_heading: int,
    stop_at_boundary: bool,
    min_expansion_before_boundary: int,
    min_words_for_heading: int,
) -> list[int]:
    """
    Walk forward from a heading-like clause, in document order, deciding
    per-candidate whether to keep expanding.

    Stops (without including the triggering clause) at the first of:
      - another heading-like clause, once at least
        `min_expansion_before_boundary` clauses have already been pulled in
        (so a suspiciously-immediate "heading" one or two clauses in — e.g.
        a snappy line of dialogue that happens to match the heuristic, or
        back-to-back short legal subsection headers — doesn't collapse
        expansion to near-zero), or
      - `max_expansion_per_heading` clauses pulled in (hard cap; always
        enforced regardless of `stop_at_boundary`, so this function can
        never run unbounded).

    If `stop_at_boundary` is False, boundary detection is skipped entirely
    and this always expands straight to the hard cap (old fixed-N-style
    behaviour, kept for rollback/comparison).

    Returns:
        Ordered list of sentence_ids to pull in as expansion context.
    """
    if max_expansion_per_heading <= 0:
        return []

    candidates = _following_clause_ids(
        heading_id, heading_paragraph, nodes_by_id, max_expansion_per_heading,
    )

    if not stop_at_boundary:
        return candidates[:max_expansion_per_heading]

    result: list[int] = []
    for cid in candidates:
        if len(result) >= max_expansion_per_heading:
            break

        node = nodes_by_id.get(cid, {})
        is_boundary = _is_heading_like(node.get("sentence", ""), min_words_for_heading)

        if is_boundary and len(result) >= min_expansion_before_boundary:
            # Genuine boundary — a new heading/section starts here, and
            # we've already pulled in a sensible minimum, so stop without
            # including the boundary clause itself.
            break

        # Either not a boundary, or a boundary encountered too early to
        # trust (fewer than min_expansion_before_boundary clauses so far) —
        # include it and keep walking.
        result.append(cid)

    return result


def expand_heading_context(
    ranked_clauses: list[RankedClause],
    graph_index: GraphIndex,
    max_expansion_per_heading: int = 15,
    min_words_for_heading: int = 8,
    stop_at_boundary: bool = True,
    min_expansion_before_boundary: int = 3,
) -> list[RankedClause]:
    """
    Rule-based context-expansion pass: for every heading-like clause among
    `ranked_clauses`, dynamically pull in the clauses that follow it (by
    document order) as additional context, regardless of their own PPR
    score, and append them to the returned list.

    Expansion length is dynamic but safely bounded: it walks forward,
    clause by clause, until either another heading-like clause is hit
    (a natural "this section is over" signal — reusing the same heuristic
    that flags the trigger clause itself) or `max_expansion_per_heading` is
    reached, whichever comes first. `max_expansion_per_heading` is now a
    hard ceiling, not a fixed target — most expansions will stop well
    before it once real content gives way to the next heading/section.

    To avoid a suspiciously-immediate false-positive boundary (e.g. a snappy
    line of dialogue right after the heading, or back-to-back short legal
    subsection headers) collapsing expansion to near-zero, a boundary is
    only honored once at least `min_expansion_before_boundary` clauses have
    already been pulled in; boundary-like clauses encountered before that
    are included anyway.

    Only the clauses PPR actually ranked are treated as expansion triggers
    (expansion never chains off clauses that were themselves added by this
    same call), which keeps the pass bounded: at most
    `len(ranked_clauses) * max_expansion_per_heading` new clauses are ever
    added, regardless of `stop_at_boundary`.

    Args:
        ranked_clauses: Output of personalized_pagerank_search() (or its
                         fallback path) — read-only; a new list is returned.
        graph_index:     The GraphIndex the ranking was produced against —
                         used to look up paragraph_id / sentence text for
                         candidate expansion clauses via nodes_by_id.
        max_expansion_per_heading: Hard cap on clauses pulled in per detected
                                    heading — always enforced, with or
                                    without boundary stopping.
        min_words_for_heading:     Word-count threshold used by the "short
                                    clause" heading heuristic.
        stop_at_boundary:           If True (default), expansion stops early
                                    at the next heading-like clause. If
                                    False, always expands straight to
                                    `max_expansion_per_heading` (old fixed-N
                                    behaviour — kept for rollback/comparison).
        min_expansion_before_boundary: Minimum clauses pulled in before an
                                    encountered heading-like clause is
                                    trusted as a real boundary.

    Returns:
        A new list: the original ranked_clauses in their original order,
        followed by any newly-added expansion clauses (each with
        `expansion_source` set to the sentence_id of the heading that
        pulled it in). Clauses already present in ranked_clauses are never
        duplicated.
    """
    if not ranked_clauses or max_expansion_per_heading <= 0:
        return list(ranked_clauses)

    existing_ids = {c.sentence_id for c in ranked_clauses}
    expanded = list(ranked_clauses)

    # Snapshot the original clauses as the only expansion triggers — clauses
    # appended during this loop must not themselves trigger further expansion.
    for clause in list(ranked_clauses):
        if not _is_heading_like(clause.sentence, min_words_for_heading):
            continue

        heading_id = clause.sentence_id
        heading_node = graph_index.nodes_by_id.get(heading_id, {})
        heading_paragraph = heading_node.get("paragraph_id")

        candidate_ids = _expand_from_heading(
            heading_id,
            heading_paragraph,
            graph_index.nodes_by_id,
            max_expansion_per_heading=max_expansion_per_heading,
            stop_at_boundary=stop_at_boundary,
            min_expansion_before_boundary=min_expansion_before_boundary,
            min_words_for_heading=min_words_for_heading,
        )

        for cid in candidate_ids:
            if cid in existing_ids:
                continue
            node = graph_index.nodes_by_id.get(cid)
            if node is None:
                continue
            expanded.append(RankedClause(
                sentence_id=cid,
                sentence=node.get("sentence", ""),
                score=0.0,
                is_seed=False,
                cluster_id=node.get("cluster_id"),
                paragraph_id=node.get("paragraph_id"),
                expansion_source=heading_id,
            ))
            existing_ids.add(cid)

    return expanded