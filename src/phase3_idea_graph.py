#!/usr/bin/env python3
"""
phase3_idea_graph.py

Phase 3 of the NLP pipeline: Build an idea graph with intelligent clustering.

Loads sentence embeddings and original sentence metadata, builds a FAISS
HNSW approximate-nearest-neighbor index over the (L2-normalised) embeddings,
applies dynamic per-node thresholding, detects topic clusters via HDBSCAN
(with Louvain as fallback, built from HNSW-derived edges), computes
per-cluster centrality metrics, and saves the enriched graph as JSON.

Milestone 2 (structural hardening): there is no brute-force O(N^2)
cosine_similarity path any more, at any corpus size. Every neighbor lookup —
clustering edges, graph edges, floor calibration — is served by a single
FAISS HNSW index built once per run.
"""

import json
import logging
import os
import warnings
from typing import Optional

import numpy as np
from sklearn.preprocessing import normalize

log = logging.getLogger("phase3")
if not log.handlers:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

# --- Optional clustering backends (graceful fallback if missing) ---
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("hdbscan not installed — falling back to Louvain clustering.")

try:
    import networkx as nx
    from community import best_partition as louvain_partition   # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    warnings.warn("networkx / python-louvain not installed — cluster step will be skipped.")

# --- FAISS is now a HARD requirement ---
# Milestone 2 removes the exact O(N^2) cosine_similarity fallback entirely,
# so there is no longer a code path that works without FAISS installed.
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError as exc:
    FAISS_AVAILABLE = False
    raise ImportError(
        "faiss is required by phase3_idea_graph.py — there is no brute-force "
        "fallback. Install with: pip install faiss-cpu  (or faiss-gpu for CUDA)"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration — loaded from config.yaml
# ---------------------------------------------------------------------------
from config_loader import load_config as _load_config
_cfg = _load_config()

EMBEDDINGS_PATH  = _cfg.paths.full("output_embeddings")
SENTENCES_PATH   = _cfg.paths.full("output_sentences")
OUTPUT_PATH      = _cfg.paths.full("output_graph")
CLUSTERS_PATH    = _cfg.paths.full("output_clusters")
MANIFEST_PATH    = _cfg.paths.full("output_manifest")

TOP_K                   = _cfg.phase3.top_k
DYNAMIC_THRESHOLD_SIGMA = _cfg.phase3.dynamic_threshold_sigma
GLOBAL_THRESHOLD_MIN    = _cfg.phase3.global_threshold_min
GLOBAL_THRESHOLD_MAX    = _cfg.phase3.global_threshold_max

HDBSCAN_MIN_CLUSTER     = _cfg.phase3.hdbscan_min_cluster
HDBSCAN_MIN_SAMPLES     = _cfg.phase3.hdbscan_min_samples
HDBSCAN_METRIC          = _cfg.phase3.hdbscan_metric

LOUVAIN_RESOLUTION      = _cfg.phase3.louvain_resolution
LOUVAIN_EDGE_THRESHOLD  = _cfg.phase3.louvain_edge_threshold

ENABLE_SEQUENTIAL_EDGES            = _cfg.phase3.enable_sequential_edges
SEQUENTIAL_WINDOW                  = _cfg.phase3.sequential_window
SEQUENTIAL_EDGE_WEIGHT              = _cfg.phase3.sequential_edge_weight
ENABLE_CROSS_PARAGRAPH_SEQUENTIAL   = _cfg.phase3.enable_cross_paragraph_sequential

HNSW_M                  = _cfg.phase3.hnsw_m
HNSW_EF_CONSTRUCTION    = _cfg.phase3.hnsw_ef_construction
HNSW_EF_SEARCH          = _cfg.phase3.hnsw_ef_search


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_embeddings(filepath: str) -> np.ndarray:
    embeddings = np.load(filepath).astype(np.float32)
    print(f"Embeddings loaded  : shape={embeddings.shape}, dtype={embeddings.dtype}")
    return embeddings


def load_sentences(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"Sentences loaded   : {len(data)} records")
    return data


def load_manifest(filepath: str) -> Optional[list[dict]]:
    """Load the Phase 2 chunk manifest if it exists (may be absent for un-chunked runs)."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    print(f"Manifest loaded    : {len(manifest)} chunk entries")
    return manifest


# ---------------------------------------------------------------------------
# Similarity — exact (small corpora) and FAISS ANN (large corpora)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Similarity — FAISS HNSW only (no brute-force path, any corpus size)
# ---------------------------------------------------------------------------

def build_hnsw_index(
    embeddings: np.ndarray,
    m: int = HNSW_M,
    ef_construction: int = HNSW_EF_CONSTRUCTION,
) -> "faiss.IndexHNSWFlat":
    """
    Build a FAISS HNSW index over L2-normalised embeddings.

    Metric: METRIC_INNER_PRODUCT. Embeddings are unit-normalised by Phase 2
    (normalize_embeddings=True in generate_embeddings), so inner product is
    exactly cosine similarity — no separate distance-to-similarity conversion
    is needed anywhere downstream.

    Args:
        embeddings:      float32 array, shape (N, D). Assumed L2-normalised.
        m:               HNSW graph connectivity (neighbors per node).
        ef_construction: candidate list size during build — quality/build-time knob.

    Returns:
        A populated faiss.IndexHNSWFlat.

    Raises:
        RuntimeError: if index construction fails (e.g. malformed embeddings).
    """
    n, d = embeddings.shape
    try:
        vecs = np.ascontiguousarray(embeddings.astype(np.float32))
        # Defensive re-normalisation — guards against upstream callers passing
        # raw (non-unit) vectors, which would silently break the cosine ≡ IP identity.
        vecs = normalize(vecs, norm="l2").astype(np.float32)

        index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(vecs)
        log.info(
            "HNSW index built: %d vectors, dim=%d, M=%d, efConstruction=%d",
            n, d, m, ef_construction,
        )
        return index
    except Exception as exc:
        raise RuntimeError(f"Failed to build FAISS HNSW index: {exc}") from exc


def compute_neighbors_hnsw(
    embeddings: np.ndarray,
    top_k: int = TOP_K,
    m: int = HNSW_M,
    ef_construction: int = HNSW_EF_CONSTRUCTION,
    ef_search: int = HNSW_EF_SEARCH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the HNSW index and run a top-k self-search in one pass.

    This is the single neighbor-search call reused for floor calibration,
    Louvain edge construction, and final graph construction — it is run
    exactly once per pipeline execution.

    Args:
        embeddings:      L2-normalised float32 array, shape (N, D).
        top_k:           Neighbors to retrieve per query (hard cap on graph degree).
        m, ef_construction: passed through to build_hnsw_index.
        ef_search:       Candidate list size at query time. Raised to at least
                          top_k+1 automatically — a smaller value would silently
                          truncate recall below what top_k promises.

    Returns:
        similarities : (N, top_k+1) float32 — cosine similarities, self at rank 0.
        indices      : (N, top_k+1) int64   — neighbor sentence indices.

    Raises:
        RuntimeError: if the search fails after a successfully built index.
    """
    n, _ = embeddings.shape
    index = build_hnsw_index(embeddings, m=m, ef_construction=ef_construction)

    # efSearch must be >= k or HNSW silently returns fewer/worse candidates.
    index.hnsw.efSearch = max(ef_search, top_k + 1)

    try:
        vecs = np.ascontiguousarray(normalize(embeddings, norm="l2").astype(np.float32))
        k = min(top_k + 1, n)   # +1 — HNSW includes the query itself as neighbor 0
        similarities, indices = index.search(vecs, k)
        log.info("HNSW search done: %d queries, top-%d neighbors each (efSearch=%d)",
                  n, k, index.hnsw.efSearch)
        return similarities, indices
    except Exception as exc:
        raise RuntimeError(f"FAISS HNSW search failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Dynamic threshold
# ---------------------------------------------------------------------------

def dynamic_threshold(
    similarities: np.ndarray,
    sigma: float = DYNAMIC_THRESHOLD_SIGMA,
    floor: float = GLOBAL_THRESHOLD_MIN,
    ceiling: float = GLOBAL_THRESHOLD_MAX,
) -> float:
    """
    Compute a per-node similarity cutoff.

    Strategy: use the mean of the *positive* similarity distribution minus
    sigma * std.  This adapts to nodes that live in dense regions (high mean
    → higher bar) vs sparse regions (low mean → lower bar), preventing
    hub-node explosion and isolated-node starvation simultaneously.

    Args:
        similarities: 1-D array of cosine scores for one source node
                      (diagonal already zeroed out).
        sigma:        How many standard deviations below the mean to place
                      the cutoff.  Lower = more edges.
        floor:        Hard minimum — never accept weaker links than this.
        ceiling:      Hard maximum — always accept links at least this strong.

    Returns:
        Scalar threshold in [floor, ceiling].
    """
    positive = similarities[similarities > 0.0]
    if positive.size == 0:
        return ceiling  # no signal → accept nothing below ceiling

    mu, sd = float(positive.mean()), float(positive.std())
    threshold = mu - sigma * sd
    # Clamp to [floor, ceiling]
    return float(np.clip(threshold, floor, ceiling))


def calibrate_floor(
    hnsw_similarities: np.ndarray,
    percentile: float = 10.0,
    hard_minimum: float = 0.20,
) -> float:
    """
    Auto-calibrate the similarity floor from the actual distribution of
    positive similarities returned by the HNSW neighbor search.

    The hardcoded GLOBAL_THRESHOLD_MIN = 0.35 was tuned on one literary
    document. A legal contract, technical manual, or academic paper has a
    very different similarity distribution — calibrating from the data means
    the same pipeline works across domains without manual tuning.

    Strategy:
        Set the floor at the Nth percentile of all positive similarities
        already retrieved by the HNSW top-k search (no extra computation —
        this reuses the same neighbor search used for graph construction).

    Args:
        hnsw_similarities : (N, K) cosine similarities from compute_neighbors_hnsw.
        percentile         : Percentile of positive similarities to use as floor.
                              10.0 = the bottom 10% of edges are excluded.
        hard_minimum       : Absolute floor — never go below this regardless of
                              corpus (0.20 = below this is noise in any embedding
                              space produced by sentence-transformers).

    Returns:
        Calibrated floor value in [hard_minimum, GLOBAL_THRESHOLD_MAX].
    """
    positives = hnsw_similarities[hnsw_similarities > 0.0].ravel()

    if positives.size == 0:
        log.warning("calibrate_floor: no positive similarities — using hardcoded floor %.4f",
                    GLOBAL_THRESHOLD_MIN)
        return GLOBAL_THRESHOLD_MIN

    calibrated = float(np.percentile(positives, percentile))
    calibrated = max(calibrated, hard_minimum)
    calibrated = min(calibrated, GLOBAL_THRESHOLD_MAX)

    log.info(
        "calibrate_floor: p%.0f of %s positive sims = %.4f  (range [%.4f, %.4f])",
        percentile, f"{positives.size:,}", calibrated, positives.min(), positives.max(),
    )
    return calibrated


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_hdbscan(embeddings: np.ndarray) -> np.ndarray:
    """
    Cluster embeddings with HDBSCAN.

    HDBSCAN naturally discovers variable-density clusters and marks
    low-confidence points as noise (label = -1), which we later treat as
    singleton clusters.

    Returns:
        Integer label array of length N.  Noise points get label == -1.
    """
    print("Clustering with HDBSCAN…")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method="eom",   # Excess of Mass — finds compact clusters
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"  HDBSCAN → {n_clusters} clusters, {n_noise} noise points")
    return labels


def cluster_louvain(
    n_nodes: int,
    hnsw_similarities: np.ndarray,
    hnsw_indices: np.ndarray,
    edge_threshold: float = LOUVAIN_EDGE_THRESHOLD,
    resolution: float = LOUVAIN_RESOLUTION,
) -> np.ndarray:
    """
    Cluster nodes with the Louvain community-detection algorithm.

    Louvain works on a graph rather than the raw embedding space. Edges are
    built directly from the HNSW top-k neighbor search — there is no full
    similarity matrix to threshold against, so any pair not surfaced by the
    HNSW search (i.e. not in either node's top-k) is simply never considered
    as a candidate edge. This is a deliberate approximation: HNSW recall is
    high enough that strong edges are reliably found from either endpoint.

    Args:
        n_nodes:         Total number of nodes (embeddings.shape[0]).
        hnsw_similarities: (N, K) cosine similarities from compute_neighbors_hnsw.
        hnsw_indices:     (N, K) corresponding neighbor indices.
        edge_threshold:   Only wire the Louvain graph with edges above this.
        resolution:       Higher = more, smaller communities.

    Returns:
        Integer label array of length N.
    """
    log.info("Clustering with Louvain (HNSW-derived edges)…")
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        for sim_val, j in zip(hnsw_similarities[i], hnsw_indices[i]):
            j = int(j)
            if j == i or j < 0:
                continue
            score = float(sim_val)
            if score < edge_threshold:
                continue
            a, b = (i, j) if i < j else (j, i)
            if G.has_edge(a, b):
                if score > G[a][b]["weight"]:
                    G[a][b]["weight"] = score
            else:
                G.add_edge(a, b, weight=score)

    partition = louvain_partition(G, weight="weight", resolution=resolution)
    labels    = np.array([partition.get(i, -1) for i in range(n_nodes)], dtype=int)

    n_clusters = len(set(labels))
    log.info("Louvain → %d communities (%d edges)", n_clusters, G.number_of_edges())
    return labels


def assign_clusters(
    embeddings: np.ndarray,
    hnsw_similarities: np.ndarray,
    hnsw_indices: np.ndarray,
) -> np.ndarray:
    """
    Pick the best available clustering algorithm.

    Priority: HDBSCAN > Louvain > no clustering (all zeros).
    """
    if HDBSCAN_AVAILABLE:
        return cluster_hdbscan(embeddings)
    if LOUVAIN_AVAILABLE:
        return cluster_louvain(len(embeddings), hnsw_similarities, hnsw_indices)

    warnings.warn("No clustering backend available — all nodes assigned cluster 0.")
    return np.zeros(len(embeddings), dtype=int)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_idea_graph(
    hnsw_similarities: np.ndarray,
    hnsw_indices: np.ndarray,
    sentence_records: list[dict],
    cluster_labels: np.ndarray,
    manifest: Optional[list[dict]],
    top_k: int = TOP_K,
    calibrated_floor: float = GLOBAL_THRESHOLD_MIN,
) -> list[dict]:
    """
    Build the idea graph from HNSW ANN results — the only graph-building
    path now (Milestone 2 removed the exact O(N^2) path entirely).

    HNSW returns pre-selected top-k neighbors so no full matrix is ever
    materialised, at any corpus size. Dynamic threshold is computed only
    over the returned neighbor similarities (not a full row), which is a
    reasonable approximation since HNSW recall is high and the global
    mean/std of similarities changes slowly with corpus size.

    Args:
        hnsw_similarities : (N, K) cosine similarities from compute_neighbors_hnsw.
        hnsw_indices       : (N, K) corresponding sentence indices.
        sentence_records   : Original sentence metadata.
        cluster_labels      : Per-node cluster assignment.
        manifest            : Optional Phase 2 chunk manifest.
        top_k               : Hard cap on neighbors per node.
        calibrated_floor    : Document-adaptive floor from calibrate_floor().

    Returns:
        List of enriched graph-node dicts.
    """
    n     = len(sentence_records)
    graph = []
    log.info("Building idea graph (HNSW, top_k=%d, floor=%.4f)…", top_k, calibrated_floor)

    for i in range(n):
        raw_sims = hnsw_similarities[i]
        raw_idxs = hnsw_indices[i]

        # Filter out self-match and invalid indices (-1 from HNSW padding when
        # the corpus is smaller than top_k+1).
        valid_mask = (raw_idxs != i) & (raw_idxs >= 0)
        sims = raw_sims[valid_mask].astype(np.float32)
        idxs = raw_idxs[valid_mask]

        thresh = dynamic_threshold(sims, floor=calibrated_floor) if len(sims) > 0 else calibrated_floor

        neighbors = []
        for sim_val, j in zip(sims, idxs):
            score = float(sim_val)
            if score < thresh:
                continue
            neighbors.append({
                "sentence_id":   int(j),
                "similarity":    round(score, 6),
                "cross_cluster": bool(cluster_labels[i] != cluster_labels[j]),
                "edge_type":     "similarity",
            })
            if len(neighbors) >= top_k:
                break

        chunk_meta = manifest[i] if manifest else {}
        graph.append({
            "sentence_id":    i,
            "sentence":       sentence_records[i]["sentence"],
            "paragraph_id":   sentence_records[i].get("paragraph_id"),
            "cluster_id":     int(cluster_labels[i]),
            "threshold_used": round(thresh, 6),
            "chunk_index":    chunk_meta.get("chunk_index", 0),
            "total_chunks":   chunk_meta.get("total_chunks", 1),
            "neighbors":      neighbors,
        })

    return graph


# ---------------------------------------------------------------------------
# Sequential / proximity edges
# ---------------------------------------------------------------------------

def add_sequential_edges(
    graph: list[dict],
    cluster_labels: np.ndarray,
    enable_sequential_edges: bool = ENABLE_SEQUENTIAL_EDGES,
    sequential_window: int = SEQUENTIAL_WINDOW,
    sequential_edge_weight: float = SEQUENTIAL_EDGE_WEIGHT,
    enable_cross_paragraph_sequential: bool = ENABLE_CROSS_PARAGRAPH_SEQUENTIAL,
) -> list[dict]:
    """
    Add narrative/document-order edges alongside the similarity edges
    already present in `graph`, mutating and returning it in place.

    Purely semantic (FAISS/cosine) edges miss a real retrieval case: a
    heading clause ("NASTENKA'S HISTORY... I have an old grandmother") is
    often *not* strongly similar to the body clauses that immediately
    follow it, even though a human reader would obviously "keep reading"
    from one into the next. Personalized PageRank can only walk edges that
    exist, so without this pass a heading-only seed match can never reach
    its own body text.

    This pass connects each node i to the next `sequential_window` nodes in
    original document order (i -> i+1 .. i+window), scoped to the same
    paragraph_id unless `enable_cross_paragraph_sequential` is set. It is
    strictly additive: existing similarity edges, clustering, and floor
    calibration are untouched.

    Node/index invariant this function relies on: build_idea_graph() always
    appends nodes in the same order as sentence_records (for i in range(n)),
    so graph[i]["sentence_id"] == i and "the next clause" is simply graph[i+1].

    Args:
        graph:                  Output of build_idea_graph() — mutated in place.
        cluster_labels:         Per-node cluster assignment (same array passed
                                 into build_idea_graph), used only to populate
                                 the "cross_cluster" field on new edges so the
                                 schema stays consistent with similarity edges.
        enable_sequential_edges: Master switch — no-op and returns graph
                                 unchanged if False.
        sequential_window:      How many following clauses each node links to.
        sequential_edge_weight: Fixed weight assigned to new sequential edges
                                 (there is no similarity score to use instead).
        enable_cross_paragraph_sequential: If False (default), a sequential
                                 edge is only added when both clauses share the
                                 same paragraph_id — prevents bridging across
                                 section boundaries, which would reintroduce
                                 the noise dynamic thresholding was built to
                                 avoid. If True, the paragraph_id check is
                                 skipped entirely.

    Returns:
        The same `graph` list, mutated in place (also returned for convenient
        chaining/assignment at the call site).
    """
    if not enable_sequential_edges:
        return graph

    n = len(graph)
    added, tagged_both = 0, 0

    for i in range(n):
        node = graph[i]
        # Existing neighbor sentence_ids for this node, for O(1) dedupe lookup.
        existing_by_id = {nb["sentence_id"]: nb for nb in node["neighbors"]}
        src_paragraph = node.get("paragraph_id")

        for offset in range(1, sequential_window + 1):
            j = i + offset
            if j >= n:
                break

            if not enable_cross_paragraph_sequential:
                if src_paragraph != graph[j].get("paragraph_id"):
                    continue

            existing = existing_by_id.get(j)
            if existing is not None:
                # A similarity edge already connects these two nodes — don't
                # create a second entry for the same (src, dst) pair, just
                # mark that document order also supports this edge.
                if existing.get("edge_type") != "both":
                    existing["edge_type"] = "both"
                    tagged_both += 1
                continue

            new_edge = {
                "sentence_id":   j,
                "similarity":    round(float(sequential_edge_weight), 6),
                "cross_cluster": bool(cluster_labels[i] != cluster_labels[j]),
                "edge_type":     "sequential",
            }
            node["neighbors"].append(new_edge)
            existing_by_id[j] = new_edge
            added += 1

    log.info(
        "Sequential edges: %d added, %d existing similarity edges tagged 'both' "
        "(window=%d, weight=%.3f, cross_paragraph=%s)",
        added, tagged_both, sequential_window, sequential_edge_weight,
        enable_cross_paragraph_sequential,
    )
    return graph


# ---------------------------------------------------------------------------
# Per-cluster centrality
# ---------------------------------------------------------------------------

def compute_cluster_centrality(
    graph: list[dict],
    cluster_labels: np.ndarray,
) -> dict[int, list[dict]]:
    """
    Compute intra-cluster degree centrality for every node.

    Approximated entirely from the neighbor similarity scores already stored
    in each graph node by build_idea_graph() — no full matrix is built at any
    point in Phase 3 any more, so this is the only path.
    """
    log.info("Computing per-cluster centrality…")
    unique_clusters = sorted(set(int(l) for l in cluster_labels))
    cluster_centrality: dict[int, list[dict]] = {}

    for cid in unique_clusters:
        members = [i for i, l in enumerate(cluster_labels) if int(l) == cid]
        if len(members) < 2:
            cluster_centrality[cid] = [{
                "sentence_id": members[0] if members else -1,
                "sentence":    graph[members[0]]["sentence"] if members else "",
                "centrality":  0.0,
            }]
            continue

        member_set = set(members)
        weighted_degree = {}
        for i in members:
            intra_sims = [
                nb["similarity"] for nb in graph[i]["neighbors"]
                if nb["sentence_id"] in member_set
            ]
            denom = max(len(members) - 1, 1)
            weighted_degree[i] = sum(intra_sims) / denom
        ranked = sorted(
            weighted_degree.items(),
            key=lambda x: x[1], reverse=True,
        )

        cluster_centrality[cid] = [
            {
                "sentence_id": sid,
                "sentence":    graph[sid]["sentence"],
                "centrality":  round(float(cent), 6),
            }
            for sid, cent in ranked
        ]

    return cluster_centrality


def build_cluster_summary(
    cluster_centrality: dict[int, list[dict]],
    cluster_labels: np.ndarray,
) -> list[dict]:
    """
    Produce a flat, human-readable cluster summary sorted by cluster size.

    Args:
        cluster_centrality: Output of compute_cluster_centrality.
        cluster_labels:     Per-node cluster labels.

    Returns:
        List of cluster summary dicts ordered by descending member count.
    """
    summaries = []
    for cid, ranked_nodes in cluster_centrality.items():
        members     = [i for i, l in enumerate(cluster_labels) if int(l) == cid]
        top_node    = ranked_nodes[0] if ranked_nodes else {}
        summaries.append(
            {
                "cluster_id":     cid,
                "is_noise":       cid == -1,
                "size":           len(members),
                "centroid_sentence_id": top_node.get("sentence_id"),
                "centroid_sentence":    top_node.get("sentence", ""),
                "centroid_centrality":  top_node.get("centrality", 0.0),
                "top_nodes":      ranked_nodes[:5],   # top-5 representatives
            }
        )
    summaries.sort(key=lambda x: x["size"], reverse=True)
    return summaries


# ---------------------------------------------------------------------------
# Persistence & reporting
# ---------------------------------------------------------------------------

def save_json(obj: object, filepath: str, label: str = "File", pretty: bool = False) -> None:
    """
    Persist *obj* as JSON.

    Args:
        pretty: If True, use indent=2 for human readability (cluster summary).
                If False (default), write compact JSON — faster and ~60% smaller,
                appropriate for machine-consumed files like idea_graph.json.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2 if pretty else None)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"{label} saved        → {filepath}  ({size_kb:.1f} KB)")


def print_summary(
    graph: list[dict],
    cluster_summary: list[dict],
) -> None:
    total_nodes  = len(graph)
    total_edges  = sum(len(n["neighbors"]) for n in graph)
    cross_edges  = sum(
        1 for node in graph for nb in node["neighbors"] if nb["cross_cluster"]
    )
    isolated     = sum(1 for n in graph if not n["neighbors"])
    all_sims     = [nb["similarity"] for n in graph for nb in n["neighbors"]]
    avg_sim      = float(np.mean(all_sims)) if all_sims else 0.0
    all_thresh   = [n["threshold_used"] for n in graph]
    avg_thresh   = float(np.mean(all_thresh)) if all_thresh else 0.0

    print("\n" + "=" * 55)
    print("IDEA GRAPH SUMMARY")
    print("=" * 55)
    print(f"  Nodes                   : {total_nodes}")
    print(f"  Directed edges          : {total_edges}")
    print(f"  Cross-cluster edges     : {cross_edges}  ({100*cross_edges/max(total_edges,1):.1f} %)")
    print(f"  Isolated nodes          : {isolated}")
    print(f"  Avg edge similarity     : {avg_sim:.4f}")
    print(f"  Avg dynamic threshold   : {avg_thresh:.4f}")
    print(f"\n  Clusters detected       : {len(cluster_summary)}")
    print("-" * 55)
    for cs in cluster_summary[:8]:  # show top-8 by size
        noise_tag = " [noise]" if cs["is_noise"] else ""
        print(
            f"  Cluster {cs['cluster_id']:>3}{noise_tag:<8}"
            f"  size={cs['size']:>4}   "
            f"  centroid: \"{cs['centroid_sentence'][:60]}…\""
        )
    if len(cluster_summary) > 8:
        print(f"  … and {len(cluster_summary) - 8} more clusters")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Phase 3 entry point: HNSW search → cluster → graph → centrality → save."""

    # 1. Load
    try:
        embeddings       = load_embeddings(EMBEDDINGS_PATH)
        sentence_records = load_sentences(SENTENCES_PATH)
        manifest          = load_manifest(MANIFEST_PATH)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        log.error("Failed to load Phase 3 inputs: %s", exc)
        raise

    # Validate and align all three data sources.
    # The authoritative length is len(embeddings) — if sentence_records or
    # manifest has a different count, truncate or raise accordingly.
    n_emb  = len(embeddings)
    n_sent = len(sentence_records)
    n_man  = len(manifest) if manifest is not None else None

    if manifest is not None and n_man != n_emb:
        raise ValueError(
            f"Embedding / manifest count mismatch: "
            f"{n_emb} embeddings vs {n_man} manifest entries. "
            "Re-run Phase 2."
        )

    if n_sent != n_emb:
        # Common cause: Phase 1 produced one extra blank/duplicate that Phase 2
        # deduplicated, leaving sentence_records 1 longer than embeddings.
        # Safe fix: truncate sentence_records to match embeddings.
        if abs(n_sent - n_emb) <= 5:
            log.warning(
                "sentence_records (%d) vs embeddings (%d) — trimming sentence_records "
                "to %d (likely a blank/duplicate removed by Phase 2 deduplication).",
                n_sent, n_emb, n_emb,
            )
            sentence_records = sentence_records[:n_emb]
        else:
            raise ValueError(
                f"Embedding / sentence count mismatch is too large to auto-fix: "
                f"{n_emb} embeddings vs {n_sent} sentence records. "
                "Re-run Phase 1 and Phase 2."
            )

    n = len(embeddings)
    log.info("Corpus size: %d sentences", n)

    # 2. Single HNSW neighbor search — reused for floor calibration, optional
    #    Louvain edges, and final graph construction. Computed exactly once.
    try:
        hnsw_similarities, hnsw_indices = compute_neighbors_hnsw(embeddings, top_k=TOP_K)
    except RuntimeError as exc:
        log.error("HNSW index/search failed — cannot continue Phase 3: %s", exc)
        raise

    calibrated_floor = calibrate_floor(hnsw_similarities)

    # 3. Cluster — HDBSCAN operates on raw embeddings directly; Louvain builds
    #    its graph from the same HNSW neighbor arrays computed above.
    if HDBSCAN_AVAILABLE:
        cluster_labels = cluster_hdbscan(embeddings)
    elif LOUVAIN_AVAILABLE:
        cluster_labels = cluster_louvain(n, hnsw_similarities, hnsw_indices)
    else:
        warnings.warn("No clustering backend available — all nodes assigned cluster 0.")
        cluster_labels = np.zeros(n, dtype=int)

    # 4. Build graph directly from the HNSW search results — no second search.
    try:
        graph = build_idea_graph(
            hnsw_similarities, hnsw_indices, sentence_records, cluster_labels, manifest,
            top_k=TOP_K, calibrated_floor=calibrated_floor,
        )
    except Exception as exc:
        log.error("Idea graph construction failed: %s", exc)
        raise

    # 4b. Sequential/proximity edges — additive pass alongside the similarity
    #     edges above; lets query-time PPR "keep reading" contiguous text.
    graph = add_sequential_edges(graph, cluster_labels)

    # 5. Per-cluster centrality (approximated from stored neighbor similarities)
    cluster_centrality = compute_cluster_centrality(graph, cluster_labels)
    cluster_summary    = build_cluster_summary(cluster_centrality, cluster_labels)

    # 6. Report
    print_summary(graph, cluster_summary)

    # 7. Persist — idea graph is machine-consumed (compact), cluster summary is human-readable (pretty)
    try:
        save_json(graph,           OUTPUT_PATH,   label="Idea graph",      pretty=False)
        save_json(cluster_summary, CLUSTERS_PATH, label="Cluster summary", pretty=True)
    except OSError as exc:
        log.error("Failed to persist Phase 3 outputs: %s", exc)
        raise


if __name__ == "__main__":
    main()