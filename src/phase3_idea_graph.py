#!/usr/bin/env python3
"""
phase3_idea_graph.py

Phase 3 of the NLP pipeline: Build an idea graph with intelligent clustering.

Loads sentence embeddings and original sentence metadata, computes pairwise
cosine similarity, applies dynamic per-node thresholding, detects topic
clusters via HDBSCAN (with Louvain as fallback), computes per-cluster
centrality metrics, and saves the enriched graph as JSON.
"""

import json
import os
import warnings
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# --- Optional heavy deps (graceful fallback if missing) ---
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

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # FAISS not required — falls back to exact O(N²) dot product.
    # Install with: pip install faiss-cpu   (or faiss-gpu for CUDA)


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

FAISS_THRESHOLD         = _cfg.phase3.faiss_threshold
FAISS_N_PROBE           = _cfg.phase3.faiss_n_probe
FAISS_N_LIST            = _cfg.phase3.faiss_n_list


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

def compute_similarity_matrix_exact(embeddings: np.ndarray) -> np.ndarray:
    """
    Full pairwise cosine similarity via matrix multiply.

    Time:   O(N² · D)
    Memory: O(N²) float32  — 5k sentences ≈ 95 MB, 10k ≈ 380 MB.
    Use for N < FAISS_THRESHOLD.
    """
    print("Computing similarity matrix (exact O(N²))…")
    normed = normalize(embeddings, norm="l2")
    sim    = (normed @ normed.T).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    print(f"Similarity matrix  : {sim.shape}  ({sim.nbytes / 1e6:.1f} MB)")
    return sim


def compute_neighbors_faiss(
    embeddings: np.ndarray,
    top_k: int = TOP_K,
    n_list: int = FAISS_N_LIST,
    n_probe: int = FAISS_N_PROBE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate nearest-neighbor search with FAISS IVF-Flat index.

    Instead of building the full N×N matrix, FAISS retrieves only the
    top_k+1 nearest neighbors per query using an inverted-file index.
    This is O(N · sqrt(N) · D) time and O(N · top_k) memory — tractable
    for corpora up to hundreds of thousands of sentences.

    Args:
        embeddings : L2-normalised float32 array, shape (N, D).
        top_k      : Number of neighbors to retrieve per query.
        n_list     : Number of Voronoi cells. Rule: sqrt(N) to 4·sqrt(N).
        n_probe    : Cells searched per query. Higher → more accurate, slower.

    Returns:
        distances  : (N, top_k+1) float32 — cosine similarities (since embeddings
                     are L2-normalised, inner product == cosine similarity).
        indices    : (N, top_k+1) int64   — neighbor indices.
        (The +1 is because FAISS includes the query itself as neighbor 0.)
    """
    print(f"Computing neighbors via FAISS IVF (n_list={n_list}, n_probe={n_probe})…")
    n, d = embeddings.shape

    # Adaptive n_list: FAISS requires n_list < n and recommends n_list ≤ sqrt(n)
    n_list  = min(n_list, max(1, int(n ** 0.5)))
    n_probe = min(n_probe, n_list)

    # Ensure contiguous float32 — FAISS is strict about this
    vecs = np.ascontiguousarray(embeddings.astype(np.float32))

    # Inner-product index on L2-normalised vectors == cosine similarity
    quantizer = faiss.IndexFlatIP(d)
    index     = faiss.IndexIVFFlat(quantizer, d, n_list, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = n_probe

    index.train(vecs)
    index.add(vecs)

    k = min(top_k + 1, n)   # +1 to account for self-match at rank 0
    distances, indices = index.search(vecs, k)

    print(f"FAISS search done  : {n} queries, top-{k} neighbors each")
    return distances, indices


def compute_similarity(
    embeddings: np.ndarray,
    top_k: int = TOP_K,
    threshold: int = FAISS_THRESHOLD,
) -> tuple[Optional[np.ndarray], Optional[tuple[np.ndarray, np.ndarray]]]:
    """
    Route similarity computation to exact or FAISS based on corpus size.

    Returns:
        (sim_matrix, None)          if exact path taken  (N < threshold or no FAISS)
        (None, (distances, indices)) if FAISS path taken  (N ≥ threshold and FAISS available)

    Callers must handle both return shapes — see build_idea_graph_routed().
    """
    n = len(embeddings)
    use_faiss = FAISS_AVAILABLE and n >= threshold

    if use_faiss:
        print(f"Corpus size {n} ≥ {threshold} — using FAISS ANN (install faiss-cpu if missing).")
        return None, compute_neighbors_faiss(embeddings, top_k=top_k)
    else:
        if n >= threshold and not FAISS_AVAILABLE:
            warnings.warn(
                f"Corpus has {n} sentences (≥ {threshold}) but FAISS is not installed. "
                "Falling back to exact O(N²) similarity — this may use "
                f"{n*n*4/1e6:.0f} MB of memory. Install faiss-cpu to avoid this."
            )
        return compute_similarity_matrix_exact(embeddings), None


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
    sim_matrix: Optional[np.ndarray] = None,
    faiss_distances: Optional[np.ndarray] = None,
    percentile: float = 10.0,
    hard_minimum: float = 0.20,
) -> float:
    """
    Auto-calibrate the similarity floor from the actual distribution of
    positive similarities in this document.

    The hardcoded GLOBAL_THRESHOLD_MIN = 0.35 was tuned on one literary
    document. A legal contract, technical manual, or academic paper has a
    very different similarity distribution — calibrating from the data means
    the same pipeline works across domains without manual tuning.

    Strategy:
        Set the floor at the Nth percentile of all positive similarities.
        ~N% of all edges are structurally excluded as noise, adapting to
        whatever the corpus density actually is.

    Args:
        sim_matrix      : Full N×N similarity matrix (exact path).
        faiss_distances : FAISS neighbor distances (N, K) (FAISS path).
                          Provide exactly one of these two.
        percentile      : Percentile of positive similarities to use as floor.
                          10.0 = the bottom 10% of edges are excluded.
        hard_minimum    : Absolute floor — never go below this regardless of
                          corpus (0.20 = below this is noise in any embedding
                          space produced by sentence-transformers).

    Returns:
        Calibrated floor value in [hard_minimum, GLOBAL_THRESHOLD_MAX].
    """
    if sim_matrix is not None:
        n = sim_matrix.shape[0]
        if n <= 3000:
            # Small enough — use full upper triangle
            positives = sim_matrix[np.triu_indices(n, k=1)]
            positives = positives[positives > 0.0]
        else:
            # Large matrix — random sample 500k pairs for speed
            rng  = np.random.default_rng(42)
            rows = rng.integers(0, n, size=500_000)
            cols = rng.integers(0, n, size=500_000)
            mask = rows != cols
            positives = sim_matrix[rows[mask], cols[mask]]
            positives = positives[positives > 0.0]

    elif faiss_distances is not None:
        positives = faiss_distances[faiss_distances > 0.0].ravel()

    else:
        print(f"  calibrate_floor: no similarity data — using hardcoded floor {GLOBAL_THRESHOLD_MIN:.4f}")
        return GLOBAL_THRESHOLD_MIN

    if positives.size == 0:
        print(f"  calibrate_floor: no positive similarities — using hardcoded floor {GLOBAL_THRESHOLD_MIN:.4f}")
        return GLOBAL_THRESHOLD_MIN

    calibrated = float(np.percentile(positives, percentile))
    calibrated = max(calibrated, hard_minimum)
    calibrated = min(calibrated, GLOBAL_THRESHOLD_MAX)

    print(
        f"  calibrate_floor: p{percentile:.0f} of {positives.size:,} positive sims "
        f"= {calibrated:.4f}  (range [{positives.min():.4f}, {positives.max():.4f}])"
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
    sim_matrix: np.ndarray,
    edge_threshold: float = LOUVAIN_EDGE_THRESHOLD,
    resolution: float = LOUVAIN_RESOLUTION,
) -> np.ndarray:
    """
    Cluster nodes with the Louvain community-detection algorithm.

    Louvain works on a graph rather than the raw embedding space, so it
    respects the topology you've already built (useful when embeddings alone
    are noisy but graph structure is informative).

    Args:
        sim_matrix:      Full pairwise similarity matrix.
        edge_threshold:  Only wire the Louvain graph with edges above this.
        resolution:      Higher = more, smaller communities.

    Returns:
        Integer label array of length N.
    """
    print("Clustering with Louvain…")
    G = nx.Graph()
    n = sim_matrix.shape[0]
    G.add_nodes_from(range(n))

    rows, cols = np.where(sim_matrix >= edge_threshold)
    for r, c in zip(rows, cols):
        if r < c:  # undirected — add once
            G.add_edge(int(r), int(c), weight=float(sim_matrix[r, c]))

    partition = louvain_partition(G, weight="weight", resolution=resolution)
    labels    = np.array([partition[i] for i in range(n)], dtype=int)

    n_clusters = len(set(labels))
    print(f"  Louvain → {n_clusters} communities")
    return labels


def assign_clusters(
    embeddings: np.ndarray,
    sim_matrix: np.ndarray,
) -> np.ndarray:
    """
    Pick the best available clustering algorithm.

    Priority: HDBSCAN > Louvain > no clustering (all zeros).
    """
    if HDBSCAN_AVAILABLE:
        return cluster_hdbscan(embeddings)
    if LOUVAIN_AVAILABLE:
        return cluster_louvain(sim_matrix)

    warnings.warn("No clustering backend available — all nodes assigned cluster 0.")
    return np.zeros(len(embeddings), dtype=int)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_idea_graph(
    sim_matrix: np.ndarray,
    sentence_records: list[dict],
    cluster_labels: np.ndarray,
    manifest: Optional[list[dict]],
    top_k: int = TOP_K,
    calibrated_floor: float = GLOBAL_THRESHOLD_MIN,
) -> list[dict]:
    """
    Build idea graph from a full similarity matrix (exact path, N < FAISS_THRESHOLD).
    See build_idea_graph_routed() for the entry point that handles both paths.

    Args:
        calibrated_floor: Floor computed by calibrate_floor() for this document.
                          Replaces the hardcoded GLOBAL_THRESHOLD_MIN so the
                          threshold adapts to the actual similarity distribution.
    """
    n     = len(sentence_records)
    graph = []
    print(f"Building idea graph — exact path (top_k={top_k}, floor={calibrated_floor:.4f})…")

    for i in range(n):
        sims   = sim_matrix[i].copy()
        thresh = dynamic_threshold(sims, floor=calibrated_floor)

        num_candidates = min(top_k * 3, n - 1)
        if num_candidates < 1:
            candidate_idx = np.array([], dtype=int)
        else:
            candidate_idx = np.argpartition(sims, -num_candidates)[-num_candidates:]
            candidate_idx = candidate_idx[np.argsort(sims[candidate_idx])[::-1]]

        neighbors = []
        for j in candidate_idx:
            if int(j) == i:
                continue
            score = float(sims[j])
            if score < thresh:
                continue
            neighbors.append({
                "sentence_id":   int(j),
                "similarity":    round(score, 6),
                "cross_cluster": bool(cluster_labels[i] != cluster_labels[j]),
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


def build_idea_graph_faiss(
    faiss_distances: np.ndarray,
    faiss_indices: np.ndarray,
    sentence_records: list[dict],
    cluster_labels: np.ndarray,
    manifest: Optional[list[dict]],
    top_k: int = TOP_K,
    calibrated_floor: float = GLOBAL_THRESHOLD_MIN,
) -> list[dict]:
    """
    Build idea graph from FAISS ANN results (large corpus path, N ≥ FAISS_THRESHOLD).

    FAISS returns pre-selected top-k neighbors so we skip the full matrix entirely.
    Dynamic threshold is computed only over the returned neighbor similarities
    (not the full row), which is a reasonable approximation for large N since
    the global mean/std of similarities changes slowly with corpus size.

    Args:
        faiss_distances  : (N, K) cosine similarities from FAISS search.
        faiss_indices    : (N, K) corresponding sentence indices.
        sentence_records : Original sentence metadata.
        cluster_labels   : Per-node cluster assignment.
        manifest         : Optional Phase 2 chunk manifest.
        top_k            : Hard cap on neighbors per node.
        calibrated_floor : Document-adaptive floor from calibrate_floor().

    Returns:
        List of enriched graph-node dicts (same schema as build_idea_graph).
    """
    n     = len(sentence_records)
    graph = []
    print(f"Building idea graph — FAISS path (top_k={top_k}, floor={calibrated_floor:.4f})…")

    for i in range(n):
        # FAISS returns self as the first neighbor (distance ≈ 1.0) — skip it
        raw_sims = faiss_distances[i]
        raw_idxs = faiss_indices[i]

        # Filter out self-match and invalid indices (-1 from FAISS padding)
        valid_mask = (raw_idxs != i) & (raw_idxs >= 0)
        sims = raw_sims[valid_mask].astype(np.float32)
        idxs = raw_idxs[valid_mask]

        # Dynamic threshold over the available neighbor similarities
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


def build_idea_graph_routed(
    embeddings: np.ndarray,
    sentence_records: list[dict],
    cluster_labels: np.ndarray,
    manifest: Optional[list[dict]],
    top_k: int = TOP_K,
    precomputed_sim_matrix: Optional[np.ndarray] = None,
) -> tuple[list[dict], Optional[np.ndarray]]:
    """
    Entry point that routes to exact or FAISS graph builder based on corpus size.

    Args:
        precomputed_sim_matrix: Pass the matrix already computed by cluster_louvain()
                                to avoid computing it a second time on the Louvain path.
                                When provided, skips compute_similarity() entirely.

    Returns:
        graph      : the built idea graph
        sim_matrix : full similarity matrix if exact path was taken, else None
                     (downstream callers that need sim_matrix must handle None)
    """
    if precomputed_sim_matrix is not None:
        # Louvain path: matrix was already computed for clustering — reuse it.
        sim_matrix   = precomputed_sim_matrix
        faiss_result = None
    else:
        sim_matrix, faiss_result = compute_similarity(embeddings, top_k=top_k)

    # Calibrate floor ONCE from the real similarity distribution of this document.
    calibrated_floor = calibrate_floor(
        sim_matrix=sim_matrix,
        faiss_distances=faiss_result[0] if faiss_result is not None else None,
    )

    if faiss_result is not None:
        distances, indices = faiss_result
        graph = build_idea_graph_faiss(
            distances, indices, sentence_records, cluster_labels, manifest,
            top_k=top_k, calibrated_floor=calibrated_floor,
        )
        return graph, None   # no full sim_matrix available on FAISS path
    else:
        graph = build_idea_graph(
            sim_matrix, sentence_records, cluster_labels, manifest,
            top_k=top_k, calibrated_floor=calibrated_floor,
        )
        return graph, sim_matrix


# ---------------------------------------------------------------------------
# Per-cluster centrality
# ---------------------------------------------------------------------------

def compute_cluster_centrality(
    graph: list[dict],
    sim_matrix: Optional[np.ndarray],
    cluster_labels: np.ndarray,
) -> dict[int, list[dict]]:
    """
    Compute intra-cluster degree centrality for every node.

    When sim_matrix is None (FAISS path), centrality is approximated from
    the neighbor similarity scores already stored in each graph node — no
    full matrix needed. When sim_matrix is available (exact path), the full
    intra-cluster sub-matrix is used for higher accuracy.
    """
    print("Computing per-cluster centrality…")
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

        if sim_matrix is not None:
            # Exact path — use full sub-matrix
            idx  = np.array(members)
            sub  = sim_matrix[np.ix_(idx, idx)]
            np.fill_diagonal(sub, 0.0)
            denom          = max(len(members) - 1, 1)
            weighted_degree = sub.sum(axis=1) / denom
            ranked = sorted(
                zip(members, weighted_degree.tolist()),
                key=lambda x: x[1], reverse=True,
            )
        else:
            # FAISS path — approximate from stored neighbor similarities
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
    """Phase 3 entry point: similarity → cluster → graph → centrality → save."""

    # 1. Load
    embeddings       = load_embeddings(EMBEDDINGS_PATH)
    sentence_records = load_sentences(SENTENCES_PATH)
    manifest         = load_manifest(MANIFEST_PATH)

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
            print(
                f"⚠️  sentence_records ({n_sent}) vs embeddings ({n_emb}) — "
                f"trimming sentence_records to {n_emb} (likely a blank/duplicate "
                f"removed by Phase 2 deduplication)."
            )
            sentence_records = sentence_records[:n_emb]
        else:
            raise ValueError(
                f"Embedding / sentence count mismatch is too large to auto-fix: "
                f"{n_emb} embeddings vs {n_sent} sentence records. "
                "Re-run Phase 1 and Phase 2."
            )

    n = len(embeddings)
    print(f"\nCorpus size: {n} sentences")
    if n >= FAISS_THRESHOLD:
        print(f"  → Large corpus (≥ {FAISS_THRESHOLD}): "
              f"{'FAISS ANN' if FAISS_AVAILABLE else 'exact (FAISS not installed)'}")
    else:
        print(f"  → Small corpus (< {FAISS_THRESHOLD}): exact similarity")

    # 2. Cluster — HDBSCAN needs embeddings not sim_matrix, so cluster first.
    #    Louvain needs the full sim_matrix; we hold onto it so build_idea_graph_routed
    #    can reuse it rather than computing it a second time (the double-compute bug).
    sim_for_louvain: Optional[np.ndarray] = None

    if HDBSCAN_AVAILABLE:
        cluster_labels = cluster_hdbscan(embeddings)
    elif LOUVAIN_AVAILABLE:
        # Louvain needs the sim_matrix — compute it once here and reuse below.
        sim_for_louvain = compute_similarity_matrix_exact(embeddings)
        cluster_labels  = cluster_louvain(sim_for_louvain)
    else:
        warnings.warn("No clustering backend available — all nodes assigned cluster 0.")
        cluster_labels = np.zeros(n, dtype=int)

    # 3. Build graph (routes to exact or FAISS internally).
    #    On the Louvain path, pass sim_for_louvain so the matrix is not computed twice.
    #    On all other paths it is None and build_idea_graph_routed computes it normally.
    graph, sim_matrix = build_idea_graph_routed(
        embeddings, sentence_records, cluster_labels, manifest,
        top_k=TOP_K,
        precomputed_sim_matrix=sim_for_louvain,
    )

    # 4. Per-cluster centrality (handles None sim_matrix gracefully)
    cluster_centrality = compute_cluster_centrality(graph, sim_matrix, cluster_labels)
    cluster_summary    = build_cluster_summary(cluster_centrality, cluster_labels)

    # 5. Report
    print_summary(graph, cluster_summary)

    # 6. Persist — idea graph is machine-consumed (compact), cluster summary is human-readable (pretty)
    save_json(graph,           OUTPUT_PATH,   label="Idea graph",      pretty=False)
    save_json(cluster_summary, CLUSTERS_PATH, label="Cluster summary", pretty=True)


if __name__ == "__main__":
    main()