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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDINGS_PATH  = os.path.join("data", "embeddings.npy")
SENTENCES_PATH   = os.path.join("data", "output_v2.json")
OUTPUT_PATH      = os.path.join("data", "idea_graph.json")
CLUSTERS_PATH    = os.path.join("data", "cluster_summary.json")
MANIFEST_PATH    = os.path.join("data", "embeddings_manifest.json")   # Phase 2 output

TOP_K                   = 5      # Hard cap on neighbors per node
DYNAMIC_THRESHOLD_SIGMA = 0.5    # Threshold = mean_sim − σ * std_sim  (per node)
GLOBAL_THRESHOLD_MIN    = 0.35   # Never accept edges below this floor
GLOBAL_THRESHOLD_MAX    = 0.90   # Never reject edges above this ceiling (high-confidence links)

# HDBSCAN params
HDBSCAN_MIN_CLUSTER     = 3
HDBSCAN_MIN_SAMPLES     = 2
HDBSCAN_METRIC          = "euclidean"   # Works on L2-normalized vecs (≡ cosine distance)

# Louvain params
LOUVAIN_RESOLUTION      = 1.0
LOUVAIN_EDGE_THRESHOLD  = 0.50          # Only wire Louvain graph with strong edges


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
# Similarity
# ---------------------------------------------------------------------------

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Full pairwise cosine similarity.  For N > ~5 000 consider switching to
    FAISS approximate nearest neighbours to avoid O(N²) memory.
    """
    print("Computing cosine similarity matrix…")
    # If embeddings are already L2-normalised (Phase 2), this is just a dot product.
    normed = normalize(embeddings, norm="l2")
    sim    = normed @ normed.T             # faster than sklearn for float32
    np.fill_diagonal(sim, 0.0)
    print(f"Similarity matrix  : {sim.shape}  (dtype={sim.dtype})")
    return sim


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
) -> list[dict]:
    """
    Build the idea graph with per-node dynamic thresholds.

    Each node retains up to *top_k* neighbors that satisfy its own adaptive
    threshold.  Cluster membership from HDBSCAN/Louvain is embedded in the
    node so downstream tools can group, colour, or filter by topic.

    Cross-cluster edges are flagged explicitly so bridge sentences (ideas
    that connect two topics) are easy to identify.

    Args:
        sim_matrix:       Pairwise cosine similarity (n × n, diagonal = 0).
        sentence_records: Original sentence metadata.
        cluster_labels:   Per-node cluster assignment (−1 = noise).
        manifest:         Optional Phase 2 chunk manifest for traceability.
        top_k:            Hard cap on neighbors per node.

    Returns:
        List of enriched graph-node dicts.
    """
    n     = len(sentence_records)
    graph = []
    print(f"Building idea graph (top_k={top_k}, dynamic threshold per node)…")

    for i in range(n):
        sims = sim_matrix[i].copy()

        # --- Dynamic per-node threshold ---
        thresh = dynamic_threshold(sims)

        # --- Candidate selection (partial sort for speed) ---
        num_candidates = min(top_k * 3, n - 1)  # over-fetch; prune below
        if num_candidates < 1:
            candidate_idx = np.array([], dtype=int)
        else:
            candidate_idx = np.argpartition(sims, -num_candidates)[-num_candidates:]
            candidate_idx = candidate_idx[np.argsort(sims[candidate_idx])[::-1]]

        # --- Filter by threshold, self-edge, and hard cap ---
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

        # --- Chunk provenance (if manifest available) ---
        chunk_meta = manifest[i] if manifest else {}

        node = {
            "sentence_id":   i,
            "sentence":      sentence_records[i]["sentence"],
            "paragraph_id":  sentence_records[i].get("paragraph_id"),
            "cluster_id":    int(cluster_labels[i]),
            "threshold_used": round(thresh, 6),
            "chunk_index":   chunk_meta.get("chunk_index", 0),
            "total_chunks":  chunk_meta.get("total_chunks", 1),
            "neighbors":     neighbors,
        }
        graph.append(node)

    return graph


# ---------------------------------------------------------------------------
# Per-cluster centrality
# ---------------------------------------------------------------------------

def compute_cluster_centrality(
    graph: list[dict],
    sim_matrix: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[int, list[dict]]:
    """
    Compute intra-cluster degree centrality for every node in each cluster.

    Global centrality on a large graph is dominated by hub nodes that may
    span unrelated topics.  Restricting centrality computation to same-cluster
    edges surfaces the *most representative* sentence per topic instead.

    Algorithm
    ---------
    For each cluster C:
      1. Build the induced subgraph on nodes in C.
      2. Degree centrality = (sum of intra-cluster edge weights) / (|C| - 1).
         This is the weighted analogue of normalised degree.
      3. Return nodes ranked by centrality (descending).

    Args:
        graph:          Idea graph (output of build_idea_graph).
        sim_matrix:     Full similarity matrix for weight look-up.
        cluster_labels: Per-node cluster assignment.

    Returns:
        Dict mapping cluster_id → sorted list of
        {sentence_id, sentence, centrality} dicts.
    """
    print("Computing per-cluster centrality…")
    unique_clusters = sorted(set(int(l) for l in cluster_labels))
    cluster_centrality: dict[int, list[dict]] = {}

    for cid in unique_clusters:
        members = [i for i, l in enumerate(cluster_labels) if int(l) == cid]
        if len(members) < 2:
            # Singleton or noise — centrality is trivially 0
            cluster_centrality[cid] = [
                {
                    "sentence_id": members[0] if members else -1,
                    "sentence":    graph[members[0]]["sentence"] if members else "",
                    "centrality":  0.0,
                }
            ]
            continue

        # Extract intra-cluster similarity sub-matrix
        idx  = np.array(members)
        sub  = sim_matrix[np.ix_(idx, idx)]   # shape (|C|, |C|)
        np.fill_diagonal(sub, 0.0)

        # Weighted degree = row sum; normalise by (|C| - 1)
        denom          = max(len(members) - 1, 1)
        weighted_degree = sub.sum(axis=1) / denom   # shape (|C|,)

        ranked = sorted(
            zip(members, weighted_degree.tolist()),
            key=lambda x: x[1],
            reverse=True,
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

def save_json(obj: object, filepath: str, label: str = "File") -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    print(f"{label} saved        → {filepath}")


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

    if manifest is not None:
        # Phase 2 chunked sentences: one embedding per chunk, not per sentence.
        # Compare against the manifest which has one entry per embedding.
        expected_len = len(manifest)
        if len(embeddings) != expected_len:
            raise ValueError(
                f"Embedding / manifest count mismatch: "
                f"{len(embeddings)} embeddings vs {expected_len} manifest entries. "
                "Re-run Phase 2."
            )
    else:
        # No chunking: one embedding per sentence.
        if len(embeddings) != len(sentence_records):
            raise ValueError(
                f"Embedding / sentence count mismatch: "
                f"{len(embeddings)} vs {len(sentence_records)}. "
                "Re-run Phase 1 and Phase 2."
            )

    # 2. Similarity matrix
    sim_matrix = compute_similarity_matrix(embeddings)

    # 3. Cluster (HDBSCAN preferred → Louvain fallback → all-zero)
    cluster_labels = assign_clusters(embeddings, sim_matrix)

    # 4. Build graph with dynamic per-node thresholding
    graph = build_idea_graph(
        sim_matrix,
        sentence_records,
        cluster_labels,
        manifest,
        top_k=TOP_K,
    )

    # 5. Per-cluster centrality
    cluster_centrality = compute_cluster_centrality(graph, sim_matrix, cluster_labels)
    cluster_summary    = build_cluster_summary(cluster_centrality, cluster_labels)

    # 6. Report
    print_summary(graph, cluster_summary)

    # 7. Persist
    save_json(graph,           OUTPUT_PATH,   label="Idea graph")
    save_json(cluster_summary, CLUSTERS_PATH, label="Cluster summary")


if __name__ == "__main__":
    main()