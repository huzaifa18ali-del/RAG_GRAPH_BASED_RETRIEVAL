#!/usr/bin/env python3
"""
phase3_visualize_matplotlib.py

Static visualization of the idea graph using networkx + matplotlib.
Compatible with Python 3.12.
"""

import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# --- Configuration ---
GRAPH_JSON_PATH = "data/idea_graph.json"
OUTPUT_IMAGE = "data/idea_graph.png"


def main() -> None:
    # --- Step 1: Load idea graph ---
    with open(GRAPH_JSON_PATH, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    print(f"Loaded {len(graph_data)} sentences from the idea graph.")

    # --- Step 2: Build networkx graph ---
    G = nx.DiGraph()

    for node in graph_data:
        G.add_node(node["sentence_id"], paragraph_id=node["paragraph_id"])
        for neighbor in node["neighbors"]:
            G.add_edge(node["sentence_id"], neighbor["sentence_id"], weight=neighbor["similarity"])

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Step 3: Layout ---
    # spring_layout distributes nodes nicely
    pos = nx.spring_layout(G, k=0.5, seed=42)

    # --- Step 4: Draw nodes and edges ---
    plt.figure(figsize=(14, 14))

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="skyblue")

    # Edges, with width proportional to similarity
    edges = G.edges(data=True)
    edge_weights = [d["weight"] * 2 for (_, _, d) in edges]  # scale for visibility
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, arrowsize=10)

    # Labels (optional, for small graphs)
    if G.number_of_nodes() <= 50:
        labels = {n: f"{n}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("Idea Graph Visualization (static)")

    # Save and show
    os.makedirs(os.path.dirname(OUTPUT_IMAGE) or ".", exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Saved visualization to {OUTPUT_IMAGE}")
    plt.show()


if __name__ == "__main__":
    main()