#!/usr/bin/env python3
"""
config_loader.py

Centralised configuration loader for the NLP pipeline.

Every phase imports this module and calls load_config() to get its
parameters. The Python source files never need to be edited for
domain changes — only config.yaml changes.

Usage (in any phase file):
    from config_loader import load_config
    cfg = load_config()                          # loads config.yaml by default
    cfg = load_config("configs/legal.yaml")      # or a custom profile

Then access parameters via:
    cfg.phase1.min_words
    cfg.phase2.model_name
    cfg.paths.data_dir
    etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Typed dataclasses — one per section in config.yaml.
# Using dataclasses (not plain dicts) gives you attribute access, IDE
# autocompletion, and a clear schema visible to any reader of this file.
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    data_dir:            str = "data"
    raw_file_suffix:     str = "_clean.txt"
    output_sentences:    str = "output_v2.json"
    output_embeddings:   str = "embeddings.npy"
    output_manifest:     str = "embeddings_manifest.json"
    output_graph:        str = "idea_graph.json"
    output_clusters:     str = "cluster_summary.json"
    output_summary_json: str = "summary.json"
    output_summary_txt:  str = "summary.txt"
    output_faiss_index:  str = "faiss.index"
    output_batch:        str = "batch_results.jsonl"
    export_dir:          str = "exports"

    def full(self, key: str) -> str:
        """Return the full path for a data-dir-relative output key."""
        return os.path.join(self.data_dir, getattr(self, key))


@dataclass
class PdfConfig:
    skip_first_page:  bool  = True
    ocr_dpi:          int   = 300
    min_text_chars:   int   = 20
    column_gap_ratio: float = 0.10


@dataclass
class Phase1Config:
    min_words: int = 4


@dataclass
class Phase2Config:
    model_name:   str = "all-mpnet-base-v2"
    batch_size:   int = 32
    max_tokens:   int = 384
    chunk_stride: int = 128


@dataclass
class Phase3Config:
    top_k:                    int   = 5
    hnsw_m:                   int   = 32
    hnsw_ef_construction:     int   = 200
    hnsw_ef_search:           int   = 64
    dynamic_threshold_sigma:  float = 0.5
    global_threshold_min:     float = 0.35
    global_threshold_max:     float = 0.90
    hdbscan_min_cluster:      int   = 3
    hdbscan_min_samples:      int   = 2
    hdbscan_metric:           str   = "euclidean"
    louvain_resolution:       float = 1.0
    louvain_edge_threshold:   float = 0.50


@dataclass
class Phase4Config:
    top_n_per_cluster: int   = 4
    max_bridges:       int   = 2
    min_sent_words:    int   = 6
    max_line_width:    int   = 120
    pagerank_damping:  float = 0.85
    include_noise:     bool  = True


@dataclass
class Phase5Config:
    embed_model_name:   str           = "all-mpnet-base-v2"
    rerank_model_name:  str           = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    default_top_k:      int           = 5
    faiss_nprobe:       int           = 8
    faiss_nlist:        int           = 64
    rerank_fetch_mult:  int           = 4
    default_alpha:      float         = 0.5
    default_approx_k:   int           = 0
    console_width:      int           = 96
    theme_dict_path:    Optional[str] = None


@dataclass
class PipelineConfig:
    """Root config object. Access sections as attributes: cfg.phase1.min_words"""
    paths:  PathsConfig  = field(default_factory=PathsConfig)
    pdf:    PdfConfig    = field(default_factory=PdfConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
    phase5: Phase5Config = field(default_factory=Phase5Config)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _apply(dataclass_instance, raw: dict) -> None:
    """Overwrite dataclass fields with values from a raw dict (in-place)."""
    for key, value in raw.items():
        if hasattr(dataclass_instance, key):
            setattr(dataclass_instance, key, value)
        else:
            import warnings
            warnings.warn(
                f"config.yaml key '{key}' under "
                f"'{type(dataclass_instance).__name__}' is unrecognised and "
                f"will be ignored. Check for typos."
            )


def load_config(path: str = "config.yaml") -> PipelineConfig:
    """
    Load config.yaml (or a custom profile) and return a typed PipelineConfig.

    Path resolution order:
      1. Explicit argument passed by the caller.
      2. PIPELINE_CONFIG environment variable (set by run_pipeline.py --config).
      3. Default "config.yaml" in the current working directory.

    Falls back to hardcoded defaults if PyYAML is not installed or the file
    does not exist — a warning is printed in both cases.

    Args:
        path: Path to the YAML config file.

    Returns:
        PipelineConfig with all sections populated.
    """
    import os
    env_path = os.environ.get("PIPELINE_CONFIG")
    if env_path and path == "config.yaml":   # only override the default, not explicit calls
        path = env_path
    cfg = PipelineConfig()   # start from defaults

    if not YAML_AVAILABLE:
        import warnings
        warnings.warn(
            "PyYAML not installed — using hardcoded defaults. "
            "Install with: pip install pyyaml"
        )
        return cfg

    config_path = Path(path)
    if not config_path.exists():
        import warnings
        warnings.warn(
            f"Config file '{path}' not found — using hardcoded defaults. "
            f"Create config.yaml in the project root to customise parameters."
        )
        return cfg

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    section_map = {
        "paths":  cfg.paths,
        "pdf":    cfg.pdf,
        "phase1": cfg.phase1,
        "phase2": cfg.phase2,
        "phase3": cfg.phase3,
        "phase4": cfg.phase4,
        "phase5": cfg.phase5,
    }

    for section_name, dataclass_instance in section_map.items():
        if section_name in raw and isinstance(raw[section_name], dict):
            _apply(dataclass_instance, raw[section_name])

    print(f"Config loaded ← {config_path.resolve()}")
    return cfg