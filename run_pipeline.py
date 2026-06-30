#!/usr/bin/env python3
"""
run_pipeline.py

Single entry point for the Thought-to-Structure NLP pipeline.

Runs all five phases in order with staleness checking — a phase is skipped
if its outputs are newer than all of its inputs, so you never re-embed 5,000
sentences just because you re-ran the script.

Usage:
    python run_pipeline.py                  # run all stale phases
    python run_pipeline.py --force          # re-run every phase regardless
    python run_pipeline.py --from phase3    # re-run from phase 3 onwards
    python run_pipeline.py --only phase2    # run exactly one phase
    python run_pipeline.py --skip phase5    # skip interactive query shell
    python run_pipeline.py --dry-run        # show what would run, do nothing
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Phase definitions — order matters
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config-aware path resolution for staleness checker
# Loaded here at module level so PHASES and DATA_DIR reflect the active profile.
# ---------------------------------------------------------------------------
def _bootstrap_config():
    """
    Load config before PHASES is defined so staleness paths are accurate.
    Uses PIPELINE_CONFIG env var if set (written by main() after arg parsing),
    otherwise falls back to config.yaml in the project root.
    """
    config_path = os.environ.get("PIPELINE_CONFIG", "config.yaml")
    try:
        import yaml
        p = Path(config_path)
        if p.exists():
            with open(p) as fh:
                raw = yaml.safe_load(fh) or {}
            data_dir = raw.get("paths", {}).get("data_dir", "data")
            paths    = raw.get("paths", {})
            return data_dir, paths
    except Exception:
        pass
    return "data", {}

_data_dir_str, _path_cfg = _bootstrap_config()

def _dp(key, default):
    """Resolve a data-dir-relative path from config paths section."""
    return os.path.join(_data_dir_str, _path_cfg.get(key, default))


PHASES = [
    {
        "name":      "pdf_to_txt",
        "label":     "Phase 0 — PDF → text",
        "module":    "pdf_to_txt",
        "inputs":    [],
        "outputs":   [],
        "pdf_phase": True,
    },
    {
        "name":      "phase1",
        "label":     "Phase 1 — Data prep (spaCy)",
        "module":    "phase1_data_prep",
        "inputs":    [],
        "outputs":   [_dp("output_sentences", "output_v2.json")],
        "txt_phase": True,
    },
    {
        "name":    "phase2",
        "label":   "Phase 2 — Embeddings (sentence-transformers)",
        "module":  "phase2_embeddings",
        "inputs":  [_dp("output_sentences", "output_v2.json")],
        "outputs": [_dp("output_embeddings", "embeddings.npy"),
                    _dp("output_manifest",   "embeddings_manifest.json")],
    },
    {
        "name":    "phase3",
        "label":   "Phase 3 — Idea graph (HDBSCAN + FAISS)",
        "module":  "phase3_idea_graph",
        "inputs":  [_dp("output_embeddings", "embeddings.npy"),
                    _dp("output_manifest",   "embeddings_manifest.json"),
                    _dp("output_sentences",  "output_v2.json")],
        "outputs": [_dp("output_graph",    "idea_graph.json"),
                    _dp("output_clusters",  "cluster_summary.json")],
    },
    {
        "name":    "phase4",
        "label":   "Phase 4 — Structured summarization (PageRank)",
        "module":  "phase4_gnn_refiner",
        "inputs":  [_dp("output_graph",    "idea_graph.json"),
                    _dp("output_clusters",  "cluster_summary.json"),
                    _dp("output_manifest",  "embeddings_manifest.json")],
        "outputs": [_dp("output_summary_json", "summary.json"),
                    _dp("output_summary_txt",  "summary.txt")],
    },
    {
        "name":        "phase5",
        "label":       "Phase 5 — Semantic query shell",
        "module":      "phase5_api",
        "inputs":      [_dp("output_embeddings", "embeddings.npy"),
                        _dp("output_sentences",  "output_v2.json"),
                        _dp("output_graph",      "idea_graph.json"),
                        _dp("output_clusters",   "cluster_summary.json"),
                        _dp("output_manifest",   "embeddings_manifest.json")],
        "outputs":     [],
        "interactive": True,
    },
]

PHASE_NAMES = [p["name"] for p in PHASES]
DATA_DIR    = Path(_data_dir_str)


# ---------------------------------------------------------------------------
# Staleness checking
# ---------------------------------------------------------------------------

def _mtime(path) -> float:
    try:
        return Path(path).stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _newest_input(phase: dict) -> float:
    mtimes = []
    if phase.get("pdf_phase"):
        pdfs = list(DATA_DIR.glob("*.pdf"))
        if not pdfs:
            return 0.0
        mtimes.extend(_mtime(p) for p in pdfs)
    elif phase.get("txt_phase"):
        txts = list(DATA_DIR.glob("*_clean.txt"))
        if not txts:
            return 0.0
        mtimes.extend(_mtime(p) for p in txts)
    else:
        mtimes.extend(_mtime(p) for p in phase["inputs"])
    return max(mtimes) if mtimes else 0.0


def _oldest_output(phase: dict) -> float:
    if not phase["outputs"]:
        return 0.0   # no outputs = always considered stale when explicitly requested
    mtimes = []
    for p in phase["outputs"]:
        t = _mtime(p)
        if t == 0.0:
            return 0.0   # missing output — must run
        mtimes.append(t)
    return min(mtimes)


def is_stale(phase: dict) -> bool:
    oldest_out = _oldest_output(phase)
    if oldest_out == 0.0:
        return True
    return _newest_input(phase) > oldest_out


# ---------------------------------------------------------------------------
# Manifest version check
# ---------------------------------------------------------------------------

def _file_hash(path, sample_bytes: int = 65536) -> str:
    """MD5 of the first 64 KB of a file — fast enough to run on every start."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            h.update(fh.read(sample_bytes))
        return h.hexdigest()
    except FileNotFoundError:
        return ""


def check_manifest_consistency() -> bool:
    """
    Verify embeddings_manifest.json was produced from the current output_v2.json.

    Phase 2 stores a hash of output_v2.json at generation time.  If output_v2.json
    has since changed (Phase 1 re-run, file edited, git checkout) the manifest's
    sentence_index values now point to wrong sentences — all downstream phases
    silently produce incorrect output.  This check makes that failure loud.
    """
    hash_file      = DATA_DIR / ".output_v2_hash"
    manifest_path  = DATA_DIR / "embeddings_manifest.json"
    sentences_path = DATA_DIR / "output_v2.json"

    if not manifest_path.exists() or not sentences_path.exists():
        return True   # nothing to check on first run

    current_hash = _file_hash(sentences_path)

    if not hash_file.exists():
        # First run after adding this check — write hash and continue
        hash_file.write_text(current_hash)
        return True

    stored_hash = hash_file.read_text().strip()
    if stored_hash != current_hash:
        print(
            "\n⚠️  MANIFEST MISMATCH\n"
            "   output_v2.json has changed since embeddings_manifest.json was generated.\n"
            "   The manifest's sentence_index values now point to wrong sentences.\n"
            "   All downstream phases will produce incorrect output.\n"
            "\n"
            "   Fix: python run_pipeline.py --from phase2\n"
        )
        return False
    return True


def update_manifest_hash() -> None:
    sentences_path = DATA_DIR / "output_v2.json"
    hash_file      = DATA_DIR / ".output_v2_hash"
    if sentences_path.exists():
        hash_file.write_text(_file_hash(sentences_path))


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).parent / "src"


def _import_module(module_name: str):
    module_path = SRC_DIR / f"{module_name}.py"
    if not module_path.exists():
        raise ImportError(
            f"Cannot find {module_name}.py in {SRC_DIR}\n"
            f"Expected path: {module_path.resolve()}"
        )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod  = importlib.util.module_from_spec(spec)
    # Add src/ to sys.path so any imports inside phase scripts resolve correctly
    src_str = str(SRC_DIR.resolve())
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    spec.loader.exec_module(mod)
    return mod


def run_phase(phase: dict) -> bool:
    print(f"\n{'='*60}")
    print(f"  {phase['label']}")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    try:
        mod = _import_module(phase["module"])
        mod.main()
        elapsed = time.perf_counter() - t0
        print(f"\n  Completed in {elapsed:.1f}s")
        if phase["name"] == "phase2":
            update_manifest_hash()
        return True

    except SystemExit as e:
        # phase5 calls sys.exit(0) at the end of the interactive loop — normal
        if e.code == 0:
            elapsed = time.perf_counter() - t0
            print(f"\n  Completed in {elapsed:.1f}s")
            return True
        print(f"\n  Phase exited with code {e.code}")
        return False

    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n  FAILED after {elapsed:.1f}s — {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the Thought-to-Structure NLP pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_pipeline.py                   run all stale phases
  python run_pipeline.py --force           force-run every phase
  python run_pipeline.py --from phase3     re-run from phase3 onwards
  python run_pipeline.py --only phase2     run exactly phase2
  python run_pipeline.py --skip phase4     skip a specific phase
  python run_pipeline.py --dry-run         show what would run, do nothing
  python run_pipeline.py --include-query   also launch the phase5 query shell
        """,
    )
    p.add_argument("--force",         action="store_true",
                   help="Re-run all phases regardless of staleness.")
    p.add_argument("--from",          dest="from_phase", metavar="PHASE",
                   choices=PHASE_NAMES,
                   help="Re-run this phase and all subsequent ones.")
    p.add_argument("--only",          metavar="PHASE", choices=PHASE_NAMES,
                   help="Run exactly this one phase.")
    p.add_argument("--skip",          metavar="PHASE", action="append", default=[],
                   choices=PHASE_NAMES,
                   help="Skip this phase (repeatable).")
    p.add_argument("--dry-run",       action="store_true",
                   help="Show what would run without executing anything.")
    p.add_argument("--include-query", action="store_true",
                   help="Also run phase5 (interactive shell, skipped by default).")
    p.add_argument("--config", metavar="PATH", default="config.yaml",
                   help="Path to a YAML config profile (default: config.yaml).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Always run from the project root so relative paths resolve correctly
    os.chdir(Path(__file__).parent.resolve())
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Expose the chosen config path as an env var so config_loader.py
    # picks it up without needing to thread the argument through every phase.
    os.environ["PIPELINE_CONFIG"] = str(Path(args.config).resolve())
    if not Path(args.config).exists():
        print(f"Warning: config file not found at '{args.config}' — phases will use hardcoded defaults.")

    # --- Select phases ---
    phases_to_consider = list(PHASES)

    if args.only:
        phases_to_consider = [p for p in PHASES if p["name"] == args.only]
    elif args.from_phase:
        idx = PHASE_NAMES.index(args.from_phase)
        phases_to_consider = PHASES[idx:]
        args.force = True   # --from implies --force on the selected range

    # Apply --skip
    phases_to_consider = [p for p in phases_to_consider if p["name"] not in args.skip]

    # Phase 5 is opt-in (it blocks the terminal with an interactive shell)
    if not args.include_query and args.only != "phase5":
        phases_to_consider = [p for p in phases_to_consider if not p.get("interactive")]

    if not phases_to_consider:
        print("No phases selected.")
        sys.exit(0)

    # --- Manifest consistency check (before staleness — catches silent corruption) ---
    if not args.dry_run and not args.force:
        if not check_manifest_consistency():
            print("Run with --from phase2 to fix, or --force to skip this check.")
            sys.exit(1)

    # --- Staleness pass ---
    print(f"\n{'='*60}")
    print("  Thought-to-Structure Pipeline")
    print(f"{'='*60}")

    will_run  = []
    will_skip = []

    for phase in phases_to_consider:
        if args.force or args.only or is_stale(phase):
            will_run.append(phase)
        else:
            will_skip.append(phase)

    if will_skip:
        print("\n  Up to date (skipping):")
        for p in will_skip:
            print(f"    ⏭️   {p['label']}")

    if not will_run:
        print("\n  Everything is up to date. Nothing to do.")
        print("  Use --force to re-run anyway.\n")
        sys.exit(0)

    print(f"\n  Will run ({len(will_run)} phase{'s' if len(will_run) != 1 else ''}):")
    for phase in will_run:
        reason = ""
        if not args.force and not args.only:
            if _oldest_output(phase) == 0.0:
                reason = "  (output missing)"
            elif _newest_input(phase) > _oldest_output(phase):
                reason = "  (inputs changed)"
        print(f"    🔄  {phase['label']}{reason}")

    if args.dry_run:
        print("\n  [dry-run] nothing executed.\n")
        sys.exit(0)

    # --- Execute ---
    total_t0     = time.perf_counter()
    failed_phase = None

    for phase in will_run:
        if not run_phase(phase):
            failed_phase = phase
            break

    total_elapsed = time.perf_counter() - total_t0

    # --- Summary ---
    print(f"\n{'='*60}")
    if failed_phase:
        idx       = will_run.index(failed_phase)
        completed = will_run[:idx]
        remaining = will_run[idx + 1:]
        print(f"  Pipeline stopped at: {failed_phase['label']}")
        if completed:
            print(f"\n  Completed:")
            for p in completed:
                print(f"    ✅  {p['label']}")
        if remaining:
            print(f"\n  Not reached:")
            for p in remaining:
                print(f"    ⏸️   {p['label']}")
        print(f"\n  Total time: {total_elapsed:.1f}s")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"  Pipeline complete — {len(will_run)} phase(s) in {total_elapsed:.1f}s")
        if will_skip:
            print(f"  {len(will_skip)} phase(s) skipped (up to date)")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()