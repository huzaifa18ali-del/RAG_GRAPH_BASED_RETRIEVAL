#!/usr/bin/env python3
"""
main.py

FastAPI service wrapping the Thought-to-Structure NLP pipeline for
asynchronous document ingestion.

Milestone 3 (structural hardening): wraps PDF upload, text extraction,
clause splitting, embedding generation, and idea-graph construction in a
single background task, returning 202 Accepted with a task ID immediately
so the client is never blocked on a multi-minute pipeline run.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Usage:
    curl -F "file=@document.pdf" http://localhost:8000/api/v1/ingest
    curl http://localhost:8000/api/v1/status/<task_id>
"""


from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import shutil
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("ingest_api")

# ---------------------------------------------------------------------------
# Paths — always resolve relative to this file, regardless of cwd uvicorn
# was launched from (mirrors run_pipeline.py's os.chdir behaviour).
# ---------------------------------------------------------------------------
PROJECT_ROOT     = Path(__file__).parent.resolve()
SRC_DIR          = PROJECT_ROOT / "src"
BASE_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
UPLOAD_ROOT       = PROJECT_ROOT / "uploads"
TASK_CONFIG_DIR  = PROJECT_ROOT / ".task_configs"

os.chdir(PROJECT_ROOT)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
TASK_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf"}

# ---------------------------------------------------------------------------
# Base config — used for query-time defaults (embed model, HNSW params, PPR
# params, Ollama settings). Ingestion still clones a per-task config in
# _write_task_config(); this is the process-wide base config_loader.load_config()
# reads at import time for everything that ISN'T task-specific.
# ---------------------------------------------------------------------------
os.environ.setdefault("PIPELINE_CONFIG", str(BASE_CONFIG_PATH))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import load_config  # noqa: E402
import graph_traversal as gt            # noqa: E402
import llm_synthesizer as ls            # noqa: E402
import visualize as vz                  # noqa: E402

_BASE_CFG = load_config(str(BASE_CONFIG_PATH))

# ---------------------------------------------------------------------------
# Query-time embedding model — loaded once, lazily, on first query rather
# than at import time. Ingestion-only deployments (or tests that only hit
# /api/v1/ingest) never pay the SentenceTransformer load cost.
# ---------------------------------------------------------------------------
_QUERY_ENCODER = None
_QUERY_ENCODER_LOCK = threading.Lock()


def _get_query_encoder():
    """
    Lazily load and cache the sentence-transformers model used to embed
    incoming query text. Must match phase2.model_name — the model that
    produced the corpus embeddings each task's FAISS index is built from —
    or query vectors and corpus vectors won't live in a comparable space.
    We reuse phase5.embed_model_name (the existing convention for this
    exact constraint) rather than inventing a third config field.
    """
    global _QUERY_ENCODER
    if _QUERY_ENCODER is None:
        with _QUERY_ENCODER_LOCK:
            if _QUERY_ENCODER is None:
                from sentence_transformers import SentenceTransformer
                log.info("Loading query encoder: %s", _BASE_CFG.phase5.embed_model_name)
                _QUERY_ENCODER = SentenceTransformer(_BASE_CFG.phase5.embed_model_name)
    return _QUERY_ENCODER


# ---------------------------------------------------------------------------
# GraphIndex cache — building the FAISS HNSW index + networkx graph is the
# expensive part of a query; searching/walking it is cheap. Cache one
# GraphIndex per task data_dir, invalidated automatically if idea_graph.json
# has been modified since the cached index was built (covers re-ingestion
# of the same task_id, which shouldn't normally happen but is cheap to guard).
# ---------------------------------------------------------------------------
_GRAPH_INDEX_CACHE: dict[str, "gt.GraphIndex"] = {}
_GRAPH_INDEX_CACHE_LOCK = threading.Lock()


def _get_graph_index(data_dir: Path) -> "gt.GraphIndex":
    key = str(data_dir)
    idea_graph_path = data_dir / _BASE_CFG.paths.output_graph
    if not idea_graph_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No idea graph found for this task at {idea_graph_path}. "
                   f"Has ingestion completed?",
        )
    current_mtime = idea_graph_path.stat().st_mtime

    with _GRAPH_INDEX_CACHE_LOCK:
        cached = _GRAPH_INDEX_CACHE.get(key)
        if cached is not None and cached.source_mtime == current_mtime:
            return cached

    try:
        index = gt.build_graph_index(
            data_dir,
            embeddings_filename=_BASE_CFG.paths.output_embeddings,
            idea_graph_filename=_BASE_CFG.paths.output_graph,
            hnsw_m=_BASE_CFG.phase3.hnsw_m,
            hnsw_ef_construction=_BASE_CFG.phase3.hnsw_ef_construction,
            min_edge_weight=_BASE_CFG.ppr.min_edge_weight,
        )
    except gt.GraphIndexError as exc:
        log.error("Failed to build GraphIndex for %s: %s", data_dir, exc)
        raise HTTPException(status_code=500, detail=f"Failed to load graph artifacts: {exc}") from exc

    with _GRAPH_INDEX_CACHE_LOCK:
        _GRAPH_INDEX_CACHE[key] = index
    return index


# Phase modules run in order for a single ingestion request. Names match the
# .py files under src/ exactly — this list intentionally mirrors PHASES in
# run_pipeline.py for the ingest-relevant subset (everything through the
# idea graph; phase4 summarization and phase5 query shell are downstream of
# ingestion and are not part of the upload-triggered pipeline).
INGEST_PHASES = ["pdf_to_txt", "phase1_data_prep", "phase2_embeddings", "phase3_idea_graph"]


# ---------------------------------------------------------------------------
# In-memory task store
#
# Production note: this is intentionally an in-process dict, not Redis/a DB —
# appropriate for this structural-hardening pass on a single-worker
# deployment. A multi-worker / multi-process deployment needs a shared store
# (Redis, Postgres) instead; flagged as a follow-up rather than solved here
# to avoid scope creep on this milestone.
# ---------------------------------------------------------------------------
class TaskStatus(BaseModel):
    task_id: str
    status: str                  # queued | processing | completed | failed
    stage: Optional[str] = None
    filename: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    data_dir: Optional[str] = None


_TASKS: dict[str, TaskStatus] = {}
_TASKS_LOCK = threading.Lock()

# Serializes full pipeline runs. The underlying phase modules
# (pdf_to_txt.py, phase1_data_prep.py, ...) read their configuration via a
# process-wide PIPELINE_CONFIG environment variable at *import* time (see
# config_loader.py / run_pipeline.py) — correct for the CLI's one-shot
# execution model, but not concurrency-safe in a long-running server if two
# requests set that env var at the same time. Acquiring this lock around
# each end-to-end run keeps every request correct at the cost of serializing
# ingestion. True parallel ingestion requires threading the config through
# as an explicit function argument instead of an env var — a follow-up
# milestone, intentionally out of scope here.
_PIPELINE_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_task(task_id: str, **fields: Any) -> None:
    with _TASKS_LOCK:
        task = _TASKS[task_id]
        _TASKS[task_id] = task.model_copy(update={**fields, "updated_at": _now()})


def _get_task(task_id: str) -> TaskStatus:
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    return task


# ---------------------------------------------------------------------------
# Dynamic phase-module import — mirrors run_pipeline.py's loader exactly, so
# this service and the CLI runner share one source of truth for phase logic
# instead of duplicating it.
# ---------------------------------------------------------------------------
def _import_phase_module(module_name: str):
    module_path = SRC_DIR / f"{module_name}.py"
    if not module_path.exists():
        raise ImportError(f"Cannot find {module_name}.py in {SRC_DIR}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    src_str = str(SRC_DIR.resolve())
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    spec.loader.exec_module(mod)
    return mod


def _write_task_config(task_id: str, data_dir: Path) -> Path:
    """
    Clone config.yaml with data_dir overridden to this task's isolated
    upload folder, so each ingestion writes its own output_v2.json,
    embeddings.npy, idea_graph.json etc. without colliding with other tasks.
    """
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Base config not found: {BASE_CONFIG_PATH}")

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh) or {}

    raw_cfg.setdefault("paths", {})
    raw_cfg["paths"]["data_dir"] = str(data_dir)

    task_config_path = TASK_CONFIG_DIR / f"{task_id}.yaml"
    with open(task_config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw_cfg, fh)
    return task_config_path


def _run_ingestion_pipeline(task_id: str, data_dir: Path) -> None:
    """
    Executes pdf_to_txt -> phase1 -> phase2 -> phase3 sequentially against
    this task's isolated data directory, updating the task store as it
    progresses. Runs in a background thread via FastAPI's BackgroundTasks
    (Starlette dispatches sync callables through its threadpool automatically).
    """
    with _PIPELINE_LOCK:
        try:
            _set_task(task_id, status="processing", stage="starting")
            task_config_path = _write_task_config(task_id, data_dir)
            os.environ["PIPELINE_CONFIG"] = str(task_config_path)

            for phase_name in INGEST_PHASES:
                _set_task(task_id, stage=phase_name)
                log.info("[%s] running %s", task_id, phase_name)
                try:
                    mod = _import_phase_module(phase_name)
                    mod.main()
                except Exception as exc:
                    log.exception("[%s] phase %s failed", task_id, phase_name)
                    _set_task(
                        task_id,
                        status="failed",
                        stage=phase_name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    return

            _set_task(task_id, status="completed", stage="done", error=None)
            log.info("[%s] ingestion complete -> %s", task_id, data_dir)

        except Exception as exc:
            log.exception("[%s] ingestion pipeline failed outside phase loop", task_id)
            _set_task(task_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            os.environ.pop("PIPELINE_CONFIG", None)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Thought-to-Structure Ingestion API",
    version="1.0.0",
    description="Async PDF ingestion: upload -> text -> embeddings -> idea graph.",
)


class IngestResponse(BaseModel):
    task_id: str
    status: str
    message: str


@app.post("/api/v1/ingest", status_code=202, response_model=IngestResponse)
async def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a PDF upload, persist it to an isolated per-task data directory,
    and schedule the full ingestion pipeline (pdf_to_txt -> phase1 -> phase2
    -> phase3) as a background task.

    Returns immediately with 202 Accepted and a task_id for polling via
    GET /api/v1/status/{task_id}. Heavy work (OCR, spaCy parsing, embedding
    generation, FAISS HNSW graph construction) happens entirely after this
    response has already been sent.
    """
    if file.filename is None or Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    task_id  = str(uuid.uuid4())
    data_dir = UPLOAD_ROOT / task_id

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        dest_path = data_dir / file.filename
        with open(dest_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except OSError as exc:
        log.error("Failed to persist upload for task %s: %s", task_id, exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc
    finally:
        await file.close()

    now = _now()
    with _TASKS_LOCK:
        _TASKS[task_id] = TaskStatus(
            task_id=task_id,
            status="queued",
            stage="queued",
            filename=file.filename,
            created_at=now,
            updated_at=now,
            data_dir=str(data_dir),
        )

    background_tasks.add_task(_run_ingestion_pipeline, task_id, data_dir)

    log.info("Task %s queued for file %s", task_id, file.filename)

    return JSONResponse(
        status_code=202,
        content=IngestResponse(
            task_id=task_id,
            status="queued",
            message="File accepted. Poll GET /api/v1/status/{task_id} for progress.",
        ).model_dump(),
    )


@app.get("/api/v1/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str) -> TaskStatus:
    """Poll the status of a previously submitted ingestion task."""
    return _get_task(task_id)


# ---------------------------------------------------------------------------
# Summary endpoint — Phase 4 structured summarization on demand
# ---------------------------------------------------------------------------

_SUMMARY_CACHE: dict[str, tuple[float, list[dict]]] = {}
_SUMMARY_CACHE_LOCK = threading.Lock()


class SummaryResponse(BaseModel):
    task_id: str
    summary: list[dict]


def _compute_summary(task_id: str, data_dir: Path) -> list[dict]:
    """
    Run Phase 4 (phase4_gnn_refiner) against this task's data directory and
    return the structured summary, using the same per-task config-cloning
    pattern as ingestion so Phase 4 reads this task's idea_graph.json /
    cluster_summary.json instead of another task's.

    Serialized behind _PIPELINE_LOCK because — like the ingestion phases —
    phase4_gnn_refiner reads its paths/params from config_loader.load_config()
    at *import* time via the process-wide PIPELINE_CONFIG env var, which is
    not safe to mutate concurrently from two requests.
    """
    idea_graph_path = data_dir / _BASE_CFG.paths.output_graph
    cluster_path = data_dir / _BASE_CFG.paths.output_clusters
    if not idea_graph_path.exists() or not cluster_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Summary artifacts not found for this task at {data_dir}. "
                   f"Has ingestion completed?",
        )

    mtime = cluster_path.stat().st_mtime
    cache_key = str(data_dir)

    with _SUMMARY_CACHE_LOCK:
        cached = _SUMMARY_CACHE.get(cache_key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    with _PIPELINE_LOCK:
        task_config_path = TASK_CONFIG_DIR / f"{task_id}.yaml"
        if not task_config_path.exists():
            task_config_path = _write_task_config(task_id, data_dir)

        prev_pipeline_config = os.environ.get("PIPELINE_CONFIG")
        os.environ["PIPELINE_CONFIG"] = str(task_config_path)
        try:
            mod = _import_phase_module("phase4_gnn_refiner")
            structured_summary = mod.run(debug=False)
        except Exception as exc:
            log.exception("[%s] Phase 4 summarization failed", task_id)
            raise HTTPException(
                status_code=500, detail=f"Summarization failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if prev_pipeline_config is not None:
                os.environ["PIPELINE_CONFIG"] = prev_pipeline_config
            else:
                os.environ.pop("PIPELINE_CONFIG", None)

    with _SUMMARY_CACHE_LOCK:
        _SUMMARY_CACHE[cache_key] = (mtime, structured_summary)

    return structured_summary


@app.get("/api/v1/summary/{task_id}", response_model=SummaryResponse)
async def get_summary(task_id: str) -> SummaryResponse:
    """
    Return the Phase 4 structured summary (cluster-by-cluster paragraphs +
    cross-cluster bridge sentences) for a previously ingested document.

    404 - unknown task_id, or the idea graph / cluster summary don't exist yet.
    409 - task exists but ingestion failed or is still in progress.
    500 - Phase 4 raised while summarizing.
    """
    task = _get_task(task_id)  # raises 404 if unknown

    if task.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Task {task_id} is not ready for summarization (status={task.status}, "
                   f"stage={task.stage}). Poll GET /api/v1/status/{task_id} until completed.",
        )

    data_dir = Path(task.data_dir)
    structured_summary = await asyncio.to_thread(_compute_summary, task_id, data_dir)

    return SummaryResponse(task_id=task_id, summary=structured_summary)


# ---------------------------------------------------------------------------
# Graph explorer endpoint — on-demand interactive idea-graph HTML, reusing
# visualize.py's build_html() verbatim (same script the CLI uses to write
# idea_graph_explorer.html) instead of duplicating any rendering logic.
# ---------------------------------------------------------------------------

_GRAPH_HTML_CACHE: dict[str, tuple[float, str]] = {}
_GRAPH_HTML_CACHE_LOCK = threading.Lock()


def _compute_graph_html(data_dir: Path) -> str:
    """
    Load this task's idea_graph.json (+ optional cluster_summary.json /
    embeddings_manifest.json) and render the same self-contained explorer
    HTML visualize.py's CLI writes to disk, via its build_html().

    Cached per data_dir, invalidated by idea_graph.json's mtime — mirrors
    the GraphIndex / summary caching pattern above.
    """
    idea_graph_path = data_dir / _BASE_CFG.paths.output_graph
    if not idea_graph_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No idea graph found for this task at {idea_graph_path}. "
                   f"Has ingestion completed?",
        )

    mtime = idea_graph_path.stat().st_mtime
    cache_key = str(data_dir)

    with _GRAPH_HTML_CACHE_LOCK:
        cached = _GRAPH_HTML_CACHE.get(cache_key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    # cluster_summary.json / embeddings_manifest.json are both optional
    # enrichments for build_html() — a missing file just means a plainer
    # explorer, not an error. cluster filename comes from the shared config
    # (same field _compute_summary uses); the manifest has no config entry
    # of its own, so fall back to visualize.py's own default basename.
    cluster_path = data_dir / _BASE_CFG.paths.output_clusters
    manifest_path = data_dir / os.path.basename(vz.DEFAULT_MANIFEST_PATH)

    try:
        with idea_graph_path.open("r", encoding="utf-8") as fh:
            graph = json.load(fh)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse idea_graph.json: {exc}"
        ) from exc

    cluster_summary = None
    if cluster_path.exists():
        try:
            with cluster_path.open("r", encoding="utf-8") as fh:
                cluster_summary = json.load(fh)
        except json.JSONDecodeError as exc:
            log.warning("Failed to parse cluster_summary.json for %s: %s", data_dir, exc)

    manifest = None
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except json.JSONDecodeError as exc:
            log.warning("Failed to parse embeddings_manifest.json for %s: %s", data_dir, exc)

    # validate_graph() is written for the CLI and calls sys.exit() on
    # malformed input — fine for a one-shot script, fatal for a live
    # server. Catch it here and convert to a normal HTTP error instead.
    try:
        vz.validate_graph(graph)
    except SystemExit as exc:
        log.error("Graph validation failed for %s: %s", data_dir, exc)
        raise HTTPException(
            status_code=500, detail=f"Graph validation failed: {exc}"
        ) from exc

    html = vz.build_html(graph, cluster_summary, manifest, str(idea_graph_path))

    with _GRAPH_HTML_CACHE_LOCK:
        _GRAPH_HTML_CACHE[cache_key] = (mtime, html)

    return html


@app.get("/api/v1/graph/{task_id}", response_class=HTMLResponse)
async def get_graph_explorer(task_id: str) -> HTMLResponse:
    """
    Render the interactive idea-graph explorer (nodes = clauses, edges =
    similarity/sequential links, colored by cluster) for a previously
    ingested document.

    404 - unknown task_id, or idea_graph.json doesn't exist yet.
    409 - task exists but ingestion failed or is still in progress.
    500 - the idea graph failed schema validation, or an artifact is malformed JSON.
    """
    task = _get_task(task_id)  # raises 404 if unknown

    if task.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Task {task_id} is not ready for graph exploration (status={task.status}, "
                   f"stage={task.stage}). Poll GET /api/v1/status/{task_id} until completed.",
        )

    data_dir = Path(task.data_dir)
    html = await asyncio.to_thread(_compute_graph_html, data_dir)

    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Query endpoint — Milestone 2: dynamic PPR graph surf + local LLM synthesis
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    task_id:       str
    question:      str = Field(..., min_length=1, max_length=2000)
    seed_top_n:    Optional[int] = Field(default=None, ge=1, le=50)
    result_top_m:  Optional[int] = Field(default=None, ge=1, le=30)


class QuerySource(BaseModel):
    sentence_id:  int
    sentence:     str
    score:        float
    is_seed:      bool
    cluster_id:   Optional[int] = None
    paragraph_id: Optional[int] = None
    expansion_source: Optional[int] = None


class QueryResponse(BaseModel):
    task_id:          str
    question:         str
    answer:           str
    model:            str
    sources:          list[QuerySource]
    ppr_fallback_used: bool
    latency_ms:       float


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """
    Answer a natural-language question against a previously ingested
    document using dynamic query-time graph traversal (Personalized
    PageRank) followed by local LLM synthesis via Ollama.

    Pipeline:
      1. Look up the task and confirm ingestion completed successfully.
      2. Embed the question with the same sentence-transformers model used
         at ingestion time.
      3. Load (or reuse a cached) FAISS HNSW index + networkx idea graph
         for this task, and run a personalized PageRank walk seeded from
         the question's nearest clauses.
      4. Feed the top-ranked clauses to a local Ollama model with a strict
         factual-grounding prompt and return the synthesized prose answer.

    Error mapping:
      404 - unknown task_id, or ingestion hasn't produced a graph yet.
      409 - task exists but ingestion failed or is still in progress.
      503 - the Ollama daemon is unreachable (not running).
      504 - the Ollama daemon is reachable but did not respond in time.
      502 - the Ollama daemon responded with an error or malformed payload.
    """
    task = _get_task(req.task_id)  # raises 404 if unknown

    if task.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Task {req.task_id} is not ready for querying (status={task.status}, "
                   f"stage={task.stage}). Poll GET /api/v1/status/{req.task_id} until completed.",
        )

    data_dir = Path(task.data_dir)

    graph_index = _get_graph_index(data_dir)

    try:
        encoder = _get_query_encoder()
        query_vector = await asyncio.to_thread(
            lambda: encoder.encode(req.question, convert_to_numpy=True, show_progress_bar=False)
        )
        query_vector = np.asarray(query_vector, dtype=np.float32)
    except Exception as exc:
        log.exception("Failed to encode query for task %s", req.task_id)
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {exc}") from exc

    seed_top_n   = req.seed_top_n or _BASE_CFG.ppr.seed_top_n
    result_top_m = req.result_top_m or _BASE_CFG.ppr.result_top_m

    try:
        ranked_clauses, fallback_used = await asyncio.to_thread(
            gt.personalized_pagerank_search,
            query_vector,
            graph_index,
            seed_top_n,
            result_top_m,
            _BASE_CFG.ppr.pagerank_alpha,
            _BASE_CFG.ppr.max_iter,
            _BASE_CFG.ppr.tol,
            _BASE_CFG.phase3.hnsw_ef_search,
        )
    except gt.GraphIndexError as exc:
        log.error("PPR graph surf failed for task %s: %s", req.task_id, exc)
        raise HTTPException(status_code=500, detail=f"Graph traversal failed: {exc}") from exc

    if not ranked_clauses:
        raise HTTPException(
            status_code=404,
            detail="No relevant clauses were found for this query in the ingested document.",
        )

    if _BASE_CFG.ppr.enable_heading_expansion:
        ranked_clauses = gt.expand_heading_context(
            ranked_clauses,
            graph_index,
            max_expansion_per_heading=_BASE_CFG.ppr.max_expansion_per_heading,
            min_words_for_heading=_BASE_CFG.ppr.min_words_for_heading,
            stop_at_boundary=_BASE_CFG.ppr.expansion_stop_at_boundary,
            min_expansion_before_boundary=_BASE_CFG.ppr.min_expansion_before_boundary,
        )

    context_texts = [c.sentence for c in ranked_clauses]

    t0 = asyncio.get_event_loop().time()
    try:
        synthesis = await ls.synthesize_answer(
            query=req.question,
            context_clauses=context_texts,
            ollama_base_url=_BASE_CFG.llm.ollama_base_url,
            model_name=_BASE_CFG.llm.model_name,
            connect_timeout_seconds=_BASE_CFG.llm.connect_timeout_seconds,
            request_timeout_seconds=_BASE_CFG.llm.request_timeout_seconds,
            temperature=_BASE_CFG.llm.temperature,
            num_predict=_BASE_CFG.llm.num_predict,
            max_context_chars=_BASE_CFG.llm.max_context_chars,
        )
    except ls.OllamaUnavailableError as exc:
        log.error("Ollama unavailable for task %s: %s", req.task_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ls.OllamaTimeoutError as exc:
        log.error("Ollama timed out for task %s: %s", req.task_id, exc)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ls.OllamaResponseError as exc:
        log.error("Ollama returned an error for task %s: %s", req.task_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    total_latency_ms = (asyncio.get_event_loop().time() - t0) * 1000

    return QueryResponse(
        task_id=req.task_id,
        question=req.question,
        answer=synthesis.answer,
        model=synthesis.model,
        sources=[
            QuerySource(
                sentence_id=c.sentence_id,
                sentence=c.sentence,
                score=round(c.score, 6),
                is_seed=c.is_seed,
                cluster_id=c.cluster_id,
                paragraph_id=c.paragraph_id,
                expansion_source=c.expansion_source,
            )
            for c in ranked_clauses
        ],
        ppr_fallback_used=fallback_used,
        latency_ms=round(total_latency_ms, 1),
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}