# Thought-to-Structure Neural System (Stasis)

A **fully local graph-based RAG system**. Upload a PDF, and it is split into clauses, embedded, and linked into an "idea graph". Questions are answered by walking that graph with Personalized PageRank and synthesizing the answer with a local LLM.

No cloud APIs. No GPU. Runs CPU-only on a 16 GB machine.

---

## Why this exists

Most RAG tools retrieve isolated chunks by vector similarity and lose the structure of the document. This project keeps that structure: clauses are nodes in a graph, connected by semantic similarity and by narrative proximity, so retrieval can follow the way ideas connect instead of only matching keywords. It is built for **researchers first**, then business use.

## Features

- **PDF ingestion** with text extraction, column handling, and OCR fallback for scanned pages
- **Clause-level splitting** (spaCy) with rule-based heading detection and bounded expansion
- **FAISS HNSW** index over L2-normalized sentence embeddings (inner-product metric)
- **Idea graph** with semantic edges plus sequential proximity edges (weight 0.6) to preserve narrative flow
- **Personalized PageRank** retrieval, with personalization vectors weighted by seed cosine similarity
- **Local answer synthesis** via Ollama (`llama3.1`) with a strict factual-grounding prompt
- **Async FastAPI service**: ingestion runs as a background task and returns `202 Accepted` with a task ID
- **Cluster-aware structured summaries** and an **interactive graph visualization**
- **Domain profiles** (`academic`, `clinical`, `legal`) switchable by YAML, with no code changes
- **Streamlit UI** for upload and querying

## Results

| Test | Result |
|---|---|
| 13-question suite on Dostoevsky's *White Nights* | **89% pass rate** |
| Real research papers (manual checking) | **~88% accuracy** |
| Fast GraphRAG (CircleMind) on the same hardware | Failed to run locally |

Full output of the test run is in [`test_results.txt`](test_results.txt).

## Architecture

```
PDF ──► pdf_to_txt ──► Phase 1: clause splitting
                            │
                            ▼
                     Phase 2: embeddings (all-mpnet-base-v2)
                            │
                            ▼
                     Phase 3: idea graph (FAISS HNSW + networkx)
                            │
                            ▼
                     Phase 4: cluster-aware summarization
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     Phase 5: query-time PPR        Graph visualization
     graph traversal                 (/api/v1/graph)
              │
              ▼
     Ollama (llama3.1) synthesis ──► answer + cited source clauses
```

## Project structure

```
├── main.py                  # FastAPI service (ingest / status / summary / graph / query)
├── run_pipeline.py          # CLI entry point; runs all phases with staleness checking
├── config.yaml              # Default pipeline configuration (heavily commented)
├── configs/                 # Domain profiles: academic.yaml, clinical.yaml, legal.yaml
├── src/
│   ├── pdf_to_txt.py        # PDF extraction + OCR
│   ├── phase1_data_prep.py  # Clause splitting
│   ├── phase2_embeddings.py # Sentence embeddings
│   ├── phase3_idea_graph.py # Graph construction
│   ├── phase4_gnn_refiner.py# Structured summarization
│   ├── phase5_api.py        # Query logic
│   ├── graph_traversal.py   # Personalized PageRank
│   ├── llm_synthesizer.py   # Ollama answer synthesis
│   └── config_loader.py
├── static/app.py            # Streamlit UI
├── notebooks/visualize_graph.ipynb
└── test_results.txt
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- ~16 GB RAM (CPU-only is fine)
- Optional: Tesseract, for OCR on scanned PDFs

## Installation

```bash
git clone https://github.com/huzaifa18ali-del/RAG_GRAPH_BASED_RETRIEVAL.git
cd RAG_GRAPH_BASED_RETRIEVAL

python -m venv .venv
source .venv/bin/activate

pip install fastapi "uvicorn[standard]" python-multipart pydantic httpx requests pyyaml \
            numpy scikit-learn networkx sentence-transformers faiss-cpu \
            pdfplumber spacy matplotlib streamlit pytesseract

python -m spacy download en_core_web_sm

# Pull the local LLM
ollama pull llama3.1
```

## Usage

### 1. Start the API

```bash
export PYTHONPATH=src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Ingest a document

```bash
curl -F "file=@path/to/document.pdf" http://localhost:8000/api/v1/ingest
# -> {"task_id": "...", "status": "...", "message": "..."}
```

### 3. Check progress

```bash
curl http://localhost:8000/api/v1/status/<task_id>
```

### 4. Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"task_id": "<task_id>", "question": "Who is the narrator talking to?"}'
```

Optional fields: `seed_top_n` (1-50) and `result_top_m` (1-30).

The response contains the synthesized `answer`, the `model` used, and the ranked `sources` (clause text, score, whether it was a seed, cluster, and paragraph).

### 5. Other endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/summary/{task_id}` | Structured, cluster-ordered summary |
| `GET` | `/api/v1/graph/{task_id}` | Interactive idea-graph visualization |
| `GET` | `/healthz` | Health check |

### Streamlit UI

```bash
streamlit run static/app.py
```

### Run the pipeline from the CLI

```bash
python run_pipeline.py                         # run all stale phases
python run_pipeline.py --force                 # re-run everything
python run_pipeline.py --from phase3           # re-run from phase 3 onward
python run_pipeline.py --config configs/legal.yaml
```

## Configuration

Everything tunable lives in `config.yaml`, with a comment on each parameter explaining why it exists. To switch domains, point at a profile:

```bash
python run_pipeline.py --config configs/academic.yaml
```

Key parameters: `phase2.model_name` (embedding model), `phase3` (HNSW and edge settings), `ppr` (PageRank damping and iterations), `llm` (Ollama URL and model).

## Known limitations

- Task IDs are held in memory, so they **reset on server restart** and documents must be re-ingested.
- Query latency on CPU is higher than ideal; speeding this up is in progress.
- Only PDF input is supported, and only Ollama is implemented as an LLM provider.

## Notes

- Run manual commands with `PYTHONPATH=src`.
- Run test scripts with `bash`, not `zsh`.

## License

Add a license of your choice (e.g. MIT).
