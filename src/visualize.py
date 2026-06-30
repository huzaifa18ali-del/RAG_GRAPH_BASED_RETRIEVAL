#!/usr/bin/env python3
"""
visualize.py

Idea Graph Visualizer — Phase 3 output explorer.

Reads  : data/idea_graph.json   (required)
         data/cluster_summary.json  (optional — enriches cluster panel)
         data/embeddings_manifest.json  (optional — adds chunk info)

Writes : data/idea_graph_explorer.html   (self-contained, no server needed)

Usage:
    python visualize.py                          # default paths
    python visualize.py --graph path/to/graph.json
    python visualize.py --out my_explorer.html
    python visualize.py --graph g.json --cluster c.json --port 8080
    python visualize.py --serve                  # open browser automatically

The output is a single .html file — no CDN calls, no server required.
Share it by copying one file.  All JS/CSS is inlined.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_GRAPH_PATH   = os.path.join("data", "idea_graph.json")
DEFAULT_CLUSTER_PATH = os.path.join("data", "cluster_summary.json")
DEFAULT_MANIFEST_PATH= os.path.join("data", "embeddings_manifest.json")
DEFAULT_OUT_PATH     = os.path.join("data", "idea_graph_explorer.html")
DEFAULT_PORT         = 8000


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------

def _load_json(path: str, label: str, required: bool = True) -> Optional[Any]:
    p = Path(path)
    if not p.exists():
        if required:
            sys.exit(f"[visualize] ERROR: {label} not found: {path}\n"
                     f"  Run Phase 3 first, or pass --graph <path>.")
        print(f"[visualize] Optional {label} not found — skipping: {path}")
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        print(f"[visualize] Loaded {label:<28} ← {path}  ({p.stat().st_size // 1024} KB)")
        return data
    except json.JSONDecodeError as exc:
        sys.exit(f"[visualize] ERROR: Failed to parse {label}: {exc}")


def validate_graph(graph: list) -> None:
    """Abort early with a clear message if the graph schema looks wrong."""
    if not isinstance(graph, list) or len(graph) == 0:
        sys.exit("[visualize] ERROR: idea_graph.json must be a non-empty JSON array.")
    required_keys = {"sentence_id", "sentence", "neighbors"}
    sample = graph[0]
    missing = required_keys - set(sample.keys())
    if missing:
        sys.exit(f"[visualize] ERROR: First node is missing keys: {missing}\n"
                 f"  Expected schema from Phase 3: sentence_id, sentence, cluster_id, "
                 f"neighbors, paragraph_id, threshold_used.")
    print(f"[visualize] Graph validated — {len(graph)} nodes.")


# ---------------------------------------------------------------------------
# Graph stats (computed Python-side so JS doesn't have to)
# ---------------------------------------------------------------------------

def compute_graph_stats(graph: list[dict]) -> dict:
    """Pre-compute summary stats embedded in the HTML for the info panel."""
    n_nodes  = len(graph)
    all_edges: set[tuple] = set()
    cross_edges = 0
    sims: list[float] = []
    cluster_counts: dict[int, int] = {}
    degree_sum = 0

    for node in graph:
        cid = node.get("cluster_id", -1)
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        degree_sum += len(node.get("neighbors", []))
        for nb in node.get("neighbors", []):
            key = (min(node["sentence_id"], nb["sentence_id"]),
                   max(node["sentence_id"], nb["sentence_id"]))
            if key not in all_edges:
                all_edges.add(key)
                sims.append(nb["similarity"])
                if nb.get("cross_cluster"):
                    cross_edges += 1

    avg_sim = sum(sims) / len(sims) if sims else 0.0
    avg_deg = degree_sum / n_nodes if n_nodes else 0.0
    isolated = sum(1 for n in graph if not n.get("neighbors"))

    return {
        "n_nodes":       n_nodes,
        "n_edges":       len(all_edges),
        "n_cross_edges": cross_edges,
        "n_clusters":    len(cluster_counts),
        "avg_sim":       round(avg_sim, 4),
        "avg_degree":    round(avg_deg, 2),
        "n_isolated":    isolated,
        "cluster_sizes": {str(k): v for k, v in
                          sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)},
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _json_embed(obj: Any) -> str:
    """Compact JSON string safe to embed in a <script> block."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_html(
    graph: list[dict],
    cluster_summary: Optional[list[dict]],
    manifest: Optional[list[dict]],
    source_path: str,
) -> str:
    stats = compute_graph_stats(graph)
    graph_json    = _json_embed(graph)
    cluster_json  = _json_embed(cluster_summary or [])
    stats_json    = _json_embed(stats)
    source_name   = Path(source_path).name

    # Inline everything — zero external deps, works offline
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Idea Graph Explorer — {source_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg0:#ffffff;--bg1:#f7f6f3;--bg2:#efede8;--bg3:#e5e2db;
  --text0:#1a1916;--text1:#4a4845;--text2:#777470;
  --border:#d3d1c7;--border2:#b4b2a9;
  --purple:#7f77dd;--teal:#1d9e75;--coral:#d85a30;--blue:#378add;
  --pink:#d4537e;--green:#639922;--amber:#ba7517;--red:#e24b4a;
  --gray:#888780;--noise:#aaa9a4;
  --accent:#534ab7;
  --radius:8px;--radius-lg:12px;
  --shadow:0 1px 3px rgba(0,0,0,.08);
}}
@media(prefers-color-scheme:dark){{
  :root{{
    --bg0:#1a1916;--bg1:#222118;--bg2:#2c2c2a;--bg3:#373735;
    --text0:#f1efe8;--text1:#b4b2a9;--text2:#777470;
    --border:#444441;--border2:#5f5e5a;
  }}
}}
html,body{{height:100%;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;font-size:14px;color:var(--text0);background:var(--bg1)}}
#app{{display:flex;flex-direction:column;height:100vh;overflow:hidden}}

/* ── Toolbar ── */
#toolbar{{
  display:flex;align-items:center;gap:8px;padding:8px 14px;
  background:var(--bg0);border-bottom:1px solid var(--border);
  flex-wrap:wrap;flex-shrink:0;z-index:20
}}
#toolbar h1{{font-size:13px;font-weight:600;color:var(--text0);white-space:nowrap;display:flex;align-items:center;gap:6px}}
#toolbar h1 svg{{flex-shrink:0}}
.tb-sep{{width:1px;height:22px;background:var(--border);margin:0 2px;flex-shrink:0}}
.tb-group{{display:flex;align-items:center;gap:5px;flex-shrink:0}}
.tb-label{{font-size:11px;color:var(--text2);white-space:nowrap}}
select,input[type=text]{{
  font-size:12px;padding:3px 7px;height:28px;
  border-radius:var(--radius);border:1px solid var(--border);
  background:var(--bg0);color:var(--text0);outline:none
}}
select:focus,input[type=text]:focus{{border-color:var(--accent)}}
input[type=range]{{width:80px;accent-color:var(--accent);cursor:pointer}}
.btn{{
  font-size:12px;padding:3px 10px;height:28px;
  border-radius:var(--radius);border:1px solid var(--border);
  background:var(--bg0);color:var(--text0);cursor:pointer;
  display:flex;align-items:center;gap:4px;white-space:nowrap
}}
.btn:hover{{background:var(--bg2)}}
.btn.active{{background:var(--accent);border-color:var(--accent);color:#fff}}
.btn svg{{flex-shrink:0}}
#search{{width:160px}}
#zoom-label{{font-size:11px;color:var(--text2);min-width:38px;text-align:right}}

/* ── Stats bar ── */
#statsbar{{
  display:flex;gap:20px;padding:5px 14px;
  background:var(--bg1);border-bottom:1px solid var(--border);
  font-size:11px;color:var(--text2);flex-shrink:0;flex-wrap:wrap
}}
.stat{{display:flex;gap:5px}}.stat b{{color:var(--text0);font-weight:600}}

/* ── Legend ── */
#legend{{
  display:flex;flex-wrap:wrap;gap:5px;padding:6px 14px;
  background:var(--bg0);border-bottom:1px solid var(--border);
  flex-shrink:0
}}
.cl-chip{{
  display:flex;align-items:center;gap:4px;font-size:11px;color:var(--text1);
  padding:2px 8px;border-radius:20px;border:1px solid var(--border);
  cursor:pointer;user-select:none;transition:opacity .15s
}}
.cl-chip:hover{{border-color:var(--border2);background:var(--bg2)}}
.cl-chip.muted{{opacity:.35}}
.cl-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}

/* ── Main area ── */
#main{{display:flex;flex:1;overflow:hidden}}
#canvas-wrap{{flex:1;position:relative;overflow:hidden;background:var(--bg1);cursor:grab}}
#canvas-wrap.panning{{cursor:grabbing}}
canvas{{position:absolute;top:0;left:0;display:block}}

/* ── Right panel ── */
#panel{{
  width:320px;border-left:1px solid var(--border);
  background:var(--bg0);display:flex;flex-direction:column;
  overflow:hidden;transition:width .2s ease
}}
#panel.closed{{width:0;border-left:none}}
#panel-inner{{width:320px;overflow-y:auto;padding:14px;flex:1}}
#panel-inner h2{{font-size:13px;font-weight:600;margin-bottom:12px;color:var(--text0);display:flex;align-items:center;gap:6px}}
.pf{{margin-bottom:12px}}
.pf-label{{font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--text2);margin-bottom:4px}}
.pf-val{{
  font-size:12px;color:var(--text0);line-height:1.55;
  background:var(--bg1);border:1px solid var(--border);
  border-radius:var(--radius);padding:6px 9px
}}
.pf-val.sentence{{font-size:13px;font-style:italic;line-height:1.65}}
.nb-list{{display:flex;flex-direction:column;gap:4px}}
.nb-item{{
  display:flex;align-items:flex-start;gap:6px;padding:6px 8px;
  border-radius:var(--radius);border:1px solid var(--border);
  cursor:pointer;background:var(--bg0);transition:background .1s
}}
.nb-item:hover{{background:var(--bg2)}}
.nb-sim{{font-size:11px;font-weight:700;min-width:38px;flex-shrink:0;padding-top:1px}}
.nb-badge{{font-size:9px;font-weight:600;padding:1px 5px;border-radius:10px;flex-shrink:0;margin-top:2px}}
.nb-badge.cross{{background:#faeeda;color:#854f0b}}
@media(prefers-color-scheme:dark){{.nb-badge.cross{{background:#412402;color:#fac775}}}}
.nb-text{{font-size:11px;color:var(--text1);line-height:1.4;flex:1}}
.panel-footer{{padding:10px 14px;border-top:1px solid var(--border);display:flex;flex-direction:column;gap:6px;flex-shrink:0}}
.panel-footer .btn{{justify-content:center;width:100%}}

/* ── Cluster info panel (stats tab) ── */
#stats-panel{{
  width:320px;border-left:1px solid var(--border);
  background:var(--bg0);display:flex;flex-direction:column;
  overflow:hidden;transition:width .2s ease
}}
#stats-panel.closed{{width:0;border-left:none}}
#stats-inner{{width:320px;overflow-y:auto;padding:14px;flex:1}}
#stats-inner h2{{font-size:13px;font-weight:600;margin-bottom:12px;color:var(--text0)}}
.cluster-card{{
  border:1px solid var(--border);border-radius:var(--radius);
  padding:10px;margin-bottom:8px;background:var(--bg1)
}}
.cluster-card-header{{display:flex;align-items:center;gap:8px;margin-bottom:6px}}
.cluster-card h3{{font-size:12px;font-weight:600;color:var(--text0)}}
.cluster-card p{{font-size:11px;color:var(--text1);line-height:1.5;font-style:italic}}
.cluster-card .meta{{font-size:10px;color:var(--text2);margin-top:4px}}

/* ── Tooltip ── */
#tooltip{{
  position:absolute;pointer-events:none;z-index:100;
  background:var(--bg0);border:1px solid var(--border);
  border-radius:var(--radius);padding:7px 10px;font-size:11px;
  box-shadow:var(--shadow);max-width:260px;line-height:1.5;
  color:var(--text0);display:none;white-space:pre-wrap;
  transition:opacity .1s
}}
#tooltip b{{font-size:10px;font-weight:600;letter-spacing:.04em;color:var(--text2);display:block;margin-bottom:2px}}

/* ── Minimap ── */
#minimap{{
  position:absolute;bottom:14px;left:14px;width:140px;height:90px;
  background:var(--bg0);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;cursor:pointer;
  box-shadow:var(--shadow);
}}
#minimap canvas{{position:absolute;top:0;left:0}}
#minimap-viewport{{
  position:absolute;border:1.5px solid var(--accent);
  border-radius:3px;pointer-events:none;
}}

/* ── File open button (when no graph yet) ── */
#welcome{{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:16px;
  background:var(--bg1);z-index:50
}}
#welcome h2{{font-size:18px;font-weight:600;color:var(--text0)}}
#welcome p{{font-size:13px;color:var(--text1);text-align:center;max-width:360px;line-height:1.7}}
#drop-zone{{
  width:300px;height:120px;border:2px dashed var(--border2);
  border-radius:var(--radius-lg);display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:8px;cursor:pointer;
  transition:border-color .15s,background .15s
}}
#drop-zone:hover,#drop-zone.drag{{border-color:var(--accent);background:rgba(83,74,183,.06)}}
#drop-zone span{{font-size:12px;color:var(--text2)}}
#file-input{{display:none}}
</style>
</head>
<body>
<div id="app">

  <!-- Toolbar -->
  <div id="toolbar">
    <h1>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="5" r="3"/><circle cx="5" cy="19" r="3"/><circle cx="19" cy="19" r="3"/><line x1="12" y1="8" x2="5" y2="16"/><line x1="12" y1="8" x2="19" y2="16"/></svg>
      Idea Graph Explorer
    </h1>
    <div class="tb-sep"></div>
    <div class="tb-group">
      <span class="tb-label">Layout</span>
      <select id="sel-layout" title="Graph layout algorithm">
        <option value="force">Force directed</option>
        <option value="radial">Cluster radial</option>
        <option value="grid">Grid</option>
      </select>
    </div>
    <div class="tb-sep"></div>
    <div class="tb-group">
      <span class="tb-label">Min sim</span>
      <input type="range" id="sld-sim" min="0" max="100" value="0" step="1" title="Minimum edge similarity">
      <span class="tb-label" id="sim-val" style="min-width:32px">0.00</span>
    </div>
    <div class="tb-sep"></div>
    <div class="tb-group">
      <span class="tb-label">Node size</span>
      <select id="sel-size" title="Node size encoding">
        <option value="uniform">Uniform</option>
        <option value="degree">Degree</option>
      </select>
    </div>
    <div class="tb-sep"></div>
    <div class="tb-group">
      <span class="tb-label">Edge color</span>
      <select id="sel-ecol" title="Edge color encoding">
        <option value="cluster">Cluster</option>
        <option value="sim">Similarity</option>
      </select>
    </div>
    <div class="tb-sep"></div>
    <input type="text" id="search" class="btn" placeholder="&#128269;  Search…" title="Highlight matching nodes" aria-label="Search sentences">
    <div class="tb-sep"></div>
    <button class="btn" id="btn-cross" title="Show only cross-cluster edges">Cross-cluster</button>
    <button class="btn" id="btn-iso" title="Show/hide isolated nodes">Hide isolated</button>
    <div class="tb-sep"></div>
    <button class="btn" id="btn-stats" title="Open cluster stats panel">Stats</button>
    <button class="btn" id="btn-open" title="Open another JSON file">Open file</button>
    <input type="file" id="file-input2" accept=".json" style="display:none">
    <div class="tb-sep"></div>
    <button class="btn" id="btn-fit" title="Zoom to fit (F)">Fit</button>
    <button class="btn" id="btn-png" title="Export canvas as PNG">Export PNG</button>
    <span id="zoom-label">100%</span>
  </div>

  <!-- Stats bar -->
  <div id="statsbar">
    <div class="stat"><span>Nodes</span><b id="st-nodes">—</b></div>
    <div class="stat"><span>Edges</span><b id="st-edges">—</b></div>
    <div class="stat"><span>Clusters</span><b id="st-clusters">—</b></div>
    <div class="stat"><span>Cross-cluster edges</span><b id="st-cross">—</b></div>
    <div class="stat"><span>Avg similarity</span><b id="st-sim">—</b></div>
    <div class="stat"><span>Avg degree</span><b id="st-deg">—</b></div>
    <div class="stat"><span>Isolated</span><b id="st-iso">—</b></div>
  </div>

  <!-- Cluster legend -->
  <div id="legend"></div>

  <!-- Main -->
  <div id="main">
    <div id="canvas-wrap">
      <canvas id="bg-canvas"></canvas>
      <canvas id="fg-canvas"></canvas>
      <div id="tooltip"></div>
      <div id="minimap">
        <canvas id="mm-canvas"></canvas>
        <div id="minimap-viewport"></div>
      </div>
      <div id="welcome">
        <h2>Idea Graph Explorer</h2>
        <p>Drop your <code>idea_graph.json</code> to visualize the sentence similarity network built by Phase 3.</p>
        <div id="drop-zone" role="button" tabindex="0" aria-label="Drop area for idea_graph.json">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          <span>Drop idea_graph.json here</span>
          <span>or click to browse</span>
        </div>
        <input type="file" id="file-input" accept=".json">
      </div>
    </div>

    <!-- Node detail panel -->
    <div id="panel" class="closed">
      <div id="panel-inner">
        <h2>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          Node detail
        </h2>
        <div id="panel-content"></div>
      </div>
      <div class="panel-footer">
        <button class="btn" id="btn-path" title="Find shortest path from this node">Find path to…</button>
        <button class="btn" id="btn-close-panel">Close</button>
      </div>
    </div>

    <!-- Cluster stats panel -->
    <div id="stats-panel" class="closed">
      <div id="stats-inner">
        <h2>Cluster summary</h2>
        <div id="stats-content"></div>
      </div>
    </div>
  </div>
</div>

<script>
// ============================================================
//  DATA  (embedded by Python)
// ============================================================
const GRAPH_DATA    = {graph_json};
const CLUSTER_DATA  = {cluster_json};
const STATS_DATA    = {stats_json};
const SOURCE_NAME   = {json.dumps(source_name)};

// ============================================================
//  PALETTE
// ============================================================
const PALETTE = [
  '#7f77dd','#1d9e75','#d85a30','#378add','#d4537e',
  '#639922','#ba7517','#e24b4a','#534ab7','#0f6e56',
  '#993c1d','#185fa5','#993556','#3b6d11','#854f0b','#a32d2d',
];
const NOISE_COL = '#888780';
function clusterColor(cid) {{
  if (cid === -1 || cid === undefined) return NOISE_COL;
  return PALETTE[((cid % PALETTE.length) + PALETTE.length) % PALETTE.length];
}}

// ============================================================
//  STATE
// ============================================================
let nodes = [], edges = [], nodeMap = {{}};
let clusterIds = [], clusterVis = {{}};
let selectedId = null, hoveredId = null;
let simFloor   = 0;
let crossOnly  = false;
let hideIso    = false;
let searchTerm = '';
let sizeMode   = 'uniform';
let ecolMode   = 'cluster';
let layoutMode = 'force';
let findPathMode = false;
let pathFrom   = null;

// Camera
let camX=0, camY=0, camZ=1;
let panning=false, panStart=null, camStart=null;
let draggingNode=null, dragMoved=false;

// Force sim
let simRunning=false, simTick=0, animId=null;

// ============================================================
//  DOM
// ============================================================
const wrap    = document.getElementById('canvas-wrap');
const bgC     = document.getElementById('bg-canvas');
const fgC     = document.getElementById('fg-canvas');
const bgCtx   = bgC.getContext('2d');
const fgCtx   = fgC.getContext('2d');
const mmC     = document.getElementById('mm-canvas');
const mmCtx   = mmC.getContext('2d');
const tooltip = document.getElementById('tooltip');
const panel   = document.getElementById('panel');
const statsPanel = document.getElementById('stats-panel');
const welcome = document.getElementById('welcome');

// ============================================================
//  RESIZE
// ============================================================
let _resizeTimer = null;
function resize() {{
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (!w || !h) return;
  bgC.width = fgC.width = w; bgC.height = fgC.height = h;
  mmC.width = 140; mmC.height = 90;
  // Re-fit camera so nodes don't disappear after resize.
  // Debounced so rapid resize events (panel open/close) don't thrash.
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {{
    if (nodes.length && !simRunning) zoomToFit();
    else draw();
  }}, 120);
}}
new ResizeObserver(resize).observe(wrap);

// ============================================================
//  INIT GRAPH
// ============================================================
function initGraph(data) {{
  nodes = data.map(n => ({{
    id:          n.sentence_id,
    sentence:    n.sentence,
    para:        n.paragraph_id,
    cid:         n.cluster_id,
    thresh:      n.threshold_used,
    chunk:       n.chunk_index,
    totalChunks: n.total_chunks,
    neighbors:   n.neighbors || [],
    x: (Math.random()-0.5)*600,
    y: (Math.random()-0.5)*400,
    vx:0, vy:0,
    degree: (n.neighbors||[]).length,
  }}));

  nodeMap = {{}};
  nodes.forEach(n => nodeMap[n.id] = n);

  // Deduplicate undirected edges
  const seen = new Set();
  edges = [];
  data.forEach(n => {{
    (n.neighbors||[]).forEach(nb => {{
      const lo = Math.min(n.sentence_id, nb.sentence_id);
      const hi = Math.max(n.sentence_id, nb.sentence_id);
      const key = lo+'-'+hi;
      if (!seen.has(key)) {{
        seen.add(key);
        edges.push({{a:n.sentence_id, b:nb.sentence_id, sim:nb.similarity, cross:!!nb.cross_cluster}});
      }}
    }});
  }});

  // Normalize degree
  const maxDeg = Math.max(...nodes.map(n=>n.degree), 1);
  nodes.forEach(n => n.normDeg = n.degree/maxDeg);

  clusterIds = [...new Set(nodes.map(n=>n.cid))].sort((a,b)=>a-b);
  clusterVis = {{}};
  clusterIds.forEach(c => clusterVis[c] = true);

  updateStats();
  buildLegend();
  welcome.style.display = 'none';

  if (layoutMode === 'force') startForce();
  else applyLayout();
}}

// ============================================================
//  LAYOUT
// ============================================================
function applyLayout() {{
  if (layoutMode === 'radial') radialLayout();
  else if (layoutMode === 'grid') gridLayout();
  else startForce();
  zoomToFit();
}}

function radialLayout() {{
  stopForce();
  const groups = {{}};
  nodes.forEach(n => {{
    if (!groups[n.cid]) groups[n.cid] = [];
    groups[n.cid].push(n);
  }});
  const cids = Object.keys(groups);
  const nc   = cids.length;
  const R    = Math.min(fgC.width, fgC.height) * 0.36;
  cids.forEach((cid, ci) => {{
    const angle = (ci/nc)*Math.PI*2 - Math.PI/2;
    const cx = Math.cos(angle)*R, cy = Math.sin(angle)*R;
    const members = groups[cid];
    const r2 = Math.min(70, members.length*8 + 20);
    members.forEach((n,mi) => {{
      const a2 = (mi/members.length)*Math.PI*2;
      n.x = cx + Math.cos(a2)*r2; n.y = cy + Math.sin(a2)*r2;
      n.vx=0; n.vy=0;
    }});
  }});
  draw();
}}

function gridLayout() {{
  stopForce();
  const cols = Math.ceil(Math.sqrt(nodes.length));
  const gap  = 28;
  nodes.forEach((n,i) => {{
    n.x = (i%cols - cols/2)*gap;
    n.y = (Math.floor(i/cols) - Math.floor(nodes.length/cols)/2)*gap;
    n.vx=0; n.vy=0;
  }});
  draw();
}}

// ============================================================
//  FORCE SIMULATION
// ============================================================
const FORCE_ITERS  = 250;
const REPULSION    = 1400;
const SPRING_LEN_S = 55;   // same cluster
const SPRING_LEN_X = 200;  // cross cluster
const SPRING_K     = 0.3;
const DAMPING      = 0.78;

function startForce() {{
  stopForce();
  simRunning = true; simTick = 0;
  animId = requestAnimationFrame(stepForce);
}}
function stopForce() {{
  simRunning = false;
  if (animId) {{ cancelAnimationFrame(animId); animId=null; }}
}}
function stepForce() {{
  if (!simRunning || simTick >= FORCE_ITERS) {{ simRunning=false; zoomToFit(); return; }}
  const alpha = Math.max(0.01, 1 - simTick/FORCE_ITERS);
  // Damping
  nodes.forEach(n => {{ n.vx*=DAMPING; n.vy*=DAMPING; }});
  // Repulsion (Barnes-Hut approximation via grid bucketing for speed)
  const CELL = 80;
  const grid = {{}};
  nodes.forEach(n => {{
    const gx = Math.floor(n.x/CELL), gy = Math.floor(n.y/CELL);
    const k = gx+','+gy;
    if (!grid[k]) grid[k]=[];
    grid[k].push(n);
  }});
  nodes.forEach(a => {{
    const gx = Math.floor(a.x/CELL), gy = Math.floor(a.y/CELL);
    for (let dx=-2;dx<=2;dx++) for (let dy=-2;dy<=2;dy++) {{
      const bucket = grid[(gx+dx)+','+(gy+dy)];
      if (!bucket) continue;
      bucket.forEach(b => {{
        if (b.id >= a.id) return;
        let ddx=b.x-a.x, ddy=b.y-a.y;
        const d2=ddx*ddx+ddy*ddy+0.1, d=Math.sqrt(d2);
        const f=REPULSION/(d2)*alpha;
        a.vx-=ddx/d*f; a.vy-=ddy/d*f;
        b.vx+=ddx/d*f; b.vy+=ddy/d*f;
      }});
    }}
  }});
  // Spring attraction
  edges.forEach(e => {{
    const a=nodeMap[e.a], b=nodeMap[e.b];
    if (!a||!b) return;
    const ddx=b.x-a.x, ddy=b.y-a.y;
    const d=Math.sqrt(ddx*ddx+ddy*ddy)||1;
    const target = e.cross ? SPRING_LEN_X : SPRING_LEN_S;
    const f=(d-target)/d*SPRING_K*alpha*e.sim;
    a.vx+=ddx*f; a.vy+=ddy*f;
    b.vx-=ddx*f; b.vy-=ddy*f;
  }});
  // Gravity toward origin
  nodes.forEach(n => {{ n.vx-=n.x*0.002*alpha; n.vy-=n.y*0.002*alpha; }});
  // Integrate
  nodes.forEach(n => {{ n.x+=n.vx; n.y+=n.vy; }});
  simTick++;
  draw();
  animId = requestAnimationFrame(stepForce);
}}

// ============================================================
//  CAMERA
// ============================================================
function worldToScreen(x,y) {{
  return {{ x: x*camZ + fgC.width/2 + camX, y: y*camZ + fgC.height/2 + camY }};
}}
function screenToWorld(sx,sy) {{
  return {{ x:(sx - fgC.width/2 - camX)/camZ, y:(sy - fgC.height/2 - camY)/camZ }};
}}
function zoomToFit(pad=60) {{
  if (!nodes.length) return;
  const xs=nodes.map(n=>n.x), ys=nodes.map(n=>n.y);
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  const gw=maxX-minX||1, gh=maxY-minY||1;
  const w=fgC.width-pad*2, h=fgC.height-pad*2;
  camZ = Math.min(w/gw, h/gh, 4);
  camX = -(minX+gw/2)*camZ;
  camY = -(minY+gh/2)*camZ;
  updateZoomLabel();
  draw();
}}
function updateZoomLabel() {{
  document.getElementById('zoom-label').textContent = Math.round(camZ*100)+'%';
}}

// ============================================================
//  FILTERS
// ============================================================
function visEdges() {{
  return edges.filter(e => {{
    const a=nodeMap[e.a], b=nodeMap[e.b];
    if (!a||!b) return false;
    if (!clusterVis[a.cid]||!clusterVis[b.cid]) return false;
    if (e.sim < simFloor) return false;
    if (crossOnly && !e.cross) return false;
    return true;
  }});
}}
function visNodes() {{
  if (!hideIso && simFloor===0 && !crossOnly) {{
    return nodes.filter(n => clusterVis[n.cid]);
  }}
  const connected = new Set();
  visEdges().forEach(e => {{ connected.add(e.a); connected.add(e.b); }});
  return nodes.filter(n => {{
    if (!clusterVis[n.cid]) return false;
    if (hideIso || simFloor>0 || crossOnly) return connected.has(n.id);
    return true;
  }});
}}

function nodeR(n) {{
  const base = sizeMode==='degree' ? 4+n.normDeg*10 : 6;
  return base;
}}

// ============================================================
//  DRAW
// ============================================================
function simColor(s) {{
  // Blue→Green interpolation by similarity
  const t = Math.max(0, Math.min(1, (s-0.3)/0.65));
  const r = Math.round(55+t*(29-55));
  const g = Math.round(138+t*(158-138));
  const dd = Math.round(221+t*(117-221));
  return `rgb(${{r}},${{g}},${{dd}})`;
}}

function draw() {{
  if (!fgC.width) return;
  bgCtx.clearRect(0,0,bgC.width,bgC.height);
  fgCtx.clearRect(0,0,fgC.width,fgC.height);

  if (!nodes.length) return;

  const vEdges   = visEdges();
  const vNodes   = visNodes();
  const vNodeSet = new Set(vNodes.map(n=>n.id));

  // ── Edges ──
  vEdges.forEach(e => {{
    const a=nodeMap[e.a], b=nodeMap[e.b];
    if (!a||!b||!vNodeSet.has(e.a)||!vNodeSet.has(e.b)) return;
    const pa=worldToScreen(a.x,a.y), pb=worldToScreen(b.x,b.y);
    const alpha = 0.12 + e.sim*0.55;
    const col = ecolMode==='sim'
      ? simColor(e.sim)
      : (e.cross ? '#ba7517' : '#7f77dd');
    bgCtx.beginPath();
    bgCtx.moveTo(pa.x,pa.y); bgCtx.lineTo(pb.x,pb.y);
    bgCtx.strokeStyle = col.replace('rgb','rgba').replace(')',`,  ${{alpha}})`).replace('rgba','rgba');
    // Simpler alpha approach:
    bgCtx.globalAlpha = alpha;
    bgCtx.strokeStyle = col;
    bgCtx.lineWidth   = e.cross ? 1.5 : Math.max(0.5, e.sim*2.5);
    bgCtx.stroke();
    bgCtx.globalAlpha = 1;
  }});

  // ── Path highlight (if active) ──
  // (drawn over edges but under nodes)

  // ── Nodes ──
  const isSearch = searchTerm.length > 0;
  const query    = searchTerm.toLowerCase();

  vNodes.forEach(n => {{
    const p  = worldToScreen(n.x, n.y);
    const r  = nodeR(n) * Math.max(0.6, Math.min(1.4, camZ));
    const col = clusterColor(n.cid);
    const isSelected = n.id === selectedId;
    const isHovered  = n.id === hoveredId;
    const matched    = isSearch && n.sentence.toLowerCase().includes(query);
    const dimmed     = isSearch && !matched && !isSelected;

    if (dimmed) fgCtx.globalAlpha = 0.15;

    if (isSelected || isHovered) {{
      fgCtx.beginPath();
      fgCtx.arc(p.x, p.y, r+5, 0, Math.PI*2);
      fgCtx.fillStyle = isSelected ? col+'50' : col+'30';
      fgCtx.fill();
    }}

    fgCtx.beginPath();
    fgCtx.arc(p.x, p.y, r, 0, Math.PI*2);
    fgCtx.fillStyle = col;
    fgCtx.fill();

    if (matched) {{
      fgCtx.beginPath();
      fgCtx.arc(p.x, p.y, r+2.5, 0, Math.PI*2);
      fgCtx.strokeStyle = '#e24b4a';
      fgCtx.lineWidth = 2;
      fgCtx.stroke();
    }}

    if (findPathMode && pathFrom !== null && n.id !== pathFrom) {{
      fgCtx.beginPath();
      fgCtx.arc(p.x, p.y, r+3, 0, Math.PI*2);
      fgCtx.strokeStyle = '#1d9e75';
      fgCtx.lineWidth = 1.5;
      fgCtx.stroke();
    }}

    fgCtx.globalAlpha = 1;

    // Label when zoomed in or selected/hovered
    if (camZ > 1.2 || isSelected || isHovered) {{
      const label = n.sentence.length > 45 ? n.sentence.slice(0,45)+'…' : n.sentence;
      fgCtx.font = `${{Math.max(10,Math.min(12,10*camZ))}}px -apple-system,sans-serif`;
      fgCtx.fillStyle = 'var(--text0)';
      // Shadow for readability
      fgCtx.shadowColor = 'rgba(0,0,0,0.35)';
      fgCtx.shadowBlur = 3;
      fgCtx.fillText(label, p.x+r+4, p.y+4);
      fgCtx.shadowBlur = 0;
    }}
  }});

  // Highlight selected node's edges
  if (selectedId !== null) {{
    const sel = nodeMap[selectedId];
    if (sel) {{
      sel.neighbors.forEach(nb => {{
        const nb_n = nodeMap[nb.sentence_id];
        if (!nb_n || !vNodeSet.has(nb.sentence_id)) return;
        const pa = worldToScreen(sel.x, sel.y);
        const pb = worldToScreen(nb_n.x, nb_n.y);
        fgCtx.beginPath();
        fgCtx.moveTo(pa.x,pa.y); fgCtx.lineTo(pb.x,pb.y);
        fgCtx.strokeStyle = '#7f77dd';
        fgCtx.lineWidth = 2;
        fgCtx.stroke();
      }});
    }}
  }}

  drawMinimap(vNodes, vEdges);
  updateVisStats(vNodes.length, vEdges.length);
}}

// ============================================================
//  MINIMAP
// ============================================================
function drawMinimap(vNodes, vEdges) {{
  mmCtx.clearRect(0,0,140,90);
  if (!vNodes.length) return;
  const xs=vNodes.map(n=>n.x), ys=vNodes.map(n=>n.y);
  const minX=Math.min(...xs)||0, maxX=Math.max(...xs)||1;
  const minY=Math.min(...ys)||0, maxY=Math.max(...ys)||1;
  const gw=maxX-minX||1, gh=maxY-minY||1;
  const scX=130/gw, scY=80/gh, sc=Math.min(scX,scY);
  const offX=(140-gw*sc)/2, offY=(90-gh*sc)/2;
  const tx = x => (x-minX)*sc+offX;
  const ty = y => (y-minY)*sc+offY;

  mmCtx.globalAlpha=0.4;
  vEdges.forEach(e => {{
    const a=nodeMap[e.a], b=nodeMap[e.b];
    if (!a||!b) return;
    mmCtx.beginPath();
    mmCtx.moveTo(tx(a.x),ty(a.y)); mmCtx.lineTo(tx(b.x),ty(b.y));
    mmCtx.strokeStyle='#888'; mmCtx.lineWidth=0.5; mmCtx.stroke();
  }});
  mmCtx.globalAlpha=1;

  vNodes.forEach(n => {{
    mmCtx.beginPath();
    mmCtx.arc(tx(n.x),ty(n.y),2,0,Math.PI*2);
    mmCtx.fillStyle=clusterColor(n.cid); mmCtx.fill();
  }});

  // Viewport rect
  const vp = document.getElementById('minimap-viewport');
  const tl = screenToWorld(0,0), br = screenToWorld(fgC.width,fgC.height);
  const vpX = (tl.x-minX)*sc+offX, vpY = (tl.y-minY)*sc+offY;
  const vpW = (br.x-tl.x)*sc, vpH = (br.y-tl.y)*sc;
  vp.style.left   = Math.max(0,vpX)+'px';
  vp.style.top    = Math.max(0,vpY)+'px';
  vp.style.width  = Math.min(140,vpW)+'px';
  vp.style.height = Math.min(90,vpH)+'px';
}}

// ============================================================
//  STATS
// ============================================================
function updateStats() {{
  document.getElementById('st-nodes').textContent    = STATS_DATA.n_nodes;
  document.getElementById('st-edges').textContent    = STATS_DATA.n_edges;
  document.getElementById('st-clusters').textContent = STATS_DATA.n_clusters;
  document.getElementById('st-cross').textContent    = STATS_DATA.n_cross_edges;
  document.getElementById('st-sim').textContent      = STATS_DATA.avg_sim.toFixed(3);
  document.getElementById('st-deg').textContent      = STATS_DATA.avg_degree.toFixed(1);
  document.getElementById('st-iso').textContent      = STATS_DATA.n_isolated;
}}
function updateVisStats(vn, ve) {{
  document.getElementById('st-nodes').textContent = vn+'/'+STATS_DATA.n_nodes;
  document.getElementById('st-edges').textContent = ve+'/'+STATS_DATA.n_edges;
}}

// ============================================================
//  LEGEND
// ============================================================
function buildLegend() {{
  const leg = document.getElementById('legend');
  leg.innerHTML = '';
  clusterIds.forEach(cid => {{
    const cnt = nodes.filter(n=>n.cid===cid).length;
    const chip = document.createElement('div');
    chip.className = 'cl-chip';
    chip.dataset.cid = cid;
    chip.innerHTML = `<span class="cl-dot" style="background:${{clusterColor(cid)}}"></span>${{cid===-1?'Noise':'C'+cid}} <span style="opacity:.6">(${{cnt}})</span>`;
    chip.addEventListener('click', () => {{
      clusterVis[cid] = !clusterVis[cid];
      chip.classList.toggle('muted', !clusterVis[cid]);
      draw();
    }});
    leg.appendChild(chip);
  }});
}}

// ============================================================
//  NODE PANEL
// ============================================================
function openPanel(node) {{
  selectedId = node.id;
  panel.classList.remove('closed');
  statsPanel.classList.add('closed');
  const col = clusterColor(node.cid);
  const simBadge = s => {{
    const c = s>=0.7?'#1d9e75':s>=0.5?'#ba7517':'#888780';
    return `<span style="color:${{c}};font-weight:700">${{s.toFixed(3)}}</span>`;
  }};
  document.getElementById('panel-content').innerHTML = `
    <div class="pf"><div class="pf-label">Sentence</div>
      <div class="pf-val sentence">${{node.sentence}}</div></div>
    <div class="pf"><div class="pf-label">Metadata</div>
      <div class="pf-val">ID #${{node.id}} · Para ${{node.para||'?'}} · <span style="display:inline-flex;align-items:center;gap:4px"><span style="width:9px;height:9px;border-radius:50%;background:${{col}};display:inline-block"></span>${{node.cid===-1?'Noise':'Cluster '+node.cid}}</span></div></div>
    <div class="pf"><div class="pf-label">Graph metrics</div>
      <div class="pf-val">Degree ${{node.degree}} · threshold ${{(node.thresh||0).toFixed(4)}}${{node.totalChunks>1?' · chunk '+node.chunk+'/'+node.totalChunks:''}}</div></div>
    <div class="pf"><div class="pf-label">Neighbors (${{node.neighbors.length}})</div>
      <div class="nb-list">${{node.neighbors.map(nb => {{
        const nn = nodeMap[nb.sentence_id];
        const txt = nn ? nn.sentence : '#'+nb.sentence_id;
        return `<div class="nb-item" data-id="${{nb.sentence_id}}">
          ${{simBadge(nb.similarity)}}
          ${{nb.cross_cluster?'<span class="nb-badge cross">cross</span>':''}}
          <span class="nb-text" title="${{txt}}">${{txt.length>60?txt.slice(0,60)+'…':txt}}</span>
        </div>`;
      }}).join('')}}</div></div>
  `;
  document.querySelectorAll('.nb-item').forEach(el => {{
    el.addEventListener('click', () => {{
      const n = nodeMap[parseInt(el.dataset.id)];
      if (n) {{ openPanel(n); draw(); }};
    }});
  }});
  draw();
}}

// ============================================================
//  STATS PANEL
// ============================================================
function openStatsPanel() {{
  statsPanel.classList.remove('closed');
  panel.classList.add('closed');
  const el = document.getElementById('stats-content');
  if (!CLUSTER_DATA.length) {{
    el.innerHTML = '<p style="font-size:12px;color:var(--text2)">No cluster_summary.json was embedded. Run with --cluster to include it.</p>';
    return;
  }}
  el.innerHTML = CLUSTER_DATA.map(c => {{
    const col = clusterColor(c.cluster_id);
    const topNodes = (c.top_nodes||[]).slice(0,3);
    return `<div class="cluster-card">
      <div class="cluster-card-header">
        <span style="width:10px;height:10px;border-radius:50%;background:${{col}};display:inline-block;flex-shrink:0"></span>
        <h3>${{c.is_noise?'Noise cluster':'Cluster '+c.cluster_id}}</h3>
        <span style="margin-left:auto;font-size:10px;color:var(--text2)">${{c.size}} nodes</span>
      </div>
      <p>${{c.centroid_sentence||'(no centroid)'}}</p>
      ${{topNodes.length?`<div class="meta">Top nodes: ${{topNodes.map(n=>n.sentence.slice(0,40)+'…').join(' · ')}}</div>`:''}}
    </div>`;
  }}).join('');
}}

// ============================================================
//  PATH FINDING (BFS)
// ============================================================
function findShortestPath(fromId, toId) {{
  if (fromId === toId) return [fromId];
  const visited = new Set([fromId]);
  const queue = [[fromId]];
  while (queue.length) {{
    const path = queue.shift();
    const cur  = path[path.length-1];
    const node = nodeMap[cur];
    if (!node) continue;
    for (const nb of node.neighbors) {{
      if (nb.sentence_id === toId) return [...path, toId];
      if (!visited.has(nb.sentence_id)) {{
        visited.add(nb.sentence_id);
        queue.push([...path, nb.sentence_id]);
      }}
    }}
  }}
  return null;
}}

function highlightPath(path) {{
  if (!path) {{ alert('No path found between these nodes.'); return; }}
  // Temporarily select all in path
  const pathSet = new Set(path);
  // Draw path edges in bright color
  const origDraw = draw;
  // Flash highlight via alpha
  fgCtx.save();
  for (let i=0;i<path.length-1;i++) {{
    const a=nodeMap[path[i]], b=nodeMap[path[i+1]];
    if (!a||!b) continue;
    const pa=worldToScreen(a.x,a.y), pb=worldToScreen(b.x,b.y);
    fgCtx.beginPath();
    fgCtx.moveTo(pa.x,pa.y); fgCtx.lineTo(pb.x,pb.y);
    fgCtx.strokeStyle='#1d9e75'; fgCtx.lineWidth=3; fgCtx.stroke();
  }}
  fgCtx.restore();
  alert(`Path length: ${{path.length-1}} hops (${{path.join(' → ')}})`);
  findPathMode=false; pathFrom=null;
}}

// ============================================================
//  MOUSE / TOUCH EVENTS
// ============================================================
function nodeAtScreen(mx, my) {{
  let found = null;
  nodes.forEach(n => {{
    const p = worldToScreen(n.x, n.y);
    const r = nodeR(n)*Math.max(0.6,Math.min(1.4,camZ))+5;
    if (!found && Math.hypot(mx-p.x, my-p.y) < r) found = n;
  }});
  return found;
}}

fgC.addEventListener('mousemove', e => {{
  const rect = fgC.getBoundingClientRect();
  const mx = e.clientX-rect.left, my = e.clientY-rect.top;
  if (panning) {{
    camX = camStart.x + (e.clientX - panStart.x);
    camY = camStart.y + (e.clientY - panStart.y);
    updateZoomLabel(); draw(); return;
  }}
  if (draggingNode) {{
    const w = screenToWorld(mx, my);
    draggingNode.x = w.x; draggingNode.y = w.y;
    draggingNode.vx=0; draggingNode.vy=0;
    dragMoved=true; draw(); return;
  }}
  const n = nodeAtScreen(mx, my);
  hoveredId = n ? n.id : null;
  fgC.style.cursor = n ? 'pointer' : (findPathMode ? 'crosshair' : 'grab');
  // Tooltip
  if (n) {{
    tooltip.style.display='block';
    const MAX=100;
    const txt = n.sentence.length>MAX ? n.sentence.slice(0,MAX)+'…' : n.sentence;
    tooltip.innerHTML = `<b>Node #${{n.id}} · C${{n.cid}} · deg ${{n.degree}}</b>${{txt}}`;
    tooltip.style.left = Math.min(mx+12, fgC.width-270)+'px';
    tooltip.style.top  = Math.max(my-40, 4)+'px';
  }} else {{
    tooltip.style.display='none';
  }}
  draw();
}});

fgC.addEventListener('mousedown', e => {{
  const rect = fgC.getBoundingClientRect();
  const mx = e.clientX-rect.left, my = e.clientY-rect.top;
  const n = nodeAtScreen(mx, my);
  if (n) {{
    draggingNode=n; dragMoved=false;
    stopForce();
  }} else {{
    panning=true;
    panStart={{x:e.clientX,y:e.clientY}};
    camStart={{x:camX,y:camY}};
    wrap.classList.add('panning');
  }}
}});

window.addEventListener('mouseup', e => {{
  if (draggingNode) {{
    if (!dragMoved) {{
      if (findPathMode && pathFrom !== null) {{
        const path = findShortestPath(pathFrom, draggingNode.id);
        highlightPath(path);
      }} else {{
        openPanel(draggingNode);
      }}
    }}
    draggingNode=null;
  }}
  if (panning) {{ panning=false; wrap.classList.remove('panning'); }}
}});

fgC.addEventListener('click', e => {{
  if (panning||dragMoved) return;
  const rect = fgC.getBoundingClientRect();
  const n = nodeAtScreen(e.clientX-rect.left, e.clientY-rect.top);
  if (!n) {{
    selectedId=null;
    panel.classList.add('closed');
    if (findPathMode) {{ findPathMode=false; pathFrom=null; document.getElementById('btn-path').classList.remove('active'); }}
    draw();
  }}
}});

fgC.addEventListener('wheel', e => {{
  e.preventDefault();
  const rect = fgC.getBoundingClientRect();
  const mx = e.clientX-rect.left, my = e.clientY-rect.top;
  const factor = e.deltaY < 0 ? 1.1 : 0.91;
  const wx=(mx-fgC.width/2-camX)/camZ, wy=(my-fgC.height/2-camY)/camZ;
  camZ = Math.max(0.05, Math.min(10, camZ*factor));
  camX = mx-fgC.width/2-wx*camZ;
  camY = my-fgC.height/2-wy*camZ;
  updateZoomLabel(); draw();
}}, {{passive:false}});

// Touch
let lastTouch=null, lastDist=null;
fgC.addEventListener('touchstart', e=>{{
  e.preventDefault();
  if (e.touches.length===1) {{
    const t=e.touches[0];
    lastTouch={{x:t.clientX,y:t.clientY}};
    camStart={{x:camX,y:camY}};
  }} else if (e.touches.length===2) {{
    const dx=e.touches[0].clientX-e.touches[1].clientX;
    const dy=e.touches[0].clientY-e.touches[1].clientY;
    lastDist=Math.sqrt(dx*dx+dy*dy);
  }}
}},{{passive:false}});
fgC.addEventListener('touchmove', e=>{{
  e.preventDefault();
  if (e.touches.length===1 && lastTouch) {{
    const t=e.touches[0];
    camX=camStart.x+(t.clientX-lastTouch.x);
    camY=camStart.y+(t.clientY-lastTouch.y);
    draw();
  }} else if (e.touches.length===2 && lastDist) {{
    const dx=e.touches[0].clientX-e.touches[1].clientX;
    const dy=e.touches[0].clientY-e.touches[1].clientY;
    const d=Math.sqrt(dx*dx+dy*dy);
    camZ=Math.max(0.05,Math.min(10,camZ*(d/lastDist)));
    lastDist=d; draw();
  }}
}},{{passive:false}});

// ============================================================
//  KEYBOARD
// ============================================================
document.addEventListener('keydown', e => {{
  if (e.target.tagName==='INPUT') return;
  if (e.key==='f'||e.key==='F') {{ zoomToFit(); }}
  if (e.key==='Escape') {{ selectedId=null; panel.classList.add('closed'); statsPanel.classList.add('closed'); findPathMode=false; pathFrom=null; draw(); }}
}});

// ============================================================
//  TOOLBAR EVENTS
// ============================================================
document.getElementById('sld-sim').addEventListener('input', function() {{
  simFloor = this.value/100;
  document.getElementById('sim-val').textContent = simFloor.toFixed(2);
  draw();
}});
document.getElementById('sel-layout').addEventListener('change', function() {{
  layoutMode = this.value; applyLayout();
}});
document.getElementById('sel-size').addEventListener('change', function() {{
  sizeMode = this.value; draw();
}});
document.getElementById('sel-ecol').addEventListener('change', function() {{
  ecolMode = this.value; draw();
}});
document.getElementById('search').addEventListener('input', function() {{
  searchTerm = this.value.trim(); draw();
}});
document.getElementById('btn-cross').addEventListener('click', function() {{
  crossOnly = !crossOnly;
  this.classList.toggle('active', crossOnly);
  draw();
}});
document.getElementById('btn-iso').addEventListener('click', function() {{
  hideIso = !hideIso;
  this.classList.toggle('active', hideIso);
  draw();
}});
document.getElementById('btn-fit').addEventListener('click', () => zoomToFit());
document.getElementById('btn-stats').addEventListener('click', () => openStatsPanel());
document.getElementById('btn-close-panel').addEventListener('click', () => {{
  panel.classList.add('closed'); selectedId=null; draw();
}});
document.getElementById('btn-path').addEventListener('click', function() {{
  if (!selectedId) {{ alert('Select a node first, then click "Find path to…"'); return; }}
  findPathMode = true;
  pathFrom = selectedId;
  this.classList.add('active');
  this.textContent = 'Click target node';
  alert('Now click any other node to find the shortest path from node #'+selectedId);
}});

// Export PNG
document.getElementById('btn-png').addEventListener('click', () => {{
  const merged = document.createElement('canvas');
  merged.width = fgC.width; merged.height = fgC.height;
  const mctx = merged.getContext('2d');
  mctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg1') || '#f7f6f3';
  mctx.fillRect(0,0,merged.width,merged.height);
  mctx.drawImage(bgC,0,0);
  mctx.drawImage(fgC,0,0);
  const a = document.createElement('a');
  a.download = 'idea_graph.png';
  a.href = merged.toDataURL('image/png');
  a.click();
}});

// File open (toolbar)
document.getElementById('btn-open').addEventListener('click', () => document.getElementById('file-input2').click());
document.getElementById('file-input2').addEventListener('change', function() {{
  if (this.files[0]) loadFile(this.files[0]);
}});

// Drop zone (welcome screen)
const dz = document.getElementById('drop-zone');
dz.addEventListener('click', () => document.getElementById('file-input').click());
dz.addEventListener('keydown', e => {{ if (e.key==='Enter'||e.key===' ') document.getElementById('file-input').click(); }});
dz.addEventListener('dragover', e=>{{ e.preventDefault(); dz.classList.add('drag'); }});
dz.addEventListener('dragleave', ()=>dz.classList.remove('drag'));
dz.addEventListener('drop', e=>{{ e.preventDefault(); dz.classList.remove('drag'); if(e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); }});
document.getElementById('file-input').addEventListener('change', function() {{
  if (this.files[0]) loadFile(this.files[0]);
}});

// Full-window drag-and-drop
window.addEventListener('dragover', e=>e.preventDefault());
window.addEventListener('drop', e=>{{ e.preventDefault(); const f=e.dataTransfer.files[0]; if(f&&f.name.endsWith('.json')) loadFile(f); }});

function loadFile(file) {{
  const reader = new FileReader();
  reader.onload = evt => {{
    try {{
      const data = JSON.parse(evt.target.result);
      if (!Array.isArray(data)) {{ alert('Expected a JSON array (idea_graph.json).'); return; }}
      initGraph(data);
    }} catch(err) {{ alert('Failed to parse JSON: '+err.message); }}
  }};
  reader.readAsText(file);
}}

// ============================================================
//  BOOT
// ============================================================
resize();
if (GRAPH_DATA && GRAPH_DATA.length) {{
  initGraph(GRAPH_DATA);
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize idea_graph.json as an interactive HTML explorer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--graph",   default=DEFAULT_GRAPH_PATH,
                        help=f"Path to idea_graph.json (default: {DEFAULT_GRAPH_PATH})")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER_PATH,
                        help=f"Path to cluster_summary.json (optional, default: {DEFAULT_CLUSTER_PATH})")
    parser.add_argument("--manifest",default=DEFAULT_MANIFEST_PATH,
                        help=f"Path to embeddings_manifest.json (optional)")
    parser.add_argument("--out",     default=DEFAULT_OUT_PATH,
                        help=f"Output HTML path (default: {DEFAULT_OUT_PATH})")
    parser.add_argument("--serve",   action="store_true",
                        help="Start a local HTTP server and open the browser after export")
    parser.add_argument("--port",    type=int, default=DEFAULT_PORT,
                        help=f"Port for --serve mode (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    # Load data
    graph    = _load_json(args.graph,   "idea_graph.json",          required=True)
    cluster  = _load_json(args.cluster, "cluster_summary.json",     required=False)
    manifest = _load_json(args.manifest,"embeddings_manifest.json", required=False)

    validate_graph(graph)

    # Build HTML
    print("[visualize] Building HTML…")
    html = build_html(graph, cluster, manifest, args.graph)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size // 1024
    print(f"[visualize] Saved → {out_path}  ({size_kb} KB)")
    print(f"[visualize] Open in any browser:  file://{out_path.resolve()}")

    if args.serve:
        serve_dir = out_path.parent
        os.chdir(serve_dir)
        url = f"http://localhost:{args.port}/{out_path.name}"
        print(f"[visualize] Serving at {url}  (Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            HTTPServer(("", args.port), SimpleHTTPRequestHandler).serve_forever()
        except KeyboardInterrupt:
            print("\n[visualize] Server stopped.")


if __name__ == "__main__":
    main()