import json
import time
import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Statis", page_icon="◈", layout="centered")

# ---------------------------------------------------------------------------
# Visual identity: IBM Plex Sans for UI copy, IBM Plex Mono for data/scores/
# model names. Deep indigo as the single accent; muted teal reserved for
# "expanded" provenance tags so color carries meaning, not decoration.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .stTabs [data-baseweb="tab"] { font-weight: 500; }

    code, .mono { font-family: 'IBM Plex Mono', monospace !important; }

    .status-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .status-on { background: #E4EBE3; color: #2F5233; }
    .status-off { background: #F2DEDC; color: #8C2F26; }

    .origin-tag {
        display: inline-block;
        padding: 1px 9px;
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        margin-bottom: 6px;
    }
    .origin-seed     { background: #DDE3EF; color: #2B3A67; }
    .origin-expanded { background: #DCEAE7; color: #235349; }
    .origin-ppr      { background: #EFEEE9; color: #55524A; }

    .brand-rule {
        border: none;
        border-top: 1px solid #DEDCD3;
        margin: 0.6rem 0 0.9rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_backend():
    try:
        r = requests.get(f"{API_BASE}/healthz", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


if "query_log" not in st.session_state:
    st.session_state["query_log"] = []

with st.sidebar:
    st.markdown("**Stasis**")
    online = check_backend()
    pill_class = "status-on" if online else "status-off"
    pill_text = "backend online" if online else "backend offline"
    st.markdown(f'<span class="status-pill {pill_class}">{pill_text}</span>', unsafe_allow_html=True)

    st.markdown('<hr class="brand-rule">', unsafe_allow_html=True)
    st.caption("Local inference via Ollama · no external API calls")

    if st.session_state["query_log"]:
        last = st.session_state["query_log"][-1]
        st.caption(f"Last run — `{last['model']}` · {last['latency_ms']:.0f} ms · $0.00")

    if "task_id" in st.session_state:
        st.markdown('<hr class="brand-rule">', unsafe_allow_html=True)
        st.markdown("**Current document**")
        st.code(st.session_state["task_id"][:8] + "...", language=None)
        if st.button("Clear & upload new"):
            st.session_state.clear()
            st.rerun()

    if st.session_state["query_log"]:
        st.markdown('<hr class="brand-rule">', unsafe_allow_html=True)
        st.download_button(
            "Export session log (JSON)",
            data=json.dumps(st.session_state["query_log"], indent=2),
            file_name="session_log.json",
            mime="application/json",
        )

st.title("Statis")
st.caption("Upload a PDF to extract its idea graph and query it directly.")

tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Query", "Summary", "🕸 Graph Explorer"])

with tab1:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        st.success(f"{uploaded_file.name} — {uploaded_file.size / 1024:.1f} KB", icon=None)

        if st.button("Run pipeline"):
            try:
                r = requests.post(
                    f"{API_BASE}/api/v1/ingest",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                )

                if r.status_code == 202:
                    task_id = r.json()["task_id"]
                    st.session_state["task_id"] = task_id
                    st.session_state["pipeline_done"] = False
                    st.session_state["polling"] = True
                    st.session_state["show_done"] = False
                else:
                    st.error(f"Upload failed: {r.text}", icon=None)

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend. Make sure it's running on port 8000.", icon=None)

    if st.session_state.get("show_done"):
        st.success("Pipeline complete — switch to the Query tab.", icon=None)
        st.session_state["show_done"] = False

    if st.session_state.get("polling") and not st.session_state.get("pipeline_done"):
        stage_progress = {
            "queued": 5,
            "pdf_to_txt": 25,
            "phase1_data_prep": 50,
            "phase2_embeddings": 75,
            "phase3_idea_graph": 90,
            "done": 100,
        }
        stage_labels = {
            "queued": "Queued",
            "pdf_to_txt": "Extracting text from PDF",
            "phase1_data_prep": "Splitting sentences",
            "phase2_embeddings": "Generating embeddings — this takes a while",
            "phase3_idea_graph": "Building idea graph",
            "done": "Complete",
        }

        try:
            status_r = requests.get(f"{API_BASE}/api/v1/status/{st.session_state['task_id']}")
            data = status_r.json()
            stage = data["stage"]
            status = data["status"]

            pct = stage_progress.get(stage, 0)
            label = stage_labels.get(stage, stage)

            st.progress(pct, text=label)
            st.caption(f"Status: `{status}` · Stage: `{stage}`")

            if status == "completed":
                st.session_state["pipeline_done"] = True
                st.session_state["polling"] = False
                st.session_state["show_done"] = True
                st.rerun()
            elif status == "failed":
                st.session_state["polling"] = False
                st.error(f"Pipeline failed at: {stage} — {data.get('error', '')}", icon=None)
            else:
                time.sleep(2)
                st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("Lost connection to backend.", icon=None)
            st.session_state["polling"] = False

with tab2:
    st.header("Ask a question")

    if "task_id" not in st.session_state:
        st.warning("Upload and process a PDF first.", icon=None)
    else:
        with st.form("query_form"):
            question = st.text_input("Your question", placeholder="What is this document about?")
            submitted = st.form_submit_button("Search")

        if submitted:
            if question.strip() == "":
                st.warning("Enter a question.", icon=None)
            else:
                try:
                    with st.spinner("Running retrieval and generating an answer..."):
                        r = requests.post(
                            f"{API_BASE}/api/v1/query",
                            json={
                                "task_id": st.session_state["task_id"],
                                "question": question
                            }
                        )

                    if r.status_code == 200:
                        data = r.json()

                        st.session_state["query_log"].append({
                            "question": question,
                            "answer": data["answer"],
                            "model": data["model"],
                            "latency_ms": data["latency_ms"],
                            "ppr_fallback_used": data["ppr_fallback_used"],
                            "num_sources": len(data["sources"]),
                        })

                        st.subheader("Answer")
                        st.write(data["answer"])

                        cols = st.columns(3)
                        cols[0].metric("Model", data["model"])
                        cols[1].metric("Latency", f"{data['latency_ms']:.0f} ms")
                        cols[2].metric("Cost", "$0.00")

                        if data.get("ppr_fallback_used"):
                            st.info(
                                "PPR fallback used — the personalized PageRank walk didn't "
                                "converge or seed normally for this query, so results came "
                                "from a fallback ranking instead.",
                                icon=None,
                            )

                        st.markdown('<hr class="brand-rule">', unsafe_allow_html=True)
                        st.subheader("Sources")
                        for i, src in enumerate(data["sources"]):
                            if src.get("is_seed"):
                                origin_class, origin_text = "origin-seed", "seed"
                            elif src.get("expansion_source") is not None:
                                origin_class, origin_text = "origin-expanded", f"expanded from #{src['expansion_source']}"
                            else:
                                origin_class, origin_text = "origin-ppr", "ppr"

                            with st.expander(f"Source {i+1} — score {src['score']:.3f}"):
                                st.markdown(
                                    f'<span class="origin-tag {origin_class}">{origin_text}</span>',
                                    unsafe_allow_html=True,
                                )
                                st.write(src["sentence"])
                                col1, col2 = st.columns(2)
                                col1.metric("Paragraph", src.get("paragraph_id", "?"))
                                col2.metric("Cluster", src.get("cluster_id", "?"))
                    else:
                        st.error(f"Query failed: {r.text}", icon=None)

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend.", icon=None)

with tab3:
    st.header("Document summary")

    if "task_id" not in st.session_state:
        st.warning("Upload and process a PDF first.", icon=None)
    else:
        if st.button("Generate summary"):
            try:
                with st.spinner("Computing structured summary..."):
                    r = requests.get(f"{API_BASE}/api/v1/summary/{st.session_state['task_id']}")

                if r.status_code == 200:
                    data = r.json()
                    for entry in data["summary"]:
                        if entry.get("is_noise"):
                            continue
                        with st.expander(f"Cluster {entry['cluster_id']} — {entry['size']} sentences"):
                            st.markdown(f"**Topic:** {entry['centroid_sentence']}")
                            st.markdown('<hr class="brand-rule">', unsafe_allow_html=True)
                            st.write(entry["paragraph"])
                            if entry.get("bridges"):
                                st.markdown("**Related ideas:**")
                                for br in entry["bridges"]:
                                    st.markdown(f"> {br['sentence']}")
                else:
                    st.error(f"Summary failed: {r.text}", icon=None)

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend.", icon=None)

with tab4:
    st.header("Graph Explorer")
    st.caption("Interactive view of the idea graph — nodes are clauses, edges are similarity/sequential links, colored by cluster.")

    if "task_id" not in st.session_state:
        st.warning("Upload and process a PDF first.", icon=None)
    else:
        if st.button("Render graph explorer"):
            try:
                with st.spinner("Building the interactive graph..."):
                    r = requests.get(f"{API_BASE}/api/v1/graph/{st.session_state['task_id']}")

                if r.status_code == 200:
                    st.session_state["graph_html"] = r.text
                else:
                    st.error(f"Graph explorer failed: {r.text}", icon=None)

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend.", icon=None)

        if st.session_state.get("graph_html"):
            st.download_button(
                "⬇ Download as standalone HTML",
                data=st.session_state["graph_html"],
                file_name="idea_graph_explorer.html",
                mime="text/html",
            )
            st.markdown('<hr class="brand-rule">', unsafe_allow_html=True)
            st.components.v1.html(st.session_state["graph_html"], height=800, scrolling=True)