import time
import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Thought to Structure", page_icon="🧠")

def check_backend():
    try:
        r = requests.get(f"{API_BASE}/healthz", timeout=3)
        return r.status_code == 200
    except:
        return False

with st.sidebar:
    st.title("🧠 Thought to Structure")
    if check_backend():
        st.success("🟢 Backend Online")
    else:
        st.error("🔴 Backend Offline")
    
    if "task_id" in st.session_state:
        st.markdown("---")
        st.markdown("**Current Document**")
        st.code(st.session_state["task_id"][:8] + "...", language=None)
        if st.button("🗑 Clear & Upload New"):
            st.session_state.clear()
            st.rerun()

st.title("🧠 Thought to Structure")
st.write("Upload a PDF to begin")

tab1, tab2, tab3 = st.tabs(["📄 Upload", "🔍 Query", "📋 Summary"])

with tab1:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        st.success(f"✅ {uploaded_file.name} — {uploaded_file.size / 1024:.1f} KB")

        if st.button("▶ Run Pipeline"):
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
                    st.error(f"Upload failed: {r.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach backend. Make sure it's running on port 8000.")

    if st.session_state.get("show_done"):
        st.success("🎉 Pipeline complete! Switch to the Query tab.")
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
            "queued": "⏳ Queued...",
            "pdf_to_txt": "📄 Extracting text from PDF...",
            "phase1_data_prep": "✂️ Splitting sentences...",
            "phase2_embeddings": "🧠 Generating embeddings (this takes a while)...",
            "phase3_idea_graph": "🕸 Building idea graph...",
            "done": "✅ Complete!",
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
                st.error(f"❌ Pipeline failed at: {stage} — {data.get('error', '')}")
            else:
                time.sleep(2)
                st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("❌ Lost connection to backend.")
            st.session_state["polling"] = False

with tab2:
    st.header("Ask a Question")

    if "task_id" not in st.session_state:
        st.warning("⚠️ Upload and process a PDF first.")
    else:
        with st.form("query_form"):
            question = st.text_input("Your question", placeholder="What is this document about?")
            submitted = st.form_submit_button("🔍 Search")

        if submitted:
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                try:
                    query_progress = st.progress(0, text="🔍 Embedding your question...")
                    time.sleep(0.4)
                    query_progress.progress(30, text="🕸 Traversing idea graph...")
                    time.sleep(0.4)
                    query_progress.progress(60, text="🤖 Generating answer with LLM...")

                    r = requests.post(
                        f"{API_BASE}/api/v1/query",
                        json={
                            "task_id": st.session_state["task_id"],
                            "question": question
                        }
                    )

                    query_progress.progress(100, text="✅ Done!")
                    time.sleep(0.3)
                    query_progress.empty()

                    if r.status_code == 200:
                        data = r.json()

                        st.subheader("💡 Answer")
                        st.write(data["answer"])

                        st.markdown("---")
                        st.subheader("📚 Sources")
                        for i, src in enumerate(data["sources"]):
                            with st.expander(f"Source {i+1} — score: {src['score']:.3f} {'🌱 seed' if src['is_seed'] else ''}"):
                                st.write(src["sentence"])
                                col1, col2 = st.columns(2)
                                col1.metric("Paragraph", src.get("paragraph_id", "?"))
                                col2.metric("Cluster", src.get("cluster_id", "?"))
                    else:
                        st.error(f"Query failed: {r.text}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach backend.")

with tab3:
    st.header("Document Summary")

    if "task_id" not in st.session_state:
        st.warning("⚠️ Upload and process a PDF first.")
    else:
        if st.button("📋 Generate Summary"):
            try:
                summary_progress = st.progress(0, text="📊 Loading idea graph...")
                time.sleep(0.4)
                summary_progress.progress(40, text="🧮 Computing PageRank...")
                time.sleep(0.4)
                summary_progress.progress(75, text="✍️ Assembling paragraphs...")

                r = requests.get(f"{API_BASE}/api/v1/summary/{st.session_state['task_id']}")

                summary_progress.progress(100, text="✅ Summary ready!")
                time.sleep(0.3)
                summary_progress.empty()

                if r.status_code == 200:
                    data = r.json()
                    for entry in data["summary"]:
                        if entry.get("is_noise"):
                            continue
                        with st.expander(f"Cluster {entry['cluster_id']} — {entry['size']} sentences"):
                            st.markdown(f"**Topic:** {entry['centroid_sentence']}")
                            st.markdown("---")
                            st.write(entry["paragraph"])
                            if entry.get("bridges"):
                                st.markdown("**🌉 Bridging ideas:**")
                                for br in entry["bridges"]:
                                    st.markdown(f"> {br['sentence']}")
                else:
                    st.error(f"Summary failed: {r.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach backend.")