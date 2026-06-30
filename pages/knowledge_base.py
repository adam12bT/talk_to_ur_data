"""
pages/knowledge_base.py
Knowledge Base Browser — see, search, and manage everything in ChromaDB.
Run via: streamlit run app.py  (appears in sidebar as "knowledge_base")
"""
import sys
import os
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import json
import traceback

st.set_page_config(page_title="Knowledge Base Browser", page_icon="🗄️", layout="wide")


st.title("🗄️ Knowledge Base Browser")
st.caption("Inspect, search, and manage everything stored in ChromaDB")

try:
    from src.knowledge.vectorstore import (
        get_vectorstore,
        count_documents,
        reset_vectorstore,
        add_documents,
        similarity_search,
        get_all_chunks,
    )
    from src.knowledge.ingestion import ingest_file, ingest_url
    from src.knowledge.drift_tracker import record_snapshot
    from src.config import VECTOR_BACKEND
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()


# ── Top metrics ───────────────────────────────────────────────────────────────
total = count_documents()
col1, col2, col3 = st.columns(3)
col1.metric("Total chunks", total)
col2.metric("Backend", "Qdrant Cloud" if VECTOR_BACKEND == "qdrant" else "ChromaDB (local)")
col3.metric("Collection", "knowledge_base")

if total == 0:
    st.warning("The knowledge base is empty. Add documents below or from the main chat page.")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Browse all chunks",
    "🔍 Semantic search",
    "➕ Add documents",
    "🗑️ Manage",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BROWSE ALL
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("All stored chunks")
    st.caption("Every chunk currently in the knowledge base, grouped by source.")

    if total == 0:
        st.info("Nothing stored yet.")
    else:
        try:
            chunks_raw = get_all_chunks()

            # Group by source
            by_source: dict[str, list[dict]] = {}
            for chunk in chunks_raw:
                source = chunk["metadata"].get("source", "unknown")
                by_source.setdefault(source, []).append({
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "meta": chunk["metadata"],
                })

            st.markdown(f"**{len(by_source)} unique source(s), {total} total chunks**")
            st.divider()

            for source, chunks in sorted(by_source.items()):
                with st.expander(
                    f"📄 {source}  —  {len(chunks)} chunk(s)",
                    expanded=False,
                ):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Chunk {i}** `{chunk['id']}`")
                        st.text(chunk["content"][:600] + ("…" if len(chunk["content"]) > 600 else ""))
                        if chunk["meta"]:
                            with st.expander("Metadata", expanded=False):
                                st.json(chunk["meta"])
                        st.divider()

        except Exception as e:
            st.error(f"Could not read collection: {e}")
            st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEMANTIC SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Semantic search")
    st.caption("Find chunks by meaning, not just keywords.")

    query = st.text_input("Search query:", placeholder="e.g. how does RAG work?")
    k = st.slider("Number of results", 1, 20, 5)

    if st.button("Search", type="primary") and query:
        if total == 0:
            st.warning("Knowledge base is empty.")
        else:
            try:
                store = get_vectorstore()
                results = store.similarity_search_with_score(query, k=k)

                st.divider()
                st.markdown(f"### {len(results)} result(s) for: *\"{query}\"*")

                for i, (doc, distance) in enumerate(results, 1):
                    similarity = max(0.0, 1.0 - float(distance))
                    badge = "🟢" if similarity > 0.7 else "🟡" if similarity > 0.4 else "🔴"
                    source = doc.metadata.get("source", "unknown")

                    with st.expander(
                        f"{badge} #{i} — score {similarity:.3f} — {source}",
                        expanded=i <= 3,
                    ):
                        st.text(doc.page_content)
                        col1, col2 = st.columns(2)
                        col1.metric("Similarity", f"{similarity:.4f}")
                        col2.metric("Source", source)
                        st.json(doc.metadata)

                # Score bar chart
                scores = {f"#{i+1}": round(max(0.0, 1.0 - s), 4)
                          for i, (_, s) in enumerate(results)}
                st.markdown("### Score distribution")
                st.bar_chart(scores)

            except Exception as e:
                st.error(f"Search failed: {e}")
                st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ADD DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Add documents")

    st.markdown("#### From a URL")
    url = st.text_input("URL:", placeholder="https://example.com/article")
    if st.button("Ingest URL", type="primary") and url:
        with st.spinner("Loading and indexing…"):
            try:
                n = ingest_url(url)
                record_snapshot(label=f"Added URL: {url[:60]}")
                st.success(f"Added {n} chunks. Drift snapshot recorded.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    st.markdown("#### From a file")
    uploaded = st.file_uploader("Upload PDF, TXT, or MD:", type=["pdf", "txt", "md"])
    if uploaded and st.button("Ingest file", type="primary"):
        import tempfile
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        with st.spinner("Indexing…"):
            try:
                n = ingest_file(tmp_path)
                record_snapshot(label=f"Added file: {uploaded.name}")
                st.success(f"Added {n} chunks from {uploaded.name}. Drift snapshot recorded.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MANAGE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Manage the knowledge base")

    st.markdown("#### Export all chunks")
    if st.button("Export as JSON") and total > 0:
        try:
            export = get_all_chunks()
            st.download_button(
                "Download knowledge_base.json",
                data=json.dumps(export, indent=2),
                file_name="knowledge_base.json",
                mime="application/json",
            )
        except Exception as e:
            st.error(f"Export failed: {e}")

    st.divider()
    st.markdown("#### Danger zone")
    st.warning("Resetting will permanently delete all chunks from the knowledge base.")
    confirm = st.checkbox("I understand this cannot be undone")
    if confirm and st.button("Reset knowledge base", type="secondary"):
        reset_vectorstore()
        st.success("Knowledge base cleared.")
        st.rerun()