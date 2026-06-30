"""
app.py — Streamlit chat interface for the Knowledge Agent.
Run with: streamlit run app.py
"""
import sys
import os
sys.path.insert(0, ".")

# Load .env FIRST before any other imports
from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st

st.set_page_config(
    page_title="AI Knowledge Agent",
    page_icon="🤖",
    layout="wide",
)

from src.agent.agent import get_agent
from src.knowledge.ingestion import ingest_file, ingest_url
from src.knowledge.vectorstore import count_documents

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AI Knowledge Agent")
    st.caption("Autonomous research with RAG + web search")

    st.divider()

    doc_count = count_documents()
    st.metric("Chunks in knowledge base", doc_count)

    st.subheader("Add knowledge")

    url_input = st.text_input("Ingest a URL", placeholder="https://example.com/article")
    if st.button("Add URL", use_container_width=True):
        if url_input:
            with st.spinner("Loading and indexing…"):
                n = ingest_url(url_input)
            st.success(f"Added {n} chunks from URL")
            st.rerun()
        else:
            st.warning("Enter a URL first")

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md"],
        help="PDF, TXT, or Markdown files",
    )
    if uploaded and st.button("Ingest file", use_container_width=True):
        tmp_path = f"C:/Temp/{uploaded.name}"
        os.makedirs("C:/Temp", exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner("Indexing…"):
            n = ingest_file(tmp_path)
        st.success(f"Added {n} chunks from {uploaded.name}")
        st.rerun()

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        get_agent().reset_memory()
        st.rerun()

    st.divider()
    st.caption("Stack: Gemini · LangGraph · ChromaDB · DuckDuckGo · RAGAS")

# ── Main chat area ────────────────────────────────────────────────────────────
st.header("Chat with your knowledge base")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander(f"Agent used {len(msg['steps'])} tool(s)", expanded=False):
                for i, step in enumerate(msg["steps"], 1):
                    st.markdown(f"**Step {i} — {step['action']}**")
                    st.code(str(step["input"]), language="text")
                    st.caption("Observation:")
                    st.text(str(step["observation"])[:800])

if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                agent = get_agent()
                response = agent.chat(prompt)
                answer = response.answer
                steps = response.steps
            except Exception as e:
                answer = f"Error: {e}"
                steps = []

        st.markdown(answer)

        if steps:
            with st.expander(f"Agent used {len(steps)} tool(s)", expanded=False):
                for i, step in enumerate(steps, 1):
                    st.markdown(f"**Step {i} — {step['action']}**")
                    st.code(str(step["input"]), language="text")
                    st.caption("Observation:")
                    st.text(str(step["observation"])[:800])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "steps": steps,
    })