"""
pages/settings.py
Settings UI — edit all tuneable parameters from the browser.
Changes are written to .env so they persist across restarts.
"""
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv, set_key, dotenv_values
load_dotenv(override=True)

import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")


st.title("⚙️ Settings")
st.caption("Tune all agent and pipeline parameters. Changes are saved to your .env file.")

ENV_PATH = Path(".env")

# ── Load current values ───────────────────────────────────────────────────────
try:
    import src.config as cfg
except Exception as e:
    st.error(f"Could not load config: {e}")
    st.stop()

# ── Helper ────────────────────────────────────────────────────────────────────
def save_env(key: str, value: str):
    if not ENV_PATH.exists():
        ENV_PATH.write_text("")
    set_key(str(ENV_PATH), key, value)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 LLM",
    "📄 Chunking",
    "🔍 Retrieval",
    "🔑 API Keys",
    "🗄️ Vector Backend",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LLM
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("LLM settings")

    llm_model = st.selectbox(
        "Model",
        options=[
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-2.5-flash",
        ],
        index=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash"].index(cfg.LLM_MODEL)
        if cfg.LLM_MODEL in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash"]
        else 0,
        help="gemini-1.5-flash is free at 15 req/min. gemini-1.5-pro has higher quality but lower free limits.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg.LLM_TEMPERATURE),
        step=0.05,
        help="0 = deterministic, 1 = creative. Keep at 0 for factual RAG.",
    )

    max_tokens = st.number_input(
        "Max output tokens",
        min_value=256,
        max_value=8192,
        value=int(cfg.MAX_TOKENS),
        step=256,
        help="Maximum tokens in the agent's response.",
    )

    max_iterations = st.slider(
        "Max agent iterations",
        min_value=1,
        max_value=20,
        value=int(cfg.MAX_ITERATIONS),
        help="Max ReAct reasoning steps before the agent is forced to answer.",
    )

    if st.button("Save LLM settings", type="primary"):
        save_env("LLM_MODEL", llm_model)
        save_env("LLM_TEMPERATURE", str(temperature))
        save_env("MAX_TOKENS", str(max_tokens))
        save_env("MAX_ITERATIONS", str(max_iterations))
        st.success("Saved! Restart the app for changes to take effect.")

    st.divider()
    st.markdown("**Current values (from config.py):**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", cfg.LLM_MODEL)
    col2.metric("Temperature", cfg.LLM_TEMPERATURE)
    col3.metric("Max tokens", cfg.MAX_TOKENS)
    col4.metric("Max iterations", cfg.MAX_ITERATIONS)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Document chunking")
    st.caption("Changes apply to newly ingested documents only — existing chunks are not re-chunked.")

    chunk_size = st.slider(
        "Chunk size (chars)",
        min_value=128,
        max_value=2048,
        value=int(cfg.CHUNK_SIZE),
        step=64,
        help="Larger chunks = more context per retrieval. Smaller = more precise retrieval.",
    )

    chunk_overlap = st.slider(
        "Chunk overlap (chars)",
        min_value=0,
        max_value=512,
        value=int(cfg.CHUNK_OVERLAP),
        step=16,
        help="Overlap between consecutive chunks so context isn't cut at boundaries.",
    )

    st.markdown("**Preview:** with these settings, a 5000-char document would produce approximately "
                f"**{max(1, (5000 - chunk_overlap) // max(1, chunk_size - chunk_overlap))} chunks**.")

    if st.button("Save chunking settings", type="primary"):
        save_env("CHUNK_SIZE", str(chunk_size))
        save_env("CHUNK_OVERLAP", str(chunk_overlap))
        st.success("Saved!")

    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Current chunk size", cfg.CHUNK_SIZE)
    col2.metric("Current chunk overlap", cfg.CHUNK_OVERLAP)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Retrieval settings")

    top_k = st.slider(
        "Top K (chunks per query)",
        min_value=1,
        max_value=20,
        value=int(cfg.TOP_K),
        help="How many chunks to retrieve per query. More = more context but slower + noisier.",
    )

    embedding_model = st.selectbox(
        "Embedding model",
        options=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ],
        index=0
        if cfg.EMBEDDING_MODEL not in [
            "all-MiniLM-L6-v2", "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2"
        ]
        else ["all-MiniLM-L6-v2", "all-mpnet-base-v2",
              "paraphrase-multilingual-MiniLM-L12-v2"].index(cfg.EMBEDDING_MODEL),
        help="all-MiniLM-L6-v2 is fast and good. all-mpnet-base-v2 is slower but more accurate.",
    )

    chroma_dir = st.text_input(
        "ChromaDB directory",
        value=cfg.CHROMA_PERSIST_DIR,
        help="Where ChromaDB stores its files on disk.",
    )

    if st.button("Save retrieval settings", type="primary"):
        save_env("TOP_K", str(top_k))
        save_env("EMBEDDING_MODEL", embedding_model)
        save_env("CHROMA_PERSIST_DIR", chroma_dir)
        st.success("Saved! Restart the app for changes to take effect.")
        if embedding_model != cfg.EMBEDDING_MODEL:
            st.warning(
                "You changed the embedding model. "
                "You must re-ingest all documents so they use the new model — "
                "mixing embedding spaces will break retrieval."
            )

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Current TOP_K", cfg.TOP_K)
    col2.metric("Current embedding model", cfg.EMBEDDING_MODEL)
    col3.metric("Current ChromaDB dir", cfg.CHROMA_PERSIST_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — API KEYS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("API keys")

    current_key = os.getenv("GOOGLE_API_KEY", "")
    masked = ("*" * (len(current_key) - 6) + current_key[-6:]) if len(current_key) > 6 else "not set"

    st.markdown(f"**Current Google API key:** `{masked}`")
    st.caption("Get a free key at https://aistudio.google.com/app/apikey — no credit card needed.")

    new_key = st.text_input(
        "New Google API key",
        type="password",
        placeholder="AIza…",
    )

    if st.button("Save API key", type="primary") and new_key:
        save_env("GOOGLE_API_KEY", new_key)
        st.success("API key saved to .env. Restart the app to use the new key.")

    st.divider()
    st.markdown("**Environment status:**")
    col1, col2 = st.columns(2)
    col1.metric("GOOGLE_API_KEY", "✅ Set" if current_key else "❌ Missing")
    col2.metric(".env file", "✅ Exists" if ENV_PATH.exists() else "❌ Missing (copy .env.example)")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — VECTOR BACKEND
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Vector storage backend")
    st.caption(
        "ChromaDB stores embeddings on local disk — fine for development, "
        "but wiped every time a Hugging Face Space restarts. "
        "Switch to Qdrant Cloud for deployments that need persistence."
    )

    current_backend = getattr(cfg, "VECTOR_BACKEND", "chroma")

    backend = st.radio(
        "Backend",
        options=["chroma", "qdrant"],
        index=0 if current_backend == "chroma" else 1,
        format_func=lambda x: "ChromaDB (local disk)" if x == "chroma" else "Qdrant Cloud (persistent, free tier)",
        horizontal=True,
    )

    if backend == "qdrant":
        st.info(
            "Get a free cluster (no credit card) at "
            "[cloud.qdrant.io](https://cloud.qdrant.io) — free tier includes "
            "1GB RAM / 4GB disk, permanently free."
        )

        current_url = getattr(cfg, "QDRANT_URL", "")
        current_key_display = (
            "*" * 20 + getattr(cfg, "QDRANT_API_KEY", "")[-6:]
            if getattr(cfg, "QDRANT_API_KEY", "")
            else "not set"
        )

        qdrant_url = st.text_input(
            "Qdrant cluster URL",
            value=current_url,
            placeholder="https://your-cluster-id.your-region.cloud.qdrant.io:6333",
        )
        st.caption(f"Current API key: `{current_key_display}`")
        qdrant_key = st.text_input("Qdrant API key", type="password", placeholder="leave blank to keep current")
        qdrant_collection = st.text_input(
            "Collection name",
            value=getattr(cfg, "QDRANT_COLLECTION_NAME", "knowledge_base"),
        )

        if st.button("Save Qdrant settings", type="primary"):
            if not qdrant_url:
                st.error("Qdrant cluster URL is required.")
            else:
                save_env("VECTOR_BACKEND", "qdrant")
                save_env("QDRANT_URL", qdrant_url)
                if qdrant_key:
                    save_env("QDRANT_API_KEY", qdrant_key)
                save_env("QDRANT_COLLECTION_NAME", qdrant_collection)
                st.success(
                    "Saved! Restart the app for changes to take effect. "
                    "Note: switching backends does NOT migrate existing data — "
                    "you'll start with an empty knowledge base on Qdrant."
                )
    else:
        st.info("Using local ChromaDB. Data persists on disk but is lost on Space restarts.")
        if st.button("Switch to ChromaDB", type="primary"):
            save_env("VECTOR_BACKEND", "chroma")
            st.success("Saved! Restart the app for changes to take effect.")

    st.divider()
    st.markdown("**Current backend:**")
    st.metric("VECTOR_BACKEND", current_backend)