"""
pages/drift.py
RAG Drift Monitor — tracks how retrieval quality changes as the KB grows.

Every time you ingest a document (via KB Browser or here), a snapshot is
recorded. This page plots average retrieval score vs chunk count over time
so you can spot when adding docs starts hurting retrieval quality.
"""
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import json
import traceback
import pandas as pd

st.set_page_config(page_title="RAG Drift Monitor", page_icon="📈", layout="wide")


st.title("📈 RAG Drift Monitor")
st.caption(
    "Track how retrieval quality changes as your knowledge base grows. "
    "A dropping score means new documents are diluting retrieval — RAG drift."
)

try:
    from src.knowledge.drift_tracker import (
        get_drift_log,
        record_snapshot,
        clear_drift_log,
        get_benchmark_queries,
        DEFAULT_BENCHMARKS,
    )
    from src.knowledge.vectorstore import count_documents
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ── Controls (moved from sidebar into the page body) ───────────────────────────
st.subheader("Controls")
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

with ctrl_col1:
    if st.button("📸 Record snapshot now", use_container_width=True, type="primary"):
        with st.spinner("Scoring benchmark queries…"):
            try:
                snap = record_snapshot(label="Manual snapshot")
                st.success(f"Snapshot saved — avg score: {snap['avg_score']:.4f}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

with ctrl_col2:
    if st.button("🗑️ Clear drift log", use_container_width=True):
        clear_drift_log()
        st.warning("Drift log cleared.")
        st.rerun()

with ctrl_col3:
    st.caption("Snapshots are recorded automatically when you add documents via the KB Browser page.")

# ── Load log ──────────────────────────────────────────────────────────────────
log = get_drift_log()
total_chunks = count_documents()

# ── Top metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Snapshots recorded", len(log))
col2.metric("Current chunks in KB", total_chunks)

if len(log) >= 2:
    first_score = log[0]["avg_score"]
    last_score = log[-1]["avg_score"]
    delta = round(last_score - first_score, 4)
    col3.metric("First avg score", f"{first_score:.4f}")
    col4.metric("Latest avg score", f"{last_score:.4f}", delta=f"{delta:+.4f}",
                delta_color="normal" if delta >= 0 else "inverse")
elif len(log) == 1:
    col3.metric("Avg score", f"{log[0]['avg_score']:.4f}")
    col4.metric("Trend", "Need ≥2 snapshots")
else:
    col3.metric("Avg score", "—")
    col4.metric("Trend", "No data yet")

st.divider()

# ── No data state ─────────────────────────────────────────────────────────────
if not log:
    st.info(
        "No drift data yet. "
        "Add documents via the **KB Browser** page (snapshots are recorded automatically), "
        "or click **Record snapshot now** in the sidebar to take a baseline."
    )
    st.markdown("### Benchmark queries that will be tracked:")
    for i, q in enumerate(DEFAULT_BENCHMARKS, 1):
        st.markdown(f"{i}. {q}")
    st.stop()

# ── Build dataframe ───────────────────────────────────────────────────────────
df = pd.DataFrame(log)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ── Main drift chart ──────────────────────────────────────────────────────────
st.subheader("Average retrieval score over time")
st.caption("X = number of chunks in KB at snapshot time. Y = average top-1 similarity score across all benchmark queries.")

# Plot avg_score vs chunk_count
chart_df = df[["chunk_count", "avg_score"]].copy()
chart_df = chart_df.rename(columns={"chunk_count": "Chunks in KB", "avg_score": "Avg score"})
chart_df = chart_df.set_index("Chunks in KB")
st.line_chart(chart_df, use_container_width=True)

# ── Per-query drift chart ─────────────────────────────────────────────────────
st.subheader("Per-query score drift")
st.caption("Individual benchmark query scores over snapshots — spot which queries degrade most.")

try:
    # Expand query_scores into columns
    query_scores_df = pd.json_normalize(df["query_scores"].tolist())
    query_scores_df.index = df["chunk_count"].values

    # Shorten column names for display
    query_scores_df.columns = [
        q[:50] + "…" if len(q) > 50 else q
        for q in query_scores_df.columns
    ]
    query_scores_df.index.name = "Chunks in KB"

    st.line_chart(query_scores_df, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render per-query chart: {e}")

# ── Snapshot history table ────────────────────────────────────────────────────
st.subheader("Snapshot history")

display_df = df[["timestamp", "chunk_count", "avg_score", "label"]].copy()
display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
display_df.columns = ["Time", "Chunks in KB", "Avg score", "Label"]
display_df["Avg score"] = display_df["Avg score"].round(4)

# Color-code the score column
def color_score(val):
    if val >= 0.7:
        return "background-color: #d4edda"
    elif val >= 0.4:
        return "background-color: #fff3cd"
    else:
        return "background-color: #f8d7da"

st.dataframe(
    display_df.style.map(color_score, subset=["Avg score"]),
    use_container_width=True,
    hide_index=True,
)

# ── Drift alert ───────────────────────────────────────────────────────────────
if len(log) >= 3:
    recent_scores = [s["avg_score"] for s in log[-3:]]
    trend = recent_scores[-1] - recent_scores[0]
    if trend < -0.05:
        st.error(
            f"⚠️ **Drift detected!** Average retrieval score dropped by {abs(trend):.4f} "
            f"over the last 3 snapshots. Consider reviewing recently added documents "
            f"or increasing TOP_K in Settings."
        )
    elif trend < -0.02:
        st.warning(
            f"⚠️ Mild drift: score dropped {abs(trend):.4f} over the last 3 snapshots. "
            f"Keep an eye on this."
        )
    else:
        st.success("✅ Retrieval quality is stable across recent snapshots.")

st.divider()

# ── Export ────────────────────────────────────────────────────────────────────
st.download_button(
    "Download drift log as JSON",
    data=json.dumps(log, indent=2),
    file_name="drift_log.json",
    mime="application/json",
)

# ── Benchmark query editor ────────────────────────────────────────────────────
st.divider()
st.subheader("Benchmark queries")
st.caption(
    "These are the queries used to measure retrieval quality. "
    "To change them permanently, edit DEFAULT_BENCHMARKS in src/knowledge/drift_tracker.py."
)
for i, q in enumerate(DEFAULT_BENCHMARKS, 1):
    st.markdown(f"{i}. {q}")

if st.button("▶ Run benchmark now & record snapshot"):
    with st.spinner("Running all benchmark queries…"):
        try:
            snap = record_snapshot(label="Manual benchmark run")
            st.success(f"Done — avg score: {snap['avg_score']:.4f}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")