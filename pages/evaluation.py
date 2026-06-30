"""
pages/evaluation.py
RAGAS Evaluation UI — rewritten for ragas 0.4.x.
"""
import sys
import os

# ── Patch broken ragas 0.4.x VertexAI import FIRST ───────────────────────────
from unittest.mock import MagicMock
if "langchain_community.chat_models.vertexai" not in sys.modules:
    sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import json
import traceback
import time

st.set_page_config(page_title="RAG Evaluation", page_icon="📊", layout="wide")


st.title("📊 RAG Evaluation")
st.caption("Score your RAG pipeline using RAGAS metrics — faithfulness, relevancy, recall, precision")

try:
    from src.agent.agent import get_agent
    from src.evaluation.evaluator import run_evaluation, EvaluationResult
    from src.knowledge.vectorstore import count_documents
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ── Info banner ───────────────────────────────────────────────────────────────
total = count_documents()
if total == 0:
    st.warning("Knowledge base is empty — add documents first so retrieval has something to score.")
else:
    st.info(f"Knowledge base has **{total} chunks** ready for evaluation.")

st.divider()

# ── Q&A pair builder ──────────────────────────────────────────────────────────
st.subheader("1. Build your evaluation set")
st.caption(
    "Add question / expected-answer pairs. "
    "RAGAS compares the agent's actual answer against your expected answer to compute scores."
)

if "eval_pairs" not in st.session_state:
    st.session_state.eval_pairs = [
        {
            "question": "What is retrieval-augmented generation?",
            "ground_truth": "RAG combines information retrieval with text generation to produce grounded answers.",
        }
    ]


def add_pair():
    st.session_state.eval_pairs.append({"question": "", "ground_truth": ""})


def remove_pair(idx: int):
    st.session_state.eval_pairs.pop(idx)


for i, pair in enumerate(st.session_state.eval_pairs):
    with st.expander(f"Q{i+1}: {pair['question'][:60] or '(empty)'}…", expanded=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.session_state.eval_pairs[i]["question"] = st.text_input(
                "Question", value=pair["question"], key=f"q_{i}"
            )
            st.session_state.eval_pairs[i]["ground_truth"] = st.text_area(
                "Expected answer (ground truth)", value=pair["ground_truth"],
                key=f"gt_{i}", height=80
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("🗑️", key=f"del_{i}", help="Remove this pair"):
                remove_pair(i)
                st.rerun()

col1, col2 = st.columns([2, 8])
with col1:
    st.button("➕ Add question", on_click=add_pair, use_container_width=True)
with col2:
    uploaded = st.file_uploader(
        "Or import Q&A pairs from JSON",
        type=["json"],
        label_visibility="collapsed",
    )
    if uploaded:
        try:
            imported = json.loads(uploaded.read())
            if isinstance(imported, list) and all("question" in p for p in imported):
                st.session_state.eval_pairs = imported
                st.success(f"Imported {len(imported)} pairs.")
                st.rerun()
            else:
                st.error("JSON must be a list of {question, ground_truth} objects.")
        except Exception as e:
            st.error(f"Import failed: {e}")

st.divider()

# ── Run evaluation ────────────────────────────────────────────────────────────
st.subheader("2. Run evaluation")

valid_pairs = [
    p for p in st.session_state.eval_pairs
    if p["question"].strip() and p["ground_truth"].strip()
]
st.caption(f"{len(valid_pairs)} valid pair(s) ready to evaluate.")

if st.button("▶ Run RAGAS evaluation", type="primary", disabled=len(valid_pairs) == 0):
    if total == 0:
        st.error("Cannot evaluate — knowledge base is empty.")
    else:
        questions = [p["question"] for p in valid_pairs]
        ground_truths = [[p["ground_truth"]] for p in valid_pairs]

        # Step 1: get agent answers
        st.markdown("### Running…")
        progress = st.progress(0, text="Getting agent answers…")
        answers = []
        agent = get_agent()

        for i, q in enumerate(questions):
            try:
                resp = agent.chat(q)
                answers.append(resp.answer)
            except Exception as e:
                answers.append(f"ERROR: {e}")
            progress.progress(
                (i + 1) / (len(questions) * 2),
                text=f"Answered {i+1}/{len(questions)}…"
            )

        # Step 2: RAGAS scoring
        progress.progress(0.6, text="Running RAGAS metrics (this takes ~30s)…")
        start = time.time()
        try:
            result: EvaluationResult = run_evaluation(questions, answers, ground_truths)
            duration = round(time.time() - start, 1)
            progress.progress(1.0, text="Done!")

            st.session_state["last_eval"] = {
                "result": result,
                "answers": answers,
                "questions": questions,
                "ground_truths": ground_truths,
                "duration": duration,
            }

        except Exception as e:
            st.error(f"RAGAS evaluation failed: {e}")
            st.code(traceback.format_exc())

# ── Results display ───────────────────────────────────────────────────────────
if "last_eval" in st.session_state:
    ev = st.session_state["last_eval"]
    result: EvaluationResult = ev["result"]

    st.divider()
    st.subheader(f"3. Results  ·  {ev['duration']}s")

    def score_delta(score: float) -> str:
        if score >= 0.85:
            return "Excellent"
        if score >= 0.70:
            return "Good"
        return "Needs work"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Faithfulness", f"{result.faithfulness:.3f}", score_delta(result.faithfulness))
    col2.metric("Answer relevancy", f"{result.answer_relevancy:.3f}", score_delta(result.answer_relevancy))
    col3.metric("Context recall", f"{result.context_recall:.3f}", score_delta(result.context_recall))
    col4.metric("Context precision", f"{result.context_precision:.3f}", score_delta(result.context_precision))
    col5.metric("Average", f"{result.average():.3f}", score_delta(result.average()))

    st.markdown("### Score breakdown")
    st.bar_chart({
        "Faithfulness": result.faithfulness,
        "Answer relevancy": result.answer_relevancy,
        "Context recall": result.context_recall,
        "Context precision": result.context_precision,
    })

    st.markdown("### Per-question answers")
    for i, (q, a, gt) in enumerate(
        zip(ev["questions"], ev["answers"], ev["ground_truths"]), 1
    ):
        with st.expander(f"Q{i}: {q[:80]}…", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Expected answer:**")
                st.info(gt[0])
            with col2:
                st.markdown("**Agent answer:**")
                st.success(a)

    st.divider()
    export = {
        "scores": result.as_dict(),
        "average": result.average(),
        "pairs": [
            {"question": q, "ground_truth": gt[0], "agent_answer": a}
            for q, a, gt in zip(ev["questions"], ev["answers"], ev["ground_truths"])
        ],
    }
    st.download_button(
        "Download results as JSON",
        data=json.dumps(export, indent=2),
        file_name="ragas_results.json",
        mime="application/json",
    )