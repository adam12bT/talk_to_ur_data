"""
src/evaluation/evaluator.py
RAGAS evaluation pipeline — rewritten for ragas 0.4.x.

ragas 0.4.x has a broken import of ChatVertexAI from langchain-community 0.4+
(the module was removed). We patch it before importing ragas so the app
doesn't crash on startup.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

# ── Patch broken ragas 0.4.x import before anything else loads ragas ──────────
if "langchain_community.chat_models.vertexai" not in sys.modules:
    sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    _faithfulness as faithfulness,
    _answer_relevancy as answer_relevancy,
    _context_recall as context_recall,
    _context_precision as context_precision,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.console import Console
from rich.table import Table

from src.config import GOOGLE_API_KEY, EMBEDDING_MODEL, LLM_MODEL
from src.knowledge.vectorstore import similarity_search

console = Console()


@dataclass
class EvaluationResult:
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float

    def as_dict(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
        }

    def average(self) -> float:
        values = list(self.as_dict().values())
        return sum(values) / len(values)


def _build_dataset(
    questions: list[str],
    answers: list[str],
    ground_truths: list[list[str]],
) -> EvaluationDataset:
    samples = []
    for question, answer, gt_list in zip(questions, answers, ground_truths):
        # Retrieve context chunks for this question
        docs = similarity_search(question)
        contexts = [doc.page_content for doc in docs]

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=gt_list[0] if gt_list else "",
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def run_evaluation(
    questions: list[str],
    answers: list[str],
    ground_truths: list[list[str]],
) -> EvaluationResult:
    console.print("\n[blue]Building evaluation dataset…[/blue]")
    dataset = _build_dataset(questions, answers, ground_truths)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    console.print("[blue]Running RAGAS metrics…[/blue]")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    scores = result.to_pandas()

    return EvaluationResult(
        faithfulness=float(scores["faithfulness"].mean()),
        answer_relevancy=float(scores["answer_relevancy"].mean()),
        context_recall=float(scores["context_recall"].mean()),
        context_precision=float(scores["context_precision"].mean()),
    )


def print_results(result: EvaluationResult) -> None:
    table = Table(title="RAGAS Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Rating", justify="center")

    def rating(score: float) -> str:
        if score >= 0.85:
            return "[green]Excellent[/green]"
        if score >= 0.70:
            return "[yellow]Good[/yellow]"
        return "[red]Needs work[/red]"

    for metric, score in result.as_dict().items():
        table.add_row(metric.replace("_", " ").title(), f"{score:.3f}", rating(score))

    table.add_section()
    table.add_row("Average", f"{result.average():.3f}", rating(result.average()))
    console.print(table)