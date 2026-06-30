"""
scripts/evaluate.py — Run the RAGAS evaluation suite.

Edit the EVAL_SET below to match your domain, then run:
  python scripts/evaluate.py

The script will:
1. Ask the agent each question
2. Retrieve context chunks for each question
3. Score with RAGAS metrics
4. Print a results table
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from src.agent.agent import get_agent
from src.evaluation.evaluator import print_results, run_evaluation

console = Console()

# ── Edit this dataset to match your knowledge base ────────────────────────────
EVAL_SET = [
    {
        "question": "What is retrieval-augmented generation (RAG)?",
        "ground_truth": "RAG is a technique that combines information retrieval with language model generation. The model first retrieves relevant documents from a knowledge base, then uses them as context to generate an accurate answer.",
    },
    {
        "question": "What is the difference between semantic search and keyword search?",
        "ground_truth": "Semantic search uses vector embeddings to find conceptually similar content regardless of exact word matches. Keyword search matches exact terms. Semantic search is better for natural language queries.",
    },
    {
        "question": "How does a vector database store and retrieve information?",
        "ground_truth": "A vector database stores data as high-dimensional numerical vectors (embeddings). Retrieval works by converting a query to a vector and finding the nearest stored vectors using distance metrics like cosine similarity.",
    },
]


def main():
    console.print("[bold]AI Knowledge Agent — Evaluation Suite[/bold]\n")
    console.print(f"Running {len(EVAL_SET)} evaluation questions…\n")

    agent = get_agent()
    questions = []
    answers = []
    ground_truths = []

    for item in EVAL_SET:
        q = item["question"]
        gt = item["ground_truth"]
        console.print(f"[blue]Q:[/blue] {q}")
        response = agent.chat(q)
        console.print(f"[green]A:[/green] {response.answer[:200]}…\n")
        questions.append(q)
        answers.append(response.answer)
        ground_truths.append([gt])

    result = run_evaluation(questions, answers, ground_truths)
    print_results(result)

    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame([result.as_dict()])
    out_path = Path("evaluation_results.csv")
    df.to_csv(out_path, index=False)
    console.print(f"\n[dim]Results saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()
