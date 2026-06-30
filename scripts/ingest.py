"""
scripts/ingest.py — CLI for adding documents to the knowledge base.

Usage:
  python scripts/ingest.py                          # ingest all files in data/docs/
  python scripts/ingest.py --file path/to/doc.pdf   # ingest a specific file
  python scripts/ingest.py --url https://example.com # ingest a web page
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from src.knowledge.ingestion import ingest_directory, ingest_file, ingest_url
from src.knowledge.vectorstore import count_documents

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the AI Knowledge Agent knowledge base"
    )
    parser.add_argument("--file", "-f", help="Path to a single file to ingest")
    parser.add_argument("--url", "-u", help="URL of a web page to ingest")
    parser.add_argument(
        "--dir", "-d",
        help="Directory to ingest (default: data/docs/)",
        default=None,
    )
    args = parser.parse_args()

    before = count_documents()
    console.print(f"[dim]Knowledge base currently has {before} chunks.[/dim]\n")

    if args.file:
        ingest_file(args.file)
    elif args.url:
        ingest_url(args.url)
    else:
        ingest_directory(args.dir or "./data/docs")

    after = count_documents()
    console.print(f"\n[bold]Knowledge base now has {after} chunks.[/bold] (+{after - before})")


if __name__ == "__main__":
    main()
