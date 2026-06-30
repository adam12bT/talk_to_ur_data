"""
src/knowledge/ingestion.py
Load documents from files or URLs, split them into chunks,
and store them in the vector database.
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import track

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR
from src.knowledge.vectorstore import add_documents

console = Console()

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_file(path: str | Path) -> list[Document]:
    """Load a single file and return raw Documents."""
    path = Path(path)
    ext = path.suffix.lower()
    loader_cls = SUPPORTED_EXTENSIONS.get(ext)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {list(SUPPORTED_EXTENSIONS)}"
        )
    loader = loader_cls(str(path))
    return loader.load()


def load_url(url: str) -> list[Document]:
    """Load a web page and return raw Documents."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    # Tag source metadata
    for doc in docs:
        doc.metadata["source"] = url
    return docs


def ingest_file(path: str | Path) -> int:
    """Load, chunk, and store a single file. Returns chunk count."""
    path = Path(path)
    console.print(f"[blue]Loading[/blue] {path.name}…")
    raw_docs = load_file(path)
    splitter = _get_splitter()
    chunks = splitter.split_documents(raw_docs)
    count = add_documents(chunks)
    console.print(f"[green]✓[/green] {path.name} → {count} chunks stored")
    return count


def ingest_url(url: str) -> int:
    """Load, chunk, and store a web page. Returns chunk count."""
    console.print(f"[blue]Loading[/blue] {url}…")
    raw_docs = load_url(url)
    splitter = _get_splitter()
    chunks = splitter.split_documents(raw_docs)
    count = add_documents(chunks)
    console.print(f"[green]✓[/green] URL → {count} chunks stored")
    return count


def ingest_directory(directory: str | Path = DOCS_DIR) -> int:
    """Ingest all supported files in a directory. Returns total chunk count."""
    directory = Path(directory)
    if not directory.exists():
        console.print(f"[yellow]Directory {directory} not found, creating it.[/yellow]")
        directory.mkdir(parents=True, exist_ok=True)
        return 0

    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        console.print(f"[yellow]No supported documents found in {directory}[/yellow]")
        return 0

    total = 0
    for file in track(files, description="Ingesting documents…"):
        try:
            total += ingest_file(file)
        except Exception as e:
            console.print(f"[red]Error loading {file.name}: {e}[/red]")

    console.print(f"\n[bold green]Done.[/bold green] {total} chunks stored from {len(files)} files.")
    return total