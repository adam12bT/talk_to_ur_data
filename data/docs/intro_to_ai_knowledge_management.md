# Introduction to AI Knowledge Management

## What is a Knowledge Management System?

A knowledge management system (KMS) is a platform that captures, organises,
stores, and retrieves information within an organisation or project. In the
context of AI agents, a KMS serves as the long-term memory and reference
library that the agent can consult when answering questions.

## Retrieval-Augmented Generation (RAG)

RAG is a technique that improves the accuracy of AI-generated answers by
grounding them in retrieved documents. The process works in two phases:

1. **Retrieval phase** — the user's question is converted into a vector
   embedding and compared against a database of pre-indexed document vectors.
   The most semantically similar chunks are returned as context.

2. **Generation phase** — the language model receives both the original
   question and the retrieved context, and generates an answer that is
   grounded in the provided information.

RAG significantly reduces hallucinations because the model can cite specific
passages rather than relying purely on parametric memory.

## Vector Databases

Vector databases are specialised storage systems optimised for storing and
searching high-dimensional vectors. Unlike traditional SQL databases that
match exact values, vector databases use approximate nearest-neighbour (ANN)
algorithms to find semantically similar items quickly.

Popular vector databases include:
- **ChromaDB** — open-source, easy to set up locally
- **Pinecone** — managed cloud service, highly scalable
- **Weaviate** — open-source with hybrid keyword + semantic search
- **pgvector** — PostgreSQL extension for teams already using Postgres

## Embedding Models

An embedding model converts text into a fixed-size numerical vector that
captures semantic meaning. Two sentences with similar meaning will have
vectors that are geometrically close in the embedding space.

Common embedding models:
- `all-MiniLM-L6-v2` — compact, fast, free, runs locally (384 dimensions)
- `text-embedding-3-small` — OpenAI's efficient embedding model
- `embed-multilingual-v3` — Cohere's multilingual model

## Evaluation Metrics

Evaluating RAG systems requires specialised metrics:

- **Faithfulness** — measures whether the answer is supported by the
  retrieved context (detects hallucinations)
- **Answer relevancy** — measures whether the answer actually addresses
  the question asked
- **Context recall** — measures whether the retrieved chunks contain
  all the information needed to answer
- **Context precision** — measures whether the retrieved chunks are
  actually useful (penalises noisy retrieval)

RAGAS is the leading open-source library for computing these metrics
automatically using an LLM as a judge.
