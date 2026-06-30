import sys
import os
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import time
import traceback
import json

st.set_page_config(page_title="RAG Deep Diagnostics", page_icon="🔬", layout="wide")

st.title("🔬 RAG Deep Diagnostics")
st.caption("Full visibility into every step of the RAG pipeline")

try:
    from src.agent.agent import get_agent
    from src.knowledge.vectorstore import add_documents, similarity_search, count_documents, reset_vectorstore, get_vectorstore
    from src.knowledge.embeddings import get_embeddings
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()

def run_test(fn):
    start = time.time()
    try:
        msg = fn()
        return True, msg or "OK", round(time.time() - start, 2)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", round(time.time() - start, 2)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Chunking",
    "2. Embeddings",
    "3. Vector Store",
    "4. Retrieval",
    "5. Tools",
    "6. Agent",
    "7. Full Pipeline",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Chunking Diagnostics")
    st.caption("See exactly how your documents are split into chunks before being stored.")

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk size (chars)", 100, 2000, 512, 50)
    with col2:
        chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, 50, 10)

    sample_text = st.text_area(
        "Paste any text to see how it gets chunked:",
        value="""Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. 
        
When a user asks a question, the system first searches a knowledge base for relevant documents. 
These documents are then passed as context to a language model, which generates an answer grounded in the retrieved information.

RAG significantly reduces hallucinations because the model can cite specific passages rather than relying on memory.
Vector databases store documents as numerical embeddings, enabling fast semantic search.
The embedding model converts text into high-dimensional vectors that capture semantic meaning.""",
        height=200
    )

    if st.button("Chunk this text", type="primary"):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = [Document(page_content=sample_text)]
        chunks = splitter.split_documents(docs)

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total characters", len(sample_text))
        col2.metric("Chunks created", len(chunks))
        col3.metric("Avg chunk size", f"{sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

        st.markdown("### Chunks")
        for i, chunk in enumerate(chunks, 1):
            with st.expander(f"Chunk {i} — {len(chunk.page_content)} chars", expanded=i == 1):
                st.text(chunk.page_content)
                st.caption(f"Start: ...{chunk.page_content[:30]}... | End: ...{chunk.page_content[-30:]}...")

        # Show overlap visualization
        if len(chunks) > 1:
            st.markdown("### Overlap between chunks")
            for i in range(len(chunks) - 1):
                end_of_current = chunks[i].page_content[-chunk_overlap:] if chunk_overlap > 0 else ""
                start_of_next = chunks[i+1].page_content[:chunk_overlap] if chunk_overlap > 0 else ""
                if end_of_current and start_of_next:
                    st.markdown(f"**Chunk {i+1} → Chunk {i+2}**")
                    col1, col2 = st.columns(2)
                    col1.caption("End of chunk:")
                    col1.code(end_of_current)
                    col2.caption("Start of next:")
                    col2.code(start_of_next)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Embedding Diagnostics")
    st.caption("See the actual vectors produced by the embedding model.")

    from src.config import EMBEDDING_MODEL
    st.info(f"Model: **{EMBEDDING_MODEL}** — runs locally, no API key needed")

    text_to_embed = st.text_input("Text to embed:", value="What is a vector database?")

    col1, col2 = st.columns(2)
    with col1:
        embed_btn = st.button("Generate embedding", type="primary")
    with col2:
        compare_text = st.text_input("Compare with:", value="ChromaDB stores vectors")

    if embed_btn and text_to_embed:
        with st.spinner("Generating embeddings…"):
            try:
                embedder = get_embeddings()
                start = time.time()
                vec1 = embedder.embed_query(text_to_embed)
                t1 = round(time.time() - start, 3)

                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Vector dimensions", len(vec1))
                col2.metric("Embedding time", f"{t1}s")
                col3.metric("Model", EMBEDDING_MODEL.split("/")[-1])

                st.markdown("### Vector preview (first 20 dimensions)")
                vec_preview = vec1[:20]
                st.bar_chart(vec_preview)

                st.markdown("### Raw values (first 20)")
                st.code(str([round(v, 4) for v in vec_preview]))

                if compare_text:
                    import math
                    vec2 = embedder.embed_query(compare_text)

                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(vec1, vec2))
                    mag1 = math.sqrt(sum(a * a for a in vec1))
                    mag2 = math.sqrt(sum(b * b for b in vec2))
                    similarity = dot / (mag1 * mag2)

                    st.divider()
                    st.markdown("### Similarity comparison")
                    col1, col2 = st.columns(2)
                    col1.code(f'"{text_to_embed}"')
                    col2.code(f'"{compare_text}"')

                    score_color = "green" if similarity > 0.7 else "orange" if similarity > 0.4 else "red"
                    st.metric("Cosine similarity", f"{similarity:.4f}",
                              delta="High similarity" if similarity > 0.7 else "Medium" if similarity > 0.4 else "Low similarity")
                    st.progress(float(similarity))

            except Exception as e:
                st.error(f"Embedding failed: {e}")
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Vector Store Diagnostics")
    st.caption("Inspect what is currently stored in ChromaDB.")

    col1, col2, col3 = st.columns(3)
    total = count_documents()
    col1.metric("Total chunks stored", total)
    col2.metric("Storage", "./chroma_db")
    col3.metric("Collection", "knowledge_base")

    if total == 0:
        st.warning("Vector store is empty. Add documents via the main chat page first.")
    else:
        st.success(f"{total} chunks ready for retrieval")

    st.divider()
    st.markdown("### Add test documents")
    test_docs_text = st.text_area("Add sample text to store:", height=100,
        placeholder="Paste text here to add to the vector store for testing...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add to store", type="primary") and test_docs_text:
            docs = [Document(page_content=test_docs_text, metadata={"source": "manual_test"})]
            n = add_documents(docs)
            st.success(f"Added {n} chunk(s). Total now: {count_documents()}")
            st.rerun()
    with col2:
        if st.button("Reset store", type="secondary"):
            reset_vectorstore()
            st.warning("Vector store cleared.")
            st.rerun()

    st.divider()
    st.markdown("### Browse stored chunks")
    browse_query = st.text_input("Search for chunks:", placeholder="vector database")
    k = st.slider("Number of results", 1, 20, 5)

    if st.button("Browse") and browse_query:
        docs = similarity_search(browse_query, k=k)
        if docs:
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Chunk {i} — source: {doc.metadata.get('source', 'unknown')}", expanded=True):
                    st.text(doc.page_content)
                    st.json(doc.metadata)
        else:
            st.info("No chunks found.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Retrieval Diagnostics")
    st.caption("Test semantic search and see scores, rankings and chunk content.")

    query = st.text_input("Retrieval query:", value="What is RAG?")
    top_k = st.slider("Top K results", 1, 10, 5)

    if st.button("Run retrieval", type="primary") and query:
        with st.spinner("Searching…"):
            try:
                start = time.time()

                # Get results with scores
                store = get_vectorstore()
                results_with_scores = store.similarity_search_with_score(query, k=top_k)
                duration = round(time.time() - start, 3)

                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Results found", len(results_with_scores))
                col2.metric("Query time", f"{duration}s")
                col3.metric("Query length", f"{len(query)} chars")

                st.markdown("### Retrieved chunks (ranked by relevance)")

                if not results_with_scores:
                    st.warning("No results found. Add documents to the knowledge base first.")
                else:
                    for i, (doc, score) in enumerate(results_with_scores, 1):
                        relevance = 1 - score  # Convert distance to similarity
                        badge = "🟢" if relevance > 0.7 else "🟡" if relevance > 0.4 else "🔴"
                        with st.expander(
                            f"{badge} Rank {i} — Score: {relevance:.4f} — Source: {doc.metadata.get('source', 'unknown')}",
                            expanded=i == 1
                        ):
                            st.markdown("**Content:**")
                            st.text(doc.page_content)
                            col1, col2 = st.columns(2)
                            col1.metric("Relevance score", f"{relevance:.4f}")
                            col1.metric("Distance", f"{score:.4f}")
                            col2.metric("Chunk length", f"{len(doc.page_content)} chars")
                            col2.metric("Source", doc.metadata.get("source", "unknown"))
                            st.markdown("**Metadata:**")
                            st.json(doc.metadata)

                # Score distribution chart
                if results_with_scores:
                    st.markdown("### Score distribution")
                    scores = {f"Chunk {i+1}": round(1 - score, 4)
                              for i, (_, score) in enumerate(results_with_scores)}
                    st.bar_chart(scores)

            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TOOLS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Tool Diagnostics")
    st.caption("Test each agent tool individually and see raw output.")

    tool_choice = st.selectbox("Select tool to test:", [
        "retrieval_tool — Search knowledge base",
        "web_search_tool — Search the web",
        "file_reader_tool — Read a file or URL",
    ])

    tool_input = st.text_input("Tool input:", value="What is LangChain?")

    if st.button("Run tool", type="primary") and tool_input:
        with st.spinner("Running tool…"):
            try:
                start = time.time()

                if "retrieval_tool" in tool_choice:
                    from src.tools.retrieval_tool import retrieval_tool
                    result = retrieval_tool.invoke(tool_input)
                    tool_name = "retrieval_tool"

                elif "web_search_tool" in tool_choice:
                    from src.tools.web_search_tool import web_search_tool
                    result = web_search_tool.invoke(tool_input)
                    tool_name = "web_search_tool"

                elif "file_reader_tool" in tool_choice:
                    from src.tools.file_reader_tool import file_reader_tool
                    result = file_reader_tool.invoke(tool_input)
                    tool_name = "file_reader_tool"

                duration = round(time.time() - start, 2)

                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Tool", tool_name)
                col2.metric("Duration", f"{duration}s")
                col3.metric("Output length", f"{len(result)} chars")

                st.markdown("### Raw tool output")
                st.text_area("Output:", value=result, height=400)

                # Parse chunks if retrieval tool
                if "retrieval_tool" in tool_choice and "Chunk" in result:
                    chunks = result.split("---")
                    st.markdown(f"### {len(chunks)} chunks returned")
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(f"Chunk {i}", expanded=i == 1):
                            st.text(chunk.strip())

            except Exception as e:
                st.error(f"Tool failed: {e}")
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AGENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Agent Diagnostics")
    st.caption("Full visibility into the agent's reasoning, tool calls, and memory.")

    question = st.text_input("Question:", value="What is retrieval-augmented generation?")

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("Run agent", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear memory", use_container_width=True):
            get_agent().reset_memory()
            st.success("Memory cleared.")

    if run_btn and question:
        with st.spinner("Agent running…"):
            try:
                start = time.time()
                agent = get_agent()
                response = agent.chat(question)
                duration = round(time.time() - start, 2)

                st.divider()

                # Summary
                col1, col2, col3 = st.columns(3)
                col1.metric("Total time", f"{duration}s")
                col2.metric("Tool calls", len(response.steps))
                col3.metric("Tools used", ", ".join(response.sources) if response.sources else "none")

                # Answer
                st.markdown("### Final Answer")
                st.markdown(response.answer)

                # Step by step
                if response.steps:
                    st.markdown("### Step-by-step reasoning")
                    for i, step in enumerate(response.steps, 1):
                        with st.expander(f"Step {i} — {step['action']}", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Tool called:**")
                                st.code(step["action"])
                                st.markdown("**Input:**")
                                st.code(str(step["input"]))
                            with col2:
                                st.markdown("**Observation (output):**")
                                st.text_area(
                                    "Output",
                                    value=str(step["observation"])[:1000],
                                    height=200,
                                    key=f"obs_{i}"
                                )

                # Memory state
                st.markdown("### Current memory state")
                try:
                    history = agent._memory.get_history()
                    st.metric("Messages in memory", len(history))
                    if history:
                        for msg in history:
                            role = "User" if "Human" in type(msg).__name__ else "Assistant"
                            with st.expander(f"{role}: {str(msg.content)[:60]}…"):
                                st.text(str(msg.content))
                except Exception:
                    st.info("Could not read memory state.")

            except Exception as e:
                st.error(f"Agent failed: {e}")
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("Full Pipeline Test")
    st.caption("Runs the complete RAG pipeline end-to-end and shows every stage.")

    st.markdown("This test runs all stages in order:")
    st.markdown("**Ingest → Chunk → Embed → Store → Retrieve → Generate → Answer**")

    sample_doc = st.text_area(
        "Sample document to ingest:",
        value="""LangGraph is a library for building stateful, multi-actor applications with LLMs.
It extends LangChain with the ability to coordinate multiple agents or chains in a graph structure.
Each node in the graph represents an agent or a processing step.
Edges define how control flows between nodes.
LangGraph supports cycles, which makes it ideal for agentic loops where the model decides what to do next.""",
        height=150
    )

    test_question = st.text_input("Question to answer:", value="What is LangGraph used for?")

    if st.button("Run full pipeline test", type="primary"):
        stages = []

        # Stage 1: Chunk
        with st.status("Running full pipeline…", expanded=True) as status:

            st.write("Stage 1: Chunking document…")
            start = time.time()
            try:
                splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
                chunks = splitter.split_documents([Document(page_content=sample_doc)])
                t = round(time.time() - start, 3)
                stages.append(("1. Chunking", True, f"Created {len(chunks)} chunks in {t}s", chunks))
                st.write(f"✅ Created {len(chunks)} chunks")
            except Exception as e:
                stages.append(("1. Chunking", False, str(e), []))
                st.write(f"❌ Chunking failed: {e}")

            # Stage 2: Embed & Store
            st.write("Stage 2: Embedding and storing chunks…")
            start = time.time()
            try:
                n = add_documents(chunks)
                t = round(time.time() - start, 3)
                stages.append(("2. Embed & Store", True, f"Stored {n} chunks in {t}s", n))
                st.write(f"✅ Embedded and stored {n} chunks")
            except Exception as e:
                stages.append(("2. Embed & Store", False, str(e), 0))
                st.write(f"❌ Storing failed: {e}")

            # Stage 3: Retrieve
            st.write("Stage 3: Retrieving relevant chunks…")
            start = time.time()
            try:
                retrieved = similarity_search(test_question, k=3)
                t = round(time.time() - start, 3)
                stages.append(("3. Retrieval", True, f"Retrieved {len(retrieved)} chunks in {t}s", retrieved))
                st.write(f"✅ Retrieved {len(retrieved)} relevant chunks")
            except Exception as e:
                stages.append(("3. Retrieval", False, str(e), []))
                st.write(f"❌ Retrieval failed: {e}")

            # Stage 4: Generate
            st.write("Stage 4: Generating answer with agent…")
            start = time.time()
            try:
                agent = get_agent()
                response = agent.chat(test_question)
                t = round(time.time() - start, 3)
                stages.append(("4. Generation", True, f"Generated answer in {t}s", response))
                st.write(f"✅ Generated answer in {t}s")
            except Exception as e:
                stages.append(("4. Generation", False, str(e), None))
                st.write(f"❌ Generation failed: {e}")

            status.update(label="Pipeline complete!", state="complete")

        # Results
        st.divider()
        passed = sum(1 for _, ok, _, _ in stages if ok)
        st.markdown(f"### Pipeline Results: {passed}/{len(stages)} stages passed")
        st.progress(passed / len(stages))

        for name, ok, msg, data in stages:
            with st.expander(f"{'✅' if ok else '❌'} {name} — {msg}", expanded=True):
                if name == "1. Chunking" and ok:
                    for i, chunk in enumerate(data, 1):
                        st.markdown(f"**Chunk {i}** ({len(chunk.page_content)} chars):")
                        st.text(chunk.page_content)

                elif name == "3. Retrieval" and ok:
                    for i, doc in enumerate(data, 1):
                        st.markdown(f"**Retrieved {i}:**")
                        st.text(doc.page_content)

                elif name == "4. Generation" and ok and data:
                    st.markdown("**Answer:**")
                    st.markdown(data.answer)
                    if data.steps:
                        st.markdown("**Tools used:**")
                        for step in data.steps:
                            st.code(f"{step['action']} → {str(step['observation'])[:200]}")

                elif not ok:
                    st.error(msg)