"""
Khmer Movie Search — RAG-based recommendation system.

Run with:
    streamlit run app.py

Searches a collection of Khmer and Cambodian films using semantic embeddings
(sentence-transformers) and optionally generates recommendations via DeepSeek.
"""

import os

import streamlit as st

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer

DATA_FOLDER = "data/sample_docs"

st.set_page_config(page_title="Khmer Movie Search", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_store():
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


store, docs, chunks = load_store()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    mode = st.radio("Answer mode", ["extractive", "llm"], index=0,
                     help="Extractive works with no setup. LLM mode uses DeepSeek API.")
    if mode == "llm":
        api_key = st.text_input(
            "DeepSeek API Key", type="password",
            placeholder="sk-...",
            help="Enter your DeepSeek API key. Not stored permanently."
        )
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    else:
        api_key = ""
    st.divider()
    st.caption(f"Indexed **{len(docs)}** movies → **{len(chunks)}** chunks")
    with st.expander("Movies in this index"):
        for d in docs:
            st.write(f"- {d['title']}")

st.title("🎬 Khmer Movie Search")
st.caption("Search for Cambodian and Khmer-language films by plot, genre, director, or era.")

query = st.text_input("Your question", placeholder="e.g. Recommend a Khmer horror movie from the 2000s")
search_clicked = st.button("Search", type="primary")

if search_clicked and query.strip():
    retrieved = store.query(query, top_k=top_k)
    answer = generate_answer(query, retrieved, mode=mode, api_key=api_key)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for chunk, score in retrieved:
        with st.expander(f"{chunk.doc_title}  ·  similarity {score:.2f}"):
            st.write(chunk.text)
elif search_clicked:
    st.warning("Type a question first.")
