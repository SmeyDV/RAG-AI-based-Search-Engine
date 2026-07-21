"""
Khmer Movie Search — RAG-based recommendation system.

Run with:
    streamlit run app.py

Searches a collection of Khmer and Cambodian films using semantic embeddings
(sentence-transformers) and optionally generates recommendations via DeepSeek.
"""

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
                     help="Extractive needs no setup. LLM mode uses DEEPSEEK_API_KEY from the environment.")
    if mode == "llm":
        st.caption("DeepSeek credentials are loaded from the server environment.")
    st.divider()
    st.caption(f"Indexed **{len(docs)}** movies \u2192 **{len(chunks)}** chunks")
    with st.expander("Movies in this index"):
        for d in docs:
            year = d["metadata"].get("year")
            st.write(f"- {d['title']} ({year})" if year else f"- {d['title']}")

st.title("🎬 Khmer Movie Search")
st.caption("Search for Cambodian and Khmer-language films by plot, genre, director, or era.")

query = st.text_input("Your question", placeholder="e.g. Recommend a Khmer horror movie from the 2000s")
search_clicked = st.button("Search", type="primary")

if search_clicked and query.strip():
    retrieved = store.query(query, top_k=top_k)
    answer = generate_answer(query, retrieved, mode=mode)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for chunk, score in retrieved:
        with st.expander(f"{chunk.doc_title}  \u00b7  similarity {score:.2f}"):
            metadata = chunk.metadata
            details = []
            if metadata.get("year"):
                details.append(str(metadata["year"]))
            if metadata.get("directors"):
                details.append("Directed by " + ", ".join(metadata["directors"]))
            if metadata.get("genres"):
                details.append("Genres: " + ", ".join(metadata["genres"]))
            if details:
                st.caption(" \u00b7 ".join(details))
            st.write(chunk.text)
            for source_url in metadata.get("source_urls", []):
                st.markdown(f"[Dataset source]({source_url})")
elif search_clicked:
    st.warning("Type a question first.")
