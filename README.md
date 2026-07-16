# Khmer Movie Search — RAG-Based Recommendation System

A **Retrieval-Augmented Generation** search system that answers questions about
Khmer and Cambodian films. Users type a query (e.g. *"Recommend a Khmer horror
movie from the 2000s"*) and the system retrieves relevant movie chunks using
semantic embeddings, then generates a grounded answer with source citations.

Built for CS382 final project.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### LLM mode (recommended)

Switch the sidebar to **llm** mode and paste your DeepSeek API key in the
password field. Answers will be generated with citations instead of just
showing raw passages.

---

## Dataset

59 Khmer and Cambodian movie documents spanning **1966–2024**, including:

| Era | Examples |
|-----|----------|
| 1960s–70s Golden Age | Apsara, Puthisen Neang Kangrey, The Snake King's Wife |
| 1990s Revival | Rice People, Bophana, One Evening After the War |
| 2000s Horror Boom | The Crocodile, Nieng Arp, Ghost Banana Tree |
| 2010s International | Diamond Island, First They Killed My Father, Jailbreak |
| 2020s | White Building, Karmalink, Tenement, Meeting with Pol Pot |

Each document contains: title (English + Khmer), year, director, genre, cast,
plot summary, reception, and awards.

---

## Architecture

```
Khmer Movie Search/
├── app.py                   # Streamlit interface
├── requirements.txt
├── data/sample_docs/        # 59 Khmer movie .txt files
└── rag/
    ├── ingest.py            # Load .txt files, word-count chunking (80 word chunks, 20 word overlap)
    ├── embed_store.py       # SentenceTransformer (all-MiniLM-L6-v2) embeddings + cosine similarity search
    └── generate.py          # Extractive (no key) or DeepSeek LLM (with key) answer generation
```

### Pipeline

1. **Ingest** — documents are loaded and split into overlapping chunks.
2. **Embed** — each chunk is encoded into a 384-dimensional vector using
   `all-MiniLM-L6-v2` from sentence-transformers.
3. **Retrieve** — user query is embedded, cosine similarity ranks chunks,
   top-k are returned with scores.
4. **Generate** — retrieved chunks + query → DeepSeek LLM produces a
   natural-language answer citing source movies.
5. **Interface** — Streamlit UI with query box, answer panel, and expandable
   source list with similarity scores.

---

## Usage

- **Top-k slider** — set how many chunks to retrieve (default: 3)
- **Answer mode** — `extractive` (no key needed) or `llm` (needs DeepSeek API key)
- **Search** — type a question about Khmer movies and click Search

### Example queries

- *"Recommend a Khmer horror movie from the 2000s"*
- *"What are some Cambodian documentaries about the Khmer Rouge?"*
- *"Films directed by Rithy Panh"*
- *"Show me martial arts movies from Cambodia"*
- *"A romantic drama set in modern Phnom Penh"*

---

## Known Limitations

- **Plain text only** — only `.txt` files are loaded (no PDF support yet)
- **Word-count chunking** — fixed 80-word chunks with 20-word overlap;
  sentence-aware chunking would improve coherence
- **In-memory vector store** — fine for 298 chunks, but won't scale past a few
  thousand; upgrade to FAISS or Chroma for larger corpora
- **English-only** — documents and queries are in English; a multilingual
  embedding model and Khmer-language content would need a model swap
- **No conversation history** — each query is independent

---

## Upgrading from the Starter

This project began from the CS382 `final_project_starter.zip` and was upgraded:

| Layer | Starter | Current |
|-------|---------|---------|
| Embeddings | TF-IDF (keyword) | sentence-transformers `all-MiniLM-L6-v2` (semantic) |
| LLM | Stub / extractive only | DeepSeek API with citation generation |
| Dataset | 4 sample English docs | 59 Khmer/Cambodian movie documents |
| Interface | Generic "RAG Search" | "Khmer Movie Search" with API key input |

---

## Evaluation

| Query | Retrieval Quality | Generation Quality | Notes |
|-------|-------------------|-------------------|-------|
| *"Recommend a Khmer horror movie from the 2000s"* | Good — top result was The Crocodile (2005), score 0.67 | Good — correctly identified and described The Crocodile with citations | Strong semantic match on horror genre |
| *"What are some Cambodian documentaries about the Khmer Rouge?"* | Moderate — returned relevant films (The Red Sense, Lost Loves) but not clearly marked as docs | Good — LLM noted uncertainty about documentary status, still cited relevant films | Demonstrates graceful failure |

(Expand with 6–8 more test queries as you go.)
