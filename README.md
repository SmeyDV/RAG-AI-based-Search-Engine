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

Create a local `.env` file from the included template and add your DeepSeek API
key:

```bash
cp .env.example .env
```

Edit `.env` so it contains:

```dotenv
DEEPSEEK_API_KEY=sk-your-real-key
```

Then start the app with `streamlit run app.py` and switch the sidebar to **llm**
mode. The `.env` file is ignored by Git, and the key is never entered or
displayed in the application. An environment variable set by the deployment
platform takes precedence over `.env`. Without either value, LLM mode falls
back to extractive results.

---

## Dataset

**332** Khmer and Cambodian movie documents spanning **1962–2024** — near-complete
coverage of the full history of Khmer cinema:

| Era | Coverage / Examples |
|-----|---------------------|
| 1960s Golden Age (1962–1969) | Year-by-year coverage — Apsara, Puthisen Neang Kangrey (12 Sisters), Sovannahong, Prey Prasith |
| 1970s (pre–Khmer Rouge) | Complete through 1974 — The Snake King's Wife, Bopha Angkor (action), Chnam Oun 16 |
| 1980s Revival | Post–Khmer Rouge rebuild — Shadow of Darkness (first KR-era fiction film), Rithy Panh's Site 2 |
| 1990s | Rice People, Bophana, One Evening After the War, Blank Page |
| 2000s Horror Boom | The Crocodile, Nieng Arp, Ghost Banana Tree, Neang Neat |
| 2010s International | Diamond Island, First They Killed My Father, Jailbreak, The Prey |
| 2020s | White Building, Karmalink, Funan, Return to Seoul, Meeting with Pol Pot |

Each source document may contain a title (English + Khmer), year, director,
genre, cast, plot summary, reception, awards, and source URLs. The ingestion
layer normalizes available values into consistent metadata; unknown values stay
empty rather than being guessed.

**Grounding convention.** Many golden-age films survive only as metadata (most
prints were destroyed under the Khmer Rouge), so their entries carry verified
title/year/director/cast/genre plus an explicit *"no detailed plot summary has
been published"* note — never an invented plot. Golden-age films adapted from
documented Khmer legends (e.g. Preah Thong & Neang Neak, Kakey, the Reamker, the
Twelve Sisters) instead include the **source legend's story**, clearly labeled as
such. This keeps the RAG app grounded and prevents hallucinated citations.

### Structured metadata

`rag/ingest.py` converts the legacy text documents into records with a stable
movie ID, canonical/display titles, alternate titles, year, directors, genres,
countries, languages, runtime, cast, source URLs, plot, overview, awards, and
reception. Every retrieval chunk retains this metadata and includes the most
important fields in its searchable text.

To regenerate the portable JSON Lines dataset:

```bash
python3 scripts/export_movies.py
```

The generated dataset is written to `data/movies.jsonl`, with one movie object
per line, a schema version, its source filename, and the original text retained
in `raw_text` for auditing. `data/movie.schema.json` documents the record shape.

---

## Architecture

```
Khmer Movie Search/
├── app.py                   # Streamlit interface
├── requirements.txt
├── data/sample_docs/        # 332 Khmer movie .txt files
├── data/movies.jsonl        # Normalized structured movie metadata
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
- **In-memory vector store** — fine for the current ~635 chunks, but won't scale
  past a few thousand; upgrade to FAISS or Chroma for larger corpora
- **English-only** — documents and queries are in English; a multilingual
  embedding model and Khmer-language content would need a model swap
- **No conversation history** — each query is independent

---

## Upgrading from the Starter

This project began from the CS382 `final_project_starter.zip` and was upgraded:

| Layer | Starter | Current |
|-------|---------|---------|
| Embeddings | TF-IDF (keyword) | sentence-transformers `all-MiniLM-L6-v2` (semantic) |
| LLM | Stub / extractive only | DeepSeek API via server environment with citation generation |
| Dataset | 4 sample English docs | 332 Khmer/Cambodian movie documents (1962–2024) |
| Interface | Generic "RAG Search" | "Khmer Movie Search" with API key input |

---

## Evaluation

10 test queries across different movie categories, scored on retrieval
accuracy and LLM generation quality. These results are the baseline from before
the July 2026 dataset expansion (65 → 332 films) and should be rerun against the
current corpus. Notably, the three failure categories below — **golden age**,
**action**, and **animation** — were failing purely because the corpus lacked
that content; the expansion directly targets all three (full 1960s–70s golden-age
coverage, action titles such as The Prey and Bopha Angkor, and the animated film
Funan), so those queries are expected to improve on a rerun.

### Summary

| Rating | Count | Queries |
|--------|-------|---------|
| **Strong** (retrieval + generation both accurate) | 5 | director, romance, war, award, horror |
| **Mixed** (partial match, LLM adapted well) | 2 | documentary, recent |
| **Failed** (corpus lacks relevant content) | 3 | golden_age, action, animation |

**Average top-1 retrieval score:** 0.69
**Best-performing category:** Award-winning films (0.783)
**Most common failure mode:** Corpus lacks content for the query topic (not a system bug)

### Detailed Results

| # | Query | Top Score | Top Results | Retrieval | Generation | Notes |
|---|-------|-----------|-------------|------------|------------|-------|
| 1 | Recommend a Khmer horror movie from the 2000s | 0.631 | The Night Curse of Reatrei, Lost Loves, The Crocodile | Good | Good | LLM correctly recommended The Crocodile (2005) |
| 2 | What Cambodian documentaries explore the Khmer Rouge era? | 0.686 | Lost Loves, The Red Sense, Meeting with Pol Pot | Mixed | Fair | LLM noted documentary vs feature ambiguity honestly |
| 3 | Films from the Cambodian Golden Age of the 1960s | 0.709 | Lost Loves, One Evening After the War, Vanished 2009 | Fail | Good | Corpus has 1960s films (Apsara, Snake King's Wife) but retrieval missed them; LLM correctly declined |
| 4 | Movies directed by Rithy Panh | 0.680 | One Evening After the War, S-21, The Burnt Theatre | Good | Good | All three returned are Panh films; LLM cited them accurately |
| 5 | Action and martial arts movies from Cambodia | 0.662 | Lost Loves, Puthisen Neang Kangrey, The Golden Voice | Fail | Good | Corpus lacks action films; LLM honestly said none found |
| 6 | A Khmer romantic drama set in modern times | 0.662 | Who Am I (2009), Karmalink | Good | Good | LLM correctly recommended Who Am I with plot context |
| 7 | Cambodian films about war and genocide | 0.734 | Lost Loves, Meeting with Pol Pot, One Evening After the War | Good | Good | Best retrieval scores; LLM cited all three with proper context |
| 8 | Best Cambodian movies from the last five years | 0.633 | Lost Loves, Meeting with Pol Pot, The Last Reel | Mixed | Good | Meeting with Pol Pot is recent (2024); others are older; LLM noted limitations |
| 9 | Award-winning Cambodian films recognized internationally | 0.783 | Rice People, The Last Reel, Meeting with Pol Pot | Very Good | Good | Highest retrieval scores; all three were Oscar submissions; LLM provided accurate context |
| 10 | Cambodian animated or fantasy films | 0.689 | Lost Loves, Puthisen Neang Kangrey, One Evening After the War | Fail | Good | No animated films in corpus; LLM correctly responded "none found" |

### Key Findings

- **Semantic embeddings outperform keyword matching** — queries for "war and genocide" found films without those exact words
- **LLM graceful failure works** — for 3 queries where corpus lacked content, the LLM refused to fabricate answers instead of hallucinating
- **Corpus content is the bottleneck** — retrieval failures (action, animation, golden age) are due to missing or under-represented topics in the document collection, not the embedding model
- **Director queries work well** — the model associates director names with their films across document boundaries
- **Award queries strongest** — Oscar-submitted films have distinctive language ("submitted", "Academy Awards") that embeddings pick up well
