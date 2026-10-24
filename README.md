# 📚 RAG for Research

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline for indexing and semantically searching research papers, extended with **NeuroAI Ontology integration** for hybrid question answering.

## 🚀 Features

- 📄 PDF text extraction
- 🧩 Structured document parsing
- ✂️ Overlapping chunk generation
- 🧠 Semantic embeddings using Sentence Transformers (all-MiniLM-L6-v2)
- 📦 Local embedding storage
- ☁️ Pinecone vector database integration
- 🔍 Semantic similarity search
- 📝 Metadata-aware retrieval
- 💬 Answer generation from retrieved context (Google Gemini)
- 🧬 **NeuroAI Ontology integration** - Load and query OWL/XML ontologies with human-readable labels
- 🔀 **Hybrid QA (RAG + Ontology)** - Combine paper evidence with ontology assertions, clearly distinguishing sources and limitations

## 🏗️ Architecture

```text
                  Research Paper (PDF)
                          │
                          ▼
                   Text Extraction
                          │
                          ▼
                 Structured JSON Blocks
               (page, heading, content)
                          │
                          ▼
                   Chunk Generation
                  (overlapping chunks)
                          │
                          ▼
         SentenceTransformer Embeddings
           (all-MiniLM-L6-v2, 384 dims)
                          │
           ┌──────────────┴──────────────┐
           │                             │
           ▼                             ▼
    Local embeddings.json         Pinecone Index
           │                             │
           └──────────────┬──────────────┘
                          │
                          ▼
                   Semantic Retrieval
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
           Paper Chunks      NeuroAI Ontology
                │                   │
                └─────────┬─────────┘
                          ▼
                    Hybrid Context
                          │
                          ▼
                  Grounded Answer with
                Citations & Ontology Notes
```

## 🔀 How hybrid QA works

1. **Question received** - User asks a NeuroAI-related question.
2. **Retrieve relevant passages** - The existing RAG pipeline embeds the query and retrieves top-k chunks from the vector store (Pinecone if configured, otherwise local `data/processed/embeddings.json`).
3. **Query the ontology** - `Ontology/neuroAI.owx` is parsed (OWL/XML) to find matching individuals/classes and related object property assertions (e.g., `inspires`, `appliedTo`, `studiedIn`).
4. **Combine context** - Both sources are included in the prompt: paper passages labeled `[P1]..[Pn]` (with source filename and pages) and ontology assertions labeled `[O1]..[On]` (stated as ontology relationships).
5. **Generate grounded answer** - Gemini generates an answer citing only paper passages with `[P n]`. Ontology relationships are **explicitly presented separately** and are **not treated as independently verified scientific evidence** unless supported by retrieved paper passages.
6. **Cite sources** - The response lists paper sources with filenames, pages, and scores; ontology context is shown distinctly.

**Important distinction:** Paper passages provide textual evidence from the corpus. Ontology relationships provide structured/explicit knowledge from the ontology schema/assertions - they may suggest connections but are not automatically verified by the papers unless the retrieved passages support them.

## 📂 Project Structure

```text
RAGforResearch/
│
└── rag-paper-qa/
    ├── data/
    │   ├── papers/          # Input PDFs (user-provided)
    │   ├── processed/       # Extracted blocks and embeddings
    │   ├── chunks.json      # Chunked passages
    │   └── README.md
    ├── Ontology/
    │   ├── neuroAI.owx      # NeuroAI ontology (OWL/XML)
    │   └── README.md
    ├── src/
    │   ├── ingest.py        # PDF extraction
    │   ├── chunk.py         # Text chunking
    │   ├── embed_and_store.py
    │   ├── query.py         # Vector search
    │   ├── ask.py           # RAG QA with Gemini
    │   ├── ontology_query.py # Ontology loading/querying
    │   ├── hybrid_qa.py     # Hybrid QA logic
    │   └── hybrid_ask.py    # Hybrid QA CLI
    ├── tests/
    │   └── test_ontology.py
    ├── requirements.txt
    ├── .env.example
    └── README.md
```

## ⚙️ Tech Stack

- Python
- Sentence Transformers (all-MiniLM-L6-v2)
- Pinecone
- Google Gemini (generation)
- NumPy
- PyPDF
- python-dotenv
- xml.etree.ElementTree (ontology parsing)

## 🔑 Setup and Environment Variables

All commands below assume you are working from `rag-paper-qa/` unless stated otherwise.

Create a `.env` file in `rag-paper-qa/` with the following variables (check against actual code):
```env
PINECONE_API_KEY=YOUR_API_KEY        # required for Pinecone backend
PINECONE_INDEX_NAME=rag-paper-qa     # default used by code
PINECONE_CLOUD=aws                  # used when creating index
PINECONE_REGION=us-east-1           # used when creating index
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY  # required for answer generation (ask.py, hybrid_ask.py)
GEMINI_MODEL=gemini-3.5-flash       # default in code (can override)
```

Notes:
- `ask.py` defaults to model from `GEMINI_MODEL` env or `gemini-3.5-flash`. `hybrid_ask.py` also defaults to `gemini-3.5-flash`.
- Embedding model is fixed in code to `sentence-transformers/all-MiniLM-L6-v2` (384 dims).
- If no `PINECONE_API_KEY` is set, both RAG and hybrid flows fall back to local embeddings at `data/processed/embeddings.json` (relative to working directory).

## 🔄 Pipeline & Reproducibility

### How to obtain and place papers
1. Obtain PDFs from their original sources (journals/publishers/preprint servers). **Do not assume they are redistributable** - check each paper's license/terms before committing or redistributing.
2. Place PDF files in `rag-paper-qa/data/papers/`.
3. (Optional) Some sources may be HTML - the current `ingest.py` only processes `*.pdf` files via `pypdf`; adjust if needed.

### Regenerate the index
From `rag-paper-qa/`:
```bash
# 1) Extract blocks from all PDFs in data/papers/
python src/ingest.py

# 2) Chunk blocks
python src/chunk.py

# 3) Generate embeddings and store (local only)
python src/embed_and_store.py --skip-upload

# Or upload to Pinecone (requires PINECONE_API_KEY)
python src/embed_and_store.py
```

Embeddings will be written to `data/processed/embeddings.json` and chunks to `data/chunks.json`. The hybrid/RAG flows use these paths by default when running locally.

**Working directory note:** Commands are shown relative to `rag-paper-qa/`. `hybrid_ask.py` can be run from `src/` with `--embeddings ../data/processed/embeddings.json`, or from `rag-paper-qa/` with `--embeddings data/processed/embeddings.json`.

## 💬 Usage

### Basic RAG Query
```bash
# Local
python src/query.py --query "What is the transformer architecture?" --backend local

# Pinecone
python src/query.py --query "What is the transformer architecture?" --backend pinecone
```

### Generate Answer with RAG
```bash
python src/ask.py --query "What is the transformer architecture?" --backend local
```

### Hybrid NeuroAI QA (RAG + Ontology)
From `rag-paper-qa/`:
```bash
python src/hybrid_ask.py --query "How does hippocampal memory inspire AI architectures?" --backend local --embeddings data/processed/embeddings.json
```
Or from `rag-paper-qa/src/`:
```bash
python hybrid_ask.py --query "How does hippocampal memory inspire AI architectures?" --backend local --embeddings ../data/processed/embeddings.json
```

## 🧪 Real example with actual output

**Question:** "How does hippocampal memory inspire AI architectures?"

**Short answer (from actual run):** Hippocampal memory inspires AI architectures through biologically plausible designs: hybrid models combining autoassociation networks (e.g., modern Hopfield networks) with generative models to link neocortex and hippocampus; adoption of complementary memory systems (episodic/hippocampal vs semantic/neocortical); and attractor dynamics/pattern completion using local associative learning rules as alternatives to backpropagation.

**Paper citations (from corpus):**
- [P1] PIIS2405844024079969.pdf (pages 5,6,7, score=0.6415)
- [P2] PIIS2405844024079969.pdf (pages 3,4,5, score=0.5874)
- [P3] The_Application_of_Artificial_Intelligence_in_Neur.pdf (pages 4,5,6, score=0.5871)
- [P4] PIIS2405844024079969.pdf (pages 1,2,3, score=0.5846)
- [P5] s41467-023-37180-x.pdf (pages 2,3,4, score=0.5806)

**Ontology relationships retrieved (stated, not independently verified):**
- `HippocampalMemory --inspires--> AssociativeMemory`
- `HippocampalMemory --studiedIn--> NeuroAIPaper1`
- `MemoryAugmentedNetwork --appliedTo--> ContinualLearning`
- `AssociativeMemory --implements--> AssociativeMemory`

**Note:** Paper passages provide textual evidence from the corpus (cited as [P n]). The ontology assertions above are explicit relationships from `Ontology/neuroAI.owx` presented separately as structured knowledge; they are not presented as independently verified scientific evidence unless supported by the retrieved paper passages.

## 🧬 Ontology

The NeuroAI ontology is at `Ontology/neuroAI.owx` (OWL/XML). Parsed with `xml.etree.ElementTree` in `src/ontology_query.py`. Features:
- Classes/individuals/properties with human-readable labels
- Relationship queries (`inspires`, `implements`, `appliedTo`, `studiedIn`, `supportedBy`, etc.)
- Concept-based search returning matches and related relations
- Clear errors for missing/malformed files

Test:
```bash
cd tests
python -c "import test_ontology as t; t.test_load(); t.test_relations(); t.test_query_concept(); print('ok')"
```

## 📊 Evaluation

**Approach:** We evaluate retrieval and answer quality manually with a small test set. For each question we inspect:
- Retrieval: Are the top-k chunks relevant to the question? (manual check of sources/pages/snippets)
- Grounding: Does the answer cite paper passages with [P n]? Are claims traceable to retrieved text?
- Hybrid behavior: Are ontology assertions clearly labeled as ontology assertions (not cited as paper sources)? Does the system state limitations when ontology/papers don't support a claim?

**Current manual test set (tested against local embeddings):**
1. "How does hippocampal memory inspire AI architectures?" - Retrieved relevant papers (Rolls et al. on hippocampal memory/systems) discussing autoassociative networks, complementary memory systems; ontology surfaced `HippocampalMemory --inspires--> AssociativeMemory` and related assertions. Answer distinguished paper evidence from ontology.
2. "Which computational principles are associated with a biological memory mechanism?" - Retrieved passages on predictive coding, path integration, local associative learning; ontology relations shown separately.
3. "Which research papers support the relationship between hippocampal memory and associative memory?" - Identified relevant papers from corpus (PIIS2405844024079969.pdf discusses CA3 attractor/autoassociation networks); ontology assertions listed distinctly.
4. "How does associative memory relate to continual learning?" - Found connections via complementary systems, generative replay; noted ontology lacks direct `AssociativeMemory--continual learning` link but has `MemoryAugmentedNetwork --appliedTo--> ContinualLearning`.

**Results:** Hybrid QA correctly separates paper evidence from ontology assertions and cites paper sources. No quantitative comparison (e.g., accuracy metrics) has been run yet between hybrid QA and ordinary RAG - we do not claim improved accuracy until such a comparison is performed and reported with actual results.

## 📊 Current Capabilities

✅ PDF ingestion  
✅ Structured parsing  
✅ Chunk generation  
✅ Embedding generation  
✅ Pinecone indexing  
✅ Semantic retrieval  
✅ Answer generation (Gemini)  
✅ NeuroAI ontology loading and querying  
✅ Hybrid RAG + Ontology QA with source distinction  

## 🚧 Future Improvements

- Support multiple research papers
- Add FAISS and ChromaDB backends
- Hybrid search (BM25 + Dense Retrieval)
- Cross-encoder reranking
- Systematic evaluation (Recall@k, Precision@k, groundedness) comparing hybrid vs RAG
- Streamlit web interface
- Docker support

## 🎯 Learning Objectives

This project was built to gain hands-on experience with the core stages of Retrieval-Augmented Generation (RAG), including:
- Document preprocessing
- Semantic chunking
- Dense vector embeddings
- Vector databases
- Similarity search
- Information retrieval pipelines
- Knowledge representation with ontologies
- Hybrid neuro-symbolic approaches

## 📚 References

- Sentence Transformers
- Pinecone
- Retrieval-Augmented Generation (RAG)
- Dense Passage Retrieval (DPR)
- OWL/RDF ontologies

## 👩‍💻 Author

**Nour Shili**  
Final-year ICT Engineering Student @ SUP'COM  
Passionate about Artificial Intelligence, Machine Learning, Large Language Models, Retrieval-Augmented Generation (RAG), and AI Systems.

GitHub: https://github.com/NOURSHILI-debug

## ⭐ If you found this project useful

Feel free to **star the repository** and contribute with ideas or improvements!
