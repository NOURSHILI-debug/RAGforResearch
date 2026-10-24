# 📚 RAG for Research

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline for indexing and semantically searching research papers, extended with **NeuroAI Ontology integration** for hybrid question answering.

This project demonstrates the core components of a modern RAG system by processing academic PDFs into searchable vector embeddings using **Sentence Transformers** and **Pinecone**. It now also supports combining retrieved paper evidence with explicit relationships from a NeuroAI ontology to answer domain-specific questions.

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

## 📂 Project Structure

```text
RAGforResearch/
│
└── rag-paper-qa/
    │
    ├── data/
    │   ├── papers/          # Input PDFs
    │   ├── processed/       # Extracted blocks and embeddings
    │   ├── chunks.json      # Chunked passages
    │   └── README.md
    │
    ├── Ontology/
    │   ├── neuroAI.owx      # NeuroAI ontology (OWL/XML)
    │   └── README.md
    │
    ├── src/
    │   ├── ingest.py        # PDF extraction
    │   ├── chunk.py         # Text chunking
    │   ├── embed_and_store.py
    │   ├── query.py         # Vector search
    │   ├── ask.py           # RAG QA with Gemini
    │   ├── ontology_query.py # Ontology loading/querying
    │   ├── hybrid_qa.py     # Hybrid QA logic
    │   └── hybrid_ask.py    # Hybrid QA CLI
    │
    ├── tests/
    │   └── test_ontology.py
    │
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

## 🔄 Pipeline

### 1. Document Ingestion
Extract text from PDF research papers into structured JSON blocks (page, heading, content).

```bash
cd rag-paper-qa
python src/ingest.py
```

### 2. Chunk Generation
Split structured blocks into overlapping chunks to preserve context.

```bash
python src/chunk.py
```

### 3. Embedding Generation
Generate embeddings and store locally or upload to Pinecone.

```bash
python src/embed_and_store.py
# Or skip upload for local only
python src/embed_and_store.py --skip-upload
```

### 4. Semantic Search
Search for relevant chunks using vector similarity.

```bash
python src/query.py --backend local
# or
python src/query.py --backend pinecone
```

### 5. Answer Generation (RAG)
Generate grounded answers with citations.

```bash
python src/ask.py --query "What is the transformer architecture?" --backend local
```

### 6. Hybrid QA (RAG + Ontology)
Combine paper evidence with NeuroAI ontology relationships.

```bash
cd src
python hybrid_ask.py --query "How does hippocampal memory inspire AI architectures?" --backend local --embeddings ../data/processed/embeddings.json
```

Example queries:
- "How does hippocampal memory inspire AI architectures?"
- "Which computational principles are associated with a biological memory mechanism?"
- "Which research papers support the relationship between hippocampal memory and associative memory?"
- "How does associative memory relate to continual learning?"

## 🛠️ Installation

```bash
git clone https://github.com/NOURSHILI-debug/RAGforResearch.git
cd RAGforResearch/rag-paper-qa
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in `rag-paper-qa/`:

```env
PINECONE_API_KEY=YOUR_API_KEY
PINECONE_INDEX_NAME=rag-paper-qa
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.5-flash
```

## 🧬 Ontology

The NeuroAI ontology (`Ontology/neuroAI.owx`) is parsed with Python's standard library. The `src/ontology_query.py` module provides:
- `list_classes()`, `list_individuals()`, `list_properties()` with human-readable labels
- `get_relations(property_name=..., subject=..., object=...)` for relationship queries
- `query_by_concept(concept)` to find relevant concepts and relations

Tests: `cd tests && python -c "import test_ontology as t; t.test_load(); t.test_relations(); t.test_query_concept(); print('ok')"`

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
- Evaluation metrics (Recall@k, Precision@k, MRR)
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
