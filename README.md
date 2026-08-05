📚 RAG for Research Paper Question Answering

A lightweight Retrieval-Augmented Generation (RAG) pipeline for indexing and retrieving information from research papers.

This project demonstrates the core stages of a modern RAG system by processing academic PDFs into searchable vector embeddings using Sentence Transformers and Pinecone.

Features
PDF ingestion
Automatic text extraction
Structured document parsing
Overlapping chunk generation
Semantic embeddings with MiniLM
Local vector search
Pinecone integration
Metadata-aware retrieval
Architecture
Research Paper (PDF)
        │
        ▼
Text Extraction
        │
        ▼
Structured Blocks
(page, heading, source)
        │
        ▼
Chunking
(overlapping passages)
        │
        ▼
SentenceTransformer
(all-MiniLM-L6-v2)
        │
        ▼
Embedding Vectors
        │
        ├──────────────► Local Storage
        │
        ▼
Pinecone Vector Database
        │
        ▼
Semantic Retrieval
        │
        ▼
Relevant Context
Project Structure
rag-paper-qa/
│
├── data/
│   ├── papers/
│   ├── processed/
│   └── chunks.json
│
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed_and_store.py
│   └── query.py
│
├── eval/
├── notebooks/
├── requirements.txt
└── PROJECT_BRIEF.md
Pipeline
1. Ingestion

ingest.py

Reads one or more research papers
Extracts text page by page
Detects headings
Produces structured JSON blocks
2. Chunking

chunk.py

Converts structured blocks into overlapping chunks suitable for embedding.

Default:

Chunk size: 3 blocks
Overlap: 1 block
3. Embedding

embed_and_store.py

Embeddings are generated using

sentence-transformers/all-MiniLM-L6-v2

(384-dimensional vectors)

Vectors can be:

saved locally
uploaded to Pinecone
4. Retrieval

query.py

Supports:

Local semantic search
Pinecone vector search

Queries are embedded using the same embedding model before cosine similarity search.

Installation
git clone https://github.com/NOURSHILI-debug/RAGforResearch.git

cd RAGforResearch/rag-paper-qa

python -m venv .venv

source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
Environment Variables

Create a .env file:

PINECONE_API_KEY=your_key

PINECONE_INDEX_NAME=rag-paper-qa

PINECONE_REGION=us-east-1

PINECONE_CLOUD=aws
Usage
Extract PDFs
python src/ingest.py
Generate chunks
python src/chunk.py
Create embeddings
python src/embed_and_store.py
Query locally
python src/query.py --backend local
Query Pinecone
python src/query.py --backend pinecone
Technologies
Python
Sentence Transformers
Pinecone
NumPy
PyPDF
python-dotenv
Future Work
Full LLM answer generation
Hybrid search (BM25 + Dense Retrieval)
Cross-encoder reranking
Evaluation metrics (Recall@k, MRR)
Multi-document support
Web interface (Streamlit/Gradio)
FAISS and ChromaDB backends
Learning Objectives

This project was developed to understand the complete RAG indexing pipeline, including document ingestion, preprocessing, embedding generation, vector storage, and semantic retrieval.
