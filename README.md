# 📚 RAG for Research

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline for indexing and semantically searching research papers.

This project demonstrates the core components of a modern RAG system by processing academic PDFs into searchable vector embeddings using **Sentence Transformers** and **Pinecone**. It is designed as a learning project to understand how retrieval systems work before integrating a Large Language Model (LLM).

---

## 🚀 Features

- 📄 PDF text extraction
- 🧩 Structured document parsing
- ✂️ Overlapping chunk generation
- 🧠 Semantic embeddings using Sentence Transformers
- 📦 Local embedding storage
- ☁️ Pinecone vector database integration
- 🔍 Semantic similarity search
- 📝 Metadata-aware retrieval
- 💬 Answer generation from retrieved context (Google Gemini)

---

# 🏗️ Architecture

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
                         ▼
                  Semantic Retrieval
                         │
                         ▼
                 Most Relevant Chunks
```

---

# 📂 Project Structure

```text
rag-paper-qa/
│
├── data/
│   ├── papers/
│   ├── processed/
│   ├── chunks.json
│   └── embeddings.json
│
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed_and_store.py
│   ├── query.py
│   └── ask.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# ⚙️ Tech Stack

- Python
- Sentence Transformers
- Pinecone
- Google Gemini (generation)
- NumPy
- PyPDF
- python-dotenv

---

# 🔄 Pipeline

## 1. Document Ingestion

The ingestion step extracts text from one or more PDF research papers and converts them into structured JSON blocks containing metadata such as:

- Page number
- Heading
- Paragraph content

Run:

```bash
python src/ingest.py
```

---

## 2. Chunk Generation

The structured document is divided into overlapping chunks to preserve context during retrieval.

Run:

```bash
python src/chunk.py
```

---

## 3. Embedding Generation

Each chunk is converted into a dense vector representation using the **all-MiniLM-L6-v2** Sentence Transformer model.

Embeddings can be:

- Stored locally
- Uploaded to Pinecone

Run:

```bash
python src/embed_and_store.py
```

---

## 4. Semantic Search

A user query is embedded using the same embedding model and compared against stored vectors using cosine similarity.

Run:

```bash
python src/query.py --backend local
```

or

```bash
python src/query.py --backend pinecone
```

---

## 5. Answer Generation

The retrieved chunks are passed as context to a Gemini model, which generates a grounded answer with inline citations to the source chunks.

Run:

```bash
python src/ask.py --query "What is the transformer architecture?" --backend local
```

or

```bash
python src/ask.py --query "What is the transformer architecture?" --backend pinecone
```

Options: `--top-k` (context chunks, default 5), `--model` (override `GEMINI_MODEL`), `--embeddings`.

---

# 🛠️ Installation

Clone the repository

```bash
git clone https://github.com/NOURSHILI-debug/RAGforResearch.git
```

Navigate to the project

```bash
cd RAGforResearch/rag-paper-qa
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root.

```env
PINECONE_API_KEY=YOUR_API_KEY
PINECONE_INDEX_NAME=rag-paper-qa
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.5-flash
```

---

# 💻 Usage

### Step 1 — Extract the PDF

```bash
python src/ingest.py
```

### Step 2 — Generate chunks

```bash
python src/chunk.py
```

### Step 3 — Create embeddings

```bash
python src/embed_and_store.py
```

### Step 4 — Search

Local search

```bash
python src/query.py --backend local
```

Pinecone search

```bash
python src/query.py --backend pinecone
```

### Step 5 — Ask a question (RAG)

```bash
python src/ask.py --query "What is the transformer architecture?" --backend local
```

---

# 📊 Current Capabilities

✅ PDF ingestion

✅ Structured parsing

✅ Chunk generation

✅ Embedding generation

✅ Pinecone indexing

✅ Semantic retrieval

✅ Answer generation (Gemini)

---

# 🚧 Future Improvements

- Support multiple research papers
- Add FAISS and ChromaDB backends
- Hybrid search (BM25 + Dense Retrieval)
- Cross-encoder reranking
- Evaluation metrics (Recall@k, Precision@k, MRR)
- Streamlit web interface
- Docker support

---

# 🎯 Learning Objectives

This project was built to gain hands-on experience with the core stages of Retrieval-Augmented Generation (RAG), including:

- Document preprocessing
- Semantic chunking
- Dense vector embeddings
- Vector databases
- Similarity search
- Information retrieval pipelines

---

# 📚 References

- Sentence Transformers
- Pinecone
- Retrieval-Augmented Generation (RAG)
- Dense Passage Retrieval (DPR)

---

# 👩‍💻 Author

**Nour Shili**

Final-year ICT Engineering Student @ SUP'COM

Passionate about Artificial Intelligence, Machine Learning, Large Language Models, Retrieval-Augmented Generation (RAG), and AI Systems.

GitHub: https://github.com/NOURSHILI-debug

---

# ⭐ If you found this project useful

Feel free to **star the repository** and contribute with ideas or improvements!
