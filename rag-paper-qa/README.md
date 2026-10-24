# RAG for Research - NeuroAI Hybrid QA

This project implements a hybrid question-answering system combining:
1. **RAG Pipeline** - Retrieve relevant passages from research papers using vector similarity (Pinecone or local embeddings)
2. **Ontology Integration** - Query the NeuroAI ontology for explicit relationships between concepts
3. **Hybrid QA** - Combine both sources with clear distinction between paper evidence and ontology assertions

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file with:
```env
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=rag-paper-qa
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
GOOGLE_API_KEY=your_key
GEMINI_MODEL=gemini-3.5-flash  # optional
```

## Data Ingestion Pipeline

1. **Add papers** - Place PDF files in `data/papers/`
2. **Ingest** - Extract text blocks from PDFs
```bash
python src/ingest.py
```
3. **Chunk** - Split blocks into embeddable passages
```bash
python src/chunk.py
```
4. **Embed and store** - Generate embeddings and upload to Pinecone (or store locally)
```bash
python src/embed_and_store.py
# Or skip upload for local only
python src/embed_and_store.py --skip-upload
```

## Usage

### Basic RAG Query
```bash
# Local embeddings
python src/query.py --query "What is the transformer architecture?" --backend local

# With Pinecone
python src/query.py --query "What is the transformer architecture?" --backend pinecone
```

### Generate Answer with RAG
```bash
python src/ask.py --query "What is the transformer architecture?" --backend local
```

### Hybrid NeuroAI QA (RAG + Ontology)
```bash
cd src
python hybrid_ask.py --query "How does hippocampal memory inspire AI architectures?" --backend local --embeddings ../data/processed/embeddings.json
```

Example queries:
- "How does hippocampal memory inspire AI architectures?"
- "Which computational principles are associated with a biological memory mechanism?"
- "Which research papers support the relationship between hippocampal memory and associative memory?"
- "How does associative memory relate to continual learning?"

## Ontology

The NeuroAI ontology is located at `Ontology/neuroAI.owx` (OWL/XML format). The ontology module (`src/ontology_query.py`) parses it directly with Python's `xml.etree.ElementTree`.

**Features:**
- Load classes, individuals, and object properties with human-readable labels
- Query relationships (e.g., `inspires`, `implements`, `appliedTo`, `studiedIn`, `supportedBy`)
- Find concepts and their related relationships
- Handles missing files and malformed data with clear errors

**Test ontology:**
```bash
cd tests
python -c "import test_ontology as t; t.test_load(); t.test_relations(); t.test_query_concept(); print('ok')"
```

## Project Structure

```
rag-paper-qa/
├── data/
│   ├── papers/          # Input PDFs
│   ├── processed/       # Extracted blocks and embeddings
│   └── chunks.json      # Chunked passages
├── Ontology/
│   └── neuroAI.owx      # NeuroAI ontology
├── src/
│   ├── ingest.py        # PDF extraction
│   ├── chunk.py         # Text chunking
│   ├── embed_and_store.py
│   ├── query.py         # Vector search
│   ├── ask.py           # RAG QA with Gemini
│   ├── ontology_query.py # Ontology loading/querying
│   ├── hybrid_qa.py     # Hybrid QA logic
│   └── hybrid_ask.py    # Hybrid QA CLI
└── tests/
    └── test_ontology.py
```

## Notes

- The hybrid QA clearly distinguishes between evidence from papers (cited as [P1], [P2]...) and ontology assertions (stated separately, not cited as paper sources)
- If information is not found in papers or ontology, the system explicitly states limitations
- No citations or claims are fabricated
- Uses Gemini API for answer generation (requires `GOOGLE_API_KEY`)
