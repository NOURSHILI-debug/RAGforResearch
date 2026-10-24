# RAG for Research

A lightweight Retrieval-Augmented Generation (RAG) pipeline for indexing and semantically searching research papers, extended with NeuroAI Ontology integration for hybrid question answering.

## Features

- PDF text extraction (`pypdf`)
- Structured document parsing (page, heading, content)
- Overlapping chunk generation
- Semantic embeddings with Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions)
- Local embedding storage (`data/processed/embeddings.json`)
- Pinecone vector database integration (optional)
- Semantic similarity search
- Metadata-aware retrieval
- Answer generation with Google Gemini
- NeuroAI ontology integration (OWL/XML, parsed with `xml.etree.ElementTree`)
- Hybrid QA (RAG + Ontology) with clear separation of sources and stated limitations

## Architecture

```text
Research Paper (PDF)
        │
        ▼
  Text Extraction (ingest.py)
        │
        ▼
  Structured Blocks (page, heading, text)
        │
        ▼
  Chunking (chunk.py)
        │
        ▼
  Embeddings (embed_and_store.py) ──► Local (data/processed/embeddings.json) / Pinecone
        │
        ▼
  Retrieval (query.py / hybrid_qa.py)
        │
    ┌───┴───┐
    │       │
  Papers  Ontology (Ontology/neuroAI.owx)
    │       │
    └───┬───┘
        │
        ▼
  Hybrid Context → Gemini → Grounded Answer (with [P n] citations)
```

## How hybrid QA works

1. **Receive question**: `src/hybrid_ask.py` takes `--query`.
2. **Retrieve paper passages**: `hybrid_qa.retrieve_paper_context` uses the existing RAG backend (`query_local` or `query_pinecone`) to get top-k chunks. By default, if `PINECONE_API_KEY` is not set, it falls back to local embeddings.
3. **Query ontology**: `OntologyQuery` parses `Ontology/neuroAI.owx` (OWL/XML) and finds matching individuals/classes plus related object property assertions (e.g., `inspires`, `appliedTo`, `studiedIn`, `implements`, `supportedBy`).
4. **Build combined context**: `build_hybrid_prompt` includes paper passages labeled `[P1]..[Pn]` with source filename and pages, and ontology assertions labeled `[O1]..[On]` as explicit ontology relationships.
5. **Generate answer**: Gemini is prompted to cite only paper passages with `[P n]`. Ontology relationships are presented separately and are **not treated as independently verified scientific evidence** unless supported by retrieved paper passages.
6. **Report sources**: Paper sources (filename, pages, score) are listed; ontology context is shown distinctly.

**Important distinction:** Paper passages provide textual evidence from the corpus. Ontology assertions provide structured knowledge from the ontology; they may suggest connections but require support from retrieved papers to be treated as verified evidence.

## Project structure

```text
RAGforResearch/
└── rag-paper-qa/
    ├── data/
    │   ├── papers/          # Input PDFs (user-provided)
    │   ├── processed/       # blocks.json, embeddings.json
    │   ├── chunks.json
    │   └── README.md
    ├── Ontology/
    │   ├── neuroAI.owx
    │   └── README.md
    ├── src/
    │   ├── ingest.py
    │   ├── chunk.py
    │   ├── embed_and_store.py
    │   ├── query.py
    │   ├── ask.py
    │   ├── ontology_query.py
    │   ├── hybrid_qa.py
    │   └── hybrid_ask.py
    ├── tests/
    │   └── test_ontology.py
    ├── requirements.txt
    ├── .env.example
    └── README.md
```

## Tech stack

- Python
- [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- [PyPDF](https://pypdf.readthedocs.io/)
- [Pinecone](https://docs.pinecone.io/) (optional)
- [Google GenAI](https://ai.google.dev/gemini-api/docs) (Gemini)
- python-dotenv
- Standard library XML parsing

## Setup

All commands below assume the working directory is `rag-paper-qa/` unless noted otherwise.

### Install dependencies
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables
Create `.env` in `rag-paper-qa/`:
```env
PINECONE_API_KEY=YOUR_API_KEY        # required only for Pinecone backend
PINECONE_INDEX_NAME=rag-paper-qa
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY  # required for ask.py and hybrid_ask.py
GEMINI_MODEL=gemini-3.5-flash       # optional; defaults to gemini-3.5-flash in code
```

Notes (verified against code):
- `src/ask.py` defaults to `gemini-3.5-flash` (from `DEFAULT_MODEL` or `GEMINI_MODEL` env).
- `src/hybrid_ask.py` uses `hybrid_qa.DEFAULT_MODEL` which is `gemini-3.5-flash`.
- Embedding model is hardcoded to `sentence-transformers/all-MiniLM-L6-v2` (384 dims).
- If `PINECONE_API_KEY` is missing, backends fall back to local embeddings at `data/processed/embeddings.json` (relative to cwd).

## Pipeline and reproducibility

### Add papers
1. Obtain PDFs from original sources. **Do not assume public PDFs are redistributable** - check licenses/terms before committing or redistributing.
2. Place PDFs in `rag-paper-qa/data/papers/`.
3. The current ingester only processes `*.pdf` via `pypdf`.

### Regenerate the index
From `rag-paper-qa/`:
```bash
# 1) Extract blocks
python src/ingest.py

# 2) Chunk
python src/chunk.py

# 3) Embed (local only)
python src/embed_and_store.py --skip-upload

# or embed + upload to Pinecone
python src/embed_and_store.py
```

Outputs: `data/processed/blocks.json` (intermediate), `data/chunks.json`, `data/processed/embeddings.json`.

## Usage

### RAG query
```bash
python src/query.py --query "What is the transformer architecture?" --backend local
# or --backend pinecone / --backend auto
```

### RAG answer
```bash
python src/ask.py --query "What is the transformer architecture?" --backend local
```

### Hybrid QA (RAG + Ontology)
From `rag-paper-qa/`:
```bash
python src/hybrid_ask.py --query "How does hippocampal memory inspire AI architectures?" --backend local --embeddings data/processed/embeddings.json
```
From `rag-paper-qa/src/`:
```bash
python hybrid_ask.py --query "How does hippocampal memory inspire AI architectures?" --backend local --embeddings ../data/processed/embeddings.json
```

## Example (actual run)

**Question:** How does hippocampal memory inspire AI architectures?

**Summary (from a run):** Evidence from retrieved papers points to hybrid/autoassociative designs (e.g., modern Hopfield networks combined with generative models), complementary memory systems (episodic/hippocampal vs semantic/neocortical), and attractor dynamics/pattern completion via local associative learning as alternatives to backpropagation.

**Paper citations (retrieved):**
- [P1] PIIS2405844024079969.pdf (pages 5,6,7, score=0.6415)
- [P2] PIIS2405844024079969.pdf (pages 3,4,5, score=0.5874)
- [P3] The_Application_of_Artificial_Intelligence_in_Neur.pdf (pages 4,5,6, score=0.5871)
- [P4] PIIS2405844024079969.pdf (pages 1,2,3, score=0.5846)
- [P5] s41467-023-37180-x.pdf (pages 2,3,4, score=0.5806)

**Ontology relationships retrieved (stated separately, not cited as paper evidence):**
- `HippocampalMemory --inspires--> AssociativeMemory`
- `HippocampalMemory --studiedIn--> NeuroAIPaper1`
- `MemoryAugmentedNetwork --appliedTo--> ContinualLearning`
- `AssociativeMemory --implements--> AssociativeMemory`

**Note:** Only paper passages are cited as evidence. Ontology assertions are presented separately; the system does not treat them as independently verified scientific claims unless supported by retrieved passages.

## Ontology

- Location: `Ontology/neuroAI.owx` (OWL/XML)
- Parser: `src/ontology_query.py` using `xml.etree.ElementTree`
- Capabilities: list classes/individuals/properties with labels, query relations, `query_by_concept()`
- Error handling: reports missing files/malformed XML

Quick test:
```bash
cd tests
python -c "import test_ontology as t; t.test_load(); t.test_relations(); t.test_query_concept(); print('ok')"
```

## Evaluation

- **Type:** Manual qualitative evaluation only (small test set). No automated benchmark suite is included.
- **Checks performed:** relevance of retrieved chunks, traceability to citations, clear separation of paper evidence vs ontology assertions, explicit statement of limitations when unsupported.
- **Tested questions (local embeddings):** hippocampal memory → AI architectures; computational principles of biological memory; papers supporting hippocampal–associative relationship; associative memory vs continual learning.
- **Status:** The hybrid QA separates sources correctly in tested cases. **No quantitative comparison between hybrid QA and ordinary RAG has been run or reported** - do not assume improved accuracy until such comparison is performed and actual results are shared.

## Notes

- Verify `GOOGLE_API_KEY` and (if using Pinecone) `PINECONE_API_KEY` before generation/upload.
- The Gemini model name should be confirmed against your API access; code defaults to `gemini-3.5-flash`.
- Commands assume correct working directory as noted above.
- Do not commit PDFs unless you have rights to do so.

## Author

Nour Shili - Final-year ICT Engineering Student @ SUP'COM  
GitHub: [NOURSHILI-debug](https://github.com/NOURSHILI-debug)
