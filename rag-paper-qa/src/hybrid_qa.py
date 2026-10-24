from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from query import query_local, query_pinecone
from ontology_query import OntologyQuery

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash"

HYBRID_SYSTEM_PROMPT = (
    "You are a research assistant for NeuroAI. Answer using ONLY the evidence provided.\n"
    "You will be given two sources of information:\n"
    "1) PAPER PASSAGES (from the vector store/RAG) - treat these as evidence from the corpus. Cite them as [P1], [P2], ... matching the numbered PAPER PASSAGES.\n"
    "2) ONTOLOGY ASSERTIONS (explicit relationships from the ontology) - treat these as claims/relationships asserted in the ontology, NOT necessarily verified by the papers above. Do NOT cite ontology entries as paper sources.\n\n"
    "Rules:\n"
    "- Cite every factual claim from papers with [P n].\n"
    "- Clearly distinguish 'Evidence from papers:' vs 'Ontology assertions (stated, not necessarily in papers):'.\n"
    "- If a relationship is only in the ontology and not supported by paper passages, say so explicitly.\n"
    "- Do not fabricate citations, titles, page numbers, or claims. If information is missing, state the limitation.\n"
    "- Be precise and grounded.\n\n"
)


def retrieve_paper_context(query_text: str, backend: str = "auto", top_k: int = 5, embeddings_path: Path = Path("data/processed/embeddings.json")) -> List[Dict[str, Any]]:
    if backend == "auto":
        backend = "pinecone" if os.getenv("PINECONE_API_KEY") else "local"
    if backend == "pinecone":
        try:
            return query_pinecone(query_text, top_k=top_k)
        except Exception:
            return query_local(query_text, top_k=top_k, embeddings_path=embeddings_path)
    return query_local(query_text, top_k=top_k, embeddings_path=embeddings_path)


def build_hybrid_prompt(query_text: str, paper_matches: List[Dict[str, Any]], ontology_ctx: Dict[str, Any]) -> str:
    parts = [HYBRID_SYSTEM_PROMPT]
    parts.append("=== PAPER PASSAGES (evidence) ===")
    if not paper_matches:
        parts.append("(No paper passages retrieved from vector store.)")
    else:
        for rank, match in enumerate(paper_matches, start=1):
            meta = match.get("metadata", {})
            text = (meta.get("text") or "").strip()
            source = meta.get("source", "?")
            pages = meta.get("pages", [])
            pages_str = ",".join(str(p) for p in pages) if pages else "?"
            parts.append(f"[P{rank}] Source: {source}, pages {pages_str}\n{text}")

    parts.append("\n=== ONTOLOGY ASSERTIONS (explicit relationships) ===")
    rels = ontology_ctx.get("relations", [])
    if not rels:
        parts.append("(No relevant ontology relations found for this query.)")
    else:
        for i, r in enumerate(rels, start=1):
            parts.append(f"[O{i}] {r['subject']} --{r['property']}--> {r['object']}")
    matches_info = ontology_ctx.get("matches", [])
    if matches_info:
        parts.append("\nOntology matches (individuals/classes):")
        for m in matches_info:
            parts.append(f"  - {m[0]}: {m[2]}")

    parts.append(f"\n=== QUESTION ===\n{query_text}")
    return "\n\n".join(parts)


def generate_answer(prompt: str, model: str = DEFAULT_MODEL) -> str:
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY before generating answers")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or "(empty response from model)"


def format_sources(paper_matches: List[Dict[str, Any]]) -> str:
    lines = ["\nPAPER SOURCES (from corpus):"]
    if not paper_matches:
        lines.append("(none)")
        return "\n".join(lines)
    for rank, match in enumerate(paper_matches, start=1):
        meta = match.get("metadata", {})
        source = meta.get("source", "?")
        pages = match.get("pages") or meta.get("pages", [])
        pages_str = ",".join(str(p) for p in pages) if pages else "?"
        score = match.get("score", 0.0)
        lines.append(f"[P{rank}] {source} (pages {pages_str}, score={score:.4f})")
    return "\n".join(lines)
