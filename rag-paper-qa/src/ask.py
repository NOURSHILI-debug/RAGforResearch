from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

from query import query_local, query_pinecone

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash"

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the provided "
    "context passages, which come from academic papers. "
    'Cite the source of every factual claim inline using [1], [2], ... matching the '
    "numbered sources listed in the context. If the context does not contain enough "
    "information to answer, say so clearly instead of guessing. Keep the answer "
    "focused and precise.\n\n"
    "SOURCES (numbered, cited as [n]):\n"
)


def retrieve_context(query_text: str, backend: str, top_k: int, embeddings_path: Path) -> List[dict[str, Any]]:
    if backend == "pinecone":
        try:
            return query_pinecone(query_text, top_k=top_k)
        except Exception as exc:
            print(f"Pinecone query failed ({exc}); falling back to local embeddings")
            return query_local(query_text, top_k=top_k, embeddings_path=embeddings_path)
    return query_local(query_text, top_k=top_k, embeddings_path=embeddings_path)


def build_prompt(query_text: str, matches: List[dict[str, Any]]) -> str:
    parts = [SYSTEM_PROMPT]
    for rank, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        text = (meta.get("text") or "").strip()
        source = meta.get("source", "?")
        pages = meta.get("pages", [])
        pages_str = ",".join(str(p) for p in pages) if pages else "?"
        parts.append(f"[{rank}] Source: {source}, pages {pages_str}\n{text}")
    parts.append(f"\nQUESTION: {query_text}")
    return "\n\n".join(parts)


def generate_answer(prompt: str, model: str) -> str:
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY before generating answers")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or "(empty response from model)"


def format_sources(matches: List[dict[str, Any]]) -> str:
    lines = ["\nSOURCES:"]
    for rank, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        source = meta.get("source", "?")
        pages = meta.get("pages", [])
        pages_str = ",".join(str(p) for p in pages) if pages else "?"
        score = match.get("score", 0.0)
        lines.append(f"[{rank}] {source} (pages {pages_str}, score={score:.4f})")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks and generate an answer with Gemini")
    parser.add_argument("--query", default="What is the transformer architecture?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", choices=["auto", "local", "pinecone"], default="auto")
    parser.add_argument("--embeddings", default="data/processed/embeddings.json")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "pinecone" if os.getenv("PINECONE_API_KEY") else "local"

    matches = retrieve_context(args.query, backend, args.top_k, Path(args.embeddings))
    if not matches:
        print("No relevant chunks found; cannot generate an answer.")
        return

    prompt = build_prompt(args.query, matches)
    answer = generate_answer(prompt, args.model)

    print(f"\n--- ANSWER ({args.model}) ---\n")
    print(answer)
    print(format_sources(matches))


if __name__ == "__main__":
    main()
