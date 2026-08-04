from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _embed(texts: List[str]) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(texts, normalize_embeddings=True).tolist()


def query_pinecone(query_text: str, top_k: int = 5) -> List[dict[str, Any]]:
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-paper-qa")

    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY before querying")

    vector = _embed([query_text])[0]

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return results.get("matches", [])


def query_local(query_text: str, top_k: int = 5, embeddings_path: Path = Path("data/processed/embeddings.json")) -> List[dict[str, Any]]:
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_path}. Run embed_and_store.py first (or set PINECONE_API_KEY to use Pinecone)."
        )

    data = json.loads(embeddings_path.read_text(encoding="utf-8"))
    query_vector = _embed([query_text])[0]

    matches = []
    for item in data:
        vector = item["vector"]
        score = sum(a * b for a, b in zip(query_vector, vector))
        matches.append(
            {
                "id": item["chunk_id"],
                "score": score,
                "metadata": item["metadata"],
            }
        )

    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches[:top_k]


def print_matches(matches: List[dict[str, Any]]) -> None:
    for rank, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        source = meta.get("source", "?")
        pages = meta.get("pages", [])
        pages_str = ",".join(str(p) for p in pages) if pages else "?"
        snippet = (meta.get("text") or "")[:300].replace("\n", " ")
        print(f"[{rank}] score={match.get('score'):.4f} | {source} p.{pages_str} | {match.get('id')}")
        print(f"    {snippet}")
        print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Query the vector store (Pinecone if configured, otherwise local embeddings)")
    parser.add_argument("--query", default="What is the transformer architecture?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=["auto", "local", "pinecone"],
        default="auto",
        help="auto = use Pinecone when PINECONE_API_KEY is set, else local embeddings",
    )
    parser.add_argument("--embeddings", default="data/processed/embeddings.json")
    args = parser.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "pinecone" if os.getenv("PINECONE_API_KEY") else "local"

    if backend == "pinecone":
        try:
            matches = query_pinecone(args.query, top_k=args.top_k)
        except Exception as exc:
            print(f"Pinecone query failed ({exc}); falling back to local embeddings")
            matches = query_local(args.query, top_k=args.top_k, embeddings_path=Path(args.embeddings))
    else:
        matches = query_local(args.query, top_k=args.top_k, embeddings_path=Path(args.embeddings))

    print_matches(matches)


if __name__ == "__main__":
    main()
