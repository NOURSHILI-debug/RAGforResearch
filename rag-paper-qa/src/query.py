from __future__ import annotations

import argparse
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def query_pinecone(query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer

    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-paper-qa")

    if not api_key or not environment:
        raise RuntimeError("Set PINECONE_API_KEY and PINECONE_ENVIRONMENT before querying")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vector = model.encode([query_text], normalize_embeddings=True)[0].tolist()

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return results.get("matches", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Pinecone index")
    parser.add_argument("--query", default="What is the transformer architecture?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    matches = query_pinecone(args.query, top_k=args.top_k)
    for match in matches:
        print(match.get("score"), match.get("id"), match.get("metadata", {}).get("text", "")[:300])


if __name__ == "__main__":
    main()
