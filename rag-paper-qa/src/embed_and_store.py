from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_embeddings(chunks: List[Dict[str, Any]]) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, normalize_embeddings=True).tolist()


def write_local_embeddings(chunks: List[Dict[str, Any]], vectors: List[List[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for chunk, vector in zip(chunks, vectors):
        payload.append({"chunk_id": chunk["chunk_id"], "vector": vector, "metadata": {"text": chunk["text"]}})
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upload_to_pinecone(chunks: List[Dict[str, Any]], vectors: List[List[float]]) -> None:
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-paper-qa")

    if not api_key or not environment:
        raise RuntimeError("Set PINECONE_API_KEY and PINECONE_ENVIRONMENT before uploading")

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)

    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({"id": chunk["chunk_id"], "values": vector, "metadata": {"text": chunk["text"]}})

    index.upsert(vectors=records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and optionally upload them to Pinecone")
    parser.add_argument("--in", dest="input_path", default="data/chunks.json")
    parser.add_argument("--out", default="data/processed/embeddings.json")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {input_path}")

    chunks = load_chunks(input_path)
    vectors = build_embeddings(chunks)
    write_local_embeddings(chunks, vectors, Path(args.out))

    try:
        upload_to_pinecone(chunks, vectors)
        print("Embeddings uploaded to Pinecone")
    except Exception as exc:
        print(f"Local embedding file created. Pinecone upload skipped: {exc}")


if __name__ == "__main__":
    main()
