from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_embeddings(chunks: List[Dict[str, Any]]) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, normalize_embeddings=True).tolist()


def write_local_embeddings(chunks: List[Dict[str, Any]], vectors: List[List[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for chunk, vector in zip(chunks, vectors):
        payload.append(
            {
                "chunk_id": chunk["chunk_id"],
                "vector": vector,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk.get("source", "unknown"),
                    "pages": chunk.get("pages", []),
                    "headings": chunk.get("headings", []),
                },
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_pinecone_index(client: Any, index_name: str) -> None:
    from pinecone import ServerlessSpec

    if client.has_index(index_name):
        return

    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    print(f"Creating Pinecone index '{index_name}' ({EMBEDDING_DIM} dims, cosine)")
    client.create_index(
        name=index_name,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    deadline = time.time() + 60
    while time.time() < deadline:
        if client.describe_index(index_name).status.get("ready"):
            print("Pinecone index ready")
            return
        time.sleep(2)
    raise TimeoutError(f"Pinecone index '{index_name}' did not become ready in time")


def upload_to_pinecone(chunks: List[Dict[str, Any]], vectors: List[List[float]]) -> None:
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-paper-qa")

    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY to upload to Pinecone")

    client = Pinecone(api_key=api_key)
    ensure_pinecone_index(client, index_name)
    index = client.Index(index_name)

    records = []
    for chunk, vector in zip(chunks, vectors):
        pages = [str(p) for p in chunk.get("pages", [])]
        records.append(
            {
                "id": chunk["chunk_id"],
                "values": vector,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk.get("source", "unknown"),
                    "pages": pages,
                    "headings": chunk.get("headings", []),
                },
            }
        )

    index.upsert(vectors=records)
    print(f"Uploaded {len(records)} vectors to Pinecone index '{index_name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and optionally upload them to Pinecone")
    parser.add_argument("--in", dest="input_path", default="data/chunks.json")
    parser.add_argument("--out", default="data/processed/embeddings.json")
    parser.add_argument("--skip-upload", action="store_true", help="Only write the local embeddings file")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {input_path}")

    chunks = load_chunks(input_path)
    vectors = build_embeddings(chunks)
    write_local_embeddings(chunks, vectors, Path(args.out))
    print(f"Wrote {len(chunks)} embeddings -> {args.out}")

    if args.skip_upload:
        return

    try:
        upload_to_pinecone(chunks, vectors)
        print("Embeddings uploaded to Pinecone")
    except Exception as exc:
        print(f"Local embedding file created. Pinecone upload skipped: {exc}")


if __name__ == "__main__":
    main()
