from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict


def chunk_blocks(blocks: List[Dict[str, object]], chunk_size: int = 3, overlap: int = 1) -> List[Dict[str, object]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    chunks: List[Dict[str, object]] = []
    step = max(chunk_size - overlap, 1)

    for start in range(0, len(blocks), step):
        window = blocks[start : start + chunk_size]
        if not window:
            break

        combined_text = "\n\n".join(item["text"] for item in window if item.get("text"))
        chunk = {
            "chunk_id": f"chunk-{len(chunks) + 1:03d}",
            "text": combined_text,
            "pages": [item["page"] for item in window if "page" in item],
            "headings": [item["heading"] for item in window if item.get("heading")],
            "source": window[0].get("source", "unknown"),
        }
        chunks.append(chunk)

    return chunks


def save_chunks(chunks: List[Dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk structured blocks into embeddable passages")
    parser.add_argument("--in", dest="input_path", default="data/processed/blocks.json")
    parser.add_argument("--out", default="data/chunks.json")
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Blocks file not found: {input_path}")

    blocks = json.loads(input_path.read_text(encoding="utf-8"))
    chunks = chunk_blocks(blocks, chunk_size=args.chunk_size, overlap=args.overlap)
    output_path = save_chunks(chunks, Path(args.out))
    print(f"Created {len(chunks)} chunks -> {output_path}")


if __name__ == "__main__":
    main()
