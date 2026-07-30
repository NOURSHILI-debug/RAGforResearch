from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


def build_heading(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    first_line = lines[0]
    if len(first_line) < 80 and first_line[0].isupper():
        return first_line
    return ""


def extract_blocks(pdf_path: Path) -> List[Dict[str, object]]:
    reader = PdfReader(str(pdf_path))
    blocks: List[Dict[str, object]] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        for paragraph in paragraphs:
            blocks.append(
                {
                    "page": page_num,
                    "heading": build_heading(paragraph),
                    "text": paragraph,
                    "source": pdf_path.name,
                }
            )

    return blocks


def save_blocks(blocks: List[Dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(blocks, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PDF into structured page blocks")
    parser.add_argument("--pdf", default="data/papers/attention_is_all_you_need.pdf")
    parser.add_argument("--out", default="data/processed/blocks.json")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    blocks = extract_blocks(pdf_path)
    output_path = save_blocks(blocks, Path(args.out))
    print(f"Extracted {len(blocks)} blocks from {pdf_path.name} -> {output_path}")


if __name__ == "__main__":
    main()
