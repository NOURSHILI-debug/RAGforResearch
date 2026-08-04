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


def collect_pdf_paths(papers_dir: Path) -> List[Path]:
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {papers_dir}")
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDFs into structured page blocks")
    parser.add_argument("--pdf", default=None, help="Optional single PDF; defaults to all PDFs in data/papers/")
    parser.add_argument("--out", default="data/processed/blocks.json")
    args = parser.parse_args()

    pdf_paths = [Path(args.pdf)] if args.pdf else collect_pdf_paths(Path("data/papers"))

    blocks: List[Dict[str, object]] = []
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        extracted = extract_blocks(pdf_path)
        print(f"Extracted {len(extracted)} blocks from {pdf_path.name}")
        blocks.extend(extracted)

    output_path = save_blocks(blocks, Path(args.out))
    print(f"Total {len(blocks)} blocks -> {output_path}")


if __name__ == "__main__":
    main()
