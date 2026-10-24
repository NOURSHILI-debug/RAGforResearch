from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from hybrid_qa import retrieve_paper_context, build_hybrid_prompt, generate_answer, format_sources, DEFAULT_MODEL
from ontology_query import OntologyQuery

load_dotenv()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Hybrid NeuroAI QA: RAG + Ontology")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", choices=["auto", "local", "pinecone"], default="auto")
    parser.add_argument("--embeddings", default="data/processed/embeddings.json")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--ontology", default=None)
    args = parser.parse_args()

    try:
        oq = OntologyQuery(args.ontology)
    except Exception as e:
        print(f"Ontology error: {e}")
        return

    ontology_ctx = oq.query_by_concept(args.query)
    matches = retrieve_paper_context(args.query, backend=args.backend, top_k=args.top_k, embeddings_path=Path(args.embeddings))

    prompt = build_hybrid_prompt(args.query, matches, ontology_ctx)

    try:
        answer = generate_answer(prompt, model=args.model)
    except Exception as e:
        print(f"Generation error: {e}")
        print("\n---PROMPT PREVIEW (first 40 lines)---")
        for i, l in enumerate(prompt.splitlines()[:40]):
            print(l)
        return

    print(f"\n--- HYBRID ANSWER ({args.model}) ---\n")
    print(answer)
    print(format_sources(matches))
    print("\n--- ONTOLOGY CONTEXT ---")
    rels = ontology_ctx.get("relations", [])
    if not rels:
        print("(No relevant ontology relations found)")
    else:
        for r in rels:
            print(f"{r['subject']} --{r['property']}--> {r['object']}")


if __name__ == "__main__":
    main()
