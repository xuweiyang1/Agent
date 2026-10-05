"""Compare BM25, dense, and hybrid retrieval, and print the numbers.

    python scripts/rag_vector_compare.py
    python scripts/rag_vector_compare.py --save eval/rag-vector-store.json

This is the vector-store answer the ROADMAP asks for, kept honest the same
way W3.5 and W6 are: one corpus, one task set, one generator, and only the
retrieval component changes. Every run goes through ``run_baseline``, so the
hit rate, MRR, coverage and answer accuracy mean exactly what they mean in
the baseline report -- the rows are comparable by construction, not by
inspection.

What to expect, and why it is not a landslide:

- **BM25 wins on rare exact terms.** The corpus is technical notes, and a
  question that names a specific word is a lexical lookup BM25 is built for.
- **Dense wins where the wording differs.** Character n-grams catch a stem
  and a compound a word-level BM25 splits, without the shared stemmer.
- **Hybrid is usually the best or tied.** It is the reason both stores exist;
  if it were never better, fusing ranks would be a cost with no return.

The absolute numbers depend on the hashing embedder, which is a stand-in for
a trained encoder. The *comparison* is the deliverable, and it is valid for
whatever embedder is injected -- pass a real one and the shape holds while
the dense row improves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.corpus_rag import build_index_for
from retrieval.naive import ExtractiveGenerator
from retrieval.report import run_baseline
from retrieval.vector_store import DenseIndex, HybridIndex


def _stores(size: int, overlap: int) -> dict:
    """The three retrievers, built over the same chunks.

    Built from one chunk list on purpose. A dense store over differently split
    chunks would differ in two variables -- chunking and similarity -- and the
    comparison would not say which moved the numbers.
    """
    from retrieval.chunking import chunk_corpus
    from retrieval.corpus_rag import documents

    chunks = chunk_corpus(documents(), size=size, overlap=overlap)
    bm25 = build_index_for(size=size, overlap=overlap)
    dense = DenseIndex(chunks)
    return {"bm25": bm25, "dense": dense, "hybrid": HybridIndex([bm25, dense])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BM25 vs dense vs hybrid retrieval")
    parser.add_argument("--k", type=int, default=5, help="chunks retrieved per question")
    parser.add_argument("--size", type=int, default=512, help="chunk size in characters")
    parser.add_argument("--overlap", type=int, default=64, help="chunk overlap in characters")
    parser.add_argument("--save", default=None, help="write the report as JSON")
    options = parser.parse_args(argv)

    stores = _stores(options.size, options.overlap)
    reports = {
        name: run_baseline(index=store, generate=ExtractiveGenerator(), k=options.k)
        for name, store in stores.items()
    }

    n_chunks = len(stores["bm25"])
    print(f"corpus: {n_chunks} chunks, k={options.k}, size={options.size}, overlap={options.overlap}")
    print()
    print(f"{'retriever':10} {'hit':>6} {'cov':>6} {'mrr':>6} {'answer':>7}")
    print("-" * 42)
    for name, report in reports.items():
        print(
            f"{name:10} {report.hit_rate:>6.2f} {report.coverage:>6.2f} "
            f"{report.mrr:>6.2f} {report.answer_accuracy:>7.2f}"
        )

    print()
    print("multi_hop, the category the baseline is expected to fail:")
    print(f"{'retriever':10} {'hit':>6} {'cov':>6} {'mrr':>6}")
    print("-" * 34)
    for name, report in reports.items():
        score = report.per_category.get("multi_hop")
        if score is None:
            continue
        print(f"{name:10} {score.hit_rate:>6.2f} {score.coverage:>6.2f} {score.mrr:>6.2f}")

    if options.save:
        target = Path(options.save)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: report.to_dict() for name, report in reports.items()}
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nsaved: {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())