"""Compare TF-IDF, Day 6 dense retrieval, and RRF hybrid retrieval.

Gold labels are read only by the evaluator. They are never passed to a retriever.
Run from the repository root with ``python -m src.retrieval_experiment``.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from starter.baseline import TfidfIndex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "day6" / "chunks.json"
DEFAULT_QUESTIONS = ROOT / "data" / "day7" / "benchmark_dev.json"
DEFAULT_OUTPUT = ROOT / "starter" / "results" / "retrieval"
DEFAULT_STRATEGY = "B"
RRF_CONSTANT = 60


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_chunk(raw: dict[str, Any]) -> dict[str, Any]:
    """Adapt Day 6 (`chunk_id`, `content`, `source_name`) to Day 7's schema."""
    chunk = {
        "id": raw.get("id", raw.get("chunk_id")),
        "text": raw.get("text", raw.get("content")),
        "source_id": raw.get("source_id", raw.get("document_id")),
        "source": raw.get("source", raw.get("source_name")),
        "page": raw.get("page"),
        "metadata": raw.get("metadata") or {},
    }
    missing = [key for key in ("id", "text", "source_id", "source", "page") if chunk[key] is None]
    if missing:
        raise ValueError(f"Chunk missing required fields: {', '.join(missing)}")
    return chunk


def normalize_text(text: str) -> str:
    """Normalize whitespace so line wraps in source text don't break evidence checks."""
    return " ".join(text.split()).casefold()


def evidence_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    haystack = normalize_text(str(chunk.get("text", "")))
    return any(
        str(chunk.get("source_id")) == str(item.get("source_id"))
        and str(chunk.get("page")) == str(item.get("page"))
        and normalize_text(str(item.get("evidence", ""))) in haystack
        for item in gold
    )


def filter_matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    metadata = chunk.get("metadata") or {}
    return all(str(metadata.get(key)) == str(value) for key, value in (filters or {}).items())


def pack_context(results: list[dict[str, Any]], budget_chars: int) -> tuple[str, list[str]]:
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for chunk in results:
        chunk_id = str(chunk["id"])
        if chunk_id in seen:
            continue
        block = f"[{chunk_id} | {chunk['source']} | page {chunk['page']}]\n{chunk['text']}"
        candidate = "\n\n".join([*blocks, block])
        if len(candidate) <= budget_chars:
            blocks.append(block)
            ids.append(chunk_id)
            seen.add(chunk_id)
    return "\n\n".join(blocks), ids


class Day6Dense:
    """Read-only adapter to the real Day 6 BGE-M3 + PostgreSQL/pgvector code."""

    def __init__(self, starter_path: Path, strategy: str) -> None:
        starter_path = starter_path.resolve()
        source_dir = starter_path / "src"
        package_init = source_dir / "__init__.py"
        if not package_init.exists():
            raise FileNotFoundError(f"Day 6 starter package not found: {source_dir}")

        # Load Day 6's `src` package under a private name to avoid colliding with
        # this project's `src` namespace package.
        spec = importlib.util.spec_from_file_location(
            "day6src", package_init, submodule_search_locations=[str(source_dir)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Day 6 package from {source_dir}")
        package = importlib.util.module_from_spec(spec)
        sys.modules["day6src"] = package
        spec.loader.exec_module(package)

        config_spec = importlib.util.spec_from_file_location(
            "day6_config_runtime", starter_path / "day6_config.py"
        )
        if config_spec is None or config_spec.loader is None:
            raise ImportError(f"Cannot load Day 6 config from {starter_path}")
        config = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config)

        embedder_cls = importlib.import_module("day6src.embeddings").MyEmbedder
        store_cls = importlib.import_module("day6src.store").PgVectorStore
        self.embedder = embedder_cls(model_name=config.EMBEDDING_MODEL)
        self.store = store_cls(config.VECTOR_STORE_URL, config.EMBEDDING_DIMENSIONS)
        self.strategy = strategy

    def query_vector(self, query: str) -> list[float]:
        return self.embedder.embed_query(query)

    def search(self, query_vector: list[float], top_k: int,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows = self.store.search(self.strategy, query_vector, top_k=top_k, filters=filters)
        return [normalize_chunk(row) | {"score": row["score"], "method": "dense"} for row in rows]


def lexical_search(index: TfidfIndex, query: str, chunks: list[dict[str, Any]],
                   top_k: int, filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    # Search all B chunks before applying metadata filters so the lexical score
    # and IDF corpus remain identical across filtered and unfiltered questions.
    candidates = index.search(query, top_k=len(chunks))
    selected = [
        {**item, "method": "tfidf"}
        for item in candidates if filter_matches(item, filters)
    ][:top_k]
    return selected


def hybrid_search(lexical: list[dict[str, Any]], dense: list[dict[str, Any]],
                  top_k: int, rrf_constant: int = RRF_CONSTANT) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for source_results in (lexical, dense):
        for rank, item in enumerate(source_results, start=1):
            chunk_id = str(item["id"])
            entry = fused.setdefault(chunk_id, {**item, "score": 0.0, "ranks": {}})
            entry["score"] += 1.0 / (rrf_constant + rank)
            entry["ranks"][item.get("method", "retriever")] = rank
            # Keep the higher-detail dense chunk only when the text and metadata
            # are identical; the ID is the primary de-duplication key.
            if len(item.get("text", "")) > len(entry.get("text", "")):
                entry.update({key: value for key, value in item.items() if key != "score"})
    ranked = sorted(fused.values(), key=lambda item: (-item["score"], item["id"]))[:top_k]
    for item in ranked:
        item["method"] = "hybrid_rrf"
    return ranked


def validate_questions(questions: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    if not questions:
        raise ValueError("No questions found for this split")
    ids: set[str] = set()
    for question in questions:
        qid = question.get("id")
        if not isinstance(qid, str) or not qid or qid in ids:
            raise ValueError("Question IDs must be unique non-empty strings")
        ids.add(qid)
        if question.get("split") != "dev":
            raise ValueError("This experiment runner is dev-only; use a separate, finalized test runner")
        if question.get("answerable") is True:
            gold = question.get("gold")
            if not isinstance(gold, list) or not gold:
                raise ValueError(f"{qid}: answerable cases need gold evidence")
            for item in gold:
                if not any(evidence_hit(chunk, [item]) for chunk in chunks):
                    raise ValueError(f"{qid}: evidence does not match a source chunk: {item}")
        elif question.get("answerable") is not False:
            raise ValueError(f"{qid}: set answerable to true or false")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.method == "all":
        summaries: dict[str, Any] = {}
        for method in ("tfidf", "dense", "hybrid"):
            child_args = argparse.Namespace(**vars(args))
            child_args.method = method
            summaries[method] = run(child_args)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "all-summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summaries

    raw_chunks = read_json(args.chunks)
    chunks = [normalize_chunk(item) for item in raw_chunks if item.get("strategy") == args.strategy]
    if not chunks:
        raise ValueError(f"No chunks found for Day 6 strategy {args.strategy}")
    questions = [q for q in read_json(args.questions) if q.get("split") == "dev"]
    validate_questions(questions, chunks)

    tfidf_index_seconds = 0.0
    index: TfidfIndex | None = None
    if args.method in ("tfidf", "hybrid"):
        tfidf_started = time.perf_counter()
        index = TfidfIndex(chunks)
        tfidf_index_seconds = time.perf_counter() - tfidf_started

    dense: Day6Dense | None = None
    precomputed_dense: dict[str, list[dict[str, Any]]] = {}
    dense_startup_seconds = 0.0
    if args.method in ("dense", "hybrid"):
        if args.dense_traces is not None:
            # Reuse a recorded Day 6 dev run when the local database is offline.
            # Query, split, strategy, and filter must agree exactly.
            source_traces = read_json(args.dense_traces)
            for item in source_traces:
                if item.get("split") != "dev" or item.get("strategy") != args.strategy:
                    continue
                precomputed_dense[item["id"]] = [
                    normalize_chunk(row) | {"score": row["score"], "method": "dense"}
                    for row in item.get("hits", [])
                ]
            source_by_id = {
                item["id"]: item for item in source_traces
                if item.get("split") == "dev" and item.get("strategy") == args.strategy
            }
            for question in questions:
                item = source_by_id.get(question["id"])
                if item is None:
                    if question["answerable"]:
                        raise ValueError(f"No recorded Day 6 dense trace for {question['id']}")
                    continue
                if item.get("question") != question["question"]:
                    raise ValueError(f"Recorded dense query differs for {question['id']}")
                if (item.get("filter") or {}) != (question.get("filter") or {}):
                    raise ValueError(f"Recorded dense filter differs for {question['id']}")
        else:
            if args.day6_starter is None:
                raise ValueError("Pass --day6-starter or --dense-traces for dense/hybrid")
            dense_started = time.perf_counter()
            dense = Day6Dense(args.day6_starter, args.strategy)
            dense_startup_seconds = time.perf_counter() - dense_started

    traces: list[dict[str, Any]] = []
    for question in questions:
        query = question["question"]
        filters = question.get("filter") or None
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "query_id": question["id"],
            "split": "dev",
            "question": query,
            "answerable": question["answerable"],
            "filters": filters or {},
            "method": args.method,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "budget_chars": args.budget_chars,
            "timing_source": (
                "recorded_day6_result"
                if args.method == "dense" and args.dense_traces is not None
                else "hybrid_fusion_with_recorded_day6_result"
                if args.method == "hybrid" and args.dense_traces is not None
                else "live"
            ),
        }
        if (
            args.method in ("dense", "hybrid") and dense is None and args.dense_traces is not None
            and question["id"] not in precomputed_dense
        ):
            trace.update({
                "status": "not_run",
                "error": "No recorded dense result for this question; unanswerable case is outside the Day 6 benchmark.",
                "elapsed_ms": None,
            })
            trace["absence_reason"] = question["absence_reason"]
            traces.append(trace)
            continue
        try:
            lexical = (
                lexical_search(index, query, chunks, args.candidate_k, filters)
                if index is not None else []
            )
            dense_results: list[dict[str, Any]] = []
            if dense is not None:
                query_vector = dense.query_vector(query)
                dense_results = dense.search(query_vector, args.candidate_k, filters)
            elif question["id"] in precomputed_dense:
                dense_results = precomputed_dense[question["id"]][:args.candidate_k]

            if args.method == "tfidf":
                results = lexical[:args.top_k]
            elif args.method == "dense":
                results = dense_results[:args.top_k]
            else:
                results = hybrid_search(lexical, dense_results, args.top_k)

            context, context_ids = pack_context(results, args.budget_chars)
            gold = question.get("gold", []) if question["answerable"] else []
            marked = [{**item, "evidence_hit": evidence_hit(item, gold)} for item in results]
            trace.update({
                "status": "ok",
                "results": marked,
                "context": context,
                "context_ids": context_ids,
                "context_chars": len(context),
                "hit_at_1": bool(marked and marked[0]["evidence_hit"]),
                "evidence_hit_at_3": any(item["evidence_hit"] for item in marked[:3]),
                "context_evidence_hit": any(
                    item["evidence_hit"] and item["id"] in context_ids for item in marked
                ),
            })
            if args.method == "hybrid":
                trace["component_ranks"] = {
                    "tfidf": [{"id": row["id"], "score": row["score"]} for row in lexical],
                    "dense": [{"id": row["id"], "score": row["score"]} for row in dense_results],
                }
        except Exception as exc:  # preserve retrieval failures in the experiment trace
            trace.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        trace["elapsed_ms"] = (
            None if args.method == "dense" and args.dense_traces is not None
            else (time.perf_counter() - started) * 1000
        )
        # Gold stays in the offline evaluation trace and is never passed to search.
        if question.get("answerable"):
            trace["gold"] = question["gold"]
        else:
            trace["absence_reason"] = question["absence_reason"]
        traces.append(trace)

    answerable = [item for item in traces if item["answerable"]]
    successful = [item for item in answerable if item["status"] == "ok"]
    hit_at_1_n = sum(item.get("hit_at_1", False) for item in successful)
    evidence_hit_at_3_n = sum(item.get("evidence_hit_at_3", False) for item in successful)
    context_evidence_n = sum(item.get("context_evidence_hit", False) for item in successful)
    elapsed_values = [item["elapsed_ms"] for item in traces if isinstance(item.get("elapsed_ms"), (int, float))]
    summary = {
        "method": args.method,
        "split": "dev",
        "strategy": args.strategy,
        "embedding_model": "BAAI/bge-m3" if dense is not None or (args.method in ("dense", "hybrid") and args.dense_traces is not None) else None,
        "vector_store": "PostgreSQL + pgvector (Day 6)" if dense is not None or (args.method in ("dense", "hybrid") and args.dense_traces is not None) else None,
        "dense_source": (
            str(args.dense_traces) if args.dense_traces is not None else "live Day 6 pgvector"
        ) if args.method in ("dense", "hybrid") else None,
        "rrf_constant": RRF_CONSTANT if args.method == "hybrid" else None,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "budget_chars": args.budget_chars,
        "questions_n": len(traces),
        "answerable_n": len(answerable),
        "errors_n": sum(item["status"] == "error" for item in traces),
        "hit_at_1": hit_at_1_n,
        "hit_at_1_rate": hit_at_1_n / len(answerable) if answerable else 0,
        "evidence_hit_at_3": evidence_hit_at_3_n,
        "evidence_hit_at_3_rate": evidence_hit_at_3_n / len(answerable) if answerable else 0,
        "context_evidence_n": context_evidence_n,
        "context_evidence_rate": context_evidence_n / len(answerable) if answerable else 0,
        "metrics_denominator": len(answerable),
        "successful_answerable_n": len(successful),
        "average_context_chars": statistics.mean(
            [item.get("context_chars", 0) for item in traces if item["status"] == "ok"]
        ) if any(item["status"] == "ok" for item in traces) else 0,
        "median_query_ms": statistics.median(elapsed_values) if elapsed_values else None,
        "tfidf_index_seconds": tfidf_index_seconds,
        "dense_startup_seconds": dense_startup_seconds,
        "service_cost_usd": 0,
        "timing_scope": (
            "Cached dense replay excludes the original BGE-M3 embedding and pgvector query time; hybrid time excludes those same dense retrieval costs."
            if args.dense_traces is not None and args.method in ("dense", "hybrid")
            else "Per-query wall time includes retrieval and context packing."
        ),
        "notes": "Local inference and local pgvector; service cost excludes hardware/electricity.",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.method}-traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / f"{args.method}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=("all", "tfidf", "dense", "hybrid"), default="all")
    parser.add_argument("--strategy", choices=("A", "B", "C"), default=DEFAULT_STRATEGY)
    parser.add_argument("--day6-starter", type=Path, help="Path to Day 6 starter directory (contains src/ and day6_config.py)")
    parser.add_argument("--dense-traces", type=Path, help="Reuse recorded Day 6 traces for dev if pgvector is offline")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--budget-chars", type=int, default=3000)
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k or args.budget_chars < 1:
        parser.error("Require top-k >= 1, candidate-k >= top-k, and budget-chars >= 1")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
