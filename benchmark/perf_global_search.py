import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List


def _load_queries(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(q, str) for q in data):
        raise ValueError(f"Expected a JSON list[str] in {path}")
    return data


async def _run_query_once(
    query: str,
    *,
    cold_start: bool,
    top_k_communities: int,
    strategy: str,
    shared_retriever: Any = None,
    extra_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    from retrievers.advanced_graphrag_retriever import LightweightAdvancedGlobalOnlyRetriever

    extra_kwargs = extra_kwargs or {}
    init_ms = 0.0

    if cold_start:
        t_init0 = time.time()
        retriever = LightweightAdvancedGlobalOnlyRetriever()
        t_init1 = time.time()
        init_ms = (t_init1 - t_init0) * 1000.0
    else:
        retriever = shared_retriever
        if retriever is None:
            raise ValueError("shared_retriever is required when cold_start=false")

    t_q0 = time.time()
    result = await retriever.global_search_query(
        query,
        top_k_communities=top_k_communities,
        strategy=strategy,
        **extra_kwargs,
    )
    t_q1 = time.time()

    if cold_start:
        retriever.close()

    perf = result.get("performance_metrics", {}) if isinstance(result, dict) else {}
    final_answer = result.get("final_answer") if isinstance(result, dict) else None
    success = isinstance(final_answer, str) and not final_answer.lower().startswith("error")

    return {
        "query": query,
        "strategy": strategy,
        "cold_start": cold_start,
        "top_k_communities": top_k_communities,
        "success": success,
        "final_answer_preview": (final_answer[:200] + "...") if isinstance(final_answer, str) and len(final_answer) > 200 else final_answer,
        "init_ms": init_ms,
        "query_ms": (t_q1 - t_q0) * 1000.0,
        "total_ms": init_ms + (t_q1 - t_q0) * 1000.0,
        "neo4j_ms": perf.get("neo4j_ms"),
        "context_ms": perf.get("context_ms"),
        "llm_ms": perf.get("llm_ms"),
        "communities_used": perf.get("communities_used"),
        "llm_calls": perf.get("llm_calls"),
        "prompt_tokens": perf.get("prompt_tokens"),
        "output_tokens": perf.get("output_tokens"),
        "total_tokens": perf.get("total_tokens"),
    }


async def main_async(args: argparse.Namespace) -> int:
    queries = _load_queries(args.queries)
    if args.limit and args.limit > 0:
        queries = queries[: args.limit]

    rows: List[Dict[str, Any]] = []

    shared_retriever = None
    if not args.cold_start:
        from retrievers.advanced_graphrag_retriever import LightweightAdvancedGlobalOnlyRetriever

        t_init0 = time.time()
        shared_retriever = LightweightAdvancedGlobalOnlyRetriever()
        t_init1 = time.time()
        rows.append(
            {
                "event": "warm_retriever_init",
                "init_ms": (t_init1 - t_init0) * 1000.0,
            }
        )

    try:
        for run_idx in range(args.runs):
            for q in queries:
                row = await _run_query_once(
                    q,
                    cold_start=args.cold_start,
                    top_k_communities=args.top_k_communities,
                    strategy=args.strategy,
                    shared_retriever=shared_retriever,
                )
                row["run_idx"] = run_idx
                rows.append(row)
    finally:
        if shared_retriever is not None:
            shared_retriever.close()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"global_perf_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote results to: {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark Advanced GraphRAG global search performance.")
    p.add_argument("--queries", default="benchmark/queries/global.json", help="Path to JSON list of queries.")
    p.add_argument("--runs", type=int, default=1, help="Number of repeats.")
    p.add_argument("--limit", type=int, default=0, help="Optional limit on number of queries.")
    p.add_argument(
        "--cold-start",
        action="store_true",
        help="If set, create a fresh retriever per query (simulates serverless cold starts).",
    )
    p.add_argument("--top-k-communities", type=int, default=8, help="Top-K communities to include.")
    p.add_argument(
        "--strategy",
        default="single_pass",
        choices=["single_pass", "map_reduce"],
        help="Global search strategy to benchmark.",
    )
    p.add_argument("--out-dir", default="benchmark/results", help="Output directory for JSON results.")

    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())


