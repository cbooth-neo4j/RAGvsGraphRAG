"""
Benchmark package for RAG vs GraphRAG comparison

This package contains modules for benchmarking different RAG approaches
using the RAGAS evaluation framework.

Submodules:
- hotpotqa: HotpotQA fullwiki benchmark with Wikipedia corpus
- ragas_benchmark: Core RAGAS evaluation utilities
- visualizations: Benchmark visualization tools
"""

from .ragas_benchmark import (
    load_benchmark_data,
    load_benchmark_data_jsonl,
    collect_evaluation_data_simple,
    evaluate_with_ragas_simple,
    create_multi_approach_comparison_table
)

# HotpotQA benchmark exports
try:
    from .hotpotqa import (
        run_hotpotqa_benchmark,
        prepare_corpus,
        WikiCorpusIngester,
        BENCHMARK_PRESETS
    )
    HOTPOTQA_AVAILABLE = True
except ImportError:
    HOTPOTQA_AVAILABLE = False

__version__ = "1.0.0" 

__all__ = [
    # Core benchmark
    "load_benchmark_data",
    "load_benchmark_data_jsonl",
    "collect_evaluation_data_simple",
    "evaluate_with_ragas_simple",
    "create_multi_approach_comparison_table",
    # HotpotQA
    "run_hotpotqa_benchmark",
    "prepare_corpus",
    "WikiCorpusIngester",
    "BENCHMARK_PRESETS",
    "HOTPOTQA_AVAILABLE"
] 