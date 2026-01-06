"""
HotpotQA Fullwiki Benchmark Module

This module provides a complete pipeline for benchmarking RAG systems
using the HotpotQA fullwiki dataset with Wikipedia articles.
"""

from .data_loader import (
    load_hotpotqa_fullwiki,
    extract_referenced_titles,
    download_wikipedia_articles,
    prepare_corpus
)
from .wiki_ingester import WikiCorpusIngester
from .benchmark_pipeline import run_hotpotqa_benchmark
from .configs import BENCHMARK_PRESETS, get_preset_config

__all__ = [
    # Data loading
    "load_hotpotqa_fullwiki",
    "extract_referenced_titles", 
    "download_wikipedia_articles",
    "prepare_corpus",
    # Ingestion
    "WikiCorpusIngester",
    # Benchmark
    "run_hotpotqa_benchmark",
    # Configuration
    "BENCHMARK_PRESETS",
    "get_preset_config"
]

__version__ = "1.0.0"

