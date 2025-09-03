"""
RAGBench Integration Module

This module provides tools to ingest RAGBench dataset documents into Neo4j graphs
and create evaluation datasets for ragas_benchmark.py
"""

from .simple_ingester import SimpleRAGBenchIngester as RAGBenchIngester
from .evaluator import RAGBenchEvaluator  
from .configs import INGESTION_PRESETS

__all__ = ['RAGBenchIngester', 'RAGBenchEvaluator', 'INGESTION_PRESETS']
