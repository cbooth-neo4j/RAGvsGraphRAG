"""
HotpotQA Benchmark Pipeline

Main orchestrator for running complete HotpotQA benchmarks with RAGAS evaluation.
Integrates data loading, graph ingestion, retriever evaluation, and reporting.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.graph_rag_logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def run_hotpotqa_benchmark(
    preset: str = "smoke",
    retrievers: Optional[List[str]] = None,
    build_database: bool = False,
    output_dir: str = "benchmark_outputs/hotpotqa",
    cache_dir: str = "data/hotpotqa"
) -> Dict[str, Any]:
    """
    Run complete HotpotQA benchmark with RAGAS evaluation.
    
    Pipeline steps:
    1. Download HotpotQA questions + Wikipedia articles
    2. Ingest articles into Neo4j (only if build_database=True)
    3. Run each retriever on all questions
    4. Evaluate with RAGAS metrics
    5. Generate comparison report
    
    Args:
        preset: Benchmark preset (smoke, dev, full, mini)
        retrievers: List of retrievers to test (overrides preset)
        build_database: If True, clear Neo4j and ingest Wikipedia articles
        output_dir: Directory for benchmark outputs
        cache_dir: Directory for caching downloaded data
        
    Returns:
        Complete benchmark results dictionary
    """
    from .configs import get_preset_config, print_preset_info
    from .data_loader import prepare_corpus, load_cached_corpus, save_corpus
    from .wiki_ingester import WikiCorpusIngester
    
    # Import RAGAS benchmark utilities
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ragas_benchmark import (
        collect_evaluation_data_simple,
        evaluate_with_ragas_simple,
        create_multi_approach_comparison_table,
        save_results_selective
    )
    from visualizations import create_visualizations
    
    # Get configuration
    config = get_preset_config(preset)
    print_preset_info(preset)
    
    # Override retrievers if specified
    if retrievers:
        config["retrievers"] = retrievers
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "preset": preset,
        "config": config,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phases": {}
    }
    
    # =========================================================================
    # PHASE 1: DATA PREPARATION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: DATA PREPARATION")
    print("="*70)
    
    phase1_start = time.time()
    
    # Try to load cached corpus first
    corpus = load_cached_corpus(cache_dir)
    
    if corpus and len(corpus.get("questions", [])) >= (config["question_limit"] or 0):
        print("[CACHE] Using cached corpus")
    else:
        # Prepare fresh corpus
        corpus = prepare_corpus(
            questions_split="dev",
            cache_dir=cache_dir,
            question_limit=config["question_limit"]
        )
        # Save for future use
        save_corpus(corpus, cache_dir)
    
    questions = corpus["questions"]
    articles = corpus["articles"]
    
    # Apply question limit if needed
    if config["question_limit"] and len(questions) > config["question_limit"]:
        questions = questions[:config["question_limit"]]
    
    results["phases"]["data_preparation"] = {
        "questions_loaded": len(questions),
        "articles_loaded": len(articles),
        "duration_seconds": time.time() - phase1_start
    }
    
    print(f"\n[PHASE 1 COMPLETE] {len(questions)} questions, {len(articles)} articles")
    
    # =========================================================================
    # PHASE 2: GRAPH INGESTION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2: GRAPH INGESTION")
    print("="*70)
    
    phase2_start = time.time()
    
    if build_database:
        print("[BUILD] Building database from Wikipedia articles...")
        print("        ⚠️  This will CLEAR the existing Neo4j database!")
        ingester = WikiCorpusIngester()
        try:
            ingestion_result = ingester.ingest_corpus(
                articles=articles,
                clear_db=True,
                run_advanced_processing=True,
                domain_hint="qa"
            )
            results["phases"]["ingestion"] = {
                "skipped": False,
                "chunks_created": ingestion_result.get("total_chunks_created", 0),
                "entities_created": ingestion_result.get("total_entities_created", 0),
                "duration_seconds": time.time() - phase2_start
            }
        finally:
            ingester.close()
    else:
        print("[TEST] Testing against existing graph data")
        print("       ℹ️  Use --build-database to rebuild from HotpotQA Wikipedia articles")
        results["phases"]["ingestion"] = {
            "skipped": True,
            "reason": "build_database not specified (test-only mode)"
        }
    
    print(f"\n[PHASE 2 COMPLETE] Duration: {time.time() - phase2_start:.1f}s")
    
    # =========================================================================
    # PHASE 3: RETRIEVER EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 3: RETRIEVER EVALUATION")
    print("="*70)
    
    phase3_start = time.time()
    
    # Convert HotpotQA questions to benchmark format
    benchmark_data = []
    for q in questions:
        benchmark_data.append({
            "question": q["question"],
            "ground_truth": q["answer"],
            "question_id": q["id"],
            "question_type": q["type"],
            "question_level": q["level"]
        })
    
    # Collect evaluation data for each retriever
    datasets = {}
    for retriever in config["retrievers"]:
        print(f"\n[EVAL] Collecting data for {retriever}...")
        try:
            datasets[retriever] = collect_evaluation_data_simple(
                benchmark_data, 
                approach=retriever
            )
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {retriever}: {e}")
            datasets[retriever] = []
    
    results["phases"]["evaluation_collection"] = {
        "retrievers_tested": list(datasets.keys()),
        "questions_per_retriever": {k: len(v) for k, v in datasets.items()},
        "duration_seconds": time.time() - phase3_start
    }
    
    print(f"\n[PHASE 3 COMPLETE] Evaluated {len(datasets)} retrievers")
    
    # =========================================================================
    # PHASE 4: RAGAS EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 4: RAGAS EVALUATION")
    print("="*70)
    
    phase4_start = time.time()
    
    approach_names = {
        'chroma': 'ChromaDB RAG',
        'graphrag': 'GraphRAG',
        'text2cypher': 'Text2Cypher',
        'advanced_graphrag': 'Advanced GraphRAG',
        'drift_graphrag': 'DRIFT GraphRAG',
        'neo4j_vector': 'Neo4j Vector RAG',
        'hybrid_cypher': 'Hybrid Cypher RAG'
    }
    
    ragas_results = {}
    for retriever in config["retrievers"]:
        if datasets.get(retriever):
            print(f"\n[RAGAS] Evaluating {retriever}...")
            try:
                ragas_results[retriever] = evaluate_with_ragas_simple(
                    datasets[retriever],
                    approach_names.get(retriever, retriever)
                )
            except Exception as e:
                print(f"[ERROR] RAGAS evaluation failed for {retriever}: {e}")
                ragas_results[retriever] = {
                    "response_relevancy": 0.0,
                    "factual_correctness": 0.0,
                    "semantic_similarity": 0.0,
                    "error": str(e)
                }
    
    results["phases"]["ragas_evaluation"] = {
        "results": ragas_results,
        "duration_seconds": time.time() - phase4_start
    }
    
    print(f"\n[PHASE 4 COMPLETE] RAGAS evaluation done")
    
    # =========================================================================
    # PHASE 5: REPORTING
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 5: GENERATING REPORTS")
    print("="*70)
    
    phase5_start = time.time()
    
    # Create comparison table
    comparison_table = create_multi_approach_comparison_table(
        ragas_results, 
        approach_names
    )
    
    # Print results
    print("\n" + "="*80)
    print("HOTPOTQA BENCHMARK RESULTS")
    print("="*80)
    print(comparison_table.to_string(index=False))
    
    # Calculate overall scores
    print("\n OVERALL PERFORMANCE:")
    print("-"*50)
    averages = {}
    for col in comparison_table.columns:
        if col != 'Metric':
            try:
                avg = comparison_table[col].mean()
                averages[col] = avg
                print(f"   {col}: {avg:.4f}")
            except:
                pass
    
    if averages:
        winner = max(averages, key=averages.get)
        print(f"\n   Best Overall: {winner} ({averages[winner]:.4f})")
    
    # Save results
    save_results_selective(
        datasets=datasets,
        results=ragas_results,
        comparison_table=comparison_table,
        approaches=config["retrievers"],
        output_dir=output_dir
    )
    
    # Create visualizations
    try:
        create_visualizations(comparison_table, output_dir=output_dir)
    except Exception as e:
        print(f"[WARN] Visualization creation failed: {e}")
    
    # Save comprehensive results JSON
    results["phases"]["reporting"] = {
        "comparison_table": comparison_table.to_dict(),
        "averages": averages,
        "winner": winner if averages else None,
        "duration_seconds": time.time() - phase5_start
    }
    
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["total_duration_seconds"] = sum(
        p.get("duration_seconds", 0) for p in results["phases"].values()
    )
    
    # Save full results
    results_file = os.path.join(output_dir, "hotpotqa_benchmark_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*70}")
    print(f"   Total duration: {results['total_duration_seconds']:.1f} seconds")
    print(f"   Results saved to: {output_dir}/")
    print(f"{'='*70}\n")
    
    return results


def main():
    """CLI entry point for HotpotQA benchmark."""
    parser = argparse.ArgumentParser(
        description="HotpotQA Fullwiki Benchmark for RAG vs GraphRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test retrievers against existing graph (default - no database changes)
  python -m benchmark.hotpotqa.benchmark_pipeline smoke
  python -m benchmark.hotpotqa.benchmark_pipeline mini --retrievers graphrag neo4j_vector

  # Build database first, then test (CLEARS existing Neo4j data!)
  python -m benchmark.hotpotqa.benchmark_pipeline smoke --build-database
  python -m benchmark.hotpotqa.benchmark_pipeline dev --build-database --retrievers chroma graphrag

  # Full evaluation (all ~7400 questions)
  python -m benchmark.hotpotqa.benchmark_pipeline full --build-database

Note: Questions are from HotpotQA dataset. For best results, use --build-database 
      first to ingest the corresponding Wikipedia articles, then run tests without it.
        """
    )
    
    parser.add_argument(
        "preset",
        nargs="?",
        default="smoke",
        choices=["mini_smoke", "smoke", "dev", "full", "mini"],
        help="Benchmark preset (default: smoke)"
    )
    
    parser.add_argument(
        "--retrievers",
        nargs="+",
        help="Specific retrievers to test (overrides preset)"
    )
    
    parser.add_argument(
        "--build-database",
        action="store_true",
        help="Clear Neo4j and ingest Wikipedia articles before testing (default: test only)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs/hotpotqa",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="data/hotpotqa",
        help="Cache directory for downloaded data"
    )
    
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_presets:
        from .configs import list_presets
        list_presets()
        return
    
    # Run benchmark
    results = run_hotpotqa_benchmark(
        preset=args.preset,
        retrievers=args.retrievers,
        build_database=args.build_database,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    return results


if __name__ == "__main__":
    main()

