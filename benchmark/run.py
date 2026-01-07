"""
Unified Benchmark Runner

Usage: python -m benchmark [preset] --<metrics> --<retriever> [options]

Examples:
  # HotpotQA metrics (fast, deterministic)
  python -m benchmark micro --hotpotqa --agentic-text2cypher
  
  # RAGAS metrics (LLM-based)
  python -m benchmark mini --ragas --agentic-text2cypher
  
  # Both metric types
  python -m benchmark mini --all-metrics --agentic-text2cypher
  
  # Compare multiple retrievers
  python -m benchmark smoke --hotpotqa --chroma --graphrag
"""

import os
import sys
import argparse
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="RAG Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark micro --hotpotqa --agentic-text2cypher
  python -m benchmark mini --ragas --chroma --graphrag
  python -m benchmark smoke --all-metrics --agentic-text2cypher

Presets:
  micro  - 1 question (sanity check)
  mini   - 10 questions (development)
  smoke  - 50 questions (standard)
  dev    - 200 questions (thorough)
  full   - All questions (complete)

Metrics (required - choose one):
  --hotpotqa    Exact Match + F1 (fast, deterministic)
  --ragas       LLM-based semantic evaluation (slower)
  --all-metrics Both HotpotQA and RAGAS
        """
    )
    
    # ===== PRESET (positional) =====
    parser.add_argument(
        "preset",
        nargs="?",
        default="mini",
        choices=["micro", "mini", "smoke", "dev", "full"],
        help="Question count preset (default: mini)"
    )
    
    # ===== DATASET =====
    parser.add_argument(
        "--dataset", "-d",
        default="hotpotqa",
        choices=["hotpotqa", "pdfs", "custom"],
        help="Dataset source (default: hotpotqa)"
    )
    
    # ===== METRICS (required) =====
    metrics_group = parser.add_mutually_exclusive_group(required=True)
    metrics_group.add_argument(
        "--hotpotqa", "--em-f1",
        action="store_true",
        dest="use_hotpotqa",
        help="Use HotpotQA metrics (Exact Match + F1) - fast, deterministic"
    )
    metrics_group.add_argument(
        "--ragas",
        action="store_true",
        dest="use_ragas",
        help="Use RAGAS metrics (LLM-based semantic evaluation) - slower"
    )
    metrics_group.add_argument(
        "--all-metrics",
        action="store_true",
        help="Use both HotpotQA and RAGAS metrics"
    )
    
    # ===== RETRIEVERS =====
    retriever_group = parser.add_argument_group("Retrievers", "Select one or more retrievers to benchmark")
    retriever_group.add_argument("--chroma", action="store_true", help="ChromaDB RAG")
    retriever_group.add_argument("--graphrag", action="store_true", help="GraphRAG")
    retriever_group.add_argument("--text2cypher", action="store_true", help="Text2Cypher")
    retriever_group.add_argument("--agentic-text2cypher", action="store_true", help="Agentic Text2Cypher (Deep Agent)")
    retriever_group.add_argument("--advanced-graphrag", action="store_true", help="Advanced GraphRAG")
    retriever_group.add_argument("--drift-graphrag", action="store_true", help="DRIFT GraphRAG")
    retriever_group.add_argument("--neo4j-vector", action="store_true", help="Neo4j Vector RAG")
    retriever_group.add_argument("--hybrid-cypher", action="store_true", help="Hybrid Cypher RAG")
    
    # ===== OPTIONS =====
    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "--build-database",
        action="store_true",
        help="Rebuild database before testing (CLEARS existing data)"
    )
    options_group.add_argument(
        "--skip-advanced",
        action="store_true",
        help="Skip community detection when building database"
    )
    options_group.add_argument(
        "--output-dir",
        default="benchmark_outputs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Build retrievers list
    retrievers = []
    if args.chroma:
        retrievers.append('chroma')
    if args.graphrag:
        retrievers.append('graphrag')
    if args.text2cypher:
        retrievers.append('text2cypher')
    if getattr(args, 'agentic_text2cypher', False):
        retrievers.append('agentic-text2cypher')
    if getattr(args, 'advanced_graphrag', False):
        retrievers.append('advanced-graphrag')
    if getattr(args, 'drift_graphrag', False):
        retrievers.append('drift-graphrag')
    if getattr(args, 'neo4j_vector', False):
        retrievers.append('neo4j-vector')
    if getattr(args, 'hybrid_cypher', False):
        retrievers.append('hybrid-cypher')
    
    if not retrievers:
        print("ERROR: No retrievers specified. Use --chroma, --graphrag, --agentic-text2cypher, etc.")
        parser.print_help()
        sys.exit(1)
    
    # Determine metrics
    include_ragas = args.use_ragas or args.all_metrics
    include_hotpotqa = args.use_hotpotqa or args.all_metrics
    
    metrics_str = []
    if include_hotpotqa:
        metrics_str.append("HotpotQA (EM/F1)")
    if include_ragas:
        metrics_str.append("RAGAS")
    
    # Route to appropriate benchmark runner based on dataset
    print(f"\n{'='*60}")
    print("BENCHMARK CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Preset:     {args.preset}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Retrievers: {', '.join(retrievers)}")
    print(f"  Metrics:    {' + '.join(metrics_str)}")
    print(f"{'='*60}\n")
    
    if args.dataset == "hotpotqa":
        from benchmark.hotpotqa.benchmark_pipeline import run_hotpotqa_benchmark
        
        results = run_hotpotqa_benchmark(
            preset=args.preset,
            retrievers=retrievers,
            build_database=args.build_database,
            run_advanced=not args.skip_advanced,
            output_dir=f"{args.output_dir}/hotpotqa",
            include_ragas=include_ragas
        )
        
    elif args.dataset == "pdfs":
        # PDF benchmark uses RAGAS metrics (better for longer answers)
        print("NOTE: PDF benchmarks use RAGAS metrics (better for longer answers)")
        from benchmark.ragas_benchmark import run_benchmark
        
        results = run_benchmark(
            approaches=retrievers,
            output_dir=f"{args.output_dir}/pdfs"
        )
        
    elif args.dataset == "custom":
        from benchmark.ragas_benchmark import run_benchmark
        
        results = run_benchmark(
            approaches=retrievers,
            csv_path="benchmark/benchmark.csv",
            output_dir=f"{args.output_dir}/custom"
        )
    
    return results


if __name__ == "__main__":
    main()

