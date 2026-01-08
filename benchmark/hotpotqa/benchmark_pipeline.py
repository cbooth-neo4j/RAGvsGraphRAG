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

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.graph_rag_logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def run_hotpotqa_benchmark(
    preset: str = "smoke",
    retrievers: Optional[List[str]] = None,
    build_database: bool = False,
    run_advanced: bool = True,
    output_dir: str = "benchmark_outputs/hotpotqa",
    cache_dir: str = "data/hotpotqa",
    include_ragas: bool = False
) -> Dict[str, Any]:
    """
    Run complete HotpotQA benchmark with native EM/F1 evaluation.
    
    Pipeline steps:
    1. Download HotpotQA questions + Wikipedia articles
    2. Ingest articles into Neo4j (only if build_database=True)
    3. Run each retriever on all questions
    4. Evaluate with HotpotQA metrics (EM/F1)
    5. Optionally evaluate with RAGAS metrics (if --ragas flag)
    6. Generate comparison report
    
    Args:
        preset: Benchmark preset (smoke, dev, full, mini)
        retrievers: List of retrievers to test (overrides preset)
        build_database: If True, clear Neo4j and ingest Wikipedia articles
        run_advanced: If True, run advanced processing (community detection + summarization)
        output_dir: Base directory for benchmark outputs (timestamped folder will be created inside)
        cache_dir: Directory for caching downloaded data
        include_ragas: If True, also run RAGAS metrics (slower, LLM-based evaluation)
        
    Returns:
        Complete benchmark results dictionary
    """
    from .configs import get_preset_config, print_preset_info
    from .data_loader import prepare_corpus, load_cached_corpus, save_corpus
    from .wiki_ingester import WikiCorpusIngester
    from .metrics import evaluate_retriever_hotpotqa
    
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
    
    # Create timestamped output directory for this run
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(output_dir, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    
    print(f"\n📁 Results will be saved to: {run_output_dir}/")
    
    results = {
        "preset": preset,
        "config": config,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_timestamp": run_timestamp,
        "output_directory": run_output_dir,
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
    
    # Check for ingestion state in Neo4j (supports both completed and partial ingestion)
    from neo4j import GraphDatabase
    
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    neo4j_db = os.environ.get('CLIENT_NEO4J_DATABASE', 'neo4j')
    
    ingested_articles = None
    ingestion_source = None
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session(database=neo4j_db) as session:
            # Priority 1: Check for __TestableQuestions__ node in Neo4j (database-specific)
            result = session.run("""
                MATCH (t:__TestableQuestions__ {id: 'current'})
                RETURN t.question_ids as question_ids,
                       t.testable_count as testable_count,
                       t.computed_at as computed_at
            """)
            record = result.single()
            if record and record['question_ids']:
                testable_ids = set(record['question_ids'])
                print(f"[TESTABLE] Using pre-computed testable questions from Neo4j :__TestableQuestions__")
                print(f"           {len(testable_ids)} questions with complete article coverage")
                print(f"           Computed at: {record.get('computed_at', 'unknown')}")
                
                # Filter questions by testable IDs
                original_count = len(questions)
                questions = [q for q in questions if q.get('_id') in testable_ids]
                print(f"[FILTER] Filtered questions from {original_count} to {len(questions)}")
                ingestion_source = "testable_neo4j"
            
            # Priority 2: Fall back to testable_questions.json file
            if not ingestion_source:
                testable_file = Path(cache_dir) / "testable_questions.json"
                if testable_file.exists():
                    try:
                        with open(testable_file, 'r', encoding='utf-8') as f:
                            testable_data = json.load(f)
                        testable_ids = set(testable_data.get('testable_question_ids', []))
                        if testable_ids:
                            print(f"[TESTABLE] Using pre-computed testable questions from {testable_file}")
                            print(f"           {len(testable_ids)} questions with complete article coverage")
                            
                            # Filter questions by testable IDs
                            original_count = len(questions)
                            questions = [q for q in questions if q.get('_id') in testable_ids]
                            print(f"[FILTER] Filtered questions from {original_count} to {len(questions)}")
                            ingestion_source = "testable_file"
                    except Exception as e:
                        print(f"[WARNING] Could not read testable_questions.json: {e}")
            
            # Priority 3: Check __IngestionManifest__ (completed ingestion)
            if not ingestion_source:
                result = session.run("""
                    MATCH (m:__IngestionManifest__ {id: 'current'})
                    RETURN m.source as source, m.ingested_at as ingested_at,
                           m.questions_count as questions_count, m.articles_count as articles_count,
                           m.article_titles as article_titles, m.question_ids as question_ids,
                           m.lean_mode as lean_mode
                """)
                record = result.single()
                if record:
                    manifest = dict(record)
                    ingested_articles = set(manifest.get('article_titles', []) or [])
                    print(f"[MANIFEST] Found completed ingestion: {manifest.get('questions_count', '?')} questions, {manifest.get('articles_count', '?')} articles")
                    print(f"           Ingested at: {manifest.get('ingested_at', 'unknown')}")
                    print(f"           Mode: {'LEAN' if manifest.get('lean_mode') else 'FULL'}")
                    ingestion_source = "manifest"
            
            # Priority 4: Check __IngestionProgress__ (interrupted/partial ingestion)
            if not ingestion_source:
                result = session.run("""
                    MATCH (p:__IngestionProgress__ {id: 'current'})
                    RETURN p.processed_titles as processed_titles,
                           p.total_expected as total_expected,
                           p.questions_count as questions_count,
                           p.last_updated as last_updated
                """)
                record = result.single()
                if record:
                    ingested_articles = set(record['processed_titles'] or [])
                    print(f"[PROGRESS] Found partial ingestion: {len(ingested_articles)}/{record.get('total_expected', '?')} articles")
                    print(f"           Last updated: {record.get('last_updated', 'unknown')}")
                    print(f"           Note: Ingestion may still be in progress or was interrupted")
                    ingestion_source = "progress"
            
            # Priority 5: Fall back to scanning Document nodes
            if not ingestion_source:
                result = session.run("""
                    MATCH (d:Document)
                    WHERE d.name STARTS WITH 'wiki_'
                    WITH d.name as name
                    WITH split(name, '_') as parts
                    WITH parts[1..size(parts)-1] as title_parts
                    WITH reduce(s = '', x IN title_parts | s + CASE WHEN s = '' THEN '' ELSE '_' END + x) as title
                    RETURN collect(DISTINCT title) as titles
                """)
                record = result.single()
                if record and record['titles']:
                    ingested_articles = set(record['titles'])
                    print(f"[SCAN] Scanned database directly: found {len(ingested_articles)} articles")
                    ingestion_source = "scan"
        
        driver.close()
    except Exception as e:
        print(f"[WARNING] Could not connect to Neo4j: {e}")
    
    # Filter questions based on ingested articles (skip if already filtered by testable questions)
    if ingested_articles and ingestion_source not in ("testable_neo4j", "testable_file"):
        original_count = len(questions)
        filtered_questions = []
        for q in questions:
            # Get supporting article titles for this question
            supporting = set()
            for sf in q.get('supporting_facts', []):
                if isinstance(sf, (list, tuple)) and len(sf) >= 1:
                    supporting.add(sf[0])
            
            # Only include if ALL supporting articles were ingested
            if supporting and supporting.issubset(ingested_articles):
                filtered_questions.append(q)
        
        questions = filtered_questions
        print(f"[FILTER] Filtered questions from {original_count} to {len(questions)} (matching ingested articles)")
        
        if len(questions) == 0:
            print("\n[ERROR] No questions have complete article coverage!")
            print("        Either:")
            print("        1. Wait for ingestion to complete, or")
            print("        2. Run: python ingest.py --check-testable --quantity N")
            print("           to see which questions can be tested")
            return {"error": "No matching questions", "phases": {}}
    elif not ingested_articles:
        print("[WARNING] No ingestion state found - testing may include questions without matching articles")
        print("          Run: python ingest.py --source hotpotqa --quantity N --lean --new")
    
    # Apply question limit if needed (after manifest filtering)
    if config["question_limit"] and len(questions) > config["question_limit"]:
        questions = questions[:config["question_limit"]]
    
    # Always ensure articles match the questions being used
    # Extract titles referenced by current questions (supporting facts + context/distractors)
    referenced_titles = set()
    supporting_titles = set()  # Just the gold supporting facts
    for q in questions:
        # Extract from context (includes distractors)
        for ctx in q.get("context", []):
            if isinstance(ctx, (list, tuple)) and len(ctx) >= 1:
                referenced_titles.add(ctx[0])
        # Extract from supporting facts (gold articles)
        for sf in q.get("supporting_facts", []):
            if isinstance(sf, (list, tuple)) and len(sf) >= 1:
                referenced_titles.add(sf[0])
                supporting_titles.add(sf[0])
    
    # Filter articles to only include referenced ones
    original_article_count = len(articles)
    articles = [a for a in articles if a.get("title") in referenced_titles]
    
    if len(articles) != original_article_count:
        print(f"[FILTER] Filtered articles from {original_article_count} to {len(articles)} (matching {len(questions)} questions)")
    
    print(f"[INFO] Article breakdown: {len(supporting_titles)} gold supporting + {len(referenced_titles) - len(supporting_titles)} distractors")
    
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
        if run_advanced:
            print("        📊 Advanced processing enabled (community detection + summarization)")
        else:
            print("        ⏭️  Advanced processing SKIPPED (use without --skip-advanced to enable)")
        ingester = WikiCorpusIngester()
        try:
            ingestion_result = ingester.ingest_corpus(
                articles=articles,
                clear_db=True,
                run_advanced_processing=run_advanced,
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
        print("       ℹ️  To build graph, run: python ingest.py --source hotpotqa --quantity N --lean")
        results["phases"]["ingestion"] = {
            "skipped": True,
            "reason": "test-only mode (use ingest.py to build)"
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
    # HotpotQA uses short factoid answers - use "hotpotqa" answer style for EM/F1 metrics
    # RAGAS can handle verbose answers - use "ragas" answer style when include_ragas is True
    answer_style = "ragas" if include_ragas and not include_ragas else "hotpotqa"
    # Actually: HotpotQA benchmark always needs short answers for EM/F1, even if RAGAS is also enabled
    answer_style = "hotpotqa"
    
    datasets = {}
    for retriever in config["retrievers"]:
        print(f"\n[EVAL] Collecting data for {retriever}...")
        try:
            datasets[retriever] = collect_evaluation_data_simple(
                benchmark_data, 
                approach=retriever,
                answer_style=answer_style
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
    # PHASE 4: EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    metrics_label = "HotpotQA Metrics" + (" + RAGAS" if include_ragas else "")
    print(f"PHASE 4: EVALUATION ({metrics_label})")
    print("="*70)
    
    phase4_start = time.time()
    
    approach_names = {
        'chroma': 'ChromaDB RAG',
        'graphrag': 'GraphRAG',
        'text2cypher': 'Text2Cypher',
        'agentic-text2cypher': 'Agentic Text2Cypher',
        'advanced-graphrag': 'Advanced GraphRAG',
        'drift-graphrag': 'DRIFT GraphRAG',
        'neo4j-vector': 'Neo4j Vector RAG',
        'hybrid-cypher': 'Hybrid Cypher RAG'
    }
    
    # ---- HotpotQA Native Metrics (EM/F1) - Always run ----
    print("\n[HOTPOTQA] Computing native metrics (Exact Match / F1)...")
    hotpotqa_results = {}
    for retriever in config["retrievers"]:
        if datasets.get(retriever):
            try:
                hotpotqa_results[retriever] = evaluate_retriever_hotpotqa(
                    datasets[retriever],
                    approach_names.get(retriever, retriever)
                )
            except Exception as e:
                print(f"[ERROR] HotpotQA evaluation failed for {retriever}: {e}")
                hotpotqa_results[retriever] = {
                    "exact_match": 0.0,
                    "f1": 0.0,
                    "error": str(e)
                }
    
    results["phases"]["hotpotqa_evaluation"] = {
        "results": hotpotqa_results,
        "duration_seconds": time.time() - phase4_start
    }
    
    # ---- RAGAS Metrics (Optional) ----
    ragas_results = {}
    if include_ragas:
        print("\n[RAGAS] Computing RAGAS metrics (--ragas flag enabled)...")
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
            "duration_seconds": time.time() - phase4_start - results["phases"]["hotpotqa_evaluation"]["duration_seconds"]
        }
    else:
        print("\n[INFO] RAGAS metrics skipped (use --ragas to enable)")
    
    print(f"\n[PHASE 4 COMPLETE] Evaluation done")
    
    # =========================================================================
    # PHASE 5: REPORTING
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 5: GENERATING REPORTS")
    print("="*70)
    
    phase5_start = time.time()
    
    # ---- Print HotpotQA Results First (Primary Metrics) ----
    print("\n" + "="*80)
    print("HOTPOTQA NATIVE METRICS (EM / F1)")
    print("="*80)
    
    # Build HotpotQA metrics table
    hotpot_rows = []
    for retriever in config["retrievers"]:
        if retriever in hotpotqa_results:
            hotpot_rows.append({
                "Retriever": approach_names.get(retriever, retriever),
                "Exact Match": hotpotqa_results[retriever].get("exact_match", 0.0),
                "F1 Score": hotpotqa_results[retriever].get("f1", 0.0)
            })
    
    if hotpot_rows:
        hotpot_df = pd.DataFrame(hotpot_rows)
        print(hotpot_df.to_string(index=False))
        
        # Print best performer for HotpotQA metrics
        print("\n HOTPOTQA PERFORMANCE:")
        print("-"*50)
        best_em = max(hotpot_rows, key=lambda x: x["Exact Match"])
        best_f1 = max(hotpot_rows, key=lambda x: x["F1 Score"])
        print(f"   Best EM:  {best_em['Retriever']} ({best_em['Exact Match']:.4f})")
        print(f"   Best F1:  {best_f1['Retriever']} ({best_f1['F1 Score']:.4f})")
    
    # ---- Print RAGAS Results (only if enabled) ----
    comparison_table = None
    averages = {}
    
    if include_ragas and ragas_results:
        print("\n" + "="*80)
        print("RAGAS METRICS")
        print("="*80)
        
        # Create comparison table
        comparison_table = create_multi_approach_comparison_table(
            ragas_results, 
            approach_names
        )
        print(comparison_table.to_string(index=False))
        
        # Calculate overall scores
        print("\n RAGAS AVERAGE PERFORMANCE:")
        print("-"*50)
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
            print(f"\n   Best RAGAS Average: {winner} ({averages[winner]:.4f})")
        
        # Save RAGAS results
        save_results_selective(
            datasets=datasets,
            results=ragas_results,
            comparison_table=comparison_table,
            approaches=config["retrievers"],
            output_dir=run_output_dir
        )
    
    # Flush stdout to prevent duplicate output on Windows
    sys.stdout.flush()
    
    # Save HotpotQA metrics
    hotpotqa_results_file = os.path.join(run_output_dir, "hotpotqa_metrics.json")
    with open(hotpotqa_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics_type": "hotpotqa_native",
            "description": "HotpotQA official evaluation metrics (Exact Match and F1)",
            "results": hotpotqa_results
        }, f, indent=2)
    print(f"  - hotpotqa_metrics.json")
    
    # Save results CSV
    result_rows = []
    for retriever in config["retrievers"]:
        row = {
            "retriever": approach_names.get(retriever, retriever),
            "exact_match": hotpotqa_results.get(retriever, {}).get("exact_match", 0.0),
            "f1": hotpotqa_results.get(retriever, {}).get("f1", 0.0),
        }
        # Add RAGAS metrics only if enabled
        if include_ragas:
            row["response_relevancy"] = ragas_results.get(retriever, {}).get("response_relevancy", 0.0)
            row["factual_correctness"] = ragas_results.get(retriever, {}).get("factual_correctness", 0.0)
            row["semantic_similarity"] = ragas_results.get(retriever, {}).get("semantic_similarity", 0.0)
        result_rows.append(row)
    
    if result_rows:
        results_df = pd.DataFrame(result_rows)
        results_csv = os.path.join(run_output_dir, "benchmark_results.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"  - benchmark_results.csv")
    
    # Create visualizations
    try:
        from .visualizations import create_hotpotqa_visualizations
        create_hotpotqa_visualizations(
            hotpotqa_results=hotpotqa_results,
            approach_names=approach_names,
            output_dir=run_output_dir,
            ragas_results=ragas_results if include_ragas else None
        )
    except Exception as e:
        print(f"[WARN] Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save comprehensive results JSON
    results["phases"]["reporting"] = {
        "hotpotqa_results": hotpotqa_results,
        "ragas_comparison_table": comparison_table.to_dict() if comparison_table is not None else None,
        "ragas_averages": averages if averages else None,
        "duration_seconds": time.time() - phase5_start
    }
    
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["total_duration_seconds"] = sum(
        p.get("duration_seconds", 0) for p in results["phases"].values()
    )
    
    # Save full results
    results_file = os.path.join(run_output_dir, "hotpotqa_benchmark_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*70}")
    print(f"   Total duration: {results['total_duration_seconds']:.1f} seconds")
    print(f"   Results saved to: {run_output_dir}/")
    print(f"{'='*70}\n")
    
    return results


def main():
    """CLI entry point for HotpotQA benchmark."""
    parser = argparse.ArgumentParser(
        description="HotpotQA Fullwiki Benchmark for RAG vs GraphRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with HotpotQA metrics (EM/F1) - default
  python -m benchmark.hotpotqa.benchmark_pipeline micro --agentic-text2cypher
  
  # Mini benchmark (10 questions)
  python -m benchmark.hotpotqa.benchmark_pipeline mini --graphrag --neo4j-vector

  # Include RAGAS metrics (slower, LLM-based)
  python -m benchmark.hotpotqa.benchmark_pipeline mini --agentic-text2cypher --ragas

  # Build database first, then test (CLEARS existing Neo4j data!)
  python -m benchmark.hotpotqa.benchmark_pipeline smoke --build-database --chroma

  # Full evaluation (all ~7400 questions)
  python -m benchmark.hotpotqa.benchmark_pipeline full --build-database

Metrics:
  Default: HotpotQA native metrics (Exact Match + F1) - fast, deterministic
  --ragas: Also include RAGAS metrics (Response Relevancy, Factual Correctness, 
           Semantic Similarity) - slower, uses LLM for evaluation

Note: HotpotQA uses short factoid answers. EM/F1 metrics are recommended.
      Use --ragas only if you need LLM-based semantic evaluation.
        """
    )
    
    parser.add_argument(
        "preset",
        nargs="?",
        default="smoke",
        choices=["micro", "mini", "mini_smoke", "smoke", "dev", "full"],
        help="Benchmark preset (default: smoke)"
    )
    
    # Individual retriever flags (consistent with ragas_benchmark.py)
    parser.add_argument(
        "--chroma",
        action="store_true",
        help="Include ChromaDB RAG in testing"
    )
    parser.add_argument(
        "--graphrag",
        action="store_true",
        help="Include GraphRAG in testing"
    )
    parser.add_argument(
        "--text2cypher",
        action="store_true",
        help="Include Text2Cypher in testing"
    )
    parser.add_argument(
        "--advanced-graphrag",
        action="store_true",
        help="Include Advanced GraphRAG (intelligent global/local/hybrid) in testing"
    )
    parser.add_argument(
        "--drift-graphrag",
        action="store_true",
        help="Include DRIFT GraphRAG (iterative refinement) in testing"
    )
    parser.add_argument(
        "--neo4j-vector",
        action="store_true",
        help="Include Neo4j Vector RAG (pure vector similarity) in testing"
    )
    parser.add_argument(
        "--hybrid-cypher",
        action="store_true",
        help="Include Hybrid Cypher RAG (hybrid + generic neighborhood) in testing"
    )
    parser.add_argument(
        "--agentic-text2cypher",
        action="store_true",
        help="Include Agentic Text2Cypher (Deep Agent-powered graph exploration) in testing"
    )
    
    # NOTE: Database building removed. Use ingest.py instead:
    #   python ingest.py --source hotpotqa --quantity 100 --lean
    # Then run this benchmark to evaluate.
    
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
        "--ragas",
        action="store_true",
        help="Also run RAGAS metrics (slower, LLM-based evaluation)"
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
    
    # Build retrievers list from individual flags
    retrievers = []
    if args.chroma:
        retrievers.append('chroma')
    if args.graphrag:
        retrievers.append('graphrag')
    if args.text2cypher:
        retrievers.append('text2cypher')
    if getattr(args, 'advanced_graphrag', False):
        retrievers.append('advanced-graphrag')
    if getattr(args, 'drift_graphrag', False):
        retrievers.append('drift-graphrag')
    if getattr(args, 'neo4j_vector', False):
        retrievers.append('neo4j-vector')
    if getattr(args, 'hybrid_cypher', False):
        retrievers.append('hybrid-cypher')
    if getattr(args, 'agentic_text2cypher', False):
        retrievers.append('agentic-text2cypher')
    
    # If no retrievers specified, use preset defaults (None lets run_hotpotqa_benchmark use preset config)
    if not retrievers:
        retrievers = None
    
    # Run benchmark (test-only, use ingest.py to build database first)
    results = run_hotpotqa_benchmark(
        preset=args.preset,
        retrievers=retrievers,
        build_database=False,  # Always false - use ingest.py to build
        run_advanced=False,    # Not applicable for test-only mode
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        include_ragas=args.ragas
    )
    
    return results


if __name__ == "__main__":
    main()

