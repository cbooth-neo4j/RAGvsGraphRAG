"""
RAGBench Pipeline

Complete pipeline to:
1. Ingest RAGBench documents into Neo4j graph
2. Create evaluation datasets 
3. Run RAGAS benchmark comparisons
4. Generate detailed human-readable reports

Usage:
    python benchmark/ragbench_pipeline.py micro
    python benchmark/ragbench_pipeline.py small --approaches chroma graphrag
    python benchmark/ragbench_pipeline.py medium --enhanced
"""

import argparse
import sys
from pathlib import Path

# Add benchmark directory to path
sys.path.append(str(Path(__file__).parent))

from ragbench import RAGBenchIngester, RAGBenchEvaluator, INGESTION_PRESETS


def run_full_pipeline(preset_name: str, 
                     approaches: list = None,
                     processor_type: str = None,
                     skip_ingestion: bool = False):
    """
    Run the complete RAGBench pipeline.
    
    Args:
        preset_name: Ingestion preset (micro, small, medium, large, full_test)
        approaches: List of retriever approaches to test
        processor_type: Override processor type (basic, advanced)
        skip_ingestion: Skip ingestion if graph already exists
    """
    
    if preset_name not in INGESTION_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(INGESTION_PRESETS.keys())}")
    
    config = INGESTION_PRESETS[preset_name]
    
    print(f"🚀 RAGBench Complete Pipeline: {preset_name}")
    print("=" * 60)
    print(f"📋 {config['description']}")
    print(f"💾 Estimated storage: {config['estimated_storage_gb']} GB")
    print(f"🧠 Estimated RAM: {config['estimated_ram_gb']} GB")
    print(f"💰 Estimated cost: ${config['estimated_cost_usd']}")
    
    # Use processor type from config unless overridden
    final_processor_type = processor_type or config['processor_type']
    
    # Default approaches if not specified
    if approaches is None:
        if preset_name == "nano":
            approaches = ["chroma", "graphrag"]  # Start ultra-simple
        elif preset_name == "micro":
            approaches = ["chroma", "graphrag"]  # Start simple
        elif preset_name in ["small", "medium"]:
            approaches = ["chroma", "graphrag", "advanced_graphrag"]
        else:
            approaches = ["chroma", "graphrag", "advanced_graphrag", "drift_graphrag"]
    
    print(f"🔧 Processor type: {final_processor_type}")
    print(f"🎯 Testing approaches: {approaches}")
    
    # Confirm for expensive operations
    if config['estimated_cost_usd'] > 100:
        confirm = input(f"\n⚠️  This operation may cost ~${config['estimated_cost_usd']}. Continue? [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            print("❌ Pipeline cancelled by user")
            return
    
    # Phase 1: Document Ingestion
    if not skip_ingestion:
        print(f"\n" + "="*60)
        print("PHASE 1: DOCUMENT INGESTION")
        print("="*60)
        
        ingester = RAGBenchIngester(processor_type=final_processor_type)
        
        try:
            ingestion_result = ingester.run_preset(preset_name)
            
            if ingestion_result["status"] != "completed":
                print(f"❌ Ingestion failed: {ingestion_result}")
                return
            
            print(f"✅ Ingestion completed successfully!")
            
            # Show ingestion statistics
            stats = ingestion_result["stats"]
            print(f"\n📊 Ingestion Statistics:")
            print(f"   Records processed: {stats['records_loaded']}")
            print(f"   Documents processed: {stats['documents_processed']}")
            print(f"   Chunks created: {stats['chunks_created']}")
            print(f"   Success rate: {stats['success_rate']:.1f}%")
            
            eval_data_path = ingestion_result["evaluation_data_path"]
            
        finally:
            ingester.close()
    else:
        print(f"\n⏭️  Skipping ingestion (using existing graph)")
        eval_data_path = f"benchmark/ragbench/data/{preset_name}_eval.jsonl"
        
        if not Path(eval_data_path).exists():
            print(f"❌ Evaluation data not found: {eval_data_path}")
            print("   Cannot skip ingestion without existing evaluation data")
            return
    
    # Phase 2: Create Benchmark Dataset
    print(f"\n" + "="*60)
    print("PHASE 2: BENCHMARK DATASET CREATION")
    print("="*60)
    
    evaluator = RAGBenchEvaluator(eval_data_path)
    evaluator.print_evaluation_summary()
    
    # Create benchmark JSONL
    benchmark_jsonl = evaluator.create_benchmark_jsonl(
        output_path=f"benchmark/ragbench__{preset_name}_benchmark.jsonl"
    )
    
    # Phase 3: Run RAGAS Evaluation
    print(f"\n" + "="*60)
    print("PHASE 3: RAGAS EVALUATION")
    print("="*60)
    
    # Import and run ragas_benchmark
    from ragas_benchmark import main_selective
    
    # Create timestamped output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = f"benchmark_outputs/{preset_name}/run_{timestamp}"
    
    # Run benchmark with specified approaches
    benchmark_results = main_selective(
        approaches=approaches,
        output_dir=timestamped_output_dir
    )
    
    # Phase 4: Final Summary
    print(f"\n" + "="*60)
    print("PIPELINE COMPLETE! 🎉")
    print("="*60)
    
    print(f"📁 All outputs saved to: {timestamped_output_dir}/")
    print(f"🌐 Open detailed_results.html in your browser for human review")
    print(f"📊 Check summary_report.json for aggregated statistics")
    
    if benchmark_results and 'comparison_table' in benchmark_results:
        print(f"\n🏆 Quick Results Summary:")
        comparison_table = benchmark_results['comparison_table']
        
        # Show top performer for each metric
        for _, row in comparison_table.iterrows():
            metric = row['Metric']
            scores = {col: row[col] for col in comparison_table.columns 
                     if col not in ['Metric', 'Improvement'] and isinstance(row[col], (int, float))}
            
            if scores:
                best_approach = max(scores, key=scores.get)
                best_score = scores[best_approach]
                print(f"   {metric}: {best_approach} ({best_score:.3f})")
    
    return benchmark_results


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="RAGBench Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark/ragbench_pipeline.py micro
    python benchmark/ragbench_pipeline.py small --approaches chroma graphrag
    python benchmark/ragbench_pipeline.py medium --enhanced --skip-ingestion
    python benchmark/ragbench_pipeline.py large --approaches advanced_graphrag drift_graphrag
        """
    )
    
    parser.add_argument(
        'preset',
        choices=list(INGESTION_PRESETS.keys()),
        help='Ingestion preset to use'
    )
    
    parser.add_argument(
        '--approaches',
        nargs='+',
        choices=['chroma', 'graphrag', 'text2cypher', 'advanced_graphrag', 
                'drift_graphrag', 'neo4j_vector', 'hybrid_cypher'],
        help='Retriever approaches to test (default: preset-dependent)'
    )
    
    parser.add_argument(
        '--enhanced',
        action='store_true',
        help='Use advanced processor with element summarization'
    )
    
    parser.add_argument(
        '--skip-ingestion',
        action='store_true',
        help='Skip document ingestion (use existing graph)'
    )
    
    args = parser.parse_args()
    
    # Determine processor type
    processor_type = "advanced" if args.enhanced else None
    
    try:
        run_full_pipeline(
            preset_name=args.preset,
            approaches=args.approaches,
            processor_type=processor_type,
            skip_ingestion=args.skip_ingestion
        )
    except KeyboardInterrupt:
        print(f"\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
