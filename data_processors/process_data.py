#!/usr/bin/env python3
"""
Unified CLI for data processing - choose between PDF processing and RAGBench ingestion.
Similar interface to ragas_benchmark.py for consistency.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.graph_rag_logger import setup_logging, get_logger
from dotenv import load_dotenv

load_dotenv()

setup_logging()
logger = get_logger(__name__)

def process_pdfs(pdf_dir: str = "PDFs", 
                perform_resolution: bool = True,
                enhanced_discovery: bool = True) -> dict:
    """Process PDF documents from a directory."""
    from build_graph import CustomGraphProcessor
    processor = CustomGraphProcessor()
    print("Using enhanced CustomGraphProcessor with advanced features (refactored with research-based discovery)")
    logger.debug("Using enhanced CustomGraphProcessor with advanced features (refactored with research-based discovery)")
    
    try:
        if not Path(pdf_dir).exists():
            raise FileNotFoundError(f"PDF directory '{pdf_dir}' not found!")
        
        # Prompt user for mode and advanced processing choice at the START
        mode, run_advanced = processor.prompt_for_mode()
        
        print(f"\n[*] Processing PDFs from '{pdf_dir}' folder...")
        print(f"   Mode: {mode}")
        print(f"   Enhanced discovery: {enhanced_discovery}")
        print(f"   Entity resolution: {perform_resolution}")
        print(f"   Advanced processing: {run_advanced}")
        print("=" * 60)
        
        # Process all PDFs in the directory
        result = processor.process_directory(
            pdf_dir=pdf_dir,
            perform_resolution=perform_resolution,
            prompt_for_advanced=run_advanced,  # Now a boolean choice, not a flag
            mode=mode
        )
        
        return result
        
    finally:
        processor.close()

def process_ragbench(preset: Optional[str] = None,
                    datasets: Optional[list] = None,
                    records: Optional[int] = None,
                    enhanced_discovery: bool = True,
                    domain_hint: Optional[str] = None) -> dict:
    """Process RAGBench dataset using specified preset."""
    from pathlib import Path
    # Add parent directory to path so we can import benchmark module
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from benchmark.ragbench import RAGBenchIngester
    
    processor_type = 'basic'  # Always use basic (now includes advanced features)
    
    if preset:
        print(f"[*] Processing RAGBench preset: '{preset}'")
    else:
        print(f"[*] Processing RAGBench custom datasets: {', '.join(datasets)}")
        print(f"   Records per dataset: {records or 50}")
    
    print(f"   Enhanced discovery: {enhanced_discovery}")
    print(f"   Domain hint: {domain_hint or 'auto-detect'}")
    print(f"   Processor: Dynamic Graph Processor")
    print("=" * 60)
    
    # Initialize ingester
    ingester = RAGBenchIngester(processor_type=processor_type)
    
    if preset:
        # Process using preset
        result = ingester.load_and_process_preset(preset)
    else:
        # Process using custom datasets
        result = ingester.load_and_process_custom(
            datasets=datasets,
            records_per_dataset=records or 50
        )
    
    return result

def print_results_summary(result: dict, source_type: str):
    """Print a formatted summary of processing results."""
    print("\n" + "=" * 60)
    print(f"[STATS] **{source_type.upper()} PROCESSING COMPLETE**")
    print("=" * 60)
    
    # Handle 'add' mode (just advanced processing)
    if result.get('mode') == 'add':
        print("[OK] Advanced processing completed on existing graph")
        adv_result = result.get('advanced_processing', {})
        if adv_result.get('status') == 'completed':
            print("[INFO] Element summarization: Completed")
            print("[INFO] Community detection: Completed")
        return
    
    if source_type == "pdf":
        print(f"[OK] Documents processed: {result['successful_documents']}/{result['total_documents']}")
        print(f"[INFO] Total chunks created: {result['total_chunks_created']:,}")
        print(f"[INFO] Total entities created: {result['total_entities_created']:,}")
        print(f"[INFO] Entity types discovered: {len(result['entity_types_discovered'])}")
        print(f"   Types: {', '.join(result['entity_types_discovered'])}")
        
        if result['failed_documents'] > 0:
            print(f"[WARNING] Failed documents: {result['failed_documents']}")
    
    elif source_type == "ragbench":
        stats = result.get('stats', {})
        print(f"[OK] Records processed: {stats.get('records_loaded', 0)}")
        print(f"[INFO] Documents processed: {stats.get('documents_processed', 0)}")
        print(f"[INFO] Total chunks created: {stats.get('total_chunks', 0):,}")
        print(f"[INFO] Total entities created: {stats.get('total_entities', 0):,}")
        if 'evaluation_data_path' in stats:
            print(f"[INFO] Evaluation data saved: {stats['evaluation_data_path']}")
        print(f"[INFO] Preset: {result.get('preset', 'unknown')}")
        print(f"[INFO] Config: {result.get('config', {}).get('description', 'N/A')}")
    
    print(f"\n[OK] Knowledge graph ready! You can now:")
    print(f"   • Query the graph in Neo4j Browser: http://localhost:7474")
    print(f"   • Run retrievers for testing: python -m retrievers.chroma_retriever")
    print(f"   • Run benchmarks: python benchmark/ragas_benchmark.py --all")

def main():
    """Main CLI interface for data processing."""
    parser = argparse.ArgumentParser(
        description="Process data into Neo4j knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PDFs from PDFs folder
  python process_data.py --pdfs
  python process_data.py --pdfs --pdf-dir "my_documents" --no-resolution
  python process_data.py --pdfs --no-enhanced  # Skip enhanced entity discovery
  
  # Process RAGBench datasets
  python process_data.py --ragbench --preset nano
  python process_data.py --ragbench --preset micro --domain financial
  python process_data.py --ragbench --preset small --no-enhanced
  
  # Custom dataset selection
  python process_data.py --ragbench --datasets finqa hotpotqa --records 10
  python process_data.py --ragbench --datasets finqa --records 5
  
  # List available presets
  python process_data.py --list-presets
        """
    )
    
    # Main processing options (mutually exclusive)
    processing_group = parser.add_mutually_exclusive_group(required=True)
    processing_group.add_argument(
        '--pdfs',
        action='store_true',
        help='Process PDF documents from directory'
    )
    processing_group.add_argument(
        '--ragbench',
        action='store_true', 
        help='Process RAGBench dataset'
    )
    processing_group.add_argument(
        '--list-presets',
        action='store_true',
        help='List available RAGBench presets and exit'
    )
    
    # PDF processing options
    pdf_group = parser.add_argument_group('PDF Processing Options')
    pdf_group.add_argument(
        '--pdf-dir',
        default='PDFs',
        help='Directory containing PDF files (default: PDFs)'
    )
    pdf_group.add_argument(
        '--no-resolution',
        action='store_true',
        help='Skip entity resolution and similarity analysis'
    )
    
    # RAGBench processing options  
    ragbench_group = parser.add_argument_group('RAGBench Processing Options')
    ragbench_group.add_argument(
        '--preset',
        help='RAGBench preset to use (see --list-presets)'
    )
    ragbench_group.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets to use (e.g., finqa hotpotqa). Overrides preset.'
    )
    ragbench_group.add_argument(
        '--records',
        type=int,
        help='Number of records per dataset (default: 50). Used with --datasets.'
    )
    ragbench_group.add_argument(
        '--domain',
        help='Domain hint for entity discovery (financial, medical, legal, etc.)'
    )
    
    # Common options
    parser.add_argument(
        '--no-enhanced',
        action='store_true',
        help='Disable enhanced entity discovery (use basic method)'
    )
    parser.add_argument(
        '--output-dir',
        default='benchmark_outputs',
        help='Output directory for results (default: benchmark_outputs)'
    )
    
    args = parser.parse_args()
    
    # Handle list presets
    if args.list_presets:
        from pathlib import Path
        # Add parent directory to path so we can import benchmark module
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from benchmark.ragbench.configs import INGESTION_PRESETS
        
        print("[INFO] Available RAGBench Presets:")
        print("=" * 50)
        
        for preset_name, config in INGESTION_PRESETS.items():
            print(f"\n[*] {preset_name.upper()}")
            print(f"   Description: {config['description']}")
            print(f"   Datasets: {', '.join(config['datasets'])}")
            print(f"   Records: {config.get('max_records', 'all')}")
            print(f"   Estimated docs: {config['estimated_docs']}")
            print(f"   Estimated storage: {config['estimated_storage_gb']} GB")
            print(f"   Estimated cost: ${config['estimated_cost_usd']}")
        
        print(f"\n[HELP] Usage: python process_data.py --ragbench --preset <preset_name>")
        return
    
    # Validate arguments
    if args.ragbench:
        from pathlib import Path
        # Add parent directory to path so we can import benchmark module
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from benchmark.ragbench.configs import INGESTION_PRESETS, DATASET_SIZES
        
        # Validate preset if provided
        if args.preset and args.preset not in INGESTION_PRESETS:
            print(f"[ERROR] Unknown preset '{args.preset}'")
            print(f"   Available presets: {', '.join(INGESTION_PRESETS.keys())}")
            print(f"   Use --list-presets to see details")
            sys.exit(1)
        
        # Validate datasets if provided
        if args.datasets:
            invalid_datasets = [d for d in args.datasets if d not in DATASET_SIZES]
            if invalid_datasets:
                print(f"[ERROR] Unknown datasets: {', '.join(invalid_datasets)}")
                print(f"   Available datasets: {', '.join(DATASET_SIZES.keys())}")
                sys.exit(1)
        
        # Ensure either preset or datasets is provided
        if not args.preset and not args.datasets:
            print("[ERROR] Must specify either --preset or --datasets")
            print("   Use --list-presets to see available presets")
            print("   Or use --datasets with specific dataset names")
            sys.exit(1)
    
    try:
        # Process based on selected option
        if args.pdfs:
            logger.debug(f'Processing PDFs....')
            result = process_pdfs(
                pdf_dir=args.pdf_dir,
                perform_resolution=not args.no_resolution,
                enhanced_discovery=not args.no_enhanced
            )
            print_results_summary(result, "pdf")
            
        elif args.ragbench:
            result = process_ragbench(
                preset=args.preset,
                datasets=args.datasets,
                records=args.records,
                enhanced_discovery=not args.no_enhanced,
                domain_hint=args.domain
            )
            print_results_summary(result, "ragbench")
    
    except KeyboardInterrupt:
        print("\n\n[WARNING] Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":

    main()
