#!/usr/bin/env python3
"""
Unified data ingestion pipeline for RAGvsGraphRAG.

Single entry point for building knowledge graphs from any supported data source.
Benchmark evaluation is separate - this script ONLY builds the graph.

Usage:
    python ingest.py --source pdf --quantity 10 --lean
    python ingest.py --source hotpotqa --quantity 100 --full
    
Required arguments:
    --source      Data source: pdf | hotpotqa
    --quantity    Number of documents to process
    --lean        Build minimal graph (no summaries, no communities)
    --full        Build complete graph (with summaries + communities)
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from utils.graph_rag_logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def ingest_pdfs(pdf_dir: str, quantity: int, lean_mode: bool) -> dict:
    """
    Ingest PDF documents into the knowledge graph.
    
    Args:
        pdf_dir: Directory containing PDF files
        quantity: Maximum number of PDFs to process
        lean_mode: If True, skip summaries and communities
    """
    from data_processors.build_graph import CustomGraphProcessor
    
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    # Find all PDFs
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    # Apply quantity limit
    total_available = len(pdf_files)
    if quantity >= total_available:
        print(f"[INFO] Requested {quantity}, found {total_available}. Processing all {total_available}.")
        quantity = total_available
    else:
        print(f"[INFO] Processing {quantity} of {total_available} available PDFs")
    
    # Select PDFs (first N)
    selected_pdfs = pdf_files[:quantity]
    
    print(f"\n{'='*60}")
    print(f"INGESTING PDFs")
    print(f"{'='*60}")
    print(f"   Source: {pdf_dir}")
    print(f"   Documents: {quantity}")
    print(f"   Mode: {'LEAN (no summaries/communities)' if lean_mode else 'FULL (with summaries/communities)'}")
    print(f"{'='*60}\n")
    
    # Initialize processor
    processor = CustomGraphProcessor()
    
    try:
        # Process the directory with selected files
        result = processor.process_directory(
            pdf_dir=str(pdf_dir),
            perform_resolution=True,
            prompt_for_advanced=not lean_mode,
            auto_advanced=not lean_mode,
            mode='fresh',
            lean_mode=lean_mode
        )
        return result
    finally:
        processor.close()


def ingest_hotpotqa(quantity: int, lean_mode: bool, cache_dir: str = "data/hotpotqa") -> dict:
    """
    Ingest HotpotQA Wikipedia articles into the knowledge graph.
    
    Args:
        quantity: Number of questions (and their supporting articles) to process
        lean_mode: If True, skip summaries and communities
        cache_dir: Directory for caching downloaded data
    """
    from data_processors.build_graph import CustomGraphProcessor
    from benchmark.hotpotqa.data_loader import prepare_corpus, load_cached_corpus, save_corpus
    
    print(f"\n{'='*60}")
    print(f"INGESTING HOTPOTQA WIKIPEDIA ARTICLES")
    print(f"{'='*60}")
    print(f"   Questions: {quantity}")
    print(f"   Mode: {'LEAN (no summaries/communities)' if lean_mode else 'FULL (with summaries/communities)'}")
    print(f"   Cache: {cache_dir}")
    print(f"{'='*60}\n")
    
    # Load or prepare corpus
    print("[PHASE 1] Preparing corpus...")
    corpus = load_cached_corpus(cache_dir)
    
    if corpus and len(corpus.get("questions", [])) >= quantity:
        print(f"[CACHE] Using cached corpus ({len(corpus['questions'])} questions available)")
    else:
        print(f"[DOWNLOAD] Preparing fresh corpus...")
        corpus = prepare_corpus(
            questions_split="dev",
            cache_dir=cache_dir,
            question_limit=quantity
        )
        save_corpus(corpus, cache_dir)
    
    questions = corpus["questions"][:quantity]
    articles = corpus["articles"]
    
    # Filter articles to match selected questions
    referenced_titles = set()
    for q in questions:
        for ctx in q.get("context", []):
            if isinstance(ctx, (list, tuple)) and len(ctx) >= 1:
                referenced_titles.add(ctx[0])
        for sf in q.get("supporting_facts", []):
            if isinstance(sf, (list, tuple)) and len(sf) >= 1:
                referenced_titles.add(sf[0])
    
    articles = [a for a in articles if a.get("title") in referenced_titles]
    
    print(f"[INFO] Processing {len(articles)} articles for {len(questions)} questions")
    
    # Prepare texts and sources for ingestion
    texts = [a["text"] for a in articles]
    sources = [a["title"] for a in articles]
    
    # Initialize processor
    processor = CustomGraphProcessor()
    
    try:
        # Clear database and process
        processor.clear_database()
        
        # Discover labels from corpus sample
        if not processor.discovered_labels:
            sample_text = "\n\n".join(text[:2000] for text in texts[:20])
            proposed_labels = processor.discover_labels_for_text(sample_text)
            processor.discovered_labels = processor._approve_labels_cli(proposed_labels)
        
        # Process each article
        results = []
        total_chunks = 0
        total_entities = 0
        
        for i, (text, source) in enumerate(zip(texts, sources)):
            try:
                doc_name = f"wiki_{source}_{i}"
                result = processor.process_text_document(
                    text=text,
                    doc_name=doc_name,
                    source_info=f"wikipedia:{source}"
                )
                results.append(result)
                total_chunks += result.get('chunks_created', 0)
                total_entities += result.get('entities_created', 0)
                
                if (i + 1) % 20 == 0:
                    print(f"[PROGRESS] Processed {i + 1}/{len(texts)} articles")
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {source}: {e}")
                results.append({'status': 'failed', 'error': str(e)})
        
        # Entity resolution
        print("\n[RESOLVE] Performing entity resolution...")
        processor.create_chunk_similarity_relationships()
        processor.perform_entity_resolution()
        
        # Advanced processing (if not lean mode)
        if not lean_mode:
            print("\n[ADVANCED] Running advanced processing...")
            graph_stats = processor.get_graph_statistics()
            processor.perform_advanced_processing(graph_stats)
        else:
            print("\n[LEAN] Skipping advanced processing (summaries/communities)")
        
        # Save ingestion manifest to Neo4j (source of truth for benchmark pairing)
        import json
        import time
        manifest = {
            'source': 'hotpotqa',
            'ingested_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'questions_count': len(questions),
            'articles_count': len(texts),
            'article_titles': sources,
            'question_ids': [q.get('_id', f'q_{i}') for i, q in enumerate(questions)],
            'lean_mode': lean_mode,
            'total_chunks': total_chunks,
            'total_entities': total_entities
        }
        
        # Store manifest in Neo4j as a node
        from neo4j import GraphDatabase
        from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            with driver.session() as session:
                session.run("""
                    MERGE (m:__IngestionManifest__ {id: 'current'})
                    SET m.source = $source,
                        m.ingested_at = $ingested_at,
                        m.questions_count = $questions_count,
                        m.articles_count = $articles_count,
                        m.article_titles = $article_titles,
                        m.question_ids = $question_ids,
                        m.lean_mode = $lean_mode,
                        m.total_chunks = $total_chunks,
                        m.total_entities = $total_entities
                """, **manifest)
            print(f"[MANIFEST] Saved ingestion manifest to Neo4j (:__IngestionManifest__)")
        finally:
            driver.close()
        
        summary = {
            'source': 'hotpotqa',
            'questions_requested': quantity,
            'articles_processed': len(texts),
            'total_chunks': total_chunks,
            'total_entities': total_entities,
            'lean_mode': lean_mode,
            'status': 'completed'
        }
        
        print(f"\n{'='*60}")
        print("INGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"   Articles: {len(texts)}")
        print(f"   Chunks: {total_chunks:,}")
        print(f"   Entities: {total_entities:,}")
        print(f"   Mode: {'LEAN' if lean_mode else 'FULL'}")
        print(f"   Manifest: Stored in Neo4j (:__IngestionManifest__)")
        print(f"{'='*60}\n")
        
        return summary
        
    finally:
        processor.close()


def load_cached_corpus(cache_dir: str):
    """Load cached corpus if available."""
    import json
    corpus_file = Path(cache_dir) / "prepared_corpus.json"
    if corpus_file.exists():
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Unified data ingestion for RAGvsGraphRAG knowledge graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --source pdf --quantity 10 --lean
  python ingest.py --source hotpotqa --quantity 100 --full
  python ingest.py --source hotpotqa --quantity 1000 --lean

After ingestion, run benchmarks:
  python -m benchmark.hotpotqa.benchmark_pipeline mini
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--source',
        required=True,
        choices=['pdf', 'hotpotqa'],
        help='Data source to ingest'
    )
    
    parser.add_argument(
        '--quantity',
        type=int,
        required=True,
        help='Number of documents to process (excess is capped to available)'
    )
    
    # Build mode (mutually exclusive, one required)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--lean',
        action='store_true',
        help='Build minimal graph: Document->Chunk->Entity with RELATES_TO. No summaries, no communities.'
    )
    mode_group.add_argument(
        '--full',
        action='store_true',
        help='Build complete graph with AI summaries and community detection.'
    )
    
    # Optional arguments
    parser.add_argument(
        '--pdf-dir',
        default='PDFs',
        help='Directory containing PDF files (default: ./PDFs)'
    )
    
    parser.add_argument(
        '--cache-dir',
        default='data/hotpotqa',
        help='Cache directory for HotpotQA data (default: data/hotpotqa)'
    )
    
    args = parser.parse_args()
    
    # Determine lean mode
    lean_mode = args.lean  # True if --lean, False if --full
    
    print(f"\n{'='*60}")
    print("RAGvsGraphRAG INGESTION PIPELINE")
    print(f"{'='*60}")
    print(f"   Source: {args.source}")
    print(f"   Quantity: {args.quantity}")
    print(f"   Mode: {'LEAN' if lean_mode else 'FULL'}")
    print(f"{'='*60}\n")
    
    try:
        if args.source == 'pdf':
            result = ingest_pdfs(
                pdf_dir=args.pdf_dir,
                quantity=args.quantity,
                lean_mode=lean_mode
            )
        elif args.source == 'hotpotqa':
            result = ingest_hotpotqa(
                quantity=args.quantity,
                lean_mode=lean_mode,
                cache_dir=args.cache_dir
            )
        
        print("[SUCCESS] Ingestion completed successfully")
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

