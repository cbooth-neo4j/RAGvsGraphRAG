"""
Wikipedia Corpus Ingester

Handles ingestion of Wikipedia articles into the Neo4j knowledge graph
using the existing build_graph pipeline.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.graph_rag_logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


class WikiCorpusIngester:
    """
    Ingests Wikipedia articles into Neo4j graph using the CustomGraphProcessor.
    
    This class provides a clean interface for benchmarking, reusing the existing
    graph processing infrastructure without duplication.
    """
    
    def __init__(
        self,
        relationship_strategy: str = "smart",
        auto_approve_labels: bool = True
    ):
        """
        Initialize the ingester with graph processor.
        
        Args:
            relationship_strategy: Strategy for entity relationships
                - "smart": Semantic + proximity + co-occurrence (default)
                - "semantic": Only semantic relationships
                - "proximity": Only proximity relationships
                - "implicit": No explicit relationships
            auto_approve_labels: If True, auto-approve discovered labels
        """
        from data_processors.build_graph import CustomGraphProcessor
        
        self.processor = CustomGraphProcessor(
            relationship_strategy=relationship_strategy
        )
        self.auto_approve_labels = auto_approve_labels
        
        logger.info("WikiCorpusIngester initialized")
        print("[INIT] WikiCorpusIngester ready")
    
    def clear_database(self) -> None:
        """Clear the Neo4j database."""
        print("[CLEAR] Clearing Neo4j database...")
        self.processor.clear_database()
        print("[OK] Database cleared")
    
    def ingest_corpus(
        self,
        articles: List[Dict[str, str]],
        clear_db: bool = True,
        run_advanced_processing: bool = True,
        domain_hint: str = "qa"
    ) -> Dict[str, Any]:
        """
        Ingest Wikipedia articles into Neo4j graph.
        
        Args:
            articles: List of article dictionaries with 'title' and 'text' keys
            clear_db: If True, clear database before ingestion (default True)
            run_advanced_processing: If True, run community detection etc.
            domain_hint: Domain hint for entity discovery (default "qa")
            
        Returns:
            Ingestion statistics and results
        """
        if not articles:
            raise ValueError("No articles provided for ingestion")
        
        print(f"\n{'='*60}")
        print("WIKIPEDIA CORPUS INGESTION")
        print(f"{'='*60}")
        print(f"   Articles to process: {len(articles)}")
        print(f"   Clear database: {clear_db}")
        print(f"   Advanced processing: {run_advanced_processing}")
        print(f"   Domain hint: {domain_hint}")
        print(f"{'='*60}\n")
        
        # Prepare texts and sources
        texts = []
        sources = []
        
        for article in articles:
            text = article.get("text", "")
            title = article.get("title", "Unknown")
            
            # Skip empty articles
            if not text.strip():
                logger.warning(f"Skipping empty article: {title}")
                continue
            
            texts.append(text)
            sources.append(title)
        
        if not texts:
            raise ValueError("No valid article texts found after filtering")
        
        print(f"[PROCESS] Processing {len(texts)} valid articles...")
        
        # If auto-approving labels, monkey-patch the approval method
        if self.auto_approve_labels:
            original_approve = self.processor._approve_labels_cli
            self.processor._approve_labels_cli = lambda labels: labels
        
        try:
            # Use the existing RAGBench document processor which handles:
            # - Database clearing
            # - Entity discovery
            # - Chunking
            # - Graph construction
            # - Advanced processing
            result = self.processor.process_ragbench_documents(
                texts=texts,
                sources=sources,
                use_enhanced_discovery=True,
                domain_hint=domain_hint,
                prompt_for_advanced=run_advanced_processing,
                auto_advanced=run_advanced_processing,
                mode='fresh' if clear_db else 'add',
                doc_prefix='wiki'  # Use 'wiki' prefix for Wikipedia articles
            )
            
            # Add corpus-specific stats
            result["corpus_stats"] = {
                "total_articles": len(articles),
                "processed_articles": len(texts),
                "skipped_articles": len(articles) - len(texts),
                "total_characters": sum(len(t) for t in texts),
                "avg_article_length": sum(len(t) for t in texts) // len(texts)
            }
            
            print(f"\n{'='*60}")
            print("INGESTION COMPLETE")
            print(f"{'='*60}")
            print(f"   Documents processed: {result.get('successful_documents', 0)}")
            print(f"   Chunks created: {result.get('total_chunks_created', 0)}")
            print(f"   Entities created: {result.get('total_entities_created', 0)}")
            print(f"   Entity types: {result.get('entity_types_discovered', [])}")
            print(f"{'='*60}\n")
            
            return result
            
        finally:
            # Restore original approval method
            if self.auto_approve_labels:
                self.processor._approve_labels_cli = original_approve
    
    def ingest_articles_incremental(
        self,
        articles: List[Dict[str, str]],
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Ingest articles incrementally in batches (for large corpora).
        
        Args:
            articles: List of article dictionaries
            batch_size: Number of articles per batch
            
        Returns:
            Aggregated ingestion statistics
        """
        total_articles = len(articles)
        total_chunks = 0
        total_entities = 0
        failed_articles = []
        
        print(f"[INCREMENTAL] Processing {total_articles} articles in batches of {batch_size}")
        
        for i in range(0, total_articles, batch_size):
            batch = articles[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_articles + batch_size - 1) // batch_size
            
            print(f"\n[BATCH {batch_num}/{total_batches}] Processing {len(batch)} articles...")
            
            try:
                texts = [a.get("text", "") for a in batch if a.get("text", "").strip()]
                sources = [a.get("title", f"article_{j}") for j, a in enumerate(batch) if a.get("text", "").strip()]
                
                if not texts:
                    continue
                
                for j, (text, source) in enumerate(zip(texts, sources)):
                    try:
                        result = self.processor.process_text_document(
                            text=text,
                            doc_name=f"wiki_{source}_{i+j}",
                            source_info=f"wikipedia:{source}"
                        )
                        total_chunks += result.get("chunks_created", 0)
                        total_entities += result.get("entities_created", 0)
                    except Exception as e:
                        logger.error(f"Failed to process article '{source}': {e}")
                        failed_articles.append(source)
                        
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
        
        return {
            "total_articles": total_articles,
            "processed_articles": total_articles - len(failed_articles),
            "failed_articles": len(failed_articles),
            "total_chunks": total_chunks,
            "total_entities": total_entities,
            "failed_list": failed_articles[:20]  # First 20 failures
        }
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get current graph statistics."""
        return self.processor.get_graph_statistics()
    
    def close(self) -> None:
        """Close database connections."""
        if hasattr(self.processor, 'close'):
            self.processor.close()
        print("[CLOSE] WikiCorpusIngester closed")


def ingest_from_prepared_corpus(
    corpus: Dict[str, Any],
    clear_db: bool = True,
    run_advanced: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to ingest a prepared corpus.
    
    Args:
        corpus: Corpus dictionary from prepare_corpus()
        clear_db: Whether to clear database first
        run_advanced: Whether to run advanced processing
        
    Returns:
        Ingestion results
    """
    ingester = WikiCorpusIngester()
    
    try:
        result = ingester.ingest_corpus(
            articles=corpus["articles"],
            clear_db=clear_db,
            run_advanced_processing=run_advanced
        )
        return result
    finally:
        ingester.close()


if __name__ == "__main__":
    # Test the ingester with sample data
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Wikipedia corpus ingester")
    parser.add_argument("--test", action="store_true", help="Run with test data")
    args = parser.parse_args()
    
    if args.test:
        # Create sample test articles
        test_articles = [
            {
                "title": "Test Article 1",
                "text": "This is a test article about machine learning. "
                       "Machine learning is a subset of artificial intelligence. "
                       "It enables computers to learn from data.",
                "url": "https://test.example.com/article1"
            },
            {
                "title": "Test Article 2", 
                "text": "Natural language processing is a field of AI. "
                       "It deals with the interaction between computers and humans. "
                       "NLP uses machine learning techniques.",
                "url": "https://test.example.com/article2"
            }
        ]
        
        ingester = WikiCorpusIngester()
        try:
            result = ingester.ingest_corpus(
                articles=test_articles,
                clear_db=True,
                run_advanced_processing=False  # Skip for quick test
            )
            print(f"\nTest result: {result}")
        finally:
            ingester.close()
    else:
        print("Run with --test flag to execute test ingestion")

