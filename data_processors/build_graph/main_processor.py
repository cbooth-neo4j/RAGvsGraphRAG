"""
Main graph processor - combines all mixins into a cohesive interface.
Maintains backward compatibility with existing CustomGraphProcessor.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from utils.graph_rag_logger import setup_logging, get_logger
from .entity_discovery import EntityDiscoveryMixin
from .text_processing import TextProcessingMixin
from .graph_operations import GraphOperationsMixin
from .advanced_processing import AdvancedProcessingMixin

# Import centralized configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_model_config

from dotenv import load_dotenv

load_dotenv()

setup_logging()
logger = get_logger(__name__)

class CustomGraphProcessor(EntityDiscoveryMixin, TextProcessingMixin, GraphOperationsMixin, AdvancedProcessingMixin):
    """
    Main graph processor that combines entity discovery, text processing, and graph operations.
    
    This class maintains full backward compatibility with the original CustomGraphProcessor
    while providing enhanced capabilities through the mixin architecture.
    """
    
    def __init__(self, model_config=None, relationship_strategy="smart"):
        """Initialize the graph processor with all capabilities.
        
        Args:
            model_config: Model configuration object
            relationship_strategy: Strategy for entity relationships
                - "smart": Semantic + proximity + co-occurrence (default)
                - "semantic": Only semantic relationships
                - "proximity": Only proximity relationships  
                - "implicit": No explicit relationships (rely on chunk connections)
        """
        # Set model configuration for all mixins
        self.config = model_config or get_model_config()
        self.relationship_strategy = relationship_strategy
        
        # Initialize all mixins
        super().__init__()

        logger.debug("In __init__ method of 'Enhanced' CustomGraphProcessor with mixins. Next step is setup_Database_schema...")
        # Set up database schema
        self.setup_database_schema()
        
        print("🚀 Enhanced CustomGraphProcessor initialized")
        logger.info("Enhanced CustomGraphProcessor initialized")
        print("   ✅ Entity discovery with enhanced sampling")
        print("   ✅ Text processing with PDF and table extraction")
        print("   ✅ Graph operations with Neo4j")
        print(f"   ✅ Relationship strategy: {relationship_strategy}")
        logger.info(f"Relationship strategy: {relationship_strategy}")

    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF document into the knowledge graph.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processing statistics and metadata
        """
        logger.info(f"Processing document into KG. In process_document method: {pdf_path}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        doc_name = Path(pdf_path).stem
        
        return self._process_document_text(text, doc_name, pdf_path)
    
    def process_text_document(self, text: str, doc_name: str, source_info: str = None) -> Dict[str, Any]:
        """
        Process a text document into the knowledge graph.
        
        Args:
            text: Raw text content
            doc_name: Document identifier/name
            source_info: Optional source information
            
        Returns:
            Processing statistics and metadata
        """
        print(f"Processing {doc_name}...")
        print(f"Text length: {len(text)} characters")
        return self._process_document_text(text, doc_name, source_info or doc_name)
    
    def _process_document_text(self, text: str, doc_name: str, source_info: str) -> Dict[str, Any]:
        """Internal method to process document text (shared by PDF and text processing)"""
        
        # Per-document discovery if needed (fallback if corpus-wide failed)
        if not self.discovered_labels:
            print("\n🔎 Discovering entity labels from document text...")
            logger.info("Discovering entity labels from document text...")
            proposed_labels = self.discover_labels_for_text(text)
            self.discovered_labels = self._approve_labels_cli(proposed_labels)
            print(f"\n✅ Using labels: {self.discovered_labels}")
            logger.info(f"\n✅ Using labels: {self.discovered_labels}")
        
        # Create document embedding
        doc_embedding = self.create_embedding(text) #[:8000])  # Limit for embedding
        
        # Chunk text
        chunks = self.chunk_text(text, doc_name)
        
        # Attempt table extraction and append as atomic chunks (only for PDFs)
        table_chunks = []
        if source_info.endswith('.pdf') and os.path.exists(source_info):
            table_chunks = self.extract_tables(source_info, doc_name, start_index=len(chunks))
        if table_chunks:
            chunks.extend(table_chunks)
        print(f"Created {len(chunks)} chunks (including {len(table_chunks)} table chunks)")
        logger.info(f"Created {len(chunks)} chunks (including {len(table_chunks)} table chunks)")
        # Create embeddings for chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.create_embeddings_batch(chunk_texts)
        
        # Store everything in Neo4j
        with self.driver.session() as session:
            # Create document node
            doc_id = f"doc_{doc_name}"
            self.create_document_node(
                session, doc_id, doc_name, source_info, text, doc_embedding
            )
            
            # Create chunk nodes with sequential relationships
            chunk_ids = self.create_chunk_nodes(session, chunks, doc_id, chunk_embeddings)
            
            # Extract entities from each chunk and link to specific chunks
            all_entity_ids = []
            total_entities = 0
            
            for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
                chunk_entities = self.extract_entities_dynamic(chunk['text'], self.discovered_labels)
                chunk_entity_count = sum(len(entities) for entities in chunk_entities.values())
                
                if chunk_entities:
                    entity_ids = self.create_entity_nodes_for_chunk(session, chunk_entities, chunk_id, doc_id)
                    all_entity_ids.extend(entity_ids)
                    total_entities += chunk_entity_count
                    
                    # Create chunk-level relationships based on strategy
                    if len(entity_ids) > 1 and self.relationship_strategy != "implicit":
                        self.create_entity_relationships_dynamic(session, entity_ids, chunk['text'])
            
            print(f"Extracted {total_entities} entities across {len(chunks)} chunks")
            logger.info(f"Extracted {total_entities} entities across {len(chunks)} chunks")
        
        return {
            'document_id': doc_id,
            'chunks_created': len(chunks),
            'entities_created': total_entities,
            'status': 'success',
            'processing_status': 'completed'
        }
    
    def process_directory(self, pdf_dir: str, perform_resolution: bool = True) -> Dict[str, Any]:
        """
        Process all PDF files in a directory.
        
        Args:
            pdf_dir: Directory containing PDF files
            perform_resolution: Whether to perform entity resolution after processing
            
        Returns:
            Overall processing statistics
        """
        logger.debug(f"Processing directory: {pdf_dir}. About to clear the Graph DB: {self.neo4j_db}")
        # Clear database first
        self.clear_database()

        ## Confirm this -- set db schema?
        logger.debug(f"Setting up the database: {self.neo4j_db}")
        self.setup_database_schema()
        
        pdf_dir_path = Path(pdf_dir)
        if not pdf_dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")
        
        # Find all PDF files
        pdf_files = list(pdf_dir_path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir}")
        
        print(f"📁 Processing {len(pdf_files)} PDF files from {pdf_dir}")
        logger.info(f"Processing {len(pdf_files)} PDF files from {pdf_dir}")
        
        # Discover labels corpus-wide first
        if not self.discovered_labels:
            self.discovered_labels = self.discover_corpus_labels(pdf_files)
            logger.info(f"Labels discovered: {self.discovered_labels}")
            if not self.discovered_labels:
                print("⚠️ No labels discovered, using default set")
                logger.warning("No labels discovered, using default set")
                self.discovered_labels = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"]
        
        # Process each document
        results = []
        total_chunks = 0
        total_entities = 0
        
        for pdf_path in pdf_files:
            try:
                logger.debug(f'Processing file in path: {pdf_path}')
                result = self.process_document(str(pdf_path))
                results.append(result)
                total_chunks += result['chunks_created']
                total_entities += result['entities_created']
                print(f"✅ Processed {pdf_path.name}")
            except Exception as e:
                print(f"❌ Failed to process {pdf_path.name}: {e}")
                results.append({
                    'document_id': f"failed_{pdf_path.stem}",
                    'error': str(e),
                    'processing_status': 'failed'
                })
        
        # Perform entity resolution if requested
        if perform_resolution:
            print("\n🔗 Creating chunk similarity relationships...")
            logger.info("Creating chunk similarity relationships...")
            self.create_chunk_similarity_relationships()
            
            print("\n🔍 Performing entity resolution...")
            self.perform_entity_resolution()
        
        summary = {
            'total_documents': len(pdf_files),
            'successful_documents': len([r for r in results if r.get('processing_status') == 'completed']),
            'failed_documents': len([r for r in results if r.get('processing_status') == 'failed']),
            'total_chunks_created': total_chunks,
            'total_entities_created': total_entities,
            'entity_types_discovered': self.discovered_labels,
            'results': results
        }
        
        print(f"\n📊 Processing Summary:")
        print(f"   Documents: {summary['successful_documents']}/{summary['total_documents']} successful")
        print(f"   Chunks: {summary['total_chunks_created']:,}")
        print(f"   Entities: {summary['total_entities_created']:,}")
        print(f"   Entity types: {len(summary['entity_types_discovered'])}")

        logger.info(f"Processing Summary:")
        logger.info(f"   Documents: {summary['successful_documents']}/{summary['total_documents']} successful")
        logger.info(f"   Chunks: {summary['total_chunks_created']:,}")
        logger.info(f"   Entities: {summary['total_entities_created']:,}")
        logger.info(f"   Entity types: {len(summary['entity_types_discovered'])}")
        # Automatically perform advanced processing
        print(f"\n🚀 Starting advanced processing (summarization + community detection)...")
        logger.info(f"Starting advanced processing (summarization + community detection)...")
        graph_stats = self.get_graph_statistics()
        advanced_results = self.perform_advanced_processing(graph_stats)
        
        # Add advanced results to summary
        summary["advanced_processing"] = advanced_results
        
        return summary
    
    def process_ragbench_documents(self, 
                                 texts: List[str], 
                                 sources: List[str],
                                 use_enhanced_discovery: bool = True,
                                 domain_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Process RAGBench documents with enhanced entity discovery.
        
        Args:
            texts: List of document texts
            sources: List of source identifiers
            use_enhanced_discovery: Whether to use enhanced sampling for entity discovery
            domain_hint: Optional domain hint (e.g., 'financial', 'medical')
            
        Returns:
            Processing statistics
        """
        # Clear database first
        self.clear_database()
        
        if len(texts) != len(sources):
            raise ValueError("Number of texts must match number of sources")
        
        print(f"📚 Processing {len(texts)} RAGBench documents...")
        
        # Discover labels using enhanced method if requested
        if not self.discovered_labels:
            if use_enhanced_discovery:
                print("🚀 Using enhanced entity discovery...")
                documents = self.prepare_ragbench_documents_for_sampling(texts, sources)
                self.discovered_labels = self.discover_labels_for_text_enhanced(
                    documents, domain_hint=domain_hint
                )
            else:
                print("🔍 Using standard entity discovery...")
                # Sample text for discovery (original method)
                sample_text = "\n\n".join(text[:1000] for text in texts[:10])
                self.discovered_labels = self.discover_labels_for_text(sample_text)
            
            if not self.discovered_labels:
                print("⚠️ No labels discovered, using default set")
                self.discovered_labels = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"]
            
            # Get user approval
            self.discovered_labels = self._approve_labels_cli(self.discovered_labels)
        
        # Process each document
        results = []
        total_chunks = 0
        total_entities = 0
        
        for i, (text, source) in enumerate(zip(texts, sources)):
            try:
                doc_name = f"ragbench_{source}_{i}"
                result = self.process_text_document(text, doc_name, f"ragbench:{source}")
                results.append(result)
                total_chunks += result['chunks_created']
                total_entities += result['entities_created']
                
                if (i + 1) % 10 == 0:
                    print(f"✅ Processed {i + 1}/{len(texts)} documents")
                    
            except Exception as e:
                print(f"❌ Failed to process document {i}: {e}")
                results.append({
                    'document_id': f"failed_ragbench_{i}",
                    'error': str(e),
                    'processing_status': 'failed'
                })
        
        summary = {
            'total_documents': len(texts),
            'successful_documents': len([r for r in results if r.get('processing_status') == 'completed']),
            'failed_documents': len([r for r in results if r.get('processing_status') == 'failed']),
            'total_chunks_created': total_chunks,
            'total_entities_created': total_entities,
            'entity_types_discovered': self.discovered_labels,
            'enhanced_discovery_used': use_enhanced_discovery,
            'domain_hint': domain_hint,
            'results': results
        }
        
        print(f"\n📊 RAGBench Processing Summary:")
        print(f"   Documents: {summary['successful_documents']}/{summary['total_documents']} successful")
        print(f"   Chunks: {summary['total_chunks_created']:,}")
        print(f"   Entities: {summary['total_entities_created']:,}")
        print(f"   Entity types: {len(summary['entity_types_discovered'])}")
        print(f"   Enhanced discovery: {use_enhanced_discovery}")
        
        # Automatically perform advanced processing
        print(f"\n🚀 Starting advanced processing (summarization + community detection)...")
        graph_stats = self.get_graph_statistics()
        advanced_results = self.perform_advanced_processing(graph_stats)
        
        # Add advanced results to summary
        summary["advanced_processing"] = advanced_results
        
        return summary
    
    # Backward compatibility method
    def create_entity_relationships(self, session, entity_ids: List[Tuple[str, str]]):
        """Backward compatibility method - redirects to new dynamic version."""
        return self.create_entity_relationships_dynamic(session, entity_ids)


def main():
    """Example usage of the enhanced graph processor."""
    processor = CustomGraphProcessor()
    
    try:
        # Example: Process a directory of PDFs
        # result = processor.process_directory("path/to/pdfs")
        
        # Example: Process RAGBench documents
        # texts = ["Document 1 text...", "Document 2 text..."]
        # sources = ["doc1", "doc2"]
        # result = processor.process_ragbench_documents(
        #     texts, sources, 
        #     use_enhanced_discovery=True,
        #     domain_hint='financial'
        # )
        
        print("Graph processor ready for use!")
        
    finally:
        processor.close()


if __name__ == "__main__":
    main()
