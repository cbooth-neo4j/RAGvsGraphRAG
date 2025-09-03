"""
RAGBench Document Ingester

Processes RAGBench documents into Neo4j graphs using existing graph processors.
Only ingests the 'documents' field - questions and answers are handled separately.
"""

import os
import sys
import json
import random
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .configs import INGESTION_PRESETS, DATASET_SIZES, DOMAIN_CATEGORIES


class RAGBenchIngester:
    """
    Ingests RAGBench documents into Neo4j graphs.
    
    Only processes the 'documents' field from each record.
    Questions and answers are cached separately for evaluation.
    """
    
    def __init__(self, processor_type: str = "basic"):
        """
        Initialize the ingester with specified graph processor type.
        
        Args:
            processor_type: "basic" or "advanced" (enhanced)
        """
        self.processor_type = processor_type
        self.processor = self._init_processor(processor_type)
        self.evaluation_cache = []
        self.processing_stats = {
            "records_loaded": 0,
            "documents_processed": 0,
            "chunks_created": 0,
            "entities_extracted": 0,
            "errors": []
        }
    
    def _init_processor(self, processor_type: str):
        """Initialize the appropriate graph processor"""
        try:
            if processor_type == "basic":
                from data_processors.graph_processor import CustomGraphProcessor
                return CustomGraphProcessor()
            elif processor_type == "advanced":
                from data_processors.advanced_graph_processor import AdvancedGraphProcessor
                return AdvancedGraphProcessor()
            else:
                raise ValueError(f"Unknown processor type: {processor_type}")
        except ImportError as e:
            raise ImportError(f"Could not import graph processor: {e}")
    
    def load_dataset_subset(self, 
                           dataset_names: Union[str, List[str]], 
                           split: str = "test",
                           max_records: Optional[int] = None,
                           sampling: str = "random") -> List[Dict[str, Any]]:
        """
        Load a subset of RAGBench dataset(s).
        
        Args:
            dataset_names: Single dataset name, list of names, or "all"
            split: "train", "validation", or "test" 
            max_records: Maximum records to load (None for all)
            sampling: Sampling strategy ("first", "random", "stratified", "weighted")
            
        Returns:
            List of RAGBench records
        """
        
        # Handle "all" datasets
        if dataset_names == "all":
            dataset_names = list(DATASET_SIZES.keys())
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        print(f"📥 Loading RAGBench datasets: {dataset_names}")
        print(f"   Split: {split}, Max records: {max_records}, Sampling: {sampling}")
        
        all_records = []
        
        # Load each dataset
        for dataset_name in dataset_names:
            try:
                print(f"   Loading {dataset_name}...")
                dataset = load_dataset("galileo-ai/ragbench", dataset_name, split=split)
                
                # Convert to list for easier manipulation
                records = [dict(record) for record in dataset]
                
                # Add dataset source to each record
                for record in records:
                    record['source_dataset'] = dataset_name
                    record['domain'] = self._get_domain(dataset_name)
                
                all_records.extend(records)
                print(f"   ✅ Loaded {len(records)} records from {dataset_name}")
                
            except Exception as e:
                error_msg = f"Error loading {dataset_name}: {e}"
                print(f"   ❌ {error_msg}")
                self.processing_stats["errors"].append(error_msg)
        
        print(f"📊 Total records loaded: {len(all_records)}")
        
        # Apply sampling if needed
        if max_records and len(all_records) > max_records:
            all_records = self._apply_sampling(all_records, max_records, sampling, dataset_names)
        
        self.processing_stats["records_loaded"] = len(all_records)
        return all_records
    
    def _get_domain(self, dataset_name: str) -> str:
        """Get domain category for a dataset"""
        for domain, datasets in DOMAIN_CATEGORIES.items():
            if dataset_name in datasets:
                return domain
        return "other"
    
    def _apply_sampling(self, records: List[Dict], max_records: int, 
                       sampling: str, dataset_names: List[str]) -> List[Dict]:
        """Apply sampling strategy to reduce record count"""
        
        print(f"🎯 Applying {sampling} sampling to get {max_records} records...")
        
        if sampling == "first":
            return records[:max_records]
        
        elif sampling == "random":
            return random.sample(records, max_records)
        
        elif sampling == "stratified":
            # Balance across datasets
            per_dataset = max_records // len(dataset_names)
            sampled = []
            
            for dataset_name in dataset_names:
                dataset_records = [r for r in records if r['source_dataset'] == dataset_name]
                if dataset_records:
                    sample_size = min(per_dataset, len(dataset_records))
                    sampled.extend(random.sample(dataset_records, sample_size))
            
            # Fill remaining slots randomly
            remaining = max_records - len(sampled)
            if remaining > 0:
                unused = [r for r in records if r not in sampled]
                if unused:
                    sampled.extend(random.sample(unused, min(remaining, len(unused))))
            
            return sampled
        
        elif sampling == "weighted":
            # Sample proportionally to dataset sizes
            dataset_weights = {}
            total_available = 0
            
            for dataset_name in dataset_names:
                dataset_records = [r for r in records if r['source_dataset'] == dataset_name]
                dataset_weights[dataset_name] = len(dataset_records)
                total_available += len(dataset_records)
            
            sampled = []
            for dataset_name in dataset_names:
                dataset_records = [r for r in records if r['source_dataset'] == dataset_name]
                if dataset_records:
                    proportion = dataset_weights[dataset_name] / total_available
                    sample_size = int(max_records * proportion)
                    sample_size = min(sample_size, len(dataset_records))
                    
                    if sample_size > 0:
                        sampled.extend(random.sample(dataset_records, sample_size))
            
            return sampled
        
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling}")
    
    def extract_documents_and_cache_qa(self, records: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract documents for graph processing and cache Q&A pairs separately.
        
        This is the key method that separates concerns:
        - Documents go to graph processing
        - Questions/answers go to evaluation cache
        """
        
        print(f"📄 Extracting documents from {len(records)} records...")
        
        documents = []
        
        for record_idx, record in enumerate(records):
            # Cache Q&A data for evaluation (NOT for graph processing)
            qa_data = {
                "record_id": record.get('id', f"record_{record_idx}"),
                "question": record["question"],  # This is the question
                "ground_truth": record["response"],  # This is the expected answer
                "source_dataset": record.get('source_dataset', 'unknown'),
                "domain": record.get('domain', 'unknown')
            }
            self.evaluation_cache.append(qa_data)
            
            # Extract documents for graph processing
            for doc_idx, doc_text in enumerate(record["documents"]):
                # Each record has 4 documents
                doc_id = f"ragbench_{record.get('id', record_idx)}_doc_{doc_idx}"
                
                documents.append({
                    "id": doc_id,
                    "text": doc_text,
                    "source_dataset": record.get('source_dataset', 'unknown'),
                    "domain": record.get('domain', 'unknown'), 
                    "record_id": record.get('id', f"record_{record_idx}"),
                    "doc_index": doc_idx,
                    "record_index": record_idx
                })
        
        print(f"✅ Extracted {len(documents)} documents for processing")
        print(f"✅ Cached {len(self.evaluation_cache)} Q&A pairs for evaluation")
        
        return documents
    
    def process_documents_to_graph(self, documents: List[Dict[str, Any]]):
        """
        Process documents through the graph processor.
        
        This adapts your existing PDF-based processor to handle text documents.
        """
        
        print(f"\n🔄 Processing {len(documents)} documents to Neo4j graph...")
        
        # Setup database schema
        print("   Setting up database schema...")
        self.processor.setup_database_schema()
        
        # Trigger entity label discovery and approval if needed
        if not hasattr(self.processor, 'discovered_labels') or not self.processor.discovered_labels:
            print("\n🔎 Discovering entity labels from document corpus...")
            # Sample some text from the documents for label discovery
            sample_text = ""
            for doc in documents[:5]:  # Use first 5 documents as sample
                sample_text += doc['text'][:1000] + "\n\n"  # First 1000 chars from each
            
            proposed_labels = self.processor.discover_labels_for_text(sample_text)
            self.processor.discovered_labels = self.processor._approve_labels_cli(proposed_labels)
            print(f"\n✅ Using labels: {self.processor.discovered_labels}")
        
        # Process documents in batches for better memory management
        batch_size = 50  # Process 50 documents at a time
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        processed_count = 0
        total_chunks = 0
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]
            
            print(f"   Processing batch {batch_idx + 1}/{total_batches} ({len(batch_docs)} docs)...")
            
            for doc in tqdm(batch_docs, desc=f"Batch {batch_idx + 1}"):
                try:
                    result = self._process_single_document(doc)
                    processed_count += 1
                    total_chunks += result.get('chunks_created', 0)
                    
                except Exception as e:
                    error_msg = f"Error processing document {doc['id']}: {e}"
                    print(f"      ❌ {error_msg}")
                    self.processing_stats["errors"].append(error_msg)
        
        # Update processing stats
        self.processing_stats["documents_processed"] = processed_count
        self.processing_stats["chunks_created"] = total_chunks
        
        print(f"✅ Completed processing: {processed_count}/{len(documents)} documents")
        print(f"   📊 Created {total_chunks} chunks total")
    
    def _process_single_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single document through the graph processor.
        
        Adapts your PDF processor to handle text documents.
        """
        
        # Create document embedding
        doc_text = doc['text']
        doc_embedding = self.processor.create_embedding(doc_text[:8000])  # Limit for embedding
        
        # Chunk the text
        chunks = self.processor.chunk_text(doc_text, doc['source_dataset'])
        
        # Process with Neo4j (adapted from your process_document method)
        with self.processor.driver.session() as session:
            doc_id = f"doc_{doc['id']}"
            
            # Create document node
            session.run("""
                CREATE (d:Document {
                    id: $doc_id,
                    name: $doc_name,
                    text: $text,
                    embedding: $embedding,
                    source_dataset: $source_dataset,
                    domain: $domain,
                    record_id: $record_id,
                    doc_index: $doc_index,
                    chunk_count: $chunk_count,
                    created_at: datetime()
                })
            """, 
                doc_id=doc_id,
                doc_name=doc['id'],
                text=doc_text[:1000],  # Store first 1000 chars
                embedding=doc_embedding,
                source_dataset=doc['source_dataset'],
                domain=doc['domain'],
                record_id=doc['record_id'],
                doc_index=doc['doc_index'],
                chunk_count=len(chunks)
            )
            
            # Process each chunk (using your existing chunk processing logic)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{chunk['index']}"
                
                # Create chunk embedding
                chunk_embedding = self.processor.create_embedding(chunk['text'])
                
                # Create chunk node
                session.run("""
                    CREATE (c:Chunk {
                        id: $chunk_id,
                        text: $text,
                        index: $index,
                        embedding: $embedding,
                        source: $source,
                        type: $type
                    })
                """, 
                    chunk_id=chunk_id,
                    text=chunk['text'],
                    index=chunk['index'],
                    embedding=chunk_embedding,
                    source=chunk['source'],
                    type=chunk['type']
                )
                
                # Link chunk to document
                session.run("""
                    MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id})
                    CREATE (c)-[:PART_OF]->(d)
                """, doc_id=doc_id, chunk_id=chunk_id)
                
                # Extract entities from chunk using approved labels
                entities = self.processor.extract_entities_dynamic(
                    chunk['text'], 
                    allowed_labels=self.processor.discovered_labels
                )
                
                # Process entities (using your existing entity processing logic)
                chunk_entity_ids = []
                entity_counter = 0
                
                # Process each entity type (dynamic format)
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        entity_counter += 1
                        entity_name = entity['text']  # Dynamic format uses 'text' key
                        entity_description = entity.get('description', '')
                        
                        # Create entity embedding
                        embedding_text = f"{entity_name}: {entity_description}" if entity_description else entity_name
                        entity_embedding = self.processor.create_embedding(embedding_text)
                        
                        # Use the dynamic entity type directly (already normalized)
                        entity_label = entity_type
                        
                        # Create entity node
                        session.run(f"""
                            MERGE (e:{entity_label}:__Entity__ {{name: $name}})
                            ON CREATE SET e.description = $description, e.embedding = $embedding,
                                        e.id = $name, e.entity_type = $entity_type, 
                                        e.human_readable_id = $human_id
                            ON MATCH SET e.description = CASE WHEN e.description IS NULL THEN $description ELSE e.description END,
                                       e.embedding = CASE WHEN e.embedding IS NULL THEN $embedding ELSE e.embedding END,
                                       e.id = $name, e.entity_type = $entity_type,
                                       e.human_readable_id = CASE WHEN e.human_readable_id IS NULL THEN $human_id ELSE e.human_readable_id END
                            WITH e
                            MERGE (c:Chunk {{id: $chunk_id}})
                            MERGE (c)-[:HAS_ENTITY]->(e)
                        """, 
                            name=entity_name,
                            description=entity_description,
                            embedding=entity_embedding,
                            chunk_id=chunk_id,
                            human_id=entity_counter,
                            entity_type=entity_label
                        )
                        
                        chunk_entity_ids.append((entity_label, entity_name))
                
                # Create relationships between entities in the same chunk
                if len(chunk_entity_ids) > 1:
                    self.processor.create_entity_relationships(session, chunk_entity_ids)
        
        return {
            "status": "success",
            "chunks_created": len(chunks),
            "entities_extracted": entity_counter
        }
    
    def save_evaluation_data(self, output_path: str = "benchmark/ragbench/data/evaluation_data.jsonl"):
        """Save Q&A pairs in JSONL format for evaluation"""
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSONL (one JSON object per line)
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa_pair in self.evaluation_cache:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved evaluation data to {output_path}")
        print(f"   📊 {len(self.evaluation_cache)} Q&A pairs in JSONL format")
        
        return output_path
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            "evaluation_pairs_cached": len(self.evaluation_cache),
            "success_rate": (self.processing_stats["documents_processed"] / 
                           max(1, self.processing_stats["records_loaded"] * 4)) * 100  # 4 docs per record
        }
    
    def run_preset(self, preset_name: str) -> Dict[str, Any]:
        """Run a predefined ingestion preset"""
        
        if preset_name not in INGESTION_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(INGESTION_PRESETS.keys())}")
        
        config = INGESTION_PRESETS[preset_name]
        
        print(f"🚀 Running RAGBench ingestion preset: {preset_name}")
        print(f"   {config['description']}")
        print(f"   Estimated storage: {config['estimated_storage_gb']} GB")
        print(f"   Estimated RAM: {config['estimated_ram_gb']} GB") 
        print(f"   Estimated cost: ${config['estimated_cost_usd']}")
        
        # Confirm with user for expensive operations
        if config['estimated_cost_usd'] > 50:
            confirm = input(f"\n⚠️  This operation may cost ~${config['estimated_cost_usd']}. Continue? [y/N]: ")
            if confirm.lower() not in ['y', 'yes']:
                print("❌ Operation cancelled by user")
                return {"status": "cancelled"}
        
        # Load dataset
        records = self.load_dataset_subset(
            dataset_names=config['datasets'],
            split=config['split'],
            max_records=config['max_records'],
            sampling=config['sampling']
        )
        
        # Extract documents and cache Q&A
        documents = self.extract_documents_and_cache_qa(records)
        
        # Process to graph
        self.process_documents_to_graph(documents)
        
        # Save evaluation data
        eval_path = self.save_evaluation_data(f"benchmark/ragbench/data/{preset_name}_eval.jsonl")
        
        # Return results
        stats = self.get_processing_stats()
        
        return {
            "status": "completed",
            "preset": preset_name,
            "config": config,
            "stats": stats,
            "evaluation_data_path": eval_path
        }
    
    def close(self):
        """Close database connections"""
        if hasattr(self.processor, 'close'):
            self.processor.close()


def main():
    """Example usage"""
    
    # Initialize ingester
    ingester = RAGBenchIngester(processor_type="basic")
    
    try:
        # Run micro preset
        result = ingester.run_preset("micro")
        
        if result["status"] == "completed":
            print(f"\n✅ Ingestion completed successfully!")
            print(f"📊 Processing Statistics:")
            stats = result["stats"]
            for key, value in stats.items():
                if key != "errors":
                    print(f"   {key}: {value}")
            
            if stats["errors"]:
                print(f"\n⚠️  Errors encountered:")
                for error in stats["errors"][:5]:  # Show first 5 errors
                    print(f"   - {error}")
        
    finally:
        ingester.close()


if __name__ == "__main__":
    main()
