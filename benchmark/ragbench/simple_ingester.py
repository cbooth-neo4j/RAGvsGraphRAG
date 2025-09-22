"""
Simplified RAGBench Ingester

This replaces the complex ingester with a simple wrapper around the existing graph processor.
No code duplication - just data loading and format conversion.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .configs import INGESTION_PRESETS, DATASET_SIZES, DOMAIN_CATEGORIES


class SimpleRAGBenchIngester:
    """
    Simple RAGBench ingester that just loads data and delegates to graph_processor.
    No code duplication!
    """
    
    def __init__(self, processor_type: str = "basic"):
        """Initialize with the appropriate graph processor"""
        if processor_type == "basic":
            from data_processors.build_graph.main_processor import CustomGraphProcessor
            self.processor = CustomGraphProcessor()
        elif processor_type == "advanced":  
            from data_processors.advanced_graph_processor import AdvancedGraphProcessor
            self.processor = AdvancedGraphProcessor()
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        self.evaluation_cache = []
    
    def run_preset(self, preset_name: str) -> Dict[str, Any]:
        """Load RAGBench data and process through graph processor"""
        
        if preset_name not in INGESTION_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        config = INGESTION_PRESETS[preset_name]
        
        print(f"🚀 Running RAGBench preset: {preset_name}")
        print(f"   {config['description']}")
        
        # Step 1: Load RAGBench data
        records = self._load_dataset_subset(
            dataset_names=config['datasets'],
            split=config['split'],
            max_records=config['max_records'],
            sampling=config['sampling']
        )
        
        # Step 2: Cache Q&A pairs for evaluation first
        for record_idx, record in enumerate(records):
            self.evaluation_cache.append({
                "record_id": record.get('id', f"record_{record_idx}"),
                "question": record["question"],
                "ground_truth": record["response"], 
                "source_dataset": record.get('dataset_name', 'unknown'),
                "domain": self._get_domain(record.get('dataset_name', 'unknown'))
            })
        
        # Step 3: Extract all document texts and sources for batch processing
        texts = []
        sources = []
        
        for record_idx, record in enumerate(records):
            for doc_idx, doc_text in enumerate(record["documents"]):
                texts.append(doc_text)
                sources.append(f"ragbench_{record.get('id', record_idx)}_doc_{doc_idx}")
        
        # Step 4: Use the main processor method (includes clearing DB and advanced processing)
        processing_result = self.processor.process_ragbench_documents(
            texts=texts,
            sources=sources,
            use_enhanced_discovery=True,
            domain_hint=None  # Auto-detect from dataset
        )
        
        processed_count = processing_result.get('successful_documents', 0)
        total_chunks = processing_result.get('total_chunks_created', 0)
        
        # Step 4: Save evaluation data
        eval_path = self._save_evaluation_data(f"benchmark/ragbench/data/{preset_name}_eval.jsonl")
        
        return {
            "status": "completed",
            "preset": preset_name,
            "config": config,
            "stats": {
                "records_loaded": len(records),
                "documents_processed": processed_count,
                "chunks_created": total_chunks,
                "evaluation_pairs_cached": len(self.evaluation_cache),
                "success_rate": (processed_count / (len(records) * 4)) * 100 if records else 0
            },
            "evaluation_data_path": eval_path
        }
    
    def load_and_process_custom(self, datasets: List[str], records_per_dataset: int = 50) -> Dict[str, Any]:
        """Load and process custom dataset selection"""
        
        print(f"🚀 Processing custom RAGBench datasets: {', '.join(datasets)}")
        print(f"   Records per dataset: {records_per_dataset}")
        
        # Step 1: Load data from specified datasets
        all_records = []
        for dataset_name in datasets:
            records = self._load_dataset_subset(
                dataset_names=[dataset_name],
                split="test",
                max_records=records_per_dataset,
                sampling="first"
            )
            all_records.extend(records)
            print(f"   ✅ Loaded {len(records)} records from {dataset_name}")
        
        # Step 2: Cache Q&A pairs for evaluation first
        for record_idx, record in enumerate(all_records):
            self.evaluation_cache.append({
                "record_id": record.get('id', f"record_{record_idx}"),
                "question": record["question"],
                "ground_truth": record["response"],
                "source_dataset": record.get('source_dataset', 'unknown'),
                "domain": self._get_domain(record.get('source_dataset', 'unknown'))
            })
        
        # Step 3: Extract all document texts and sources for batch processing
        texts = []
        sources = []
        
        for record_idx, record in enumerate(all_records):
            for doc_idx, doc_text in enumerate(record["documents"]):
                if doc_text and doc_text.strip():
                    texts.append(doc_text)
                    sources.append(f"custom_{record_idx}_{doc_idx}")
        
        # Step 4: Use the main processor method (includes clearing DB and advanced processing)
        processing_result = self.processor.process_ragbench_documents(
            texts=texts,
            sources=sources,
            use_enhanced_discovery=True,
            domain_hint=None  # Auto-detect from dataset
        )
        
        processed_count = processing_result.get('successful_documents', 0)
        total_chunks = processing_result.get('total_chunks_created', 0)
        
        # Step 4: Save evaluation data
        custom_name = "_".join(datasets)
        eval_path = self._save_evaluation_data(f"benchmark/ragbench/data/{custom_name}_custom_eval.jsonl")
        
        return {
            "status": "completed",
            "custom_datasets": datasets,
            "config": {
                "description": f"Custom processing: {', '.join(datasets)} with {records_per_dataset} records each",
                "datasets": datasets,
                "records_per_dataset": records_per_dataset
            },
            "stats": {
                "records_loaded": len(all_records),
                "documents_processed": processed_count,
                "chunks_created": total_chunks,
                "evaluation_pairs_cached": len(self.evaluation_cache),
                "success_rate": (processed_count / (len(all_records) * 4)) * 100 if all_records else 0
            },
            "evaluation_data_path": eval_path
        }
    
    def _load_dataset_subset(self, dataset_names: Union[str, List[str]], split: str = "test",
                           max_records: Optional[int] = None, sampling: str = "random") -> List[Dict]:
        """Load RAGBench dataset subset"""
        
        if dataset_names == "all":
            dataset_names = list(DATASET_SIZES.keys())
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        print(f"📥 Loading RAGBench datasets: {dataset_names}")
        
        all_records = []
        for dataset_name in dataset_names:
            try:
                # Load dataset (trust_remote_code deprecated in datasets 4.0+)
                dataset = load_dataset("galileo-ai/ragbench", dataset_name, split=split)
                
                # Convert to list of dictionaries - simple approach
                records = []
                for i, record in enumerate(dataset):
                    # Extract only the fields we need
                    record_dict = {
                        'id': record.get('id', f'record_{i}'),
                        'question': record.get('question', ''),
                        'documents': record.get('documents', []),
                        'response': record.get('response', ''),
                        'source_dataset': dataset_name
                    }
                    records.append(record_dict)
                    
                    # Limit records if max_records is specified for this dataset
                    if max_records and len(records) >= max_records:
                        break
                
                all_records.extend(records)
                print(f"   ✅ Loaded {len(records)} records from {dataset_name}")
            except Exception as e:
                print(f"   ❌ Error loading {dataset_name}: {e}")
                print(f"       Trying alternative loading method...")
                try:
                    # Alternative: load without specifying split first
                    dataset_full = load_dataset("galileo-ai/ragbench", dataset_name)
                    if split in dataset_full:
                        dataset = dataset_full[split]
                        records = []
                        for record in dataset:
                            record_dict = {
                                'id': record.get('id', ''),
                                'question': record.get('question', ''),
                                'documents': record.get('documents', []),
                                'response': record.get('response', ''),
                                'source_dataset': dataset_name
                            }
                            records.append(record_dict)
                        all_records.extend(records)
                        print(f"   ✅ Alternative method: Loaded {len(records)} records from {dataset_name}")
                    else:
                        print(f"   ❌ Split '{split}' not found in {dataset_name}")
                except Exception as e2:
                    print(f"   ❌ Alternative method also failed: {e2}")
        
        # Note: Sampling is now handled per-dataset during loading for better control
        
        print(f"📊 Total records to process: {len(all_records)}")
        return all_records
    
    def _get_domain(self, dataset_name: str) -> str:
        """Get domain category for a dataset"""
        for domain, datasets in DOMAIN_CATEGORIES.items():
            if dataset_name in datasets:
                return domain
        return "other"
    
    def _save_evaluation_data(self, output_path: str) -> str:
        """Save Q&A pairs in JSONL format"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa_pair in self.evaluation_cache:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved {len(self.evaluation_cache)} Q&A pairs to {output_path}")
        return output_path
    
    def close(self):
        """Close database connections"""
        if hasattr(self.processor, 'close'):
            self.processor.close()


def main():
    """Example usage"""
    ingester = SimpleRAGBenchIngester(processor_type="basic")
    
    try:
        result = ingester.run_preset("nano")
        print(f"✅ Completed: {result}")
    finally:
        ingester.close()


if __name__ == "__main__":
    main()
