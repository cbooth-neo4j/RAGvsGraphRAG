"""
RAGBench Evaluator

Bridge between RAGBench ingestion and ragas_benchmark.py evaluation.
Converts JSONL evaluation data to formats compatible with existing benchmark system.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


class RAGBenchEvaluator:
    """
    Converts RAGBench evaluation data to formats compatible with ragas_benchmark.py
    """
    
    def __init__(self, evaluation_data_path: str):
        """
        Initialize evaluator with path to JSONL evaluation data.
        
        Args:
            evaluation_data_path: Path to JSONL file with Q&A pairs
        """
        self.evaluation_data_path = evaluation_data_path
        self.qa_pairs = self._load_evaluation_data()
    
    def _load_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load Q&A pairs from JSONL file"""
        
        if not Path(self.evaluation_data_path).exists():
            raise FileNotFoundError(f"Evaluation data not found: {self.evaluation_data_path}")
        
        qa_pairs = []
        with open(self.evaluation_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    qa_pair = json.loads(line.strip())
                    qa_pairs.append(qa_pair)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping invalid JSON on line {line_num}: {e}")
        
        print(f"📊 Loaded {len(qa_pairs)} Q&A pairs from {self.evaluation_data_path}")
        return qa_pairs
    
    def create_benchmark_jsonl(self, 
                             output_path: str = "benchmark/ragbench_benchmark.jsonl",
                             filter_domain: Optional[str] = None,
                             filter_dataset: Optional[str] = None,
                             max_questions: Optional[int] = None) -> str:
        """
        Convert RAGBench Q&A pairs to JSONL format for ragas_benchmark.py.
        
        Args:
            output_path: Where to save the JSONL file
            filter_domain: Only include questions from this domain (medical, financial, etc.)
            filter_dataset: Only include questions from this dataset (covidqa, finqa, etc.)
            max_questions: Maximum number of questions to include
            
        Returns:
            Path to created JSONL file
        """
        
        # Filter Q&A pairs if needed
        filtered_pairs = self.qa_pairs
        
        if filter_domain:
            filtered_pairs = [qa for qa in filtered_pairs if qa.get('domain') == filter_domain]
            print(f"🔍 Filtered to {len(filtered_pairs)} questions from {filter_domain} domain")
        
        if filter_dataset:
            filtered_pairs = [qa for qa in filtered_pairs if qa.get('source_dataset') == filter_dataset]
            print(f"🔍 Filtered to {len(filtered_pairs)} questions from {filter_dataset} dataset")
        
        if max_questions and len(filtered_pairs) > max_questions:
            filtered_pairs = filtered_pairs[:max_questions]
            print(f"🔍 Limited to first {max_questions} questions")
        
        if not filtered_pairs:
            raise ValueError("No Q&A pairs match the specified filters")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSONL format (one JSON object per line)
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in filtered_pairs:
                # Create benchmark record with all metadata
                benchmark_record = {
                    'question': qa['question'],
                    'ground_truth': qa['ground_truth'],
                    'record_id': qa.get('record_id'),
                    'source_dataset': qa.get('source_dataset'),
                    'domain': qa.get('domain')
                }
                f.write(json.dumps(benchmark_record, ensure_ascii=False) + '\n')
        
        print(f"📊 Created benchmark JSONL: {output_path}")
        print(f"   Questions: {len(filtered_pairs)}")
        print(f"   Format: JSONL with rich metadata (compatible with ragas_benchmark.py)")
        
        return output_path
    
    def create_domain_specific_jsonl(self, output_dir: str = "benchmark/ragbench/data") -> Dict[str, str]:
        """
        Create separate JSONL files for each domain.
        
        Returns:
            Dictionary mapping domain names to JSONL file paths
        """
        
        # Group by domain
        domain_groups = {}
        for qa in self.qa_pairs:
            domain = qa.get('domain', 'unknown')
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(qa)
        
        # Create JSONL for each domain
        jsonl_paths = {}
        
        for domain, qa_pairs in domain_groups.items():
            if len(qa_pairs) < 5:  # Skip domains with too few questions
                print(f"⚠️  Skipping {domain} domain (only {len(qa_pairs)} questions)")
                continue
            
            output_path = f"{output_dir}/ragbench_{domain}_benchmark.jsonl"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    benchmark_record = {
                        'question': qa['question'],
                        'ground_truth': qa['ground_truth'],
                        'record_id': qa.get('record_id'),
                        'source_dataset': qa.get('source_dataset'),
                        'domain': qa.get('domain')
                    }
                    f.write(json.dumps(benchmark_record, ensure_ascii=False) + '\n')
            
            jsonl_paths[domain] = output_path
            print(f"📊 Created {domain} benchmark: {output_path} ({len(qa_pairs)} questions)")
        
        return jsonl_paths
    
    def create_dataset_specific_jsonl(self, output_dir: str = "benchmark/ragbench/data") -> Dict[str, str]:
        """
        Create separate JSONL files for each source dataset.
        
        Returns:
            Dictionary mapping dataset names to JSONL file paths
        """
        
        # Group by source dataset
        dataset_groups = {}
        for qa in self.qa_pairs:
            dataset = qa.get('source_dataset', 'unknown')
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(qa)
        
        # Create JSONL for each dataset
        jsonl_paths = {}
        
        for dataset, qa_pairs in dataset_groups.items():
            if len(qa_pairs) < 5:  # Skip datasets with too few questions
                print(f"⚠️  Skipping {dataset} dataset (only {len(qa_pairs)} questions)")
                continue
            
            output_path = f"{output_dir}/ragbench_{dataset}_benchmark.jsonl"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    benchmark_record = {
                        'question': qa['question'],
                        'ground_truth': qa['ground_truth'],
                        'record_id': qa.get('record_id'),
                        'source_dataset': qa.get('source_dataset'),
                        'domain': qa.get('domain')
                    }
                    f.write(json.dumps(benchmark_record, ensure_ascii=False) + '\n')
            
            jsonl_paths[dataset] = output_path
            print(f"📊 Created {dataset} benchmark: {output_path} ({len(qa_pairs)} questions)")
        
        return jsonl_paths
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get statistics about the evaluation dataset"""
        
        # Count by domain
        domain_counts = {}
        dataset_counts = {}
        
        for qa in self.qa_pairs:
            domain = qa.get('domain', 'unknown')
            dataset = qa.get('source_dataset', 'unknown')
            
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        # Calculate question/answer length statistics
        question_lengths = [len(qa['question']) for qa in self.qa_pairs]
        answer_lengths = [len(qa['ground_truth']) for qa in self.qa_pairs]
        
        return {
            "total_questions": len(self.qa_pairs),
            "domains": domain_counts,
            "datasets": dataset_counts,
            "question_length_stats": {
                "min": min(question_lengths) if question_lengths else 0,
                "max": max(question_lengths) if question_lengths else 0,
                "avg": sum(question_lengths) / len(question_lengths) if question_lengths else 0
            },
            "answer_length_stats": {
                "min": min(answer_lengths) if answer_lengths else 0,
                "max": max(answer_lengths) if answer_lengths else 0,
                "avg": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
            }
        }
    
    def print_evaluation_summary(self):
        """Print a summary of the evaluation dataset"""
        
        stats = self.get_evaluation_stats()
        
        print(f"\n📊 RAGBench Evaluation Dataset Summary")
        print("=" * 50)
        print(f"Total Questions: {stats['total_questions']}")
        
        print(f"\nBy Domain:")
        for domain, count in sorted(stats['domains'].items()):
            print(f"   {domain}: {count} questions")
        
        print(f"\nBy Source Dataset:")
        for dataset, count in sorted(stats['datasets'].items()):
            print(f"   {dataset}: {count} questions")
        
        print(f"\nQuestion Length Stats:")
        q_stats = stats['question_length_stats']
        print(f"   Min: {q_stats['min']} chars, Max: {q_stats['max']} chars, Avg: {q_stats['avg']:.1f} chars")
        
        print(f"\nAnswer Length Stats:")
        a_stats = stats['answer_length_stats']
        print(f"   Min: {a_stats['min']} chars, Max: {a_stats['max']} chars, Avg: {a_stats['avg']:.1f} chars")


import os  # Add missing import


def main():
    """Example usage"""
    
    # Path to evaluation data (created by ingester)
    eval_data_path = "benchmark/ragbench/data/micro_eval.jsonl"
    
    if not Path(eval_data_path).exists():
        print(f"❌ Evaluation data not found: {eval_data_path}")
        print("   Run the ingester first: python benchmark/ragbench/ingester.py")
        return
    
    # Initialize evaluator
    evaluator = RAGBenchEvaluator(eval_data_path)
    
    # Print summary
    evaluator.print_evaluation_summary()
    
    # Create main benchmark JSONL
    jsonl_path = evaluator.create_benchmark_jsonl("benchmark/ragbench_micro_benchmark.jsonl")
    
    # Create domain-specific JSONLs
    domain_jsonls = evaluator.create_domain_specific_jsonl()
    
    print(f"\n✅ Evaluation files created!")
    print(f"📄 Main benchmark: {jsonl_path}")
    print(f"📁 Domain-specific benchmarks: {len(domain_jsonls)} files")
    
    print(f"\n🚀 Ready for evaluation!")
    print(f"   Run: python benchmark/ragas_benchmark.py --all")
    print(f"   Or with custom JSONL: python benchmark/ragas_benchmark.py --all --jsonl {jsonl_path}")


if __name__ == "__main__":
    main()
