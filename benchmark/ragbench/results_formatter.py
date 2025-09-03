"""
RAGBench Results Formatter

Creates human-readable outputs showing questions, responses, and RAGAS scores
for easy verification and analysis of retriever performance.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class RAGBenchResultsFormatter:
    """
    Formats RAGBench evaluation results into human-readable formats
    """
    
    def __init__(self):
        self.results_data = []
    
    def add_evaluation_result(self, 
                            question: str,
                            ground_truth: str,
                            retriever_name: str,
                            retriever_response: str,
                            retrieved_contexts: List[str],
                            ragas_scores: Dict[str, float],
                            metadata: Optional[Dict] = None):
        """
        Add a single evaluation result for formatting.
        
        Args:
            question: The question that was asked
            ground_truth: Expected correct answer
            retriever_name: Name of the retriever (e.g., "ChromaDB RAG", "GraphRAG")
            retriever_response: Actual response from the retriever
            retrieved_contexts: List of document contexts retrieved
            ragas_scores: Individual RAGAS scores for this question
            metadata: Additional metadata (domain, dataset, etc.)
        """
        
        result = {
            "question": question,
            "ground_truth": ground_truth,
            "retriever_name": retriever_name,
            "retriever_response": retriever_response,
            "retrieved_contexts": retrieved_contexts,
            "ragas_scores": ragas_scores,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.results_data.append(result)
    
    def create_comparison_csv(self, 
                             output_path: str = "benchmark/ragbench/data/detailed_comparison.csv") -> str:
        """
        Create a detailed CSV with all questions, responses, and scores.
        """
        
        if not self.results_data:
            raise ValueError("No results data to format. Add results first.")
        
        # Prepare data for CSV
        csv_data = []
        
        for result in self.results_data:
            # Flatten the data structure
            row = {
                "question": result["question"],
                "ground_truth": result["ground_truth"],
                "retriever": result["retriever_name"],
                "response": result["retriever_response"],
                "context_count": len(result["retrieved_contexts"]),
                "context_preview": result["retrieved_contexts"][0][:200] + "..." if result["retrieved_contexts"] else "",
            }
            
            # Add RAGAS scores
            for metric, score in result["ragas_scores"].items():
                row[f"ragas_{metric}"] = score
            
            # Add metadata
            for key, value in result["metadata"].items():
                row[f"meta_{key}"] = value
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"📊 Created detailed comparison CSV: {output_path}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        return output_path
    
    def load_from_ragas_results(self, 
                               ragas_datasets: Dict[str, List[Dict]], 
                               ragas_results: Dict[str, Dict],
                               evaluation_metadata: Optional[Dict] = None):
        """
        Load results from ragas_benchmark.py output format.
        
        Args:
            ragas_datasets: Dictionary mapping retriever names to their datasets
            ragas_results: Dictionary mapping retriever names to their RAGAS scores
            evaluation_metadata: Additional metadata about the evaluation
        """
        
        # Clear existing results
        self.results_data = []
        
        # Process each retriever's results
        for retriever_name, dataset in ragas_datasets.items():
            retriever_scores = ragas_results.get(retriever_name, {})
            
            # Process each question in the dataset
            for i, record in enumerate(dataset):
                question = record.get("user_input", "")
                ground_truth = record.get("reference", "")
                response = record.get("response", "")
                contexts = record.get("retrieved_contexts", [])
                
                # For individual question scores, we'd need to modify ragas_benchmark.py
                # For now, use the average scores for each question
                individual_scores = {
                    metric: score for metric, score in retriever_scores.items()
                    if isinstance(score, (int, float))
                }
                
                # Add metadata
                metadata = evaluation_metadata or {}
                if hasattr(record, 'get'):
                    metadata.update({
                        "source_dataset": record.get("source_dataset", "unknown"),
                        "domain": record.get("domain", "unknown")
                    })
                
                self.add_evaluation_result(
                    question=question,
                    ground_truth=ground_truth,
                    retriever_name=retriever_name,
                    retriever_response=response,
                    retrieved_contexts=contexts,
                    ragas_scores=individual_scores,
                    metadata=metadata
                )
        
        print(f"📊 Loaded {len(self.results_data)} evaluation results from RAGAS output")


def main():
    """Example usage"""
    
    # Example of how this would be used with ragas_benchmark.py results
    formatter = RAGBenchResultsFormatter()
    
    # This would typically be called from ragas_benchmark.py after evaluation
    print("📄 RAGBench Results Formatter")
    print("   This module formats evaluation results into human-readable reports.")
    print("   Integration with ragas_benchmark.py coming next...")


if __name__ == "__main__":
    main()
