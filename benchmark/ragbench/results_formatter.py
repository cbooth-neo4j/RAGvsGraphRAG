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
import html


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
    
    def create_detailed_html_report(self, 
                                   output_path: str = "benchmark/ragbench/data/detailed_results.html",
                                   title: str = "RAGBench Evaluation Results") -> str:
        """
        Create a detailed HTML report showing all questions, responses, and scores.
        """
        
        if not self.results_data:
            raise ValueError("No results data to format. Add results first.")
        
        # Group results by question (multiple retrievers per question)
        questions_data = {}
        for result in self.results_data:
            question = result["question"]
            if question not in questions_data:
                questions_data[question] = {
                    "question": question,
                    "ground_truth": result["ground_truth"],
                    "metadata": result["metadata"],
                    "retrievers": []
                }
            
            questions_data[question]["retrievers"].append({
                "name": result["retriever_name"],
                "response": result["retriever_response"],
                "contexts": result["retrieved_contexts"],
                "scores": result["ragas_scores"]
            })
        
        # Generate HTML
        html_content = self._generate_html_template(title, questions_data)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Created detailed HTML report: {output_path}")
        print(f"   Questions: {len(questions_data)}")
        print(f"   Total evaluations: {len(self.results_data)}")
        
        return output_path
    
    def _generate_html_template(self, title: str, questions_data: Dict) -> str:
        """Generate HTML template for the detailed report"""
        
        # Calculate summary statistics
        total_questions = len(questions_data)
        retrievers = set()
        all_scores = {}
        
        for question_data in questions_data.values():
            for retriever in question_data["retrievers"]:
                retriever_name = retriever["name"]
                retrievers.add(retriever_name)
                
                if retriever_name not in all_scores:
                    all_scores[retriever_name] = {"scores": [], "count": 0}
                
                for metric, score in retriever["scores"].items():
                    if metric not in all_scores[retriever_name]:
                        all_scores[retriever_name][metric] = []
                    all_scores[retriever_name][metric].append(score)
                
                all_scores[retriever_name]["count"] += 1
        
        # Calculate averages
        summary_stats = {}
        for retriever_name, data in all_scores.items():
            summary_stats[retriever_name] = {}
            for metric, scores in data.items():
                if metric != "count" and scores:
                    summary_stats[retriever_name][metric] = sum(scores) / len(scores)
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .summary {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .summary-table th, .summary-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .summary-table th {{
            background-color: #e9ecef;
            font-weight: 600;
        }}
        .question-block {{
            margin-bottom: 40px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .question-header {{
            background: #007bff;
            color: white;
            padding: 15px 20px;
            font-weight: 600;
        }}
        .question-content {{
            padding: 20px;
        }}
        .ground-truth {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .ground-truth h4 {{
            margin-top: 0;
            color: #155724;
        }}
        .retriever-result {{
            margin-bottom: 25px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }}
        .retriever-header {{
            background: #f8f9fa;
            padding: 10px 15px;
            border-bottom: 1px solid #e0e0e0;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .scores {{
            display: flex;
            gap: 15px;
            font-size: 0.9em;
        }}
        .score {{
            background: #e9ecef;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .score.good {{ background: #d4edda; color: #155724; }}
        .score.medium {{ background: #fff3cd; color: #856404; }}
        .score.poor {{ background: #f8d7da; color: #721c24; }}
        .response {{
            padding: 15px;
            background: #f8f9fa;
        }}
        .contexts {{
            padding: 15px;
            background: #ffffff;
        }}
        .context-item {{
            background: #f1f3f4;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 4px;
            border-left: 3px solid #007bff;
            font-size: 0.9em;
        }}
        .metadata {{
            font-size: 0.8em;
            color: #6c757d;
            margin-top: 10px;
        }}
        .verification-box {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            margin-top: 15px;
            border-radius: 5px;
        }}
        .verification-box h5 {{
            margin-top: 0;
            color: #856404;
        }}
        .verification-box textarea {{
            width: 100%;
            min-height: 60px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px;
            font-family: inherit;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <h2>📊 Summary Statistics</h2>
            <p><strong>Total Questions:</strong> {total_questions}</p>
            <p><strong>Retrievers Compared:</strong> {', '.join(sorted(retrievers))}</p>
            
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Retriever</th>
                        <th>Context Recall</th>
                        <th>Faithfulness</th>
                        <th>Factual Correctness</th>
                        <th>Overall Avg</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add summary statistics
        for retriever_name in sorted(retrievers):
            stats = summary_stats.get(retriever_name, {})
            context_recall = stats.get('context_recall', 0)
            faithfulness = stats.get('faithfulness', 0)
            factual_correctness = stats.get('factual_correctness', 0)
            overall_avg = (context_recall + faithfulness + factual_correctness) / 3
            
            html_content += f"""
                    <tr>
                        <td><strong>{retriever_name}</strong></td>
                        <td>{context_recall:.3f}</td>
                        <td>{faithfulness:.3f}</td>
                        <td>{factual_correctness:.3f}</td>
                        <td><strong>{overall_avg:.3f}</strong></td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
"""
        
        # Add detailed questions and responses
        for i, (question, question_data) in enumerate(questions_data.items(), 1):
            metadata = question_data.get("metadata", {})
            domain = metadata.get("domain", "unknown")
            source_dataset = metadata.get("source_dataset", "unknown")
            
            html_content += f"""
        <div class="question-block">
            <div class="question-header">
                Question {i}: {html.escape(question)}
                <div class="metadata">
                    Domain: {domain} | Dataset: {source_dataset}
                </div>
            </div>
            <div class="question-content">
                <div class="ground-truth">
                    <h4>✅ Expected Answer (Ground Truth)</h4>
                    <p>{html.escape(question_data['ground_truth'])}</p>
                </div>
"""
            
            # Add each retriever's response
            for retriever in question_data["retrievers"]:
                scores = retriever["scores"]
                
                # Generate score badges with color coding
                score_html = ""
                for metric, score in scores.items():
                    if score >= 0.7:
                        score_class = "good"
                    elif score >= 0.4:
                        score_class = "medium"
                    else:
                        score_class = "poor"
                    
                    score_html += f'<span class="score {score_class}">{metric.replace("_", " ").title()}: {score:.3f}</span>'
                
                html_content += f"""
                <div class="retriever-result">
                    <div class="retriever-header">
                        <span>🤖 {retriever['name']}</span>
                        <div class="scores">{score_html}</div>
                    </div>
                    <div class="response">
                        <h5>Response:</h5>
                        <p>{html.escape(retriever['response'])}</p>
                    </div>
                    <div class="contexts">
                        <h5>Retrieved Context ({len(retriever['contexts'])} documents):</h5>
"""
                
                # Add retrieved contexts
                for j, context in enumerate(retriever['contexts'][:3], 1):  # Show first 3 contexts
                    truncated_context = context[:300] + "..." if len(context) > 300 else context
                    html_content += f'<div class="context-item"><strong>Context {j}:</strong> {html.escape(truncated_context)}</div>'
                
                html_content += """
                    </div>
                </div>
"""
            
            # Add verification section
            html_content += f"""
                <div class="verification-box">
                    <h5>🔍 Human Verification</h5>
                    <p>Which response was most accurate? Any observations?</p>
                    <textarea placeholder="Add your assessment here..."></textarea>
                </div>
            </div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        return html_content
    
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
    
    def create_summary_report(self, 
                             output_path: str = "benchmark/ragbench/data/summary_report.json") -> str:
        """
        Create a JSON summary report with aggregated statistics.
        """
        
        if not self.results_data:
            raise ValueError("No results data to format. Add results first.")
        
        # Group by retriever
        retriever_stats = {}
        question_count = len(set(result["question"] for result in self.results_data))
        
        for result in self.results_data:
            retriever = result["retriever_name"]
            
            if retriever not in retriever_stats:
                retriever_stats[retriever] = {
                    "total_evaluations": 0,
                    "scores": {},
                    "domains": {},
                    "datasets": {}
                }
            
            stats = retriever_stats[retriever]
            stats["total_evaluations"] += 1
            
            # Aggregate scores
            for metric, score in result["ragas_scores"].items():
                if metric not in stats["scores"]:
                    stats["scores"][metric] = []
                stats["scores"][metric].append(score)
            
            # Count domains and datasets
            domain = result["metadata"].get("domain", "unknown")
            dataset = result["metadata"].get("source_dataset", "unknown")
            
            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
            stats["datasets"][dataset] = stats["datasets"].get(dataset, 0) + 1
        
        # Calculate averages and create summary
        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": question_count,
            "total_evaluations": len(self.results_data),
            "retrievers": {}
        }
        
        for retriever, stats in retriever_stats.items():
            retriever_summary = {
                "total_evaluations": stats["total_evaluations"],
                "average_scores": {},
                "score_distribution": {},
                "domains_tested": stats["domains"],
                "datasets_tested": stats["datasets"]
            }
            
            # Calculate average scores
            for metric, scores in stats["scores"].items():
                avg_score = sum(scores) / len(scores)
                retriever_summary["average_scores"][metric] = avg_score
                
                # Score distribution
                good_count = sum(1 for s in scores if s >= 0.7)
                medium_count = sum(1 for s in scores if 0.4 <= s < 0.7)
                poor_count = sum(1 for s in scores if s < 0.4)
                
                retriever_summary["score_distribution"][metric] = {
                    "good (≥0.7)": good_count,
                    "medium (0.4-0.7)": medium_count,
                    "poor (<0.4)": poor_count
                }
            
            summary["retrievers"][retriever] = retriever_summary
        
        # Save summary
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Created summary report: {output_path}")
        
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
