"""
Simple RAGAS Benchmark: RAG vs GraphRAG Evaluation

This script evaluates both ChromaDB and GraphRAG approaches using RAGAS framework
following the exact pattern from the RAGAS documentation.
"""

import pandas as pd
import json
import time
import math
import sys
import os
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Import visualization module
try:
    from .visualizations import create_visualizations
except ImportError:
    # Fallback for when script is run directly
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from visualizations import create_visualizations

# Add parent directory to path so we can import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Basic RAGAS setup following the documentation
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# RAGAS imports
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# Import our RAG systems
from RAGvsGraphRAG import (
    query_chroma_with_llm,
    query_neo4j_with_llm,
    query_neo4j_text2cypher
)

# Initialize LLM and embeddings for RAGAS
SEED = 42
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={"seed": SEED},
    max_retries=0
)
embeddings = OpenAIEmbeddings()

def load_benchmark_data(csv_path: str = "benchmark/benchmark.csv") -> List[Dict[str, str]]:
    """Load benchmark questions and ground truth answers"""
    df = pd.read_csv(csv_path, delimiter=';')
    
    benchmark_data = []
    for _, row in df.iterrows():
        benchmark_data.append({
            'question': row['question'],
            'ground_truth': row['ground_truth']
        })
    
    print(f"✅ Loaded {len(benchmark_data)} benchmark questions")
    return benchmark_data

def collect_evaluation_data_simple(benchmark_data: List[Dict[str, str]], approach: str = "chroma") -> List[Dict[str, Any]]:
    """
    Collect evaluation data following RAGAS documentation pattern
    """
    print(f"\n🔄 Collecting evaluation data for {approach.upper()} approach...")
    
    dataset = []
    
    for i, item in enumerate(benchmark_data, 1):
        query = item['question']
        reference = item['ground_truth']
        
        print(f"  Processing question {i}/{len(benchmark_data)}: {query[:60]}...")
        
        try:
            # Query the appropriate RAG system
            if approach == "chroma":
                result = query_chroma_with_llm(query, k=1)
            elif approach == "graphrag":
                result = query_neo4j_with_llm(query, k=5)
            elif approach == "text2cypher":
                result = query_neo4j_text2cypher(query)
            else:
                raise ValueError(f"Unknown approach: {approach}")
            
            # Extract retrieved contexts as simple list of strings (RAGAS format)
            retrieved_contexts = []
            for detail in result.get('retrieval_details', []):
                content = detail.get('content', '')
                if content:
                    # Truncate very long content to avoid issues
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    retrieved_contexts.append(content)
            
            # Ensure we have at least some context
            if not retrieved_contexts:
                retrieved_contexts = ["No relevant context retrieved"]
            
            # Get the response
            response = result.get('final_answer', '')
            
            # Create RAGAS-compatible data structure with field names expected by metrics
            dataset.append({
                "user_input": query,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": reference
            })
            
            # Small delay to avoid overwhelming the systems
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    ❌ Error processing question {i}: {e}")
            # Add a failure record to maintain dataset consistency
            dataset.append({
                "user_input": query,
                "retrieved_contexts": ["Error: Could not retrieve context"],
                "response": f"Error: {str(e)}",
                "reference": reference
            })
    
    print(f"✅ Collected {len(dataset)} evaluation records for {approach.upper()}")
    return dataset

def evaluate_with_ragas_simple(dataset: List[Dict[str, Any]], approach_name: str) -> Dict[str, float]:
    """
    Evaluate the dataset using RAGAS following the documentation pattern
    """
    print(f"\n📊 Evaluating {approach_name} using RAGAS metrics...")
    
    try:
        # Debug: Print first dataset item to check format
        if dataset:
            print(f"   📋 Dataset sample format: {list(dataset[0].keys())}")
        
        # Create RAGAS evaluation dataset
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        
        # Prepare evaluator LLM (following RAGAS documentation)
        evaluator_llm = LangchainLLMWrapper(llm)
        
        # Use basic metrics that work reliably
        metrics = [
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness()
        ]
        
        # Run evaluation
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=evaluator_llm
        )
        
        print(f"✅ {approach_name} evaluation completed")
        
        # Convert result to dictionary format with correct metric names
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            print(f"   📊 Raw DataFrame columns: {df.columns.tolist()}")
            
            # Only calculate mean for numeric metric columns
            numeric_cols = []
            for col in df.columns:
                if any(metric in col.lower() for metric in ['context_recall', 'faithfulness', 'factual_correctness', 'precision', 'relevance']):
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        numeric_cols.append(col)
                    except:
                        continue
            
            if numeric_cols:
                scores = df[numeric_cols].mean().to_dict()
                print(f"   📊 Numeric scores: {scores}")
            else:
                # Fallback: look for direct score attributes
                scores = {}
                if hasattr(result, 'binary_score'):
                    scores.update(result.binary_score)
                print(f"   📊 Fallback scores: {scores}")
        else:
            scores = result
        
        # Map RAGAS metric names to our expected names
        mapped_scores = {}
        
        # Ensure we have some scores
        if not scores:
            print(f"   ⚠️  No scores found, using default values")
            return {
                'context_recall': 0.0,
                'faithfulness': 0.0,
                'factual_correctness': 0.0
            }
        
        for key, value in scores.items():
            try:
                # Convert value to float
                numeric_value = float(value) if (value is not None and not (isinstance(value, float) and math.isnan(value))) else 0.0
                
                # Map to our expected metric names
                if 'context_recall' in key.lower():
                    mapped_scores['context_recall'] = numeric_value
                elif 'faithfulness' in key.lower():
                    mapped_scores['faithfulness'] = numeric_value
                elif 'factual_correctness' in key.lower():
                    mapped_scores['factual_correctness'] = numeric_value
                else:
                    # Keep original key as fallback
                    mapped_scores[key.lower()] = numeric_value
                    
            except (ValueError, TypeError) as e:
                print(f"   ⚠️  Could not convert {key}={value} to numeric: {e}")
                continue
        
        print(f"   ✅ Final mapped scores: {mapped_scores}")
        return mapped_scores
        
    except Exception as e:
        print(f"❌ Error during {approach_name} evaluation: {e}")
        import traceback
        print(f"   🔍 Full error traceback:")
        traceback.print_exc()
        
        # Return default scores if evaluation fails
        return {
            'context_recall': 0.0,
            'faithfulness': 0.0,
            'factual_correctness': 0.0
        }

def create_comparison_table_simple(chroma_results: Dict, graphrag_results: Dict) -> pd.DataFrame:
    """Create a simple comparison table"""
    
    # Extract scores
    def extract_scores(results):
        if isinstance(results, dict):
            return results
        return {}
    
    chroma_scores = extract_scores(chroma_results)
    graphrag_scores = extract_scores(graphrag_results)
    
    # Create comparison dataframe
    metrics = []
    chroma_values = []
    graphrag_values = []
    improvements = []
    
    # Map metric names to display names
    metric_display_names = {
        'context_recall': 'Context Recall',
        'faithfulness': 'Faithfulness', 
        'factual_correctness': 'Factual Correctness'
    }
    
    for metric_key, display_name in metric_display_names.items():
        chroma_val = chroma_scores.get(metric_key, 0.0)
        graphrag_val = graphrag_scores.get(metric_key, 0.0)
        
        metrics.append(display_name)
        chroma_values.append(round(chroma_val, 4))
        graphrag_values.append(round(graphrag_val, 4))
        
        # Calculate improvement percentage
        if chroma_val > 0:
            improvement = ((graphrag_val - chroma_val) / chroma_val) * 100
            improvements.append(f"{improvement:+.2f}%")
        else:
            improvements.append("N/A")
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Metric': metrics,
        'ChromaDB RAG': chroma_values,
        'GraphRAG': graphrag_values,
        'Improvement': improvements
    })
    
    return comparison_df

def create_three_way_comparison_table(chroma_results: Dict, graphrag_results: Dict, text2cypher_results: Dict) -> pd.DataFrame:
    """Create a three-way comparison table for all approaches"""
    
    # Extract scores
    def extract_scores(results):
        if isinstance(results, dict):
            return results
        return {}
    
    chroma_scores = extract_scores(chroma_results)
    graphrag_scores = extract_scores(graphrag_results)
    text2cypher_scores = extract_scores(text2cypher_results)
    
    # Create comparison dataframe
    metrics = []
    chroma_values = []
    graphrag_values = []
    text2cypher_values = []
    
    # Map metric names to display names
    metric_display_names = {
        'context_recall': 'Context Recall',
        'faithfulness': 'Faithfulness', 
        'factual_correctness': 'Factual Correctness'
    }
    
    for metric_key, display_name in metric_display_names.items():
        chroma_val = chroma_scores.get(metric_key, 0.0)
        graphrag_val = graphrag_scores.get(metric_key, 0.0)
        text2cypher_val = text2cypher_scores.get(metric_key, 0.0)
        
        metrics.append(display_name)
        chroma_values.append(round(chroma_val, 4))
        graphrag_values.append(round(graphrag_val, 4))
        text2cypher_values.append(round(text2cypher_val, 4))
    
    # Create three-way comparison table
    comparison_df = pd.DataFrame({
        'Metric': metrics,
        'ChromaDB RAG': chroma_values,
        'GraphRAG': graphrag_values,
        'Text2Cypher': text2cypher_values
    })
    
    return comparison_df



def save_results_simple(chroma_dataset: List, graphrag_dataset: List, text2cypher_dataset: List,
                       chroma_results: Dict, graphrag_results: Dict, text2cypher_results: Dict,
                       comparison_table: pd.DataFrame, output_dir: str = "benchmark_outputs"):
    """Save results to files in organized folder structure"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    chroma_df = pd.DataFrame(chroma_dataset)
    graphrag_df = pd.DataFrame(graphrag_dataset)
    text2cypher_df = pd.DataFrame(text2cypher_dataset)
    
    chroma_df.to_csv(f'{output_dir}/simple_benchmark_chroma.csv', index=False)
    graphrag_df.to_csv(f'{output_dir}/simple_benchmark_graphrag.csv', index=False)
    text2cypher_df.to_csv(f'{output_dir}/simple_benchmark_text2cypher.csv', index=False)
    
    # Save comparison table
    comparison_table.to_csv(f'{output_dir}/simple_benchmark_three_way_comparison.csv', index=False)
    
    # Save results
    chroma_avg = comparison_table['ChromaDB RAG'].mean()
    graphrag_avg = comparison_table['GraphRAG'].mean()
    text2cypher_avg = comparison_table['Text2Cypher'].mean()
    
    # Determine best overall approach by highest average
    scores = {'ChromaDB RAG': chroma_avg, 'GraphRAG': graphrag_avg, 'Text2Cypher': text2cypher_avg}
    best_overall = max(scores, key=scores.get)
    
    with open(f'{output_dir}/simple_benchmark_results.json', 'w') as f:
        json.dump({
            'chroma_results': chroma_results,
            'graphrag_results': graphrag_results,
            'text2cypher_results': text2cypher_results,
            'comparison_summary': {
                'chroma_avg': chroma_avg,
                'graphrag_avg': graphrag_avg,
                'text2cypher_avg': text2cypher_avg,
                'best_overall': best_overall
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to '{output_dir}/' folder:")
    print("  - simple_benchmark_chroma.csv")
    print("  - simple_benchmark_graphrag.csv")
    print("  - simple_benchmark_text2cypher.csv") 
    print("  - simple_benchmark_three_way_comparison.csv")
    print("  - simple_benchmark_results.json")

def main_simple():
    """Main benchmarking function - three-way comparison"""
    print("🚀 Starting Three-Way RAGAS Benchmark: ChromaDB vs GraphRAG vs Text2Cypher")
    print("=" * 80)
    
    # Load benchmark data
    benchmark_data = load_benchmark_data()
    
    # Collect evaluation data for all three approaches
    print("\n📋 Phase 1: Data Collection")
    chroma_dataset = collect_evaluation_data_simple(benchmark_data, approach="chroma")
    graphrag_dataset = collect_evaluation_data_simple(benchmark_data, approach="graphrag")
    text2cypher_dataset = collect_evaluation_data_simple(benchmark_data, approach="text2cypher")
    
    # Evaluate all three approaches with RAGAS
    print("\n📊 Phase 2: RAGAS Evaluation")
    chroma_results = evaluate_with_ragas_simple(chroma_dataset, "ChromaDB RAG")
    graphrag_results = evaluate_with_ragas_simple(graphrag_dataset, "GraphRAG")
    text2cypher_results = evaluate_with_ragas_simple(text2cypher_dataset, "Text2Cypher")
    
    # Create three-way comparison table
    print("\n📈 Phase 3: Results Analysis")
    comparison_table = create_three_way_comparison_table(chroma_results, graphrag_results, text2cypher_results)
    
    # Display results
    print("\n" + "=" * 90)
    print("🏆 THREE-WAY BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print(comparison_table.to_string(index=False))
    
    # Calculate overall performance
    print("\n📊 OVERALL PERFORMANCE SUMMARY:")
    print("-" * 50)
    
    chroma_avg = comparison_table['ChromaDB RAG'].mean()
    graphrag_avg = comparison_table['GraphRAG'].mean()
    text2cypher_avg = comparison_table['Text2Cypher'].mean()
    
    print(f"ChromaDB RAG Average Score:  {chroma_avg:.4f}")
    print(f"GraphRAG Average Score:      {graphrag_avg:.4f}")
    print(f"Text2Cypher Average Score:   {text2cypher_avg:.4f}")
    
    # Determine overall winner
    scores = {'ChromaDB RAG': chroma_avg, 'GraphRAG': graphrag_avg, 'Text2Cypher': text2cypher_avg}
    winner = max(scores, key=scores.get)
    winner_score = scores[winner]
    
    print(f"\n🏆 Overall Winner: {winner} (Score: {winner_score:.4f})")
    
    # Show improvements compared to ChromaDB baseline
    if graphrag_avg > chroma_avg:
        graphrag_improvement = ((graphrag_avg - chroma_avg) / chroma_avg) * 100
        print(f"📈 GraphRAG vs ChromaDB:    +{graphrag_improvement:.2f}%")
    else:
        graphrag_decline = ((chroma_avg - graphrag_avg) / chroma_avg) * 100
        print(f"📉 GraphRAG vs ChromaDB:    -{graphrag_decline:.2f}%")
    
    if text2cypher_avg > chroma_avg:
        text2cypher_improvement = ((text2cypher_avg - chroma_avg) / chroma_avg) * 100
        print(f"📈 Text2Cypher vs ChromaDB: +{text2cypher_improvement:.2f}%")
    else:
        text2cypher_decline = ((chroma_avg - text2cypher_avg) / chroma_avg) * 100
        print(f"📉 Text2Cypher vs ChromaDB: -{text2cypher_decline:.2f}%")
    
    # Save detailed results
    print("\n💾 Phase 4: Saving Results")
    save_results_simple(chroma_dataset, graphrag_dataset, text2cypher_dataset,
                       chroma_results, graphrag_results, text2cypher_results, comparison_table)
    
    # Generate visualizations
    print("\n📊 Phase 5: Generating Visualizations")
    create_visualizations(comparison_table)
    
    print("\n✅ THREE-WAY BENCHMARK COMPLETE!")
    print("=" * 80)
    
    return {
        'chroma_results': chroma_results,
        'graphrag_results': graphrag_results,
        'text2cypher_results': text2cypher_results,
        'comparison_table': comparison_table
    }

if __name__ == "__main__":
    results = main_simple() 