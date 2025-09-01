"""
RAGAS Benchmark: RAG vs GraphRAG Evaluation

This script evaluates both ChromaDB and GraphRAG approaches using RAGAS framework
following the exact pattern from the RAGAS documentation.
"""

import pandas as pd
import json
import time
import math
import sys
import os
import argparse
import asyncio
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

# Import our RAG systems from the new retrievers module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from retrievers import (
        query_chroma_rag,
        query_graphrag, 
        query_advanced_graphrag,
        query_drift_graphrag,
        query_text2cypher_rag,
        query_neo4j_vector_rag,
        query_hybrid_cypher_rag,
        get_available_retrievers,
        AVAILABLE_RETRIEVERS
    )
    
    # Check availability of each retriever
    available_retrievers = get_available_retrievers()
    
    CHROMA_AVAILABLE = 'chroma' in available_retrievers
    GRAPHRAG_AVAILABLE = 'graphrag' in available_retrievers
    ADVANCED_GRAPHRAG_AVAILABLE = 'advanced_graphrag' in available_retrievers
    DRIFT_GRAPHRAG_AVAILABLE = 'drift_graphrag' in available_retrievers
    TEXT2CYPHER_AVAILABLE = 'text2cypher' in available_retrievers
    NEO4J_VECTOR_AVAILABLE = 'neo4j_vector' in available_retrievers
    HYBRID_CYPHER_AVAILABLE = 'hybrid_cypher' in available_retrievers
    
    print("✅ Retrievers module imported successfully")
    print(f"📋 Available retrievers: {list(available_retrievers.keys())}")
    
except ImportError as e:
    print(f"❌ Error importing retrievers module: {e}")
    print("   Please ensure the retrievers module is properly set up.")
    
    # Set all retrievers as unavailable if import fails
    CHROMA_AVAILABLE = False
    GRAPHRAG_AVAILABLE = False
    ADVANCED_GRAPHRAG_AVAILABLE = False
    DRIFT_GRAPHRAG_AVAILABLE = False
    TEXT2CYPHER_AVAILABLE = False
    NEO4J_VECTOR_AVAILABLE = False

# Initialize LLM and embeddings for RAGAS
SEED = 42
llm = ChatOpenAI(
    model="gpt-4.1",
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
            # Query the appropriate RAG system using new retriever functions
            if approach == "chroma" and CHROMA_AVAILABLE:
                result = query_chroma_rag(query, k=1)
            elif approach == "graphrag" and GRAPHRAG_AVAILABLE:
                result = query_graphrag(query, k=5)
            elif approach == "text2cypher" and TEXT2CYPHER_AVAILABLE:
                result = query_text2cypher_rag(query)
            elif approach == "advanced_graphrag" and ADVANCED_GRAPHRAG_AVAILABLE:
                result = query_advanced_graphrag(query, mode="hybrid", k=5)
            elif approach == "drift_graphrag" and DRIFT_GRAPHRAG_AVAILABLE:
                result = query_drift_graphrag(query, n_depth=3, max_follow_ups=3, use_modular=True)
            elif approach == "neo4j_vector" and NEO4J_VECTOR_AVAILABLE:
                result = query_neo4j_vector_rag(query, k=5)
            elif approach == "hybrid_cypher" and HYBRID_CYPHER_AVAILABLE:
                # Use the main hybrid cypher function with reduced k for benchmark stability
                result = query_hybrid_cypher_rag(query, k=5)
            else:
                # Handle unavailable retrievers or unknown approaches
                if approach == "chroma" and not CHROMA_AVAILABLE:
                    raise ValueError(f"ChromaDB retriever not available")
                elif approach == "graphrag" and not GRAPHRAG_AVAILABLE:
                    raise ValueError(f"GraphRAG retriever not available")
                elif approach == "text2cypher" and not TEXT2CYPHER_AVAILABLE:
                    raise ValueError(f"Text2Cypher retriever not available")
                elif approach == "advanced_graphrag" and not ADVANCED_GRAPHRAG_AVAILABLE:
                    raise ValueError(f"Advanced GraphRAG retriever not available")
                elif approach == "drift_graphrag" and not DRIFT_GRAPHRAG_AVAILABLE:
                    raise ValueError(f"DRIFT GraphRAG retriever not available")
                elif approach == "neo4j_vector" and not NEO4J_VECTOR_AVAILABLE:
                    raise ValueError(f"Neo4j Vector retriever not available")
                elif approach == "hybrid_cypher" and not HYBRID_CYPHER_AVAILABLE:
                    raise ValueError(f"Hybrid Cypher retriever not available")
                else:
                    raise ValueError(f"Unknown approach: {approach}")
            
            # Extract retrieved contexts as simple list of strings (RAGAS format)
            retrieved_contexts = []
            for detail in result.get('retrieval_details', []):
                content = detail.get('content', '')
                if not content:
                    # Build a compact string from hybrid details if no 'content'
                    neighbor_summaries = detail.get('neighbor_summaries')
                    anchor = detail.get('anchor')
                    if neighbor_summaries:
                        content = f"Anchor: {str(anchor)[:200]} | Neighbors: " + ", ".join(map(str, neighbor_summaries[:20]))
                if content:
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

def create_multi_approach_comparison_table(results_dict: Dict[str, Dict], approach_names: Dict[str, str]) -> pd.DataFrame:
    """Create a comparison table for multiple approaches"""
    
    # Extract scores for all approaches
    approach_scores = {}
    for approach_key, results in results_dict.items():
        if isinstance(results, dict):
            approach_scores[approach_key] = results
        else:
            approach_scores[approach_key] = {}
    
    # Create comparison dataframe
    metrics = []
    approach_columns = {}
    
    # Initialize columns for each approach
    for approach_key in approach_scores.keys():
        approach_columns[approach_names.get(approach_key, approach_key)] = []
    
    # Map metric names to display names
    metric_display_names = {
        'context_recall': 'Context Recall',
        'faithfulness': 'Faithfulness', 
        'factual_correctness': 'Factual Correctness'
    }
    
    for metric_key, display_name in metric_display_names.items():
        metrics.append(display_name)
        
        for approach_key, scores in approach_scores.items():
            approach_name = approach_names.get(approach_key, approach_key)
            score = scores.get(metric_key, 0.0)
            approach_columns[approach_name].append(round(score, 4))
    
    # Create comparison table
    comparison_data = {'Metric': metrics}
    comparison_data.update(approach_columns)
    
    comparison_df = pd.DataFrame(comparison_data)
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

def save_results_selective(datasets: Dict, results: Dict, comparison_table: pd.DataFrame, 
                          approaches: List[str], output_dir: str = "benchmark_outputs"):
    """Save results for selected approaches only"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets for selected approaches
    for approach in approaches:
        if approach in datasets and datasets[approach]:
            df = pd.DataFrame(datasets[approach])
            df.to_csv(f'{output_dir}/simple_benchmark_{approach}.csv', index=False)
            print(f"  - simple_benchmark_{approach}.csv")
    
    # Save comparison table
    comparison_table.to_csv(f'{output_dir}/simple_benchmark_comparison.csv', index=False)
    print(f"  - simple_benchmark_comparison.csv")
    
    # Calculate averages for selected approaches
    averages = {}
    for col in comparison_table.columns:
        if col != 'Metric' and col != 'Improvement':  # Skip non-numeric columns
            try:
                # Check if column contains numeric data
                if comparison_table[col].dtype in ['float64', 'int64'] or comparison_table[col].apply(lambda x: isinstance(x, (int, float))).all():
                    averages[col] = comparison_table[col].mean()
            except (TypeError, ValueError):
                continue
    
    # Determine best overall approach
    if averages:
        best_overall = max(averages, key=averages.get)
    else:
        best_overall = "None"
    
    # Save results JSON
    results_data = {
        'selected_approaches': approaches,
        'comparison_summary': {
            'averages': averages,
            'best_overall': best_overall
        }
    }
    
    # Add individual results
    for approach in approaches:
        if approach in results:
            results_data[f'{approach}_results'] = results[approach]
    
    with open(f'{output_dir}/simple_benchmark_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"  - simple_benchmark_results.json")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RAGAS Benchmark: Compare RAG approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ragas_benchmark.py --all                                # Test all available approaches  
      python ragas_benchmark.py --chroma --graphrag           # Test ChromaDB vs GraphRAG
    python ragas_benchmark.py --graphrag --advanced-graphrag # Test GraphRAG vs Advanced GraphRAG
  python ragas_benchmark.py --advanced-graphrag --drift-graphrag # Test Advanced vs DRIFT GraphRAG
  python ragas_benchmark.py --chroma --advanced-graphrag --drift-graphrag # Test 3-way comparison
  python ragas_benchmark.py --drift-graphrag                     # Test DRIFT GraphRAG only
  python ragas_benchmark.py --chroma                             # Test ChromaDB only
        """
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Test all three approaches (ChromaDB, GraphRAG, Text2Cypher)'
    )
    parser.add_argument(
        '--chroma', 
        action='store_true',
        help='Include ChromaDB RAG in testing'
    )
    parser.add_argument(
        '--graphrag', 
        action='store_true',
        help='Include GraphRAG in testing'
    )
    parser.add_argument(
        '--text2cypher', 
        action='store_true',
        help='Include Text2Cypher in testing'
    )
    parser.add_argument(
        '--advanced-graphrag', 
        action='store_true',
        help='Include Advanced GraphRAG (intelligent global/local/hybrid) in testing'
    )
    parser.add_argument(
        '--drift-graphrag', 
        action='store_true',
        help='Include DRIFT GraphRAG (iterative refinement) in testing'
    )
    parser.add_argument(
        '--neo4j-vector', 
        action='store_true',
        help='Include Neo4j Vector RAG (pure vector similarity) in testing'
    )
    parser.add_argument(
        '--hybrid-cypher', 
        action='store_true',
        help='Include Hybrid Cypher RAG (hybrid + generic neighborhood) in testing'
    )
    parser.add_argument(
        '--output-dir',
        default='benchmark_outputs',
        help='Output directory for results (default: benchmark_outputs)'
    )
    
    args = parser.parse_args()
    
    # Determine which approaches to test
    approaches = []
    if args.all:
        base_approaches = ['chroma', 'graphrag', 'text2cypher']
        if ADVANCED_GRAPHRAG_AVAILABLE:
            base_approaches.append('advanced_graphrag')
        if DRIFT_GRAPHRAG_AVAILABLE:
            base_approaches.append('drift_graphrag')
        if NEO4J_VECTOR_AVAILABLE:
            base_approaches.append('neo4j_vector')
        if HYBRID_CYPHER_AVAILABLE:
            base_approaches.append('hybrid_cypher')
        approaches = base_approaches
    else:
        if args.chroma:
            approaches.append('chroma')
        if args.graphrag:
            approaches.append('graphrag')
        if args.text2cypher:
            approaches.append('text2cypher')
        if getattr(args, 'advanced_graphrag', False) and ADVANCED_GRAPHRAG_AVAILABLE:
            approaches.append('advanced_graphrag')
        if getattr(args, 'drift_graphrag', False) and DRIFT_GRAPHRAG_AVAILABLE:
            approaches.append('drift_graphrag')
        if getattr(args, 'neo4j_vector', False) and NEO4J_VECTOR_AVAILABLE:
            approaches.append('neo4j_vector')
        if getattr(args, 'hybrid_cypher', False) and HYBRID_CYPHER_AVAILABLE:
            approaches.append('hybrid_cypher')
    
    # If no approaches specified, default to all
    if not approaches:
        print("⚠️  No approaches specified. Defaulting to all available approaches.")
        default_approaches = ['chroma', 'graphrag', 'text2cypher']
        if ADVANCED_GRAPHRAG_AVAILABLE:
            default_approaches.append('advanced_graphrag')
        if DRIFT_GRAPHRAG_AVAILABLE:
            default_approaches.append('drift_graphrag')
        if NEO4J_VECTOR_AVAILABLE:
            default_approaches.append('neo4j_vector')
        if HYBRID_CYPHER_AVAILABLE:
            default_approaches.append('hybrid_cypher')
        approaches = default_approaches
    
    return approaches, args.output_dir

def main_selective(approaches: List[str], output_dir: str = "benchmark_outputs"):
    """Main benchmarking function with selective approach testing"""
    
    approach_names = {
        'chroma': 'ChromaDB RAG',
        'graphrag': 'GraphRAG', 
        'text2cypher': 'Text2Cypher',
        'advanced_graphrag': 'Advanced GraphRAG',
        'drift_graphrag': 'DRIFT GraphRAG',
        'neo4j_vector': 'Neo4j Vector RAG',
        'hybrid_cypher': 'Hybrid Cypher RAG'
    }
    
    selected_names = [approach_names[approach] for approach in approaches]
    
    print(f"🚀 Starting Selective RAGAS Benchmark: {' vs '.join(selected_names)}")
    print("=" * 80)
    
    # Load benchmark data
    benchmark_data = load_benchmark_data()
    
    # Collect evaluation data for selected approaches
    print(f"\n📋 Phase 1: Data Collection")
    datasets = {}
    for approach in approaches:
        datasets[approach] = collect_evaluation_data_simple(benchmark_data, approach=approach)
    
    # Evaluate selected approaches with RAGAS
    print(f"\n📊 Phase 2: RAGAS Evaluation")
    results = {}
    for approach in approaches:
        results[approach] = evaluate_with_ragas_simple(datasets[approach], approach_names[approach])
    
    # Create comparison table for selected approaches
    print(f"\n📈 Phase 3: Results Analysis")
    
    if len(approaches) == 1:
        # Single approach - create simple results display
        approach = approaches[0]
        result = results[approach]
        print(f"\n📊 Results for {approach_names[approach]}:")
        print("-" * 50)
        for metric, score in result.items():
            print(f"{metric.replace('_', ' ').title()}: {score:.4f}")
        
        # Create simple comparison table
        comparison_table = pd.DataFrame({
            'Metric': [metric.replace('_', ' ').title() for metric in result.keys()],
            approach_names[approach]: list(result.values())
        })
        
    elif len(approaches) == 2:
        # Two approaches - create comparison table
        comparison_table = create_comparison_table_simple(
            results[approaches[0]], 
            results[approaches[1]]
        )
        # Rename columns to match selected approaches
        comparison_table.columns = ['Metric', approach_names[approaches[0]], approach_names[approaches[1]], 'Improvement']
        
    else:
        # Multiple approaches - create multi-approach comparison
        comparison_table = create_multi_approach_comparison_table(results, approach_names)
    
    # Display results
    print(f"\n" + "=" * 90)
    print(f"🏆 BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print(comparison_table.to_string(index=False))
    
    # Calculate overall performance for selected approaches
    print(f"\n📊 OVERALL PERFORMANCE SUMMARY:")
    print("-" * 50)
    
    averages = {}
    for col in comparison_table.columns:
        if col != 'Metric' and col != 'Improvement':  # Skip non-numeric columns
            try:
                # Check if column contains numeric data
                if comparison_table[col].dtype in ['float64', 'int64'] or comparison_table[col].apply(lambda x: isinstance(x, (int, float))).all():
                    averages[col] = comparison_table[col].mean()
                    print(f"{col} Average Score: {averages[col]:.4f}")
            except (TypeError, ValueError):
                print(f"⚠️  Skipping non-numeric column: {col}")
                continue
    
    # Determine overall winner
    if averages:
        winner = max(averages, key=averages.get)
        winner_score = averages[winner]
        print(f"\n🏆 Overall Winner: {winner} (Score: {winner_score:.4f})")
        
        # Show improvements if multiple approaches
        if len(approaches) > 1:
            print(f"\n📈 Performance Comparisons:")
            baseline = list(averages.values())[0]
            baseline_name = list(averages.keys())[0]
            
            for name, score in averages.items():
                if name != baseline_name:
                    if score > baseline:
                        improvement = ((score - baseline) / baseline) * 100
                        print(f"📈 {name} vs {baseline_name}: +{improvement:.2f}%")
                    else:
                        decline = ((baseline - score) / baseline) * 100
                        print(f"📉 {name} vs {baseline_name}: -{decline:.2f}%")
    
    # Save detailed results
    print(f"\n💾 Phase 4: Saving Results")
    save_results_selective(datasets, results, comparison_table, approaches, output_dir)
    
    # Generate visualizations
    print(f"\n📊 Phase 5: Generating Visualizations")
    create_visualizations(comparison_table)
    
    print(f"\n✅ BENCHMARK COMPLETE!")
    print("=" * 80)
    
    return {
        'approaches': approaches,
        'results': results,
        'comparison_table': comparison_table,
        'datasets': datasets
    }



if __name__ == "__main__":
    # Parse command line arguments
    approaches, output_dir = parse_arguments()
    
    # Run selective benchmark
    results = main_selective(approaches, output_dir) 