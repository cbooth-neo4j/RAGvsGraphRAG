"""
Test RAGAS Setup - Quick verification script

This script tests the RAGAS evaluation setup with a few sample questions
to ensure everything works before running the full benchmark.
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import os

# Add parent directory to path so we can import from the main project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Add benchmark directory to path for benchmark imports  
benchmark_dir = os.path.join(project_root, 'benchmark')
sys.path.insert(0, benchmark_dir)

# Now import from the benchmark module
from ragas_benchmark import (
    load_benchmark_data,
    collect_evaluation_data_simple,
    evaluate_with_ragas_simple,
    create_comparison_table_simple,
    create_multi_approach_comparison_table
)

def test_ragas_setup():
    """Test RAGAS setup with first question"""
    print("🧪 Testing RAGAS Setup with Sample Question")
    print("=" * 50)
    
    # Load benchmark data and take first question
    # Get the path to benchmark.csv in the benchmark folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    benchmark_path = os.path.join(project_root, "benchmark", "benchmark.csv")
    benchmark_data = load_benchmark_data(benchmark_path)
    sample_data = benchmark_data[:1]  # Test with first question
    
    print(f"\n📝 Testing with {len(sample_data)} sample question:")
    for i, item in enumerate(sample_data, 1):
        print(f"  {i}. {item['question']}")
    
    try:
        # Test ChromaDB approach
        print(f"\n🔵 Testing ChromaDB RAG...")
        chroma_dataset = collect_evaluation_data_simple(sample_data, approach="chroma")
        chroma_results = evaluate_with_ragas_simple(chroma_dataset, "ChromaDB RAG (Test)")
        
        # Test GraphRAG approach
        print(f"\n🟠 Testing GraphRAG...")
        simple_dataset = collect_evaluation_data_simple(sample_data, approach="graphrag")
        simple_results = evaluate_with_ragas_simple(simple_dataset, "GraphRAG (Test)")
        
        # Test Advanced GraphRAG approach  
        print(f"\n🟢 Testing Advanced GraphRAG...")
        advanced_dataset = collect_evaluation_data_simple(sample_data, approach="advanced_graphrag")
        advanced_results = evaluate_with_ragas_simple(advanced_dataset, "Advanced GraphRAG (Test)")
        
        # Test DRIFT GraphRAG approach
        print(f"\n🟡 Testing DRIFT GraphRAG...")
        drift_dataset = collect_evaluation_data_simple(sample_data, approach="drift_graphrag")
        drift_results = evaluate_with_ragas_simple(drift_dataset, "DRIFT GraphRAG (Test)")
        
        # Test Neo4j Vector approach
        print(f"\n🟣 Testing Neo4j Vector RAG...")
        neo4j_vector_dataset = collect_evaluation_data_simple(sample_data, approach="neo4j_vector")
        neo4j_vector_results = evaluate_with_ragas_simple(neo4j_vector_dataset, "Neo4j Vector RAG (Test)")
        
        # Create five-way comparison
        print(f"\n📊 Creating five-way comparison...")
        results_dict = {
            'chroma': chroma_results,
            'graphrag': simple_results,
            'advanced_graphrag': advanced_results,
            'drift_graphrag': drift_results,
            'neo4j_vector': neo4j_vector_results
        }
        approach_names = {
            'chroma': 'ChromaDB RAG',
            'graphrag': 'GraphRAG',
            'advanced_graphrag': 'Advanced GraphRAG',
            'drift_graphrag': 'DRIFT GraphRAG',
            'neo4j_vector': 'Neo4j Vector RAG'
        }
        five_way_comparison = create_multi_approach_comparison_table(results_dict, approach_names)
        
        print("\n" + "=" * 80)
        print("🎯 FIVE-WAY TEST RESULTS")
        print("=" * 80)
        print(five_way_comparison.to_string(index=False))
        
        print(f"\n✅ RAGAS setup test completed successfully!")
        print(f"✅ All five approaches working correctly:")
        print(f"   🔵 ChromaDB RAG")
        print(f"   🟠 GraphRAG") 
        print(f"   🟢 Advanced GraphRAG") 
        print(f"   🟡 DRIFT GraphRAG")
        print(f"   🟣 Neo4j Vector RAG")
        print(f"✅ Ready to run full benchmark with {len(benchmark_data)} questions")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"❌ Please check your setup before running full benchmark")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ragas_setup()
    if success:
        print(f"\n🚀 To run the full benchmark, execute: python ragas_benchmark.py") 