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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add benchmark directory to path for benchmark imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'benchmark'))

from ragas_benchmark import (
    load_benchmark_data,
    collect_evaluation_data_simple,
    evaluate_with_ragas_simple,
    create_comparison_table_simple
)

def test_ragas_setup():
    """Test RAGAS setup with first 3 questions"""
    print("🧪 Testing RAGAS Setup with Sample Questions")
    print("=" * 50)
    
    # Load benchmark data and take first 3 questions
    # Get the path to benchmark.csv in the benchmark folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    benchmark_path = os.path.join(project_root, "benchmark", "benchmark.csv")
    benchmark_data = load_benchmark_data(benchmark_path)
    sample_data = benchmark_data[:3]  # Test with first 3 questions
    
    print(f"\n📝 Testing with {len(sample_data)} sample questions:")
    for i, item in enumerate(sample_data, 1):
        print(f"  {i}. {item['question']}")
    
    try:
        # Test ChromaDB approach
        print(f"\n🔵 Testing ChromaDB RAG...")
        chroma_dataset = collect_evaluation_data_simple(sample_data, approach="chroma")
        chroma_results = evaluate_with_ragas_simple(chroma_dataset, "ChromaDB RAG (Test)")
        
        # Test GraphRAG approach  
        print(f"\n🟢 Testing GraphRAG...")
        graphrag_dataset = collect_evaluation_data_simple(sample_data, approach="graphrag")
        graphrag_results = evaluate_with_ragas_simple(graphrag_dataset, "GraphRAG (Test)")
        
        # Create comparison
        print(f"\n📊 Creating comparison table...")
        comparison_table = create_comparison_table_simple(chroma_results, graphrag_results)
        
        print("\n" + "=" * 60)
        print("🎯 TEST RESULTS")
        print("=" * 60)
        print(comparison_table.to_string(index=False))
        
        print(f"\n✅ RAGAS setup test completed successfully!")
        print(f"✅ Both approaches working correctly")
        print(f"✅ Ready to run full benchmark with {len(benchmark_data)} questions")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"❌ Please check your setup before running full benchmark")
        return False

if __name__ == "__main__":
    success = test_ragas_setup()
    if success:
        print(f"\n🚀 To run the full benchmark, execute: python ragas_benchmark.py") 