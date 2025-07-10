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
    create_three_way_comparison_table
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
        
        # Test Text2Cypher approach
        print(f"\n🟡 Testing Text2Cypher...")
        text2cypher_dataset = collect_evaluation_data_simple(sample_data, approach="text2cypher")
        text2cypher_results = evaluate_with_ragas_simple(text2cypher_dataset, "Text2Cypher (Test)")
        
        # Create three-way comparison
        print(f"\n📊 Creating three-way comparison...")
        three_way_comparison = create_three_way_comparison_table(chroma_results, graphrag_results, text2cypher_results)
        
        print("\n" + "=" * 80)
        print("🎯 THREE-WAY TEST RESULTS")
        print("=" * 80)
        print(three_way_comparison.to_string(index=False))
        
        print(f"\n✅ RAGAS setup test completed successfully!")
        print(f"✅ All three approaches working correctly:")
        print(f"   🔵 ChromaDB RAG")
        print(f"   🟢 GraphRAG") 
        print(f"   🟡 Text2Cypher")
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