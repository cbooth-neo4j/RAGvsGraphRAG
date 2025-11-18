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

def validate_results(results: dict, approach_name: str) -> tuple[bool, list[str]]:
    """Validate that results are meaningful (not all zeros or errors)"""
    issues = []
    
    # Check if all scores are zero (indicates failure)
    all_zero = all(v == 0.0 for v in results.values())
    if all_zero:
        issues.append(f"All scores are 0.0 (likely failure)")
    
    # Check for at least one non-zero score
    has_valid_score = any(v > 0.0 for v in results.values())
    if not has_valid_score:
        issues.append(f"No valid scores > 0.0")
    
    # Check for expected metrics
    expected_metrics = {'context_recall', 'faithfulness', 'factual_correctness'}
    missing_metrics = expected_metrics - set(results.keys())
    if missing_metrics:
        issues.append(f"Missing metrics: {missing_metrics}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


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
    
    # Track results and validation
    all_results = {}
    validation_status = {}
    
    try:
        # Test ChromaDB approach
        print(f"\n🔵 Testing ChromaDB RAG...")
        chroma_dataset = collect_evaluation_data_simple(sample_data, approach="chroma")
        chroma_results = evaluate_with_ragas_simple(chroma_dataset, "ChromaDB RAG (Test)")
        all_results['chroma'] = chroma_results
        is_valid, issues = validate_results(chroma_results, "ChromaDB RAG")
        validation_status['chroma'] = (is_valid, issues)
        
        # Test GraphRAG approach
        print(f"\n🟠 Testing GraphRAG...")
        simple_dataset = collect_evaluation_data_simple(sample_data, approach="graphrag")
        simple_results = evaluate_with_ragas_simple(simple_dataset, "GraphRAG (Test)")
        all_results['graphrag'] = simple_results
        is_valid, issues = validate_results(simple_results, "GraphRAG")
        validation_status['graphrag'] = (is_valid, issues)
        
        # Test Advanced GraphRAG approach  
        print(f"\n🟢 Testing Advanced GraphRAG...")
        advanced_dataset = collect_evaluation_data_simple(sample_data, approach="advanced_graphrag")
        advanced_results = evaluate_with_ragas_simple(advanced_dataset, "Advanced GraphRAG (Test)")
        all_results['advanced_graphrag'] = advanced_results
        is_valid, issues = validate_results(advanced_results, "Advanced GraphRAG")
        validation_status['advanced_graphrag'] = (is_valid, issues)
        
        # Test DRIFT GraphRAG approach
        print(f"\n🟡 Testing DRIFT GraphRAG...")
        drift_dataset = collect_evaluation_data_simple(sample_data, approach="drift_graphrag")
        drift_results = evaluate_with_ragas_simple(drift_dataset, "DRIFT GraphRAG (Test)")
        all_results['drift_graphrag'] = drift_results
        is_valid, issues = validate_results(drift_results, "DRIFT GraphRAG")
        validation_status['drift_graphrag'] = (is_valid, issues)
        
        # Test Neo4j Vector approach
        print(f"\n🟣 Testing Neo4j Vector RAG...")
        neo4j_vector_dataset = collect_evaluation_data_simple(sample_data, approach="neo4j_vector")
        neo4j_vector_results = evaluate_with_ragas_simple(neo4j_vector_dataset, "Neo4j Vector RAG (Test)")
        all_results['neo4j_vector'] = neo4j_vector_results
        is_valid, issues = validate_results(neo4j_vector_results, "Neo4j Vector RAG")
        validation_status['neo4j_vector'] = (is_valid, issues)
        
        # Create five-way comparison
        print(f"\n📊 Creating five-way comparison...")
        approach_names = {
            'chroma': 'ChromaDB RAG',
            'graphrag': 'GraphRAG',
            'advanced_graphrag': 'Advanced GraphRAG',
            'drift_graphrag': 'DRIFT GraphRAG',
            'neo4j_vector': 'Neo4j Vector RAG'
        }
        five_way_comparison = create_multi_approach_comparison_table(all_results, approach_names)
        
        print("\n" + "=" * 80)
        print("🎯 FIVE-WAY TEST RESULTS")
        print("=" * 80)
        print(five_way_comparison.to_string(index=False))
        
        # Report validation status
        print("\n" + "=" * 80)
        print("🔍 VALIDATION STATUS")
        print("=" * 80)
        
        successful_approaches = []
        failed_approaches = []
        
        for approach, (is_valid, issues) in validation_status.items():
            display_name = approach_names[approach]
            if is_valid:
                print(f"✅ {display_name}: PASSED")
                successful_approaches.append(display_name)
            else:
                print(f"❌ {display_name}: FAILED")
                for issue in issues:
                    print(f"   - {issue}")
                failed_approaches.append(display_name)
        
        # Final summary
        print("\n" + "=" * 80)
        if not failed_approaches:
            print(f"✅ All {len(successful_approaches)} approaches working correctly!")
            print(f"✅ Ready to run full benchmark with {len(benchmark_data)} questions")
            return True
        else:
            print(f"⚠️  Test completed with issues:")
            print(f"   ✅ {len(successful_approaches)} approaches succeeded: {', '.join(successful_approaches)}")
            print(f"   ❌ {len(failed_approaches)} approaches failed: {', '.join(failed_approaches)}")
            print(f"\n⚠️  Please fix the failed approaches before running full benchmark")
            return False
        
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