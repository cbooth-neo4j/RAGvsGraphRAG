#!/usr/bin/env python3
"""
Test the fixed benchmark with correct RAGAS format
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_fixed_format():
    """Test that our fixed benchmark format works correctly"""
    print("🧪 Testing fixed benchmark format...")
    
    try:
        from ragas_benchmark import collect_evaluation_data_simple, evaluate_with_ragas_simple
        
        # Create minimal test data - check the exact format expected by collect_evaluation_data_simple
        test_data = [
            {
                "record_id": "test_1",
                "question": "What is the capital of France?",
                "ground_truth": "The capital of France is Paris.",
                "source_dataset": "test",
                "domain": "geography"
            }
        ]
        
        # Debug: check what the function expects
        print(f"🔍 Input test data format: {list(test_data[0].keys())}")
        print(f"🔍 Ground truth value: '{test_data[0]['ground_truth']}'")
        print(f"🔍 Question value: '{test_data[0]['question']}')")
        
        print("🔄 Testing ChromaDB data collection...")
        
        # Test data collection (this should produce the correct format)
        try:
            chroma_dataset = collect_evaluation_data_simple(test_data, approach="chroma")
            
            if chroma_dataset:
                sample = chroma_dataset[0]
                print(f"✅ Data collection successful")
                print(f"📋 Sample keys: {list(sample.keys())}")
                
                # Check if we have exactly the required fields
                required_fields = {'user_input', 'retrieved_contexts', 'response', 'reference'}
                sample_fields = set(sample.keys())
                
                if required_fields.issubset(sample_fields):
                    print(f"✅ All required RAGAS fields present: {required_fields}")
                    
                    # Check for extra fields
                    extra_fields = sample_fields - required_fields
                    if extra_fields:
                        print(f"⚠️  Extra fields present: {extra_fields}")
                    else:
                        print(f"✅ No extra fields - perfect RAGAS format!")
                    
                    # Test evaluation
                    print(f"\n🔄 Testing RAGAS evaluation...")
                    
                    # Set short timeout for testing
                    os.environ['RAGAS_TIMEOUT'] = '60'
                    os.environ['RAGAS_MAX_WORKERS'] = '1'
                    
                    scores = evaluate_with_ragas_simple(chroma_dataset, "ChromaDB Test")
                    
                    print(f"\n📊 Results:")
                    for metric, score in scores.items():
                        print(f"   {metric}: {score:.3f}")
                    
                    # Check context recall specifically
                    context_recall = scores.get('context_recall', 0.0)
                    if context_recall > 0.0:
                        print(f"\n🎉 SUCCESS: Context recall is {context_recall:.3f} (not 0.0!)")
                        return True
                    else:
                        print(f"\n⚠️  Context recall is still 0.0, but evaluation completed")
                        return False
                        
                else:
                    missing_fields = required_fields - sample_fields
                    print(f"❌ Missing required fields: {missing_fields}")
                    return False
            else:
                print(f"❌ No data collected")
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"❌ Could not import benchmark modules: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 Testing Fixed Benchmark Format")
    print("=" * 40)
    
    success = test_fixed_format()
    
    if success:
        print(f"\n✅ Fixed format test PASSED!")
        print(f"💡 The benchmark should now work correctly with proper context recall scores")
    else:
        print(f"\n❌ Fixed format test FAILED")
        print(f"💡 There may still be issues with the format or evaluation")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
