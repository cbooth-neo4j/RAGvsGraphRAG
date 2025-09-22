#!/usr/bin/env python3
"""
Direct test of RAGAS context recall with known good data
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_context_recall_directly():
    """Test RAGAS context recall with manually created data that should work"""
    print("🧪 Testing RAGAS context recall directly...")
    
    try:
        from ragas import evaluate
        from ragas.metrics import LLMContextRecall
        from ragas.dataset_schema import EvaluationDataset
        from config.model_factory import LLMFactory
        from ragas.llms import LangchainLLMWrapper
        
        # Get our LLM
        llm = LLMFactory.create_llm()
        evaluator_llm = LangchainLLMWrapper(llm)
        
        # Create test data where context recall should be HIGH
        test_cases = [
            {
                "name": "Perfect Match",
                "data": {
                    "user_input": "What is the capital of France?",
                    "retrieved_contexts": ["The capital of France is Paris. Paris is a major European city."],
                    "response": "The capital of France is Paris.",
                    "reference": "The capital of France is Paris."
                },
                "expected_score": "> 0.8"
            },
            {
                "name": "Partial Match", 
                "data": {
                    "user_input": "What is the capital of France?",
                    "retrieved_contexts": ["France is a European country. Paris is located in France and serves as its capital city."],
                    "response": "The capital of France is Paris.",
                    "reference": "The capital of France is Paris."
                },
                "expected_score": "> 0.5"
            },
            {
                "name": "No Match (should be 0)",
                "data": {
                    "user_input": "What is the capital of France?", 
                    "retrieved_contexts": ["London is the capital of England. Berlin is the capital of Germany."],
                    "response": "I don't know.",
                    "reference": "The capital of France is Paris."
                },
                "expected_score": "= 0.0"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🔬 Testing: {test_case['name']} (expected score {test_case['expected_score']})")
            
            # Create dataset
            dataset = EvaluationDataset.from_list([test_case['data']])
            
            # Set short timeout
            os.environ['RAGAS_TIMEOUT'] = '30'
            os.environ['RAGAS_MAX_WORKERS'] = '1'
            
            # Evaluate
            result = evaluate(
                dataset=dataset,
                metrics=[LLMContextRecall()],
                llm=evaluator_llm,
                raise_exceptions=False
            )
            
            # Get score
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                if not df.empty:
                    for col in df.columns:
                        if 'context_recall' in col.lower():
                            score = df[col].iloc[0]
                            print(f"   📊 Context Recall Score: {score:.3f}")
                            
                            # Analyze result
                            if score > 0.8:
                                print(f"   ✅ EXCELLENT: High context recall achieved!")
                            elif score > 0.3:
                                print(f"   ✅ GOOD: Moderate context recall achieved!")
                            elif score > 0.0:
                                print(f"   ⚠️  LOW: Some context recall but could be better")
                            else:
                                print(f"   ❌ ZERO: No context recall detected")
                            break
            
        return True
        
    except Exception as e:
        print(f"❌ Direct context recall test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔬 Direct RAGAS Context Recall Test")
    print("=" * 40)
    
    success = test_context_recall_directly()
    
    if success:
        print(f"\n✅ Direct context recall test completed!")
        print(f"💡 This shows whether RAGAS context recall itself is working correctly")
    else:
        print(f"\n❌ Direct context recall test failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())


