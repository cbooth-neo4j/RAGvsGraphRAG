#!/usr/bin/env python3
"""
Debug script to check RAGAS format requirements and test LLMContextRecall directly
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_ragas_format():
    """Test what format RAGAS actually expects"""
    print("🔍 Testing RAGAS format requirements...")
    
    try:
        from ragas import evaluate
        from ragas.metrics import LLMContextRecall, Faithfulness
        from ragas.dataset_schema import EvaluationDataset
        from config.model_factory import LLMFactory
        
        # Get our LLM
        llm = LLMFactory.create_llm()
        print(f"✅ LLM loaded: {type(llm)}")
        
        # Test different data formats to see what works
        test_formats = [
            {
                "name": "Format 1: Standard RAGAS format",
                "data": [
                    {
                        "user_input": "What is the capital of France?",
                        "retrieved_contexts": ["France is a country in Europe. Paris is the capital city of France."],
                        "response": "The capital of France is Paris.",
                        "reference": "Paris is the capital of France."
                    }
                ]
            },
            {
                "name": "Format 2: Alternative field names",
                "data": [
                    {
                        "question": "What is the capital of France?",
                        "contexts": ["France is a country in Europe. Paris is the capital city of France."],
                        "answer": "The capital of France is Paris.",
                        "ground_truth": "Paris is the capital of France."
                    }
                ]
            },
            {
                "name": "Format 3: Mixed field names",
                "data": [
                    {
                        "user_input": "What is the capital of France?",
                        "question": "What is the capital of France?",
                        "retrieved_contexts": ["France is a country in Europe. Paris is the capital city of France."],
                        "contexts": ["France is a country in Europe. Paris is the capital city of France."],
                        "response": "The capital of France is Paris.",
                        "answer": "The capital of France is Paris.",
                        "reference": "Paris is the capital of France.",
                        "ground_truth": "Paris is the capital of France.",
                        "ground_truths": ["Paris is the capital of France."]
                    }
                ]
            }
        ]
        
        for test_format in test_formats:
            print(f"\n🧪 Testing {test_format['name']}...")
            
            try:
                # Create evaluation dataset
                dataset = EvaluationDataset.from_list(test_format['data'])
                print(f"   ✅ Dataset creation successful")
                
                # Print dataset schema info
                if hasattr(dataset, 'to_pandas'):
                    df = dataset.to_pandas()
                    print(f"   📋 Dataset columns: {list(df.columns)}")
                    print(f"   📋 Dataset shape: {df.shape}")
                    
                    # Show first row
                    if not df.empty:
                        print(f"   📋 First row keys: {list(df.iloc[0].keys())}")
                
                # Try a very quick evaluation with timeout
                print(f"   🔄 Testing LLMContextRecall evaluation...")
                
                # Set very short timeout for testing
                os.environ['RAGAS_TIMEOUT'] = '30'  # 30 seconds max
                os.environ['RAGAS_MAX_WORKERS'] = '1'
                
                from langchain_community.chat_models import ChatOllama
                from ragas.llms import LangchainLLMWrapper
                
                # Use a wrapper for the LLM
                evaluator_llm = LangchainLLMWrapper(llm)
                
                # Test just context recall
                context_recall = LLMContextRecall()
                
                result = evaluate(
                    dataset=dataset,
                    metrics=[context_recall],
                    llm=evaluator_llm,
                    raise_exceptions=False
                )
                
                if hasattr(result, 'to_pandas'):
                    result_df = result.to_pandas()
                    print(f"   ✅ Evaluation successful!")
                    print(f"   📊 Result columns: {list(result_df.columns)}")
                    
                    # Check for context recall score
                    for col in result_df.columns:
                        if 'context_recall' in col.lower():
                            score = result_df[col].iloc[0] if not result_df.empty else 0.0
                            print(f"   📈 Context Recall Score: {score}")
                            
                            if score > 0:
                                print(f"   🎉 SUCCESS: Non-zero context recall achieved!")
                            else:
                                print(f"   ⚠️  Still getting 0 context recall")
                else:
                    print(f"   ❌ Evaluation returned unexpected result type: {type(result)}")
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
                
    except ImportError as e:
        print(f"❌ Could not import RAGAS modules: {e}")
        return False
    except Exception as e:
        print(f"❌ General error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_llm_performance():
    """Check if our LLM is working properly for RAGAS"""
    print("\n🔍 Testing LLM performance for RAGAS...")
    
    try:
        from config.model_factory import LLMFactory
        
        llm = LLMFactory.create_llm()
        print(f"✅ LLM type: {type(llm)}")
        
        # Test a simple prompt
        test_prompt = "Is the statement 'Paris is the capital of France' supported by this context: 'France is a European country with Paris as its capital city.'? Answer with just 'Yes' or 'No'."
        
        print(f"🔄 Testing LLM with simple prompt...")
        response = llm.invoke(test_prompt)
        print(f"📝 LLM Response: {response}")
        
        # Extract content from response object
        response_text = ""
        if hasattr(response, 'content'):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)
        
        print(f"📝 Extracted text: {response_text}")
        
        if 'yes' in response_text.lower() or 'no' in response_text.lower():
            print(f"✅ LLM is responding correctly to context recall type questions")
            return True
        else:
            print(f"⚠️  LLM response format may be causing issues")
            return False
            
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

def main():
    """Main debug function"""
    print("🔍 RAGAS Format Debug Tool")
    print("=" * 50)
    
    # Test LLM first
    llm_ok = check_llm_performance()
    
    # Test RAGAS format
    if llm_ok:
        test_ragas_format()
    else:
        print("⚠️  Skipping RAGAS format test due to LLM issues")
    
    print("\n📊 Debug complete!")

if __name__ == "__main__":
    main()
