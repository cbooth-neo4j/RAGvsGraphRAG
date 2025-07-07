"""
RAGAS Determinism Debugging Script

This script isolates and debugs RAGAS evaluation variability issues
to identify the exact source of non-deterministic behavior.
"""

import os
import sys
import json
import hashlib
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# RAGAS setup
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# Our RAG systems for comparison
from RAGvsGraphRAG import query_chroma_with_llm, query_neo4j_with_llm

# Fixed seed for testing
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

# Initialize LLM with strict determinism
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={"seed": SEED},
    max_retries=0
)

def print_separator(title: str):
    """Print a clear separator for debugging sections"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def hash_content(content: Any) -> str:
    """Generate hash of content for comparison"""
    return hashlib.md5(str(content).encode()).hexdigest()[:8]

def test_fixed_content_determinism():
    """Test RAGAS with completely fixed content to isolate evaluation variability"""
    print_separator("FIXED CONTENT DETERMINISM TEST")
    
    # Create identical test data for multiple runs
    fixed_dataset = [
        {
            "user_input": "What is the capital of France?",
            "retrieved_contexts": [
                "Paris is the capital and most populous city of France.",
                "France is a country in Western Europe with Paris as its capital.",
                "The city of Paris serves as the political and cultural center of France."
            ],
            "response": "The capital of France is Paris.",
            "reference": "Paris"
        },
        {
            "user_input": "What year was the company founded?", 
            "retrieved_contexts": [
                "The company was established in 1995 by John Smith.",
                "Founded in 1995, the organization has grown significantly.",
                "Since its founding in 1995, the company has expanded globally."
            ],
            "response": "The company was founded in 1995.",
            "reference": "1995"
        }
    ]
    
    print(f"📋 Testing with {len(fixed_dataset)} fixed samples")
    print(f"🔒 Content hash: {hash_content(fixed_dataset)}")
    
    results = []
    for run in range(5):
        print(f"\n🔄 Run {run + 1}/5")
        
        # Create fresh evaluation dataset
        evaluation_dataset = EvaluationDataset.from_list(fixed_dataset)
        evaluator_llm = LangchainLLMWrapper(llm)
        
        # Use same metrics as main benchmark
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
        
        # Extract scores
        df = result.to_pandas()
        scores = {}
        for col in df.columns:
            if any(metric in col.lower() for metric in ['context_recall', 'faithfulness', 'factual_correctness']):
                try:
                    scores[col] = df[col].mean()
                except:
                    continue
        
        results.append(scores)
        print(f"   📊 Scores: {scores}")
        print(f"   🏷️  Hash: {hash_content(scores)}")
    
    # Analyze consistency
    print(f"\n📈 FIXED CONTENT ANALYSIS:")
    if len(results) > 1:
        first_hash = hash_content(results[0])
        all_same = all(hash_content(r) == first_hash for r in results[1:])
        print(f"   ✅ All results identical: {all_same}")
        if not all_same:
            print(f"   ⚠️  Variability detected in fixed content evaluation!")
            for i, result in enumerate(results):
                print(f"      Run {i+1}: {hash_content(result)}")
    
    return results

def test_chromadb_content_consistency():
    """Test if ChromaDB returns consistent content across runs"""
    print_separator("CHROMADB CONTENT CONSISTENCY TEST")
    
    test_query = "What city is NovaGrid Energy Corporation headquartered in?"
    print(f"🔍 Query: {test_query}")
    
    retrieval_results = []
    for run in range(5):
        print(f"\n🔄 Retrieval Run {run + 1}/5")
        
        result = query_chroma_with_llm(test_query, k=1)
        
        # Extract just the content for comparison
        contents = [detail['content'] for detail in result['retrieval_details']]
        content_hash = hash_content(contents)
        
        retrieval_results.append({
            'run': run + 1,
            'content_hash': content_hash,
            'num_chunks': len(contents),
            'first_chunk_preview': contents[0][:100] if contents else "NO CONTENT"
        })
        
        print(f"   📊 Content hash: {content_hash}")
        print(f"   📝 Chunks: {len(contents)}")
        print(f"   📄 Preview: {contents[0][:50] if contents else 'NO CONTENT'}...")
    
    # Analyze retrieval consistency
    first_hash = retrieval_results[0]['content_hash']
    all_same = all(r['content_hash'] == first_hash for r in retrieval_results[1:])
    
    print(f"\n📈 CHROMADB RETRIEVAL ANALYSIS:")
    print(f"   ✅ Content identical across runs: {all_same}")
    
    if not all_same:
        print(f"   ⚠️  ChromaDB returning different content!")
        for result in retrieval_results:
            print(f"      Run {result['run']}: {result['content_hash']}")
    
    return retrieval_results

def test_single_sample_ragas_variability():
    """Test RAGAS evaluation on a single sample multiple times"""
    print_separator("SINGLE SAMPLE RAGAS VARIABILITY TEST")
    
    # Get one consistent sample from ChromaDB
    test_query = "What city is NovaGrid Energy Corporation headquartered in?"
    chroma_result = query_chroma_with_llm(test_query, k=1)
    
    # Create single evaluation sample
    sample = {
        "user_input": test_query,
        "retrieved_contexts": [detail['content'][:500] for detail in chroma_result['retrieval_details']],
        "response": chroma_result['final_answer'],
        "reference": "Toronto"  # Known answer from benchmark
    }
    
    print(f"🔍 Testing single sample: {test_query[:50]}...")
    print(f"📝 Content hash: {hash_content(sample)}")
    
    evaluation_results = []
    for run in range(5):
        print(f"\n🔄 Evaluation Run {run + 1}/5")
        
        # Create dataset with single sample
        dataset = EvaluationDataset.from_list([sample])
        evaluator_llm = LangchainLLMWrapper(llm)
        
        metrics = [
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness()
        ]
        
        # Evaluate the same sample
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm
        )
        
        # Extract scores
        df = result.to_pandas()
        scores = {}
        for col in df.columns:
            if any(metric in col.lower() for metric in ['context_recall', 'faithfulness', 'factual_correctness']):
                try:
                    scores[col] = df[col].iloc[0]  # Single sample, get first row
                except:
                    continue
        
        evaluation_results.append(scores)
        print(f"   📊 Scores: {scores}")
        print(f"   🏷️  Hash: {hash_content(scores)}")
    
    # Analyze evaluation consistency
    print(f"\n📈 SINGLE SAMPLE ANALYSIS:")
    if len(evaluation_results) > 1:
        first_hash = hash_content(evaluation_results[0])
        all_same = all(hash_content(r) == first_hash for r in evaluation_results[1:])
        print(f"   ✅ All evaluations identical: {all_same}")
        if not all_same:
            print(f"   ⚠️  RAGAS showing variability on identical input!")
            for i, result in enumerate(evaluation_results):
                print(f"      Run {i+1}: {hash_content(result)} - {result}")
    
    return evaluation_results

def test_environment_setup():
    """Test if the environment setup is deterministic"""
    print_separator("ENVIRONMENT SETUP TEST")
    
    print(f"🔧 Python hash seed: {os.environ.get('PYTHONHASHSEED', 'NOT SET')}")
    print(f"🤖 LLM model: {llm.model_name}")
    print(f"🌡️  LLM temperature: {llm.temperature}")
    print(f"🎲 LLM seed: {llm.model_kwargs.get('seed', 'NOT SET')}")
    print(f"🔄 LLM max retries: {llm.max_retries}")
    
    # Test LLM consistency
    test_prompt = "Answer with exactly: 'Hello World'"
    responses = []
    
    for i in range(3):
        response = llm.invoke(test_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        responses.append(content)
        print(f"   LLM Test {i+1}: '{content}'")
    
    all_same = all(r == responses[0] for r in responses[1:])
    print(f"   ✅ LLM responses identical: {all_same}")
    
    return responses

def main():
    """Run all debugging tests"""
    print("🚀 RAGAS Determinism Debugging Suite")
    print("=" * 60)
    
    # Test 1: Environment setup
    env_results = test_environment_setup()
    
    # Test 2: Fixed content determinism
    fixed_results = test_fixed_content_determinism()
    
    # Test 3: ChromaDB content consistency
    retrieval_results = test_chromadb_content_consistency()
    
    # Test 4: Single sample RAGAS variability
    eval_results = test_single_sample_ragas_variability()
    
    # Summary
    print_separator("DEBUGGING SUMMARY")
    
    # Check LLM determinism
    llm_deterministic = len(set(env_results)) == 1
    print(f"🤖 LLM Deterministic: {'✅ YES' if llm_deterministic else '❌ NO'}")
    
    # Check fixed content evaluation
    if fixed_results:
        fixed_hashes = [hash_content(r) for r in fixed_results]
        fixed_deterministic = len(set(fixed_hashes)) == 1
        print(f"🔒 Fixed Content Evaluation: {'✅ DETERMINISTIC' if fixed_deterministic else '❌ VARIABLE'}")
    
    # Check ChromaDB retrieval
    if retrieval_results:
        retrieval_hashes = [r['content_hash'] for r in retrieval_results]
        retrieval_deterministic = len(set(retrieval_hashes)) == 1
        print(f"🗄️  ChromaDB Retrieval: {'✅ CONSISTENT' if retrieval_deterministic else '❌ VARIABLE'}")
    
    # Check single sample evaluation
    if eval_results:
        eval_hashes = [hash_content(r) for r in eval_results]
        eval_deterministic = len(set(eval_hashes)) == 1
        print(f"📊 Single Sample RAGAS: {'✅ DETERMINISTIC' if eval_deterministic else '❌ VARIABLE'}")
    
    print(f"\n🎯 CONCLUSION:")
    if not llm_deterministic:
        print("   🔴 Root cause: LLM non-determinism")
    elif not fixed_deterministic:
        print("   🟡 Root cause: RAGAS framework non-determinism")
    elif not retrieval_deterministic:
        print("   🟠 Root cause: ChromaDB retrieval variability")
    elif not eval_deterministic:
        print("   🟣 Root cause: RAGAS evaluation variability")
    else:
        print("   🟢 All tests passed - investigate test design")
    
    print(f"\n💡 Next steps:")
    print(f"   - Focus debugging on the failing component")
    print(f"   - Consider running with different seeds")
    print(f"   - Check OpenAI API behavior")

if __name__ == "__main__":
    main() 