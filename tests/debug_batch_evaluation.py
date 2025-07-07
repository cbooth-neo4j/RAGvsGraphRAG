"""
Batch vs Individual RAGAS Evaluation Test

This script tests whether RAGAS evaluation variability comes from
batch processing vs individual sample evaluation.
"""

import os
import sys
import hashlib
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# RAGAS setup
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# Our RAG systems
from RAGvsGraphRAG import query_chroma_with_llm

# Fixed seed
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={"seed": SEED},
    max_retries=0
)

def hash_content(content: Any) -> str:
    """Generate hash of content for comparison"""
    return hashlib.md5(str(content).encode()).hexdigest()[:8]

def get_test_samples():
    """Get the same 3 samples used in the main test"""
    queries = [
        "What city is NovaGrid Energy Corporation headquartered in?",
        "In which year is AlTahadi Aviation Group scheduled to take its inaugural commercial flight?", 
        "AtlasVentures Consulting Group is headquartered in which country?"
    ]
    
    references = ["Toronto", "2025", "United Kingdom"]  # Known answers
    
    samples = []
    for i, (query, ref) in enumerate(zip(queries, references)):
        print(f"   Generating sample {i+1}: {query[:50]}...")
        
        # Get ChromaDB result (we know this is consistent)
        result = query_chroma_with_llm(query, k=1)
        
        # Create evaluation sample
        retrieved_contexts = []
        for detail in result.get('retrieval_details', []):
            content = detail.get('content', '')
            if content:
                if len(content) > 1000:
                    content = content[:1000] + "..."
                retrieved_contexts.append(content)
        
        if not retrieved_contexts:
            retrieved_contexts = ["No relevant context retrieved"]
        
        sample = {
            "user_input": query,
            "retrieved_contexts": retrieved_contexts,
            "response": result.get('final_answer', ''),
            "reference": ref
        }
        
        samples.append(sample)
        print(f"      Sample hash: {hash_content(sample)}")
    
    return samples

def evaluate_individual_samples(samples: List[Dict], run_num: int):
    """Evaluate each sample individually and collect results"""
    print(f"\n🔄 Individual Evaluation Run {run_num}")
    
    individual_results = []
    
    for i, sample in enumerate(samples):
        print(f"   📊 Evaluating sample {i+1} individually...")
        
        # Create dataset with single sample
        dataset = EvaluationDataset.from_list([sample])
        evaluator_llm = LangchainLLMWrapper(llm)
        
        metrics = [
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness()
        ]
        
        # Evaluate single sample
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm
        )
        
        # Extract scores for this sample
        df = result.to_pandas()
        sample_scores = {}
        for col in df.columns:
            if any(metric in col.lower() for metric in ['context_recall', 'faithfulness', 'factual_correctness']):
                try:
                    sample_scores[col] = df[col].iloc[0]
                except:
                    continue
        
        individual_results.append(sample_scores)
        print(f"      Scores: {sample_scores}")
    
    # Calculate averages
    avg_scores = {}
    if individual_results:
        for key in individual_results[0].keys():
            avg_scores[key] = sum(result[key] for result in individual_results) / len(individual_results)
    
    print(f"   📈 Individual Averages: {avg_scores}")
    print(f"   🏷️  Hash: {hash_content(avg_scores)}")
    
    return avg_scores, individual_results

def evaluate_batch_samples(samples: List[Dict], run_num: int):
    """Evaluate all samples as a batch"""
    print(f"\n🔄 Batch Evaluation Run {run_num}")
    
    # Create dataset with all samples
    dataset = EvaluationDataset.from_list(samples)
    evaluator_llm = LangchainLLMWrapper(llm)
    
    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness()
    ]
    
    # Evaluate batch
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm
    )
    
    # Extract average scores
    df = result.to_pandas()
    batch_scores = {}
    for col in df.columns:
        if any(metric in col.lower() for metric in ['context_recall', 'faithfulness', 'factual_correctness']):
            try:
                batch_scores[col] = df[col].mean()
            except:
                continue
    
    print(f"   📈 Batch Averages: {batch_scores}")
    print(f"   🏷️  Hash: {hash_content(batch_scores)}")
    
    return batch_scores

def main():
    """Test individual vs batch evaluation"""
    print("🔍 BATCH vs INDIVIDUAL RAGAS EVALUATION TEST")
    print("=" * 60)
    
    # Get consistent test samples
    print("\n📋 Generating test samples...")
    samples = get_test_samples()
    print(f"✅ Generated {len(samples)} test samples")
    print(f"🔒 Samples hash: {hash_content(samples)}")
    
    # Test individual evaluation multiple times
    print(f"\n" + "="*60)
    print("🔍 INDIVIDUAL EVALUATION TESTS")
    print("=" * 60)
    
    individual_runs = []
    for run in range(3):
        avg_scores, sample_results = evaluate_individual_samples(samples, run + 1)
        individual_runs.append(avg_scores)
    
    # Test batch evaluation multiple times
    print(f"\n" + "="*60)
    print("🔍 BATCH EVALUATION TESTS")
    print("=" * 60)
    
    batch_runs = []
    for run in range(3):
        batch_scores = evaluate_batch_samples(samples, run + 1)
        batch_runs.append(batch_scores)
    
    # Compare results
    print(f"\n" + "="*60)
    print("🔍 COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Check individual consistency
    individual_hashes = [hash_content(run) for run in individual_runs]
    individual_consistent = len(set(individual_hashes)) == 1
    print(f"👤 Individual Evaluation Consistent: {'✅ YES' if individual_consistent else '❌ NO'}")
    
    if not individual_consistent:
        print("   Individual run hashes:")
        for i, hash_val in enumerate(individual_hashes):
            print(f"      Run {i+1}: {hash_val}")
    
    # Check batch consistency
    batch_hashes = [hash_content(run) for run in batch_runs]
    batch_consistent = len(set(batch_hashes)) == 1
    print(f"📦 Batch Evaluation Consistent: {'✅ YES' if batch_consistent else '❌ NO'}")
    
    if not batch_consistent:
        print("   Batch run hashes:")
        for i, hash_val in enumerate(batch_hashes):
            print(f"      Run {i+1}: {hash_val}")
    
    # Check if individual vs batch gives different results
    if individual_runs and batch_runs:
        individual_avg = individual_runs[0]
        batch_avg = batch_runs[0]
        
        print(f"\n📊 Score Comparison (First Run):")
        for key in individual_avg.keys():
            ind_val = individual_avg.get(key, 0)
            batch_val = batch_avg.get(key, 0)
            diff = abs(ind_val - batch_val)
            print(f"   {key}:")
            print(f"      Individual: {ind_val:.6f}")
            print(f"      Batch:      {batch_val:.6f}")
            print(f"      Difference: {diff:.6f}")
    
    # Final conclusion
    print(f"\n🎯 CONCLUSION:")
    if individual_consistent and batch_consistent:
        print("   🟢 Both individual and batch evaluation are consistent")
        print("   🤔 The variability must be elsewhere in the pipeline")
    elif individual_consistent and not batch_consistent:
        print("   🟡 Individual evaluation is consistent, batch evaluation is not")
        print("   💡 Issue is in RAGAS batch processing")
    elif not individual_consistent and batch_consistent:
        print("   🟠 Batch evaluation is consistent, individual evaluation is not")
        print("   💡 Issue is in individual evaluation setup")
    else:
        print("   🔴 Both individual and batch evaluation are inconsistent")
        print("   💡 Issue is fundamental to RAGAS setup")

if __name__ == "__main__":
    main() 