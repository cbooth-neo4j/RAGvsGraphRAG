"""
HotpotQA Native Evaluation Metrics

Implements the standard HotpotQA evaluation metrics:
- Exact Match (EM): Binary match after normalization
- F1 Score: Token-level overlap between prediction and answer
- Supporting Facts EM/F1: Evaluation of retrieved evidence (optional)

These metrics are designed for short, factoid answers typical of HotpotQA.
"""

import re
import string
from typing import Dict, List, Any, Tuple
from collections import Counter


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    
    Follows HotpotQA official normalization:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove articles
    articles = {"a", "an", "the"}
    words = text.split()
    words = [w for w in words if w not in articles]
    text = " ".join(words)
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute Exact Match score.
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        1.0 if normalized strings match exactly, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score based on token overlap.
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score (harmonic mean of precision and recall)
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # Handle empty cases
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    # Count token occurrences
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    
    # Find common tokens
    common = pred_counter & truth_counter
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate_answer(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Evaluate a single answer using all HotpotQA metrics.
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        Dictionary with EM and F1 scores
    """
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "f1": f1_score(prediction, ground_truth)
    }


def evaluate_batch(
    predictions: List[str], 
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Evaluate a batch of answers and compute average metrics.
    
    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        
    Returns:
        Dictionary with average EM and F1 scores
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Mismatched lengths: {len(predictions)} predictions vs {len(ground_truths)} ground truths")
    
    if not predictions:
        return {"exact_match": 0.0, "f1": 0.0, "count": 0}
    
    total_em = 0.0
    total_f1 = 0.0
    
    for pred, truth in zip(predictions, ground_truths):
        total_em += exact_match(pred, truth)
        total_f1 += f1_score(pred, truth)
    
    n = len(predictions)
    return {
        "exact_match": total_em / n,
        "f1": total_f1 / n,
        "count": n
    }


def evaluate_dataset(
    dataset: List[Dict[str, Any]],
    prediction_key: str = "response",
    ground_truth_key: str = "reference"
) -> Dict[str, Any]:
    """
    Evaluate a RAGAS-style dataset using HotpotQA metrics.
    
    Args:
        dataset: List of evaluation records with predictions and ground truths
        prediction_key: Key for model's response in each record
        ground_truth_key: Key for ground truth in each record
        
    Returns:
        Dictionary with:
        - 'metrics': averaged EM and F1
        - 'per_sample': individual scores for each sample
    """
    per_sample = []
    
    for i, record in enumerate(dataset):
        prediction = record.get(prediction_key, "")
        ground_truth = record.get(ground_truth_key, "")
        
        scores = evaluate_answer(prediction, ground_truth)
        scores["question"] = record.get("user_input", record.get("question", f"Q{i+1}"))
        scores["prediction"] = prediction
        scores["ground_truth"] = ground_truth
        per_sample.append(scores)
    
    # Compute averages
    if per_sample:
        avg_em = sum(s["exact_match"] for s in per_sample) / len(per_sample)
        avg_f1 = sum(s["f1"] for s in per_sample) / len(per_sample)
    else:
        avg_em = 0.0
        avg_f1 = 0.0
    
    return {
        "metrics": {
            "exact_match": avg_em,
            "f1": avg_f1,
            "count": len(per_sample)
        },
        "per_sample": per_sample
    }


def print_hotpotqa_results(
    results: Dict[str, Any],
    approach_name: str = "Retriever",
    show_samples: int = 5
) -> None:
    """
    Pretty-print HotpotQA evaluation results.
    
    Args:
        results: Output from evaluate_dataset()
        approach_name: Name of the retriever being evaluated
        show_samples: Number of sample results to show (0 to hide)
    """
    metrics = results["metrics"]
    per_sample = results["per_sample"]
    
    print(f"\n{'='*60}")
    print(f"HOTPOTQA METRICS: {approach_name}")
    print(f"{'='*60}")
    print(f"   Exact Match (EM): {metrics['exact_match']:.4f} ({metrics['exact_match']*100:.1f}%)")
    print(f"   F1 Score:         {metrics['f1']:.4f} ({metrics['f1']*100:.1f}%)")
    print(f"   Samples:          {metrics['count']}")
    
    if show_samples > 0 and per_sample:
        print(f"\n   Sample Results (first {min(show_samples, len(per_sample))}):")
        print(f"   {'-'*50}")
        for sample in per_sample[:show_samples]:
            em_icon = "✓" if sample["exact_match"] == 1.0 else "✗"
            print(f"   {em_icon} Q: {sample['question'][:50]}...")
            print(f"      Pred:  '{sample['prediction'][:40]}...' " if len(sample['prediction']) > 40 else f"      Pred:  '{sample['prediction']}'")
            print(f"      Truth: '{sample['ground_truth']}'")
            print(f"      EM={sample['exact_match']:.0f}, F1={sample['f1']:.2f}")
            print()


# Convenience function for integration with benchmark pipeline
def evaluate_retriever_hotpotqa(
    dataset: List[Dict[str, Any]],
    approach_name: str = "Retriever"
) -> Dict[str, float]:
    """
    Evaluate a retriever using HotpotQA metrics.
    
    This is the main integration point for the benchmark pipeline.
    
    Args:
        dataset: RAGAS-style dataset with 'response' and 'reference' keys
        approach_name: Name for logging
        
    Returns:
        Dictionary with 'exact_match' and 'f1' scores
    """
    print(f"\nEvaluating {approach_name} using HotpotQA metrics...")
    
    results = evaluate_dataset(dataset)
    
    # Print summary
    metrics = results["metrics"]
    print(f"   EM: {metrics['exact_match']:.4f}, F1: {metrics['f1']:.4f} ({metrics['count']} samples)")
    
    # Show failures for debugging
    failures = [s for s in results["per_sample"] if s["exact_match"] == 0.0]
    if failures and len(failures) <= 5:
        print(f"   Failures ({len(failures)}):")
        for f in failures[:3]:
            print(f"      - '{f['prediction'][:30]}...' vs '{f['ground_truth']}'")
    
    return {
        "exact_match": metrics["exact_match"],
        "f1": metrics["f1"]
    }

