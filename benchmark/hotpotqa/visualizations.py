"""
HotpotQA Benchmark Visualizations

Clean, focused visualizations for HotpotQA benchmark results.
Primary metrics: Exact Match (EM) and F1 Score
Optional: RAGAS metrics (when --ragas flag is used)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional


def create_hotpotqa_visualizations(
    hotpotqa_results: Dict,
    approach_names: Dict[str, str],
    output_dir: str,
    ragas_results: Optional[Dict] = None
):
    """
    Create visualizations for HotpotQA benchmark results.
    
    Args:
        hotpotqa_results: Dict of retriever -> {exact_match, f1}
        approach_names: Dict mapping retriever keys to display names
        output_dir: Directory to save charts
        ragas_results: Optional RAGAS results (only included if --ragas flag was used)
    """
    plt.style.use('default')
    Path(output_dir).mkdir(exist_ok=True)
    
    # Prepare data
    retrievers = list(hotpotqa_results.keys())
    labels = [approach_names.get(r, r) for r in retrievers]
    em_scores = [hotpotqa_results[r].get('exact_match', 0) for r in retrievers]
    f1_scores = [hotpotqa_results[r].get('f1', 0) for r in retrievers]
    
    charts_created = 0
    
    # =========================================================================
    # Chart 1: HotpotQA Metrics (Primary - Always Created)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = range(len(retrievers))
    width = 0.35
    
    # Color bars based on score
    em_colors = ['#27ae60' if s >= 0.7 else '#f39c12' if s >= 0.4 else '#e74c3c' for s in em_scores]
    f1_colors = ['#2980b9' if s >= 0.7 else '#9b59b6' if s >= 0.4 else '#c0392b' for s in f1_scores]
    
    bars1 = ax.bar([i - width/2 for i in x], em_scores, width, 
                   label='Exact Match (EM)', color=em_colors, alpha=0.85, edgecolor='black')
    bars2 = ax.bar([i + width/2 for i in x], f1_scores, width,
                   label='F1 Score', color=f1_colors, alpha=0.85, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_title('HotpotQA Benchmark Results', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Retriever', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.15)
    
    # Add threshold lines
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.4, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hotpotqa_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    charts_created += 1
    
    # =========================================================================
    # Chart 2: Leaderboard Style (if multiple retrievers)
    # =========================================================================
    if len(retrievers) > 1:
        fig, ax = plt.subplots(figsize=(10, max(6, len(retrievers) * 0.8)))
        
        # Sort by average of EM + F1
        avg_scores = [(em + f1) / 2 for em, f1 in zip(em_scores, f1_scores)]
        sorted_indices = sorted(range(len(avg_scores)), key=lambda i: avg_scores[i], reverse=True)
        
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_avg = [avg_scores[i] for i in sorted_indices]
        
        colors = ['#27ae60' if s >= 0.7 else '#f39c12' if s >= 0.4 else '#e74c3c' for s in sorted_avg]
        
        bars = ax.barh(sorted_labels, sorted_avg, color=colors, alpha=0.85, edgecolor='black')
        
        # Add value labels
        for bar, score in zip(bars, sorted_avg):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', fontweight='bold', fontsize=11)
        
        ax.set_title('Retriever Leaderboard (Avg of EM + F1)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Average Score', fontsize=12)
        ax.set_xlim(0, 1.15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Rank indicators
        for i, label in enumerate(sorted_labels[:3]):
            medal = ['🥇', '🥈', '🥉'][i]
            ax.text(-0.08, i, medal, fontsize=14, va='center', ha='center', transform=ax.get_yaxis_transform())
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/leaderboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        charts_created += 1
    
    # =========================================================================
    # Chart 3: RAGAS Metrics (Only if --ragas flag was used)
    # =========================================================================
    if ragas_results:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        metrics = ['response_relevancy', 'factual_correctness', 'semantic_similarity']
        metric_labels = ['Response\nRelevancy', 'Factual\nCorrectness', 'Semantic\nSimilarity']
        
        x = range(len(metrics))
        width = 0.8 / len(retrievers)
        
        colors = sns.color_palette("husl", len(retrievers))
        
        for i, retriever in enumerate(retrievers):
            offset = (i - len(retrievers)/2 + 0.5) * width
            values = [ragas_results[retriever].get(m, 0) for m in metrics]
            bars = ax.bar([j + offset for j in x], values, width, 
                         label=labels[i], color=colors[i], alpha=0.85, edgecolor='black')
            
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
        
        ax.set_title('RAGAS Metrics (Optional)', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.15)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ragas_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        charts_created += 1
    
    print(f"\n📊 Generated {charts_created} visualization(s) in '{output_dir}/':")
    print(f"  - hotpotqa_results.png")
    if len(retrievers) > 1:
        print(f"  - leaderboard.png")
    if ragas_results:
        print(f"  - ragas_metrics.png")

