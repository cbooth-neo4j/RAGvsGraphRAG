"""
Visualization module for RAG benchmark results

This module provides functions to create charts and visualizations for comparing
different RAG approaches using RAGAS and HotpotQA evaluation metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional


def create_visualizations(
    comparison_table: pd.DataFrame, 
    output_dir: str = "benchmark_outputs",
    hotpotqa_results: Optional[Dict] = None,
    approach_names: Optional[Dict[str, str]] = None
):
    """Create and save visualization charts for the benchmark results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get available approaches from the comparison table (exclude non-numeric columns)
    available_approaches = []
    for col in comparison_table.columns:
        if col != 'Metric' and col != 'Improvement':
            try:
                # Check if column contains numeric data
                if comparison_table[col].dtype in ['float64', 'int64'] or comparison_table[col].apply(lambda x: isinstance(x, (int, float))).all():
                    available_approaches.append(col)
            except (TypeError, ValueError):
                continue
    
    if not available_approaches:
        print("⚠️  No numeric columns found for visualization")
        return
    
    # Chart 1: Overall Performance Comparison (Bar Chart)
    plt.figure(figsize=(12, 8))
    
    # Calculate average scores for each approach
    avg_scores = [comparison_table[approach].mean() for approach in available_approaches]
    
    # Use colors based on number of approaches
    color_palette = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']  # Blue, Green, Orange, Red, Purple, Teal
    colors = color_palette[:len(available_approaches)]
    bars = plt.bar(available_approaches, avg_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, score in zip(bars, avg_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title('RAGAS Overall Performance', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Average Score', fontsize=12, fontweight='bold')
    plt.xlabel('Retriever', fontsize=12, fontweight='bold')
    plt.ylim(0, max(avg_scores) * 1.2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add winner annotation
    winner_idx = avg_scores.index(max(avg_scores))
    winner_name = available_approaches[winner_idx]
    plt.annotate(f'🏆 Winner\n{winner_name}', 
                xy=(winner_idx, avg_scores[winner_idx]), 
                xytext=(winner_idx, avg_scores[winner_idx] + 0.05),
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Detailed Metrics Comparison (Grouped Bar Chart)
    plt.figure(figsize=(16, 10))
    
    # Prepare data for grouped bar chart
    x = range(len(comparison_table['Metric']))
    width = 0.8 / len(available_approaches)  # Adjust width based on number of approaches
    
    # Create bars for each approach
    for i, approach in enumerate(available_approaches):
        offset = (i - len(available_approaches)/2 + 0.5) * width
        color = colors[i] if i < len(colors) else colors[i % len(colors)]
        
        plt.bar([j + offset for j in x], comparison_table[approach], width, 
                label=approach, color=color, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, metric in enumerate(comparison_table['Metric']):
        for j, approach in enumerate(available_approaches):
            offset = (j - len(available_approaches)/2 + 0.5) * width
            value = comparison_table.iloc[i][approach]
            plt.text(i + offset, value + 0.02, 
                    f"{value:.3f}", 
                    ha='center', va='bottom', fontsize=9, rotation=90)
    
    plt.title('RAGAS Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.xlabel('Metric', fontsize=12, fontweight='bold')
    plt.xticks(x, comparison_table['Metric'], fontsize=11)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    charts_created = 2
    
    # Chart 3: HotpotQA Metrics (if provided)
    if hotpotqa_results:
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        retrievers = list(hotpotqa_results.keys())
        retriever_labels = [approach_names.get(r, r) if approach_names else r for r in retrievers]
        em_scores = [hotpotqa_results[r].get('exact_match', 0) for r in retrievers]
        f1_scores = [hotpotqa_results[r].get('f1', 0) for r in retrievers]
        
        x = range(len(retrievers))
        width = 0.35
        
        # Create grouped bars
        bars1 = plt.bar([i - width/2 for i in x], em_scores, width, 
                       label='Exact Match (EM)', color='#2ecc71', alpha=0.8, edgecolor='black')
        bars2 = plt.bar([i + width/2 for i in x], f1_scores, width,
                       label='F1 Score', color='#3498db', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.title('HotpotQA Native Metrics: Exact Match & F1 Score', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.xlabel('Retriever', fontsize=12, fontweight='bold')
        plt.xticks(x, retriever_labels, fontsize=11)
        plt.legend(fontsize=11, loc='upper right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.ylim(0, 1.15)
        
        # Add description
        plt.figtext(0.5, 0.02,
                   'HotpotQA Metrics: Standard benchmark metrics for factoid question answering',
                   ha='center', fontsize=10, style='italic', color='gray')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hotpotqa_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        charts_created += 1
    
    # Chart 4: Combined Overview (if HotpotQA results available)
    if hotpotqa_results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: HotpotQA metrics (primary for this benchmark)
        ax1 = axes[0]
        retrievers = list(hotpotqa_results.keys())
        retriever_labels = [approach_names.get(r, r) if approach_names else r for r in retrievers]
        
        # Calculate combined HotpotQA score (average of EM and F1)
        combined_scores = [(hotpotqa_results[r].get('exact_match', 0) + hotpotqa_results[r].get('f1', 0)) / 2 
                          for r in retrievers]
        
        colors_hotpot = ['#27ae60' if s >= 0.8 else '#f39c12' if s >= 0.5 else '#e74c3c' for s in combined_scores]
        bars = ax1.bar(retriever_labels, combined_scores, color=colors_hotpot, alpha=0.8, edgecolor='black')
        
        for bar, score in zip(bars, combined_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax1.set_title('HotpotQA Score\n(Primary Metric)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average of EM + F1', fontsize=11)
        ax1.set_ylim(0, 1.15)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (0.5)')
        
        # Right: RAGAS metrics (secondary)
        ax2 = axes[1]
        ragas_avg = [comparison_table[approach].mean() for approach in available_approaches]
        colors_ragas = ['#3498db'] * len(available_approaches)
        bars2 = ax2.bar(available_approaches, ragas_avg, color=colors_ragas, alpha=0.8, edgecolor='black')
        
        for bar, score in zip(bars2, ragas_avg):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax2.set_title('RAGAS Score\n(Secondary Metric)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average RAGAS Score', fontsize=11)
        ax2.set_ylim(0, 1.15)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('Benchmark Overview: HotpotQA vs RAGAS Metrics', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/benchmark_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        charts_created += 1
    
    print(f"\n📊 Generated {charts_created} visualization charts in '{output_dir}/' folder:")
    print("  - overall_performance_comparison.png")
    print("  - detailed_metrics_comparison.png")
    if hotpotqa_results:
        print("  - hotpotqa_metrics_comparison.png")
        print("  - benchmark_overview.png")