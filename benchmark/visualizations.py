"""
Visualization module for RAG benchmark results

This module provides functions to create charts and visualizations for comparing
different RAG approaches using RAGAS evaluation metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict


def create_visualizations(comparison_table: pd.DataFrame, output_dir: str = "benchmark_outputs"):
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
    
    plt.title('RAG Approaches: Overall Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Average RAGAS Score', fontsize=12, fontweight='bold')
    plt.xlabel('Approach', fontsize=12, fontweight='bold')
    plt.ylim(0, max(avg_scores) * 1.2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add description of what RAGAS measures
    plt.figtext(0.5, 0.02, 
                'RAGAS Score: Average of Response Relevancy, Factual Correctness, and Semantic Similarity metrics', 
                ha='center', fontsize=10, style='italic', color='gray')
    
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
    
    plt.title('RAG Approaches: Detailed Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('RAGAS Score', fontsize=12, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.xticks(x, comparison_table['Metric'], fontsize=11)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, 1.1)
    
    # Add metric descriptions (updated for universal metrics)
    metric_descriptions = {
        'Response Relevancy': 'How well the answer addresses the question asked',
        'Factual Correctness': 'How accurate the facts are compared to ground truth',
        'Semantic Similarity': 'How well the meaning matches the expected answer'
    }
    
    # Add descriptions below x-axis labels
    for i, metric in enumerate(comparison_table['Metric']):
        if metric in metric_descriptions:
            plt.text(i, -0.15, metric_descriptions[metric], 
                    ha='center', va='top', fontsize=9, 
                    style='italic', color='gray', wrap=True)
    
    # Adjust layout to accommodate descriptions
    plt.subplots_adjust(bottom=0.25)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Generated 2 visualization charts in '{output_dir}/' folder:")
    print("  - overall_performance_comparison.png")
    print("  - detailed_metrics_comparison.png") 