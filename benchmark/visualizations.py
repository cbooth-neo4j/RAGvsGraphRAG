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
    
    # Chart 1: Overall Performance Comparison (Bar Chart)
    plt.figure(figsize=(12, 8))
    
    # Calculate average scores for each approach
    approaches = ['ChromaDB RAG', 'GraphRAG', 'Text2Cypher']
    avg_scores = [comparison_table[approach].mean() for approach in approaches]
    
    colors = ['#3498db', '#2ecc71', '#f39c12']  # Blue, Green, Orange
    bars = plt.bar(approaches, avg_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, score in zip(bars, avg_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title('RAG Approaches: Overall Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Average RAGAS Score', fontsize=12, fontweight='bold')
    plt.xlabel('Approach', fontsize=12, fontweight='bold')
    plt.ylim(0, max(avg_scores) * 1.2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add winner annotation
    winner_idx = avg_scores.index(max(avg_scores))
    winner_name = approaches[winner_idx]
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
    plt.figure(figsize=(14, 8))
    
    # Prepare data for grouped bar chart
    x = range(len(comparison_table['Metric']))
    width = 0.25
    
    plt.bar([i - width for i in x], comparison_table['ChromaDB RAG'], width, 
            label='ChromaDB RAG', color='#3498db', alpha=0.8, edgecolor='black')
    plt.bar(x, comparison_table['GraphRAG'], width, 
            label='GraphRAG', color='#2ecc71', alpha=0.8, edgecolor='black')
    plt.bar([i + width for i in x], comparison_table['Text2Cypher'], width, 
            label='Text2Cypher', color='#f39c12', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, metric in enumerate(comparison_table['Metric']):
        plt.text(i - width, comparison_table.iloc[i]['ChromaDB RAG'] + 0.02, 
                f"{comparison_table.iloc[i]['ChromaDB RAG']:.3f}", 
                ha='center', va='bottom', fontsize=9, rotation=90)
        plt.text(i, comparison_table.iloc[i]['GraphRAG'] + 0.02, 
                f"{comparison_table.iloc[i]['GraphRAG']:.3f}", 
                ha='center', va='bottom', fontsize=9, rotation=90)
        plt.text(i + width, comparison_table.iloc[i]['Text2Cypher'] + 0.02, 
                f"{comparison_table.iloc[i]['Text2Cypher']:.3f}", 
                ha='center', va='bottom', fontsize=9, rotation=90)
    
    plt.title('RAG Approaches: Detailed Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('RAGAS Score', fontsize=12, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.xticks(x, comparison_table['Metric'], fontsize=11)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Generated 2 visualization charts in '{output_dir}/' folder:")
    print("  - overall_performance_comparison.png")
    print("  - detailed_metrics_comparison.png") 