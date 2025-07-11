# RAGAS Benchmark: RAG vs GraphRAG vs Text2Cypher Evaluation

This benchmarking system evaluates and compares three RAG approaches using the RAGAS (RAG Assessment) framework:
- **ChromaDB RAG**: Traditional vector similarity search
- **GraphRAG**: Multi-hop graph traversal with context enhancement  
- **Text2Cypher**: Natural language to Cypher query translation

The system includes professional visualizations with charts, heatmaps, and performance comparisons automatically generated for comprehensive analysis.

## Overview

The benchmark evaluates all three approaches on 20 crafted questions from the `benchmark.csv` file using multiple RAGAS metrics:

- **Context Recall**: How well the retrieval system finds relevant information
- **Faithfulness**: How faithful the generated answer is to the retrieved context
- **Factual Correctness**: How factually accurate the response is compared to ground truth

## Files

- `ragas_benchmark.py` - Main benchmarking script (three-way comparison)
- `visualizations.py` - Visualization module for generating charts and graphs
- `test_ragas_setup.py` - Quick test script (runs on 3 questions)
- `benchmark.csv` - 20 benchmark questions with ground truth answers
- `BENCHMARK_README.md` - This documentation
- `benchmark_outputs/` - Generated results folder (CSV, JSON, PNG charts)

## Prerequisites

1. **Environment Setup**: Make sure your `.env` file contains:
   ```
   OPENAI_API_KEY=your_openai_key
   NEO4J_URI=your_neo4j_uri
   NEO4J_USERNAME=your_username
   NEO4J_PASSWORD=your_password
   ```

2. **Databases Ready**: 
   - ChromaDB should be populated with document chunks
   - Neo4j should have the knowledge graph created

3. **Dependencies**: All required packages should be installed via `requirements.txt`

## Quick Start

### 1. Test Setup (Recommended First)
```bash
python test_ragas_setup.py
```
This runs a quick test with 3 questions to verify everything works correctly.

### 2. Full Benchmark
```bash
python ragas_benchmark.py
```
This runs the complete three-way evaluation on all 20 questions with automatic visualization generation.


## Interpreting Results

### Metrics Explanation

1. **Context Recall (0.0-1.0)**: Higher = better retrieval coverage
2. **Faithfulness (0.0-1.0)**: Higher = answers stick to retrieved facts
3. **Factual Correctness (0.0-1.0)**: Higher = more accurate responses

### How Overall Performance is Calculated

#### Average Score Calculation
The overall performance for each approach is calculated as the **arithmetic mean** of all three metric scores:

```
Average Score = (Context Recall + Faithfulness + Factual Correctness) / 3
```

For example:
- ChromaDB RAG: (0.75 + 0.80 + 0.70) / 3 = 0.75
- GraphRAG: (0.85 + 0.88 + 0.82) / 3 = 0.85

#### Winner Determination
The **overall winner** is determined by comparing the average scores across all approaches.


### How RAGAS Creates Individual Scores

#### 1. Context Recall (0.0-1.0)
- **What it measures**: How well the retrieval system finds relevant information
- **How RAGAS calculates it**: Uses an LLM evaluator to determine if the retrieved contexts contain information needed to answer the question
- **Process**:
  - Takes the question and retrieved contexts
  - Asks the evaluator LLM: "Do these contexts contain the information needed to answer this question?"
  - Scores based on the evaluator's assessment

#### 2. Faithfulness (0.0-1.0)
- **What it measures**: How faithful the generated answer is to the retrieved context
- **How RAGAS calculates it**: Evaluates whether the response is supported by the retrieved documents
- **Process**:
  - Compares the generated answer against the retrieved contexts
  - Asks the evaluator LLM: "Is this answer supported by the provided context?"
  - Scores based on factual consistency with retrieved information

#### 3. Factual Correctness (0.0-1.0)
- **What it measures**: How factually accurate the response is compared to ground truth
- **How RAGAS calculates it**: Compares the generated answer against the reference answer
- **Process**:
  - Takes the generated answer and ground truth
  - Asks the evaluator LLM: "How factually correct is this answer compared to the reference?"
  - Scores based on factual accuracy and completeness

## Customization

### Adding New Metrics

To add custom RAGAS metrics, modify the `metrics` list in `evaluate_with_ragas()`:

```python
from ragas.metrics import YourCustomMetric

metrics = [
    LLMContextRecall(),
    Faithfulness(),
    YourCustomMetric(),  # Add here
    # ... other metrics
]
```
# Others can be found at: `https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/`

### Changing Question Set

Replace or modify `benchmark.csv` with your own questions:
```csv
question;ground_truth
Your question here?;Expected answer here
```

### Adjusting Retrieval Parameters

Modify the `k` parameter when calling the query functions:
```python
# Get more/fewer retrieved chunks
result = query_chroma_with_llm(query, k=10)  # Default is 3
```

### Customizing Visualizations

To modify chart appearance or add new visualizations, edit `visualizations.py`:
```python
# Change colors, styling, or add new charts
colors = ['#3498db', '#2ecc71', '#f39c12']  # Customize colors
plt.figure(figsize=(12, 8))  # Adjust chart size
```

