# RAGAS Benchmark: RAG vs GraphRAG Evaluation

This benchmarking system evaluates and compares ChromaDB RAG vs GraphRAG approaches using the RAGAS (RAG Assessment) framework.

## Overview

The benchmark evaluates both approaches on 20 carefully crafted questions from the `benchmark.csv` file using multiple RAGAS metrics:

- **Context Recall**: How well the retrieval system finds relevant information
- **Faithfulness**: How faithful the generated answer is to the retrieved context
- **Factual Correctness**: How factually accurate the response is compared to ground truth
- **Context Precision**: How precise the retrieved context is (relevance of retrieved chunks)
- **Answer Relevancy**: How relevant the generated answer is to the question
- **Context Relevancy**: How relevant the retrieved context is to the question

## Files

- `benchmark_ragas.py` - Main benchmarking script
- `test_ragas_setup.py` - Quick test script (runs on 3 questions)
- `benchmark.csv` - 20 benchmark questions with ground truth answers
- `BENCHMARK_README.md` - This documentation

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
python benchmark_ragas.py
```
This runs the complete evaluation on all 20 questions.

## What the Benchmark Does

1. **Data Collection Phase**:
   - Loads 20 questions from `benchmark.csv`
   - Runs each question through both ChromaDB and GraphRAG systems
   - Collects responses and retrieved contexts

2. **RAGAS Evaluation Phase**:
   - Evaluates both datasets using 6 RAGAS metrics
   - Uses OpenAI GPT-4o-mini as the evaluator LLM

3. **Results Analysis Phase**:
   - Creates comparison table showing performance differences
   - Calculates overall improvement percentages
   - Generates summary statistics

4. **Output Generation**:
   - Displays comprehensive results table
   - Saves detailed results to CSV files
   - Creates JSON file with raw RAGAS scores

## Output Files

After running the benchmark, you'll get:

- `benchmark_results_chroma.csv` - Detailed ChromaDB results
- `benchmark_results_graphrag.csv` - Detailed GraphRAG results  
- `benchmark_comparison_table.csv` - Side-by-side comparison table
- `benchmark_ragas_results.json` - Raw RAGAS evaluation scores

## Expected Runtime

- **Test (3 questions)**: ~3-5 minutes
- **Full benchmark (20 questions)**: ~15-25 minutes

The runtime depends on:
- OpenAI API response times
- Neo4j query complexity
- ChromaDB collection size

## Interpreting Results

### Metrics Explanation

1. **Context Recall (0.0-1.0)**: Higher = better retrieval coverage
2. **Faithfulness (0.0-1.0)**: Higher = answers stick to retrieved facts
3. **Factual Correctness (0.0-1.0)**: Higher = more accurate responses
4. **Context Precision (0.0-1.0)**: Higher = more relevant retrieved chunks
5. **Answer Relevancy (0.0-1.0)**: Higher = answers directly address questions
6. **Context Relevancy (0.0-1.0)**: Higher = retrieved context is more relevant

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

### Sample Output Table

```
                   Metric  ChromaDB RAG  GraphRAG  Improvement
            Context Recall        0.8500    0.9200      +8.24%
              Faithfulness        0.7800    0.8400      +7.69%
       Factual Correctness        0.7200    0.8100     +12.50%
         Context Precision        0.6900    0.7800     +13.04%
          Answer Relevancy        0.8200    0.8600      +4.88%
         Context Relevancy        0.7500    0.8300     +10.67%
```

## Troubleshooting

### Common Issues

1. **"No results found"**: Check that databases are populated
2. **OpenAI API errors**: Verify API key and check rate limits
3. **Neo4j connection errors**: Confirm database is running and credentials are correct
4. **Memory issues**: Consider reducing batch size or running test first

### Debug Mode

For debugging, you can run individual functions:

```python
from benchmark_ragas import load_benchmark_data, collect_evaluation_data

# Load just one question
data = load_benchmark_data("benchmark.csv")[:1]

# Test ChromaDB only
chroma_results = collect_evaluation_data(data, approach="chroma")
print(chroma_results)
```

## Performance Tips

1. **Start with test**: Always run `test_ragas_setup.py` first
2. **Monitor progress**: The script shows progress for each question
3. **Check logs**: Look for error messages if questions fail
4. **Resource usage**: Monitor API usage and database connections

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
result = query_chroma_with_llm(query, k=10)  # Default is 5
```

## Support

If you encounter issues:

1. Check the console output for specific error messages
2. Verify all prerequisites are met
3. Run the test script first to isolate issues
4. Check that both databases contain the expected data

The benchmark provides detailed logging to help identify and resolve any issues. 