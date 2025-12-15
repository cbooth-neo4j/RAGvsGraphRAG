# RAGBench Integration

This module integrates the [RAGBench dataset](https://huggingface.co/datasets/galileo-ai/ragbench) with your existing RAG vs GraphRAG evaluation framework.

## What it does

1. **Document Ingestion**: Processes RAGBench documents (not questions/answers) through your existing graph processors
2. **Evaluation Dataset Creation**: Converts RAGBench Q&A pairs to formats compatible with `ragas_benchmark.py`
3. **Detailed Reporting**: Creates human-readable HTML reports showing individual responses and RAGAS scores

## Quick Start

### Nano Test (10 records, ~$2 cost)
```bash
python benchmark/ragbench_pipeline.py nano
```

### Micro Test (50 records, ~$10 cost)
```bash
python benchmark/ragbench_pipeline.py micro
```

### Small Test (500 records, ~$100 cost)
```bash
python benchmark/ragbench_pipeline.py small --approaches chroma graphrag advanced_graphrag
```

### Medium Test with Enhanced Processing
```bash
python benchmark/ragbench_pipeline.py medium --enhanced
```

## What You Get

After running the pipeline, you'll find in `benchmark_outputs/{preset_name}/`:

### 📄 detailed_results.html
Beautiful HTML report showing:
- Each question and expected answer
- Actual responses from each retriever
- Retrieved document contexts
- Individual RAGAS scores with color coding
- Space for human verification notes

### 📊 detailed_comparison.csv
Spreadsheet with all questions, responses, and scores for analysis

### 📋 summary_report.json
Aggregated statistics and performance breakdowns

### 📈 Traditional outputs
- Comparison tables (CSV)
- Visualization charts (PNG)
- Raw evaluation data (JSON)

## Configuration Presets

| Preset | Records | Datasets | Storage | RAM | Cost | Description |
|--------|---------|----------|---------|-----|------|-------------|
| `nano` | 10 | finqa | 0.1 GB | 2 GB | $2 | Ultra-tiny test |
| `micro` | 50 | covidqa | 1 GB | 4 GB | $10 | Tiny test |
| `small` | 500 | covidqa, expertqa | 8 GB | 16 GB | $100 | Small test |
| `medium` | 2K | finqa, hotpotqa, msmarco | 32 GB | 64 GB | $400 | Medium test |
| `large` | 10K | Multiple domains | 160 GB | 256 GB | $2000 | Large test |
| `full_test` | ~15K | All 12 datasets | 240 GB | 512 GB | $3000 | Full test split |

## Data Structure

### RAGBench Record
```json
{
  "id": "1234",
  "user_input": "What are the most common viruses?",
  "documents": [
    "Title: Medical research...\nPassage: The main viruses...",
    "Title: Clinical study...\nPassage: Research shows...",
    "Title: Epidemiology...\nPassage: Data indicates...",
    "Title: Public health...\nPassage: Statistics reveal..."
  ],
  "response": "The most common viruses are...",
  "dataset_name": "covidqa"
}
```

### What Goes Where
- **documents** → Neo4j graph (processed by your graph_processor.py)
- **user_input + response** → Evaluation dataset (for ragas_benchmark.py)
- **Questions are NOT stored in the graph** (only used for evaluation)

## Advanced Usage

### Skip Ingestion (Use Existing Graph)
```bash
python benchmark/ragbench_pipeline.py small --skip-ingestion
```

### Test Specific Retrievers
```bash
python benchmark/ragbench_pipeline.py medium --approaches drift_graphrag hybrid_cypher
```

### Use Advanced Processing
```bash
python benchmark/ragbench_pipeline.py large --enhanced
```

## Module Structure

```
benchmark/ragbench/
├── __init__.py              # Module exports
├── ingester.py              # Document ingestion to Neo4j
├── evaluator.py             # Q&A dataset creation
├── results_formatter.py     # Human-readable report generation
├── configs.py               # Preset configurations
├── README.md                # This file
└── data/                    # Generated data
    ├── micro_eval.jsonl     # Q&A pairs for evaluation
    ├── detailed_results.html # Human-readable report
    └── summary_report.json   # Aggregated statistics
```

## Integration Points

### With Your Existing Code
- Uses your `graph_processor.py` and `advanced_graph_processor.py`
- Integrates with your `ragas_benchmark.py`
- Compatible with your existing retriever modules
- Follows your existing output formats

### With RAGBench Dataset
- Automatically downloads from Hugging Face
- Handles all 12 dataset subsets
- Respects train/validation/test splits (uses test only)
- Maintains domain and dataset metadata

## Domain Coverage

| Domain | Datasets | Description |
|--------|----------|-------------|
| Medical | covidqa, pubmedqa | Medical research and COVID-19 |
| Financial | finqa, tatqa | Financial documents and tables |
| Legal | cuad | Legal contracts and agreements |
| Technical | techqa, emanual | Technical documentation |
| QA | expertqa, hotpotqa, msmarco | General question answering |
| Multimodal | hagrid | Mixed content types |
| Other | delucionqa | Specialized domains |

## Output Examples

### HTML Report Features
- 📊 Summary statistics table
- 🎯 Question-by-question breakdown
- 🤖 Side-by-side retriever comparisons
- 📄 Retrieved context preview
- 🏆 Color-coded RAGAS scores
- ✍️ Human verification sections

### Sample Question Block
```
Question 1: What are the most common viruses?
Domain: medical | Dataset: covidqa

✅ Expected Answer (Ground Truth)
The most common viruses are enterovirus, respiratory syncytial virus...

🤖 ChromaDB RAG                    Relevancy: 0.85  Factual: 0.78  Semantic: 0.82
Response: Based on the retrieved documents, the most common viruses include...
Context: [3 documents] Title: Medical research... Passage: The main viruses...

🤖 GraphRAG                        Relevancy: 0.91  Factual: 0.82  Semantic: 0.88
Response: The analysis of medical literature shows that common viruses are...
Context: [5 documents] Title: Clinical study... Passage: Research shows...

🔍 Human Verification
Which response was most accurate? Any observations?
[Text area for notes]
```

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure you're running from the project root
- Check that all dependencies are installed

**Memory Issues** 
- Start with `micro` preset
- Monitor RAM usage during processing
- Use `--skip-ingestion` to test evaluation only

**API Costs**
- Check cost estimates before proceeding
- Start small and scale up
- Monitor OpenAI usage dashboard

**Graph Connection**
- Ensure Neo4j is running
- Check environment variables
- Verify database schema setup

### Getting Help

1. Check the detailed error messages
2. Review the processing statistics
3. Examine the generated log files
4. Start with smaller presets to isolate issues

## Future Enhancements

- [ ] Per-question RAGAS scores (requires ragas_benchmark.py modification)
- [ ] Interactive HTML report with filtering
- [ ] Automated performance regression detection
- [ ] Integration with more evaluation frameworks
- [ ] Support for custom RAGBench-style datasets
