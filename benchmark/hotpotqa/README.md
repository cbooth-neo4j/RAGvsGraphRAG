# HotpotQA Fullwiki Benchmark

A comprehensive benchmarking pipeline for evaluating RAG vs GraphRAG approaches using the HotpotQA fullwiki dataset.

## Overview

This benchmark module provides:

1. **Data Loading**: Downloads HotpotQA questions and referenced Wikipedia articles
2. **Graph Ingestion**: Processes Wikipedia articles into Neo4j knowledge graph
3. **Retriever Evaluation**: Tests multiple retrieval approaches
4. **RAGAS Metrics**: Evaluates response quality with industry-standard metrics
5. **Comparison Reports**: Generates detailed comparison visualizations

## Quick Start

```bash
# Smoke test (50 questions, ~5 min, ~$5)
python -m benchmark.hotpotqa.benchmark_pipeline smoke

# Development benchmark (500 questions, ~1 hour, ~$20)
python -m benchmark.hotpotqa.benchmark_pipeline dev

# Full evaluation (all ~7400 questions, ~5 hours, ~$100+)
python -m benchmark.hotpotqa.benchmark_pipeline full
```

## Presets

| Preset | Questions | Retrievers | Est. Cost | Est. Time |
|--------|-----------|------------|-----------|-----------|
| `mini` | 10 | ChromaDB only | $1 | 5 min |
| `smoke` | 50 | ChromaDB, GraphRAG | $5 | 15 min |
| `dev` | 500 | All main retrievers | $20 | 60 min |
| `full` | ~7400 | All retrievers | $100+ | 5+ hours |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   HotpotQA      │────▶│   Wikipedia     │────▶│   Neo4j Graph   │
│   Questions     │     │   Articles      │     │   + ChromaDB    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Benchmark     │◀────│   RAGAS         │◀────│   Retrievers    │
│   Report        │     │   Evaluation    │     │   (7 types)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## CLI Options

```bash
python -m benchmark.hotpotqa.benchmark_pipeline [preset] [options]

Positional Arguments:
  preset              Benchmark preset: smoke, dev, full, mini (default: smoke)

Options:
  --retrievers        Specific retrievers to test (overrides preset)
  --skip-ingestion    Skip graph ingestion (use existing graph)
  --output-dir        Output directory for results
  --cache-dir         Cache directory for downloaded data
  --list-presets      List available presets and exit

Examples:
  # Test specific retrievers
  python -m benchmark.hotpotqa.benchmark_pipeline dev --retrievers chroma graphrag hybrid_cypher

  # Skip re-ingesting if graph already populated
  python -m benchmark.hotpotqa.benchmark_pipeline dev --skip-ingestion
```

## Module Structure

```
benchmark/hotpotqa/
├── __init__.py              # Module exports
├── benchmark_pipeline.py    # Main orchestrator
├── configs.py               # Presets and settings
├── data_loader.py           # HotpotQA + Wikipedia downloader
├── wiki_ingester.py         # Neo4j graph ingestion
└── README.md                # This file
```

## Available Retrievers

| Retriever | Description |
|-----------|-------------|
| `chroma` | ChromaDB vector similarity search |
| `graphrag` | Basic GraphRAG with chunk traversal |
| `advanced_graphrag` | Intelligent global/local/hybrid routing |
| `drift_graphrag` | DRIFT iterative refinement |
| `hybrid_cypher` | Hybrid vector + Cypher neighborhood |
| `neo4j_vector` | Pure Neo4j vector similarity |
| `text2cypher` | Natural language to Cypher |

## Output Files

After running a benchmark, outputs are saved to the output directory:

```
benchmark_outputs/hotpotqa/
├── hotpotqa_benchmark_results.json   # Complete results
├── simple_benchmark_comparison.csv   # Metric comparison
├── simple_benchmark_*.csv            # Per-retriever results
├── detailed_comparison.csv           # Detailed analysis
├── overall_performance_comparison.png
└── detailed_metrics_comparison.png
```

## Metrics

The benchmark evaluates using RAGAS universal metrics:

- **Response Relevancy**: Does the response address the question?
- **Factual Correctness**: Are the facts in the response correct?
- **Semantic Similarity**: Does the response meaning match ground truth?

## Data Caching

Downloaded data is cached locally:

```
data/hotpotqa/
├── hotpot_dev_fullwiki.json          # HotpotQA questions
├── articles/                          # Cached Wikipedia articles
│   └── Article_Name_abc123.json
└── prepared_corpus.json               # Preprocessed corpus
```

## Programmatic Usage

```python
from benchmark.hotpotqa import (
    prepare_corpus,
    WikiCorpusIngester,
    run_hotpotqa_benchmark
)

# Prepare data manually
corpus = prepare_corpus(question_limit=100)

# Ingest to graph
ingester = WikiCorpusIngester()
ingester.ingest_corpus(corpus["articles"])

# Or run full benchmark
results = run_hotpotqa_benchmark(
    preset="dev",
    retrievers=["chroma", "graphrag"],
    skip_ingestion=True
)
```

## Requirements

- Neo4j database running with required indexes
- OpenAI API key (or configured alternative LLM)
- ChromaDB installed
- Required packages: `wikipedia-api>=0.6.0`

## Notes

- HotpotQA fullwiki dev set contains ~7,400 multi-hop questions
- Each question requires reasoning over 2+ Wikipedia articles
- Wikipedia articles are downloaded via API with rate limiting
- Articles are cached locally after first download
- Graph ingestion includes entity extraction and community detection

