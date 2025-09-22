# Benchmark System

Comprehensive evaluation framework for comparing RAG approaches using RAGAS metrics.

## 🚀 Quick Start

```bash
# Run full benchmark comparison
python benchmark/ragas_benchmark.py --all

# Run specific retrievers
python benchmark/ragas_benchmark.py --chroma --graphrag

# Use custom benchmark data
python benchmark/ragas_benchmark.py --all --jsonl path/to/benchmark.jsonl
```
### **Retrieval Approaches**
  --all                 Test all approaches
  --chroma              Include ChromaDB RAG in testing
  --graphrag            Include GraphRAG in testing
  --text2cypher         Include Text2Cypher in testing
  --advanced-graphrag   Include Advanced GraphRAG (intelligent global/local/hybrid) in testing
  --drift-graphrag      Include DRIFT GraphRAG (iterative refinement) in testing
  --neo4j-vector        Include Neo4j Vector RAG (pure vector similarity) in testing
  --hybrid-cypher       Include Hybrid Cypher RAG (hybrid + generic neighborhood) in testing

## 📁 Benchmark File Selection

The benchmark system uses the following **priority order** for selecting evaluation data:

1. **`--jsonl path/to/file.jsonl`** → Uses specified JSONL file (highest priority)
2. **`--csv path/to/file.csv`** → Uses specified CSV file  
3. **No file specified** → Uses default `benchmark/benchmark.csv` (18 questions)

### **Examples:**
```bash
# Uses default benchmark.csv (18 questions)
python benchmark/ragas_benchmark.py --hybrid-cypher

# Uses your custom JSONL file (10 questions from nano pipeline)
python benchmark/ragas_benchmark.py --hybrid-cypher --jsonl benchmark/ragbench__nano_benchmark.jsonl

# Uses custom CSV file
python benchmark/ragas_benchmark.py --all --csv my_custom_benchmark.csv
```

**⚠️ Important**: The approach flags (like `--hybrid-cypher`) only determine **which retriever to test**, not which benchmark file to use. File selection is controlled by `--csv` and `--jsonl` arguments.

## 📊 Evaluation Framework

### **RAGAS Metrics**
- **Context Recall**: How well retrieved context covers ground truth
- **Faithfulness**: Whether response is grounded in retrieved context  
- **Factual Correctness**: Accuracy of factual claims in response

## 🧪 RAGBench Integration

### **Dataset Processing**
```bash
# Process RAGBench dataset
python data_processors/process_data.py --ragbench --preset nano

# This creates benchmark data automatically
# -> benchmark/ragbench__nano_benchmark.jsonl
```

### **Evaluation Pipeline**
```bash
# 1. Process data
python data_processors/process_data.py --ragbench --preset micro --domain financial

# 2. Run evaluation  
python benchmark/ragas_benchmark.py --all --jsonl benchmark/ragbench__micro_benchmark.jsonl

# 3. View results
# -> benchmark_outputs/detailed_metrics_comparison.png
# -> benchmark_outputs/overall_performance_comparison.png
```

## 📁 Components

### **Core Benchmarking**
- **`ragas_benchmark.py`** - Main evaluation CLI with RAGAS integration
- **`visualizations.py`** - Chart generation and performance analysis
- **`benchmark.csv`** - Default benchmark dataset (legacy)

### **RAGBench Integration** (`ragbench/`)
- **`simple_ingester.py`** - RAGBench dataset processor
- **`evaluator.py`** - Q&A data preparation for evaluation
- **`results_formatter.py`** - Human-readable evaluation reports
- **`configs.py`** - Preset configurations (nano, micro, small, etc.)

See [ragbench/README.md](ragbench/README.md) for RAGBench details.

## 📈 Output Formats

### **Automated Visualizations**
- **Performance comparison charts** - Side-by-side metric comparisons
- **Detailed heatmaps** - Per-question performance breakdown
- **Statistical summaries** - Mean, std dev, confidence intervals

### **Data Formats**
- **CSV exports** - For spreadsheet analysis
- **JSON reports** - For programmatic processing  
- **HTML reports** - Human-readable detailed results
- **JSONL benchmarks** - Flexible Q&A format with rich metadata

## ⚙️ Configuration

### **Model Configuration**
The benchmark system now uses centralized model configuration via environment variables:

```bash
# Copy the example configuration
cp env.example .env

# Edit .env to configure your models
LLM_PROVIDER=ollama          # or 'openai'
LLM_MODEL=llama3.1:8b          # or 'gpt-4o-mini'
EMBEDDING_PROVIDER=ollama    # or 'openai'
EMBEDDING_MODEL=nomic-embed-text  # or 'text-embedding-3-small'
```

**Supported Ollama Models:**
- LLM: `llama3.1:8b`, `qwen3:8b`, `gemma3:1b`, `gemma3:12b`
- Embeddings: `nomic-embed-text`

**Supported OpenAI Models:**
- LLM: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Embeddings: `text-embedding-3-small`, `text-embedding-3-large`

### **Benchmark Data Sources**
1. **Default CSV** - Simple question/ground_truth format
2. **RAGBench JSONL** - Rich metadata with domain, dataset, record IDs
3. **Custom JSONL** - Your own benchmark questions

### **Evaluation Settings**
- **Selective testing** - Choose specific retrievers
- **Custom datasets** - Use domain-specific benchmarks
- **Output directories** - Organize results by experiment

## 🎯 Usage Patterns

### **Development Testing**
```bash
# Quick test with nano dataset
python data_processors/process_data.py --ragbench --preset nano
python benchmark/ragas_benchmark.py --chroma --graphrag
```

### **Research Evaluation**
```bash
# Comprehensive evaluation
python data_processors/process_data.py --ragbench --preset small --domain financial
python benchmark/ragas_benchmark.py --all --output-dir results/financial/
```

### **Custom Benchmarks**
```bash
# Use your own Q&A data
python benchmark/ragas_benchmark.py --all --jsonl my_benchmark.jsonl
```

