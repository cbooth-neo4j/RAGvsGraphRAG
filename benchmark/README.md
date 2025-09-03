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

## 📊 Evaluation Framework

### **RAGAS Metrics**
- **Context Recall**: How well retrieved context covers ground truth
- **Faithfulness**: Whether response is grounded in retrieved context  
- **Factual Correctness**: Accuracy of factual claims in response

### **Retrieval Approaches Tested**
1. **ChromaDB RAG** - Traditional vector similarity search
2. **GraphRAG** - Multi-hop graph traversal with context enhancement
3. **Advanced GraphRAG** - With community detection and summarization
4. **Text2Cypher** - Natural language to Cypher query translation
5. **Neo4j Vector** - Graph database vector search
6. **Hybrid Cypher** - Combined vector + graph traversal

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
LLM_MODEL=qwen3:8b          # or 'gpt-4o-mini'
EMBEDDING_PROVIDER=ollama    # or 'openai'
EMBEDDING_MODEL=nomic-embed-text  # or 'text-embedding-3-small'
```

**Supported Ollama Models:**
- LLM: `qwen3:8b`, `gemma3:7b`, `llama3:8b`, `mistral:7b`
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

## 📊 Interpreting Results

### **Key Metrics to Watch**
- **Context Recall > 0.8** - Good retrieval coverage
- **Faithfulness > 0.9** - Responses stay grounded
- **Factual Correctness > 0.7** - Accurate information

### **Approach Comparisons**
- **ChromaDB** - Fast, good for semantic similarity
- **GraphRAG** - Better for complex, multi-hop questions
- **Text2Cypher** - Excellent for structured data queries
- **Hybrid** - Balanced performance across question types

## 🔧 Troubleshooting

### **Common Issues**
- **No retrievers available**: Install retriever dependencies
- **RAGAS import error**: Install ragas package
- **Empty results**: Check if data was processed correctly
- **API limits**: Reduce batch size or use rate limiting

### **Performance Tips**
- Start with **nano** preset for quick validation
- Use **selective testing** during development
- Run **full evaluation** only for final results
- Monitor **API costs** with larger datasets

## 🚀 Advanced Features

### **Domain-Specific Evaluation**
- Financial document analysis
- Medical literature processing  
- Legal document understanding
- Technical specification parsing

### **Custom Metrics**
- Extend RAGAS with domain-specific metrics
- Add retrieval latency measurements
- Include cost-per-query analysis

### **Batch Processing**
- Process multiple presets automatically
- Compare across different domains
- Generate comparative reports
