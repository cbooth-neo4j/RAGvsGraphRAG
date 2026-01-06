# Benchmark System

Comprehensive evaluation framework for comparing RAG approaches using RAGAS metrics.

## 🚀 Quick Start

```bash
# Run full benchmark comparison (default CSV)
python benchmark/ragas_benchmark.py --all

# Run specific retrievers
python benchmark/ragas_benchmark.py --chroma --graphrag

# Use HotpotQA benchmark (recommended for research)
python -m benchmark.hotpotqa.benchmark_pipeline smoke
```

## 📊 Evaluation Framework

### **RAGAS Metrics** (Universal - Fair for All Retriever Types)
- **Response Relevancy**: How well the answer addresses the question asked
- **Factual Correctness**: Accuracy of facts compared to ground truth  
- **Semantic Similarity**: How well the meaning matches the expected answer

### **Retrieval Approaches Tested**
1. **ChromaDB RAG** - Traditional vector similarity search
2. **GraphRAG** - Multi-hop graph traversal with context enhancement
3. **Advanced GraphRAG** - With community detection and summarization
4. **Text2Cypher** - Natural language to Cypher query translation
5. **Neo4j Vector** - Graph database vector search
6. **Hybrid Cypher** - Combined vector + graph traversal
7. **DRIFT GraphRAG** - Dynamic reasoning with iterative refinement

## 🧪 HotpotQA Benchmark (Recommended)

The HotpotQA fullwiki benchmark provides a rigorous, research-grade evaluation using ~7,400 multi-hop reasoning questions with Wikipedia articles.

### **Quick Start**
```bash
# Smoke test (50 questions, ~$5, ~15 min)
python -m benchmark.hotpotqa.benchmark_pipeline smoke

# Development benchmark (500 questions, ~$20, ~1 hour)
python -m benchmark.hotpotqa.benchmark_pipeline dev

# Full evaluation (all ~7400 questions)
python -m benchmark.hotpotqa.benchmark_pipeline full
```

### **Available Presets**

| Preset | Questions | Retrievers | Est. Cost | Est. Time |
|--------|-----------|------------|-----------|-----------|
| `mini` | 10 | ChromaDB only | $1 | 5 min |
| `smoke` | 50 | ChromaDB, GraphRAG | $5 | 15 min |
| `dev` | 500 | All main retrievers | $20 | 60 min |
| `full` | ~7400 | All retrievers | $100+ | 5+ hours |

### **CLI Options**
```bash
# Test specific retrievers
python -m benchmark.hotpotqa.benchmark_pipeline dev --retrievers chroma graphrag hybrid_cypher

# Skip ingestion if graph already populated
python -m benchmark.hotpotqa.benchmark_pipeline dev --skip-ingestion

# List available presets
python -m benchmark.hotpotqa.benchmark_pipeline --list-presets
```

See [hotpotqa/README.md](hotpotqa/README.md) for detailed HotpotQA documentation.

## 📁 Components

### **Core Benchmarking**
- **`ragas_benchmark.py`** - Main evaluation CLI with RAGAS integration
- **`visualizations.py`** - Chart generation and performance analysis
- **`benchmark.csv`** - Default benchmark dataset (simple Q&A)

### **HotpotQA Integration** (`hotpotqa/`)
- **`data_loader.py`** - Downloads HotpotQA questions + Wikipedia articles
- **`wiki_ingester.py`** - Ingests Wikipedia corpus into Neo4j graph
- **`benchmark_pipeline.py`** - Main orchestrator with 5-phase pipeline
- **`configs.py`** - Preset configurations (smoke, dev, full)

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
The benchmark system uses centralized model configuration via environment variables:

```bash
# Copy the example configuration
cp env.example .env

# Edit .env to configure your models
LLM_PROVIDER=ollama          # or 'openai'
LLM_MODEL=llama3.1:8b        # or 'gpt-4o-mini'
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
1. **HotpotQA** - Research-grade multi-hop questions with Wikipedia (recommended)
2. **Default CSV** - Simple question/ground_truth format for quick tests
3. **Custom JSONL** - Your own benchmark questions

### **Evaluation Settings**
- **Selective testing** - Choose specific retrievers
- **Custom datasets** - Use domain-specific benchmarks
- **Output directories** - Organize results by experiment

## 🎯 Usage Patterns

### **Development Testing**
```bash
# Quick test with default CSV
python benchmark/ragas_benchmark.py --chroma --graphrag --limit 5
```

### **Research Evaluation**
```bash
# HotpotQA benchmark for publication-quality results
python -m benchmark.hotpotqa.benchmark_pipeline dev --retrievers chroma graphrag hybrid_cypher advanced_graphrag
```

### **Custom Benchmarks**
```bash
# Use your own Q&A data
python benchmark/ragas_benchmark.py --all --jsonl my_benchmark.jsonl
```

## 📊 Interpreting Results

### **Key Metrics to Watch**
- **Response Relevancy > 0.8** - Answers address the question well
- **Semantic Similarity > 0.8** - Answers match expected meaning
- **Factual Correctness > 0.7** - Accurate information

### **Approach Comparisons**
- **ChromaDB** - Fast, good for semantic similarity
- **GraphRAG** - Better for complex, multi-hop questions
- **Advanced GraphRAG** - Best for global/summarization queries
- **Hybrid Cypher** - Balanced performance across question types
- **DRIFT** - Excellent for iterative refinement needs

## 🔧 Troubleshooting

### **Common Issues**
- **No retrievers available**: Install retriever dependencies
- **RAGAS import error**: Install ragas package
- **Empty results**: Check if data was processed correctly
- **API limits**: Reduce batch size or use rate limiting
- **wikipedia-api not found**: Run `pip install wikipedia-api`

### **Performance Tips**
- Start with **smoke** preset for quick validation
- Use **selective testing** during development
- Run **full evaluation** only for final results
- Monitor **API costs** with larger datasets
- Use **--skip-ingestion** if graph is already populated

## 🚀 Advanced Features

### **Domain-Specific Evaluation**
- Multi-hop reasoning questions (HotpotQA)
- Financial document analysis
- Medical literature processing  
- Legal document understanding

### **Custom Metrics**
- Extend RAGAS with domain-specific metrics
- Add retrieval latency measurements
- Include cost-per-query analysis

### **Batch Processing**
- Process multiple presets automatically
- Compare across different domains
- Generate comparative reports
