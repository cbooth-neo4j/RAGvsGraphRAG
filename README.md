# RAG vs GraphRAG: Comprehensive Evaluation Framework

## 🎯 Project Overview

A comprehensive evaluation framework comparing various RAG approaches using the RAGAS evaluation framework with research-based enhancements.

### **🔍 Retrieval Approaches**
1. **ChromaDB RAG** - Traditional vector similarity search
2. **GraphRAG** - Multi-hop graph traversal with entity resolution  
3. **Advanced GraphRAG** - Community detection and element summarization
4. **Text2Cypher** - Natural language to Cypher query translation
5. **Neo4j Vector** - Graph database vector search
6. **Hybrid Cypher** - Combined vector + graph traversal
7. **DRIFT GraphRAG** - Dynamic reasoning with iterative fact-finding

### **🧠 Ontology & Entity Discovery**
- **Research-based corpus sampling** with TF-IDF clustering and stratified selection
- **Domain-aware entity extraction** (financial, medical, legal, technical, academic)
- **Multi-strategy text sampling** for optimal entity type discovery
- **Quality metrics** and performance analysis

### **🧪 RAGBench Integration**
- **Multiple dataset presets** from nano (10 docs) to full (60K docs)
- **Domain-specific benchmarks** with rich metadata
- **JSONL format** for flexible evaluation data
- **Automated Q&A pair generation** for evaluation

All approaches are evaluated using RAGAS framework with automated visualizations and comprehensive performance metrics.

### **📚 Research Foundation**
- **GraphRAG Patterns**: [Neo4j GraphRAG Field Guide](https://neo4j.com/blog/developer/graphrag-field-guide-rag-patterns/)
- **Microsoft GraphRAG**: [Community Summary Retrievers](https://graphrag.com/reference/graphrag/global-community-summary-retriever/)
- **DRIFT Algorithm**: [Microsoft DRIFT Research](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)
- **Entity Discovery**: 2025 research in ontology discovery and active learning

## 📁 Project Structure

```
RAGvsGraphRAG/
├── 📂 data_processors/              # Document processing and graph construction
│   ├── process_data.py             # 🎯 Main CLI for data processing
│   ├── build_graph/                # Graph processor
│   │   ├── main_processor.py       # Main orchestrator class
│   │   ├── entity_discovery.py     # Research-based entity discovery
│   │   ├── text_processing.py      # PDF extraction, chunking, embeddings
│   │   ├── graph_operations.py     # Neo4j operations & entity resolution
│   │   └── README.md               # Technical deep-dive documentation
│   ├── chroma_processor.py         # ChromaDB vector processing
│   ├── graph_processor.py          # Legacy processor (use build_graph instead)
│   └── advanced_graph_processor.py # Community detection and summarization
├── 📂 retrievers/                   # RAG retrieval implementations
│   ├── chroma_retriever.py         # ChromaDB vector similarity search
│   ├── graph_rag_retriever.py      # Multi-hop graph traversal
│   ├── advanced_graphrag_retriever.py # Community-enhanced GraphRAG
│   ├── text2cypher_retriever.py    # Natural language to Cypher
│   ├── neo4j_vector_retriever.py   # Neo4j vector search
│   ├── hybrid_cypher_retriever.py  # Combined vector + graph
│   ├── drift_graphrag_retriever.py # Dynamic reasoning approach
│   └── README.md                   # Retriever usage guide
├── 📂 benchmark/                    # Evaluation framework
│   ├── ragas_benchmark.py          # 🎯 Main evaluation CLI
│   ├── visualizations.py           # Automated chart generation
│   ├── benchmark.csv               # Default benchmark dataset
│   ├── ragbench/                   # RAGBench dataset integration
│   │   ├── simple_ingester.py      # Dataset processor
│   │   ├── evaluator.py            # Q&A data preparation
│   │   ├── results_formatter.py    # Human-readable reports
│   │   ├── configs.py              # Preset configurations
│   │   └── README.md               # RAGBench documentation
│   └── README.md                   # Benchmarking guide
├── 📂 benchmark_outputs/           # Generated results and visualizations
├── 📂 tests/                       # Test and validation scripts
├── 📂 PDFs/                        # Source documents for processing
├── 📂 chroma_db/                   # ChromaDB vector store data
└── 📄 requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure .env file from template
cp .env_example .env

# Configure your settings in .env:
# - Provider selection (openai, ollama, vertexai)
# - Embedding model selection
# - Neo4j connection details
# - API keys as needed

# See .env_example for all available options
```

**⚠️ IMPORTANT**: Different embedding models produce different vector dimensions.
- VertexAI/Ollama: 768 dimensions
- OpenAI (small/ada-002): 1536 dimensions  
- OpenAI (large): 3072 dimensions

**You must use the same embedding model for data ingestion AND querying!**

See [Embedding Dimensions Guide](docs/EMBEDDING_DIMENSIONS.md) for detailed information.

### 2. Start Neo4j Database
```bash
# Using Docker (recommended)
docker run --name neo4j-rag \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

### 3. Process Data (Choose One)

#### **Option A: Process Your PDFs**
```bash
# Place PDFs in PDFs/ folder, then:
python data_processors/process_data.py --pdfs
```

#### **Option B: Use RAGBench Dataset**
```bash
# Quick test with nano preset (10 documents)
python data_processors/process_data.py --ragbench --preset nano

# Or larger dataset with domain hint
python data_processors/process_data.py --ragbench --preset micro --domain financial

# See all available presets
python data_processors/process_data.py --list-presets

# This creates:
# - Neo4j graph with dynamically discovered entities and relationships
# - ChromaDB vector store for similarity search  
# - Entity resolution to merge duplicates using LLM evaluation
# - Corpus-wide entity discovery with CLI approval and caching
```

### 4. Run Evaluation
```bash
# Compare all RAG approaches (uses default benchmark.csv with 18 questions)
python benchmark/ragas_benchmark.py --all

# Use RAGBench evaluation data (automatically created during processing)
python benchmark/ragas_benchmark.py --all --jsonl benchmark/ragbench__nano_benchmark.jsonl

# Selective testing
python benchmark/ragas_benchmark.py --chroma --graphrag --text2cypher
```

#### **📁 Benchmark File Selection Priority:**
1. **`--jsonl file.jsonl`** → Uses specified JSONL file (highest priority)
2. **`--csv file.csv`** → Uses specified CSV file  
3. **No file specified** → Uses default `benchmark/benchmark.csv` (18 questions)

```bash
# Examples:
python benchmark/ragas_benchmark.py --hybrid-cypher                    # Uses default CSV
python benchmark/ragas_benchmark.py --hybrid-cypher --jsonl my.jsonl  # Uses custom JSONL
```

**⚠️ Note**: Approach flags (like `--hybrid-cypher`) determine **which retriever to test**, not which file to use.

### 5. View Results
- **Neo4j Browser**: http://localhost:7474 (explore the knowledge graph)
- **Charts**: `benchmark_outputs/` folder (performance comparisons)
- **Detailed Reports**: HTML reports with individual Q&A analysis

## ⚡ Global retriever performance benchmarking (before/after)

The repo includes a small harness to benchmark **Advanced GraphRAG global search** before/after optimizations and generate a markdown summary.

### Run benchmarks

```bash
# Optimized (global-only + single-pass, 1 LLM call)
python -m benchmark.perf_global_search --impl optimized --runs 1 --cold-start --strategy single_pass

# Baseline (legacy full-graph + map-reduce)
python -m benchmark.perf_global_search --impl baseline --runs 1 --cold-start --strategy map_reduce
```

### Generate a markdown comparison report

```bash
python -m benchmark.report_global_perf \
  --before benchmark/results/<baseline_json>.json \
  --after benchmark/results/<optimized_json>.json \
  --out benchmark/results/global_before_after_report.md
```

## 🎯 Key Features

### **🧠 Research-Based Entity Discovery**
- **Multi-strategy corpus sampling** with TF-IDF clustering and stratified selection
- **Domain-aware entity extraction** with hints for financial, medical, legal, technical domains
- **Quality metrics** including diversity scores and compression ratios
- **Interactive CLI approval** for discovered entity types

### **🧪 RAGBench Integration** 
- **Multiple dataset presets** from nano (10 docs) to full (60K docs)
- **Rich metadata** with domain, dataset, and record IDs
- **JSONL format** for flexible evaluation data
- **Automated Q&A generation** for comprehensive evaluation

### **🔍 7 Retrieval Approaches**
- **ChromaDB RAG** - Fast vector similarity search
- **GraphRAG** - Multi-hop graph traversal with entity resolution
- **Advanced GraphRAG** - Community detection and element summarization  
- **Text2Cypher** - Natural language to database queries
- **Neo4j Vector** - Graph database vector search
- **Hybrid Cypher** - Combined vector + graph approach
- **DRIFT GraphRAG** - Dynamic reasoning with iterative refinement

### **📊 Comprehensive Evaluation**
- **RAGAS metrics** - Context Recall, Faithfulness, Factual Correctness
- **Automated visualizations** - Performance charts and heatmaps
- **Detailed reports** - HTML, CSV, and JSON outputs
- **Human-readable analysis** - Individual Q&A breakdowns

## 📚 Component Documentation

- **[Data Processors](data_processors/README.md)** - Data processing and ingestion guide
- **[Build Graph](data_processors/build_graph/README.md)** - Technical deep-dive on enhanced graph processing
- **[Retrievers](retrievers/README.md)** - Retrieval approaches and usage patterns
- **[Benchmark](benchmark/README.md)** - Evaluation framework and RAGAS integration
- **[RAGBench](benchmark/ragbench/README.md)** - RAGBench dataset integration details
- **[Embedding Dimensions](docs/EMBEDDING_DIMENSIONS.md)** - ⚠️ **IMPORTANT**: Guide for handling different embedding models and dimensions

## 🛠️ Requirements

- **Python 3.8+**
- **Neo4j Database** (Docker recommended)
- **OpenAI API Key** (for embeddings and LLM processing)
- **8GB+ RAM** (for larger datasets)
- **Optional**: scikit-learn (for enhanced entity discovery)

### 2. GraphRAG  
Neo4j graph-enhanced vector search with **dynamic entity discovery**. Automatically discovers entity types from your documents with CLI approval. Includes LLM-based entity resolution to merge duplicates.

### 3. Advanced GraphRAG 
Intelligent routing between global community search and local entity search with element summarization and community detection.

### 4. DRIFT GraphRAG 
Iterative refinement algorithm with dynamic follow-ups and multi-depth exploration using NetworkX action graphs.

### 5. Text2Cypher RAG 
Natural language to Cypher query translation with direct Neo4j graph database querying and schema-aware prompt engineering.

### 6. Neo4j Vector RAG 
Pure Neo4j vector similarity search using native vector operations without graph traversal for fast retrieval - good to compare against vector only databases such as ChromaDB.

## 📈 RAGAS Evaluation Framework

### Metrics Overview

The benchmark evaluates all approaches using three key RAGAS metrics:

#### 1. Context Recall (0.0-1.0)
How well the retrieval system finds relevant information needed to answer the question.

#### 2. Faithfulness (0.0-1.0)  
How faithful the generated answer is to retrieved context without hallucination.

#### 3. Factual Correctness (0.0-1.0)
How factually accurate the response is compared to ground truth reference answers.

### Overall Performance Calculation

The **Average Score** for each approach is calculated as:
```
Average Score = (Context Recall + Faithfulness + Factual Correctness) / 3
```

## 🧠 Dynamic Entity Discovery

The graph processor now features **intelligent entity discovery** that adapts to your document content:

### How It Works
1. **Corpus Analysis**: Analyzes your entire document collection using hybrid sampling (first/last 500 chars + entity-rich patterns)
2. **LLM Proposal**: GPT-4 proposes relevant entity types based on document content
3. **CLI Approval**: You review and approve/modify the proposed entities
4. **Schema Caching**: Approved entities are cached for reuse across runs
5. **Dynamic Extraction**: Extracts only the approved entity types from each document chunk

### How to Use
Simply run the graph processor and it will automatically discover entities from your documents:
```bash
python data_processors/graph_processor.py
```

### Benefits
- **Adaptive**: Discovers entities relevant to your specific domain (contracts, medical, legal, etc.)
- **Consistent**: Single entity schema applied across all documents in a corpus
- **Efficient**: Caches approved schemas to avoid re-discovery
- **User-Controlled**: You approve all entity types before processing

### Example Discovery Output
```
🔍 Analyzing corpus with LLM...
📋 Proposed entities: Contract, Vendor, Deliverable, Timeline, Budget, Compliance
✅ Approve these entities? (y/n/edit): y
🚀 Processing documents with approved entities...
```

## 📚 Customization Guide

### Adding New Documents
1. **Add PDFs**: Place new PDF files in the `PDFs/` directory
2. **Reprocess**: Run your chosen processing command again:
   ```bash
   python data_processors/graph_processor.py  # Reprocesses all PDFs
   ```
3. **Test**: Validate with `python tests/test_ragas_setup.py`

### Custom Benchmark Questions  
1. **Edit questions**: Modify `benchmark/benchmark.csv`:
   ```csv
   question,ground_truth
   Your custom question?,Expected answer here
   ```
2. **Run benchmark**: `python benchmark/ragas_benchmark.py --all`

## 🔧 Development & Testing

### Health Checks
```bash
# Verify all systems working
python tests/check_chromaDB.py      # ChromaDB status
python tests/check_schema.py        # Neo4j schema and statistics  
python tests/test_ragas_setup.py    # All approaches validation
```

## 🚀 Next Steps

### Quick Start Checklist
- [ ] Set up environment variables in `.env`
- [ ] Run `pip install -r requirements.txt`
- [ ] Add PDFs to `PDFs/` directory
- [ ] Choose processing level: `python data_processors/graph_processor.py`
- [ ] Validate setup: `python tests/test_ragas_setup.py`
- [ ] Run benchmark: `python benchmark/ragas_benchmark.py --all`

## Cheatsheet
```
python benchmark/ragas_benchmark.py --hybrid-cypher --chroma --neo4j-vector --graphrag --advanced-graphrag  --limit 1 --jsonl benchmark/ragbench__nano_benchmark.jsonl
```
```
python benchmark/ragas_benchmark.py --hybrid-cypher --chroma --neo4j-vector --graphrag --advanced-graphrag --text2cypher --limit 1
```