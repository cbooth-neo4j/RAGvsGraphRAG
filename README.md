# RAG vs GraphRAG: Comprehensive Evaluation Framework

## 🎯 Project Overview

A comprehensive evaluation framework comparing various RAG approaches using the RAGAS evaluation framework with research-based enhancements.

### **🔍 Retrieval Approaches**
1. **ChromaDB RAG** - Traditional vector similarity search (`--chroma`)
2. **GraphRAG** - Multi-hop graph traversal with entity resolution (`--graphrag`)
3. **Advanced GraphRAG** - Community detection and element summarization (`--advanced-graphrag`)
4. **Text2Cypher** - Natural language to Cypher query translation (`--text2cypher`)
5. **Neo4j Vector** - Graph database vector search (`--neo4j-vector`)
6. **Hybrid Cypher** - Combined vector + graph traversal (`--hybrid-cypher`)
7. **DRIFT GraphRAG** - Dynamic reasoning with iterative fact-finding (`--drift-graphrag`)
8. **Agentic Text2Cypher** - Deep Agent-powered adaptive graph exploration (`--agentic-text2cypher`)

### **🧠 Ontology & Entity Discovery**
- **Research-based corpus sampling** with TF-IDF clustering and stratified selection
- **Domain-aware entity extraction** (financial, medical, legal, technical, academic)
- **Multi-strategy text sampling** for optimal entity type discovery
- **Quality metrics** and performance analysis

### **🧪 HotpotQA Benchmark Integration**
- **Multi-hop reasoning questions** from the HotpotQA fullwiki dataset (7,405 total questions)
- **Wikipedia corpus** - Articles downloaded and ingested automatically (~10 articles per question)
- **Research-grade evaluation** - Rigorous testing with ground truth answers
- **Scalable presets** - From micro (1 question, ~2 articles) to full (7,405 questions, ~10K articles)

All approaches are evaluated using RAGAS framework with automated visualizations and comprehensive performance metrics.

### **📚 Research Foundation**
- **GraphRAG Patterns**: [Neo4j GraphRAG Field Guide](https://neo4j.com/blog/developer/graphrag-field-guide-rag-patterns/)
- **Microsoft GraphRAG**: [Community Summary Retrievers](https://graphrag.com/reference/graphrag/global-community-summary-retriever/)
- **DRIFT Algorithm**: [Microsoft DRIFT Research](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)
- **Deep Agents**: [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) - Agentic planning with subagent spawning
- **HotpotQA**: [Multi-hop Question Answering Dataset](https://hotpotqa.github.io/)
- **Entity Discovery**: 2025 research in ontology discovery and active learning


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

### 3. Build Knowledge Graph (Required)

Use `ingest.py` to build the knowledge graph. You **must** specify:
- `--source`: Data source (`pdf` or `hotpotqa`)
- `--quantity`: Number of documents/questions to process
- `--lean` or `--full`: Build mode

```bash
# Build from PDFs (place files in ./PDFs/ folder)
python ingest.py --source pdf --quantity 10 --lean

# Build from HotpotQA Wikipedia articles
python ingest.py --source hotpotqa --quantity 100 --lean    # Minimal graph
python ingest.py --source hotpotqa --quantity 1000 --full   # With summaries + communities
```

| Mode | Build Time | Features |
|------|-----------|----------|
| `--lean` | Fast | Document->Chunk->Entity + RELATES_TO (query-time intelligence) |
| `--full` | Slower | Adds AI summaries + community detection |

> **Ingestion Manifest:** When ingesting HotpotQA, a manifest is saved to Neo4j (`:__IngestionManifest__` node) 
> tracking which questions/articles were ingested. The benchmark reads this to ensure question-article pairing.
> This allows multiple Neo4j instances (lean vs full) to each carry their own manifest.

### 4. Run Benchmark (Test Only)

The benchmark tests **only the questions whose articles were ingested**:

```bash
# Step 1: Ingest (creates manifest)
python ingest.py --source hotpotqa --quantity 100 --lean

# Step 2: Benchmark (reads manifest, tests matching questions)
python -m benchmark.hotpotqa.benchmark_pipeline smoke --hotpotqa --agentic-text2cypher
```

**Benchmark commands:**
```bash
# Quick smoke test
python -m benchmark.hotpotqa.benchmark_pipeline smoke --hotpotqa --agentic-text2cypher

# With RAGAS metrics (LLM-based, slower)
python -m benchmark.hotpotqa.benchmark_pipeline smoke --ragas --agentic-text2cypher

# Compare multiple retrievers
python -m benchmark.hotpotqa.benchmark_pipeline smoke --hotpotqa --chroma --graphrag --neo4j-vector
```

> **Note:** The benchmark **only tests** against existing graph data.
> Questions without matching ingested articles are automatically filtered out.

**Metrics (required - choose one):**
| Flag | Description | Speed |
|------|-------------|-------|
| `--hotpotqa` | Exact Match + F1 Score | Fast, deterministic |
| `--ragas` | LLM-based semantic evaluation | Slower |
| `--all-metrics` | Both HotpotQA and RAGAS | Slowest |

### 5. View Results
- **Neo4j Browser**: http://localhost:7474 (explore the knowledge graph)
- **Charts**: `benchmark_outputs/` folder (performance comparisons)
- **Detailed Reports**: CSV and JSON outputs with individual Q&A analysis

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

### **🧪 HotpotQA Benchmark** 
- **7,405 multi-hop questions** requiring reasoning over multiple Wikipedia articles
- **Automatic Wikipedia download** with intelligent caching (~10 articles per question)
- **Scalable presets** - from 1 question (micro) to full 7,405 questions
- **Research-grade evaluation** matching academic benchmarks

### **🔍 8 Retrieval Approaches**
- **ChromaDB RAG** - Fast vector similarity search
- **GraphRAG** - Multi-hop graph traversal with entity resolution
- **Advanced GraphRAG** - Community detection and element summarization  
- **Text2Cypher** - Natural language to database queries with iterative refinement
- **Neo4j Vector** - Graph database vector search
- **Hybrid Cypher** - Combined vector + graph approach
- **DRIFT GraphRAG** - Dynamic reasoning with iterative refinement
- **Agentic Text2Cypher** - Deep Agent-powered adaptive exploration with thinking models

### **📊 Comprehensive Evaluation**
- **RAGAS metrics** - Response Relevancy, Factual Correctness, Semantic Similarity
- **Automated visualizations** - Performance charts and heatmaps
- **Detailed reports** - CSV and JSON outputs
- **Human-readable analysis** - Individual Q&A breakdowns

## 📚 Component Documentation

- **[Data Processors](data_processors/README.md)** - Data processing and ingestion guide
- **[Build Graph](data_processors/build_graph/README.md)** - Technical deep-dive on enhanced graph processing
- **[Retrievers](retrievers/README.md)** - Retrieval approaches and usage patterns
- **[Benchmark](benchmark/README.md)** - Evaluation framework and RAGAS integration
- **[HotpotQA](benchmark/hotpotqa/README.md)** - HotpotQA benchmark documentation
- **[Embedding Dimensions](docs/EMBEDDING_DIMENSIONS.md)** - ⚠️ **IMPORTANT**: Guide for handling different embedding models and dimensions

## 🛠️ Requirements

- **Python 3.8+**
- **Neo4j Database** (Docker recommended)
- **OpenAI API Key** (for embeddings and LLM processing)
- **8GB+ RAM** (for larger datasets)
- **Optional**: scikit-learn (for enhanced entity discovery)
- **Optional**: deepagents (for `agentic_text2cypher` retriever)

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

### 7. Agentic Text2Cypher RAG
Deep Agent-powered adaptive graph exploration using an agentic loop instead of fixed pipelines. The agent autonomously explores the graph schema, generates and executes Cypher queries, inspects results, and refines its approach until it finds the answer. Supports "thinking models" (e.g., GPT-5.2, o3) for complex reasoning.

**Key features:**
- **Agentic loop**: Autonomous exploration with planning and reflection
- **Schema-aware**: Agent reads and understands the graph schema before querying
- **Multi-strategy**: Can try different query approaches if initial attempts fail
- **Thinking model support**: Optimized for reasoning-focused models
- **Direct Neo4j tools**: Uses native driver calls (mirroring MCP functionality)

**Configuration** (in `.env`):
```bash
AGENTIC_TEXT2CYPHER_MODEL=gpt-5.2    # or o3, gpt-4.1
AGENTIC_TEXT2CYPHER_PROVIDER=openai
AGENTIC_TEXT2CYPHER_MAX_ITERATIONS=15
```

## 📈 RAGAS Evaluation Framework

### Metrics Overview

The benchmark evaluates all approaches using three key RAGAS metrics:

#### 1. Response Relevancy (0.0-1.0)
How well the generated answer addresses and is relevant to the question asked.

#### 2. Factual Correctness (0.0-1.0)
How factually accurate the response is compared to ground truth reference answers.

#### 3. Semantic Similarity (0.0-1.0)  
How well the meaning of the response matches the expected answer.

### Overall Performance Calculation

The **Average Score** for each approach is calculated as:
```
Average Score = (Response Relevancy + Factual Correctness + Semantic Similarity) / 3
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
- [ ] Start Neo4j database
- [ ] Run smoke test: `python -m benchmark.hotpotqa.benchmark_pipeline smoke`
- [ ] View results in `benchmark_outputs/hotpotqa/`

## Cheatsheet

### Benchmark Commands

```bash
# Quick test
python -m benchmark micro --hotpotqa --agentic-text2cypher

# Development (10 questions)
python -m benchmark mini --hotpotqa --chroma --graphrag

# Standard test (50 questions)
python -m benchmark smoke --hotpotqa --agentic-text2cypher

# With RAGAS metrics
python -m benchmark mini --ragas --agentic-text2cypher

# Both metrics
python -m benchmark mini --all-metrics --agentic-text2cypher

# Build database first
python -m benchmark smoke --hotpotqa --graphrag --build-database
```

### Presets

| Preset | Questions | Articles (approx) | Use Case |
|--------|-----------|-------------------|----------|
| `micro` | 1 | ~10 | Sanity check |
| `mini` | 10 | ~100 | Development |
| `smoke` | 50 | ~500 | Standard test |
| `dev` | 500 | ~4,000 | Thorough test |
| `full` | 7,405 | ~10,000 | Complete benchmark |

> **Note:** Each HotpotQA question references 2 "gold" Wikipedia articles plus ~8 distractor articles. Article counts decrease per question at larger scales due to deduplication (shared articles across questions).

### Metrics

| Flag | Metrics | Speed |
|------|---------|-------|
| `--hotpotqa` | Exact Match + F1 | Fast |
| `--ragas` | Response Relevancy, Factual Correctness, Semantic Similarity | Slow |
| `--all-metrics` | Both | Slowest |

### Retrievers

| Flag | Approach |
|------|----------|
| `--chroma` | ChromaDB vector search |
| `--graphrag` | Multi-hop graph traversal |
| `--agentic-text2cypher` | Deep Agent-powered exploration |
| `--text2cypher` | Natural language to Cypher |
| `--neo4j-vector` | Neo4j vector similarity |
| `--hybrid-cypher` | Combined vector + graph |
| `--advanced-graphrag` | Community-enhanced |
| `--drift-graphrag` | Iterative refinement |

### Options

| Flag | Effect |
|------|--------|
| `--build-database` | Clear Neo4j + ingest Wikipedia |
| `--skip-advanced` | Skip community detection |
| `--dataset pdfs` | Use PDF documents instead |

### Help

```bash
python -m benchmark --help
```
