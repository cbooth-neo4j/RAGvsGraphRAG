# RAG vs GraphRAG: Comprehensive Evaluation Framework

## 🎯 Project Overview

A comprehensive evaluation framework comparing various RAG approaches using the RAGAS evaluation framework

1. **ChromaDB RAG**: Traditional vector similarity search using embeddings
2. **GraphRAG**: Basic Neo4j graph-enhanced vector search with entity resolution
3. **Advanced GraphRAG**: Intelligent routing between global (community-based) and local (entity-enhanced) search modes
4. **DRIFT GraphRAG**: Iterative refinement approach with action graphs and dynamic follow-ups
5. **Text2Cypher RAG**: Natural language to Cypher query translation
6. **Neo4j Vector RAG**: Pure vector similarity search using Neo4j vector index

All approaches are evaluated using the RAGAS framework with automated professional visualizations and comprehensive performance metrics.

### Research Foundation
- **GraphRAG Field Guide**: [Neo4j GraphRAG Patterns](https://neo4j.com/blog/developer/graphrag-field-guide-rag-patterns/)
- **Microsoft GraphRAG**: [Global Community Summary Retrievers](https://graphrag.com/reference/graphrag/global-community-summary-retriever/)
- **DRIFT Algorithm**: [Microsoft DRIFT Research](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)

*Note: This repo provides a comprehensive starting framework. Performance outputs require customisation to your specific graph schema and input data for optimal results.*

## 📁 Project Structure

```
RAGvsGraphRAG/
├── 📂 data_processors/              # Document processing and graph construction
│   ├── __init__.py                 # Module exports and factory functions
│   ├── chroma_processor.py         # PDF text extraction and chunking for ChromaDB
│   ├── graph_processor.py          # Basic graph processing + entity resolution
│   └── advanced_graph_processor.py # Advanced features (inherits from basic)
├── 📂 retrievers/                   # RAG retrieval implementations
│   ├── __init__.py                 # Universal retriever interface
│   ├── chroma_retriever.py         # ChromaDB vector similarity search
│   ├── graph_rag_retriever.py      # Basic GraphRAG with entity traversal
│   ├── advanced_graphrag_retriever.py # Global/local routing with communities
│   ├── drift_graphrag_retriever.py # Microsoft DRIFT iterative refinement
│   ├── text2cypher_retriever.py    # Natural language to Cypher translation
│   └── neo4j_vector_retriever.py   # Pure Neo4j vector search
├── 📂 benchmark/                    # Benchmark scripts and evaluation
│   ├── ragas_benchmark.py          # Main RAGAS evaluation script (6-way comparison)
│   ├── visualizations.py           # Automated chart generation
│   ├── benchmark.csv               # Test questions and ground truth
│   └── BENCHMARK_README.md         # Detailed benchmark documentation
├── 📂 benchmark_outputs/           # Generated results and visualizations
├── 📂 tests/                       # Test and validation scripts
│   ├── test_ragas_setup.py         # Quick 6-approach validation
│   ├── check_chromaDB.py           # ChromaDB health check
│   └── check_schema.py             # Neo4j schema inspection
├── 📂 PDFs/                        # Source documents for processing
├── 📂 chroma_db/                   # ChromaDB vector store data
└── 📄 requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables in .env file:
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username  
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_key
```

### 2. Process Documents

#### Option A: ChromaDB Only (for traditional RAG)
```bash
# Process PDFs and create ChromaDB vector store
python data_processors/chroma_processor.py

# This creates:
# - ChromaDB vector store in chroma_db/ directory
# - Chunked text with metadata for similarity search
```

#### Option B: Basic Graph Processing (GraphRAG)
```bash
# Process PDFs and create Neo4j graph with entity resolution
python data_processors/graph_processor.py

# This creates:
# - Neo4j graph with entities and relationships  
# - ChromaDB vector store for similarity search
# - Entity resolution to merge duplicates using LLM evaluation
```

#### Option C: Advanced Graph Processing (Full Pipeline to enable advanced GraphRAG methods)
```bash
# Full advanced processing with all features enabled
python data_processors/advanced_graph_processor.py

# This takes what was done from graph_processor and:
# - Enhances entity descriptions using batch LLM processing
# - Adds hierarchical community structure with summaries
# - Adds community detection using Leiden algorithm
```

> ⚠️ **Cost Warning**: Advanced processing includes LLM calls for entity enhancement and community summarization, which increases OpenAI API costs significantly.

### 3. Validate Setup
```bash
# Quick health checks
python tests/check_chromaDB.py
python tests/check_schema.py

# Test all six approaches with sample questions
python tests/test_ragas_setup.py
```

### 4. Run Comprehensive Benchmark

#### Six-Way Comparison (All Approaches)
```bash
# Complete evaluation with all six approaches
python benchmark/ragas_benchmark.py --all
```

#### Selective Testing
```bash
# RAG vs GraphRAG
python benchmark/ragas_benchmark.py --chroma --graphrag

# GraphRAG Comparison
python benchmark/ragas_benchmark.py --graphrag --advanced-graphrag --drift-graphrag

# Single Approach Testing
python benchmark/ragas_benchmark.py --chroma               # ChromaDB only
python benchmark/ragas_benchmark.py --graphrag             # GraphRAG only
python benchmark/ragas_benchmark.py --advanced-graphrag    # Advanced GraphRAG only
python benchmark/ragas_benchmark.py --drift-graphrag       # DRIFT GraphRAG only
python benchmark/ragas_benchmark.py --text2cypher          # Text2Cypher only
python benchmark/ragas_benchmark.py --neo4j-vector        # Neo4j Vector only

# Custom output directory
python benchmark/ragas_benchmark.py --all --output-dir my_results
```

## 🔍 RAG Approaches Explained

### 1. ChromaDB RAG
Traditional vector similarity search using OpenAI embeddings for fast document chunk retrieval.

### 2. GraphRAG  
Neo4j graph-enhanced vector search with entity relationships and LLM-based entity resolution to merge duplicates.

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