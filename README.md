# RAG vs GraphRAG Comparison

A comprehensive evaluation framework comparing traditional RAG (Retrieval-Augmented Generation) with GraphRAG approaches using the RAGAS evaluation framework.

## 🎯 Project Overview

This project implements and benchmarks two different RAG approaches:

1. **Traditional RAG**: Uses ChromaDB vector store for document retrieval
2. **GraphRAG**: Uses Neo4j knowledge graph with entity relationships and enhanced context

Both approaches are evaluated using the RAGAS (RAG Assessment) framework to measure performance across multiple dimensions including faithfulness, factual correctness, and context recall.

## 📁 Project Structure

```
RAGvsGraphRAG/
├── 📂 benchmark/           # Benchmark scripts and data
│   ├── ragas_benchmark.py  # Main RAGAS evaluation script
│   ├── benchmark.csv       # Test questions and ground truth
│   └── BENCHMARK_README.md # Detailed benchmark documentation
├── 📂 tests/               # Test and validation scripts
│   ├── test_ragas_setup.py # Quick 3-question validation
│   ├── check_chromaDB.py   # ChromaDB health check
│   └── check_schema.py     # Neo4j schema inspection
├── 📂 PDFs/                # Source documents for processing
├── 📂 chroma_db/           # ChromaDB vector store data
├── 📄 RAGvsGraphRAG.py     # Main RAG implementations
├── 📄 pdf_processor.py     # PDF document processing
├── 📄 graph_processor.py   # Neo4j graph construction
└── 📄 requirements.txt     # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
# NEO4J_URI=your_neo4j_uri
# NEO4J_USERNAME=your_username  
# NEO4J_PASSWORD=your_password
# OPENAI_API_KEY=your_openai_key
```

### 2. Process Documents
```bash
# Process PDFs and populate both ChromaDB and Neo4j
python pdf_processor.py
python graph_processor.py
```

### 3. Validate Setup
```bash
# Quick health checks
python tests/check_chromaDB.py
python tests/check_schema.py

# Test with 3 sample questions
python tests/test_ragas_setup.py
```

### 4. Run Full Benchmark
```bash
# Complete evaluation with 20 questions
python benchmark/ragas_benchmark.py
```

## 📊 Current Results (k=3 retrieval)

| Metric | ChromaDB RAG | GraphRAG | Improvement |
|--------|--------------|----------|-------------|
| Context Recall | 0.6000 | 0.6000 | +0.00% |
| Faithfulness | 0.6146 | 0.7644 | **+24.37%** |
| Factual Correctness | 0.3720 | 0.4000 | **+7.53%** |

## 🔍 Key Components

### RAG Implementations (`RAGvsGraphRAG.py`)

#### ChromaDB RAG
- Vector similarity search using OpenAI embeddings
- Simple document chunk retrieval
- Standard retrieval-augmented generation

#### GraphRAG
- Neo4j knowledge graph with entities and relationships
- Enhanced context through entity traversal
- Safe graph traversal preventing chunk cross-contamination
- Structured output with separated content types

### Graph Processing (`graph_processor.py`)
- Extracts entities from documents using LLMs
- Creates knowledge graph in Neo4j
- Establishes entity relationships based on co-occurrence
- Supports entity types: Organization, Person, Location, Date, Requirement, Financial

### Benchmark Framework (`benchmark/`)
- RAGAS-based evaluation using GPT-4o-mini as evaluator
- Measures Context Recall, Faithfulness, and Factual Correctness
- Comprehensive comparison tables and visualizations
- Detailed metric explanations and scoring methodology

## 🔧 Key Optimizations

### GraphRAG Improvements
1. **Safe Entity Traversal**: Prevents retrieval of other document chunks
2. **Entity Prioritization**: Limits to top 20 entities by frequency
3. **Embedding-Based Traversal**: Uses cosine similarity for relevance
4. **Structured Context**: Separates factual content from related context

### Performance Tuning
- Optimal k=3 retrieval for balanced performance
- Similarity thresholds: min=0.4, max=0.9
- Path limits: 10 (1-hop), 20 (2-hop)
- Entity limit: 20 per chunk

## 📈 Evaluation Metrics

### Context Recall (0.0-1.0)
**What it measures**: How well the retrieval system finds relevant information  
**Process**: LLM evaluates if retrieved contexts contain information needed to answer the question

### Faithfulness (0.0-1.0)  
**What it measures**: How faithful the generated answer is to retrieved context  
**Process**: Compares generated answer against retrieved documents for factual consistency

### Factual Correctness (0.0-1.0)
**What it measures**: How factually accurate the response is compared to ground truth  
**Process**: Compares generated answer against reference answer for accuracy

## 🛠️ Development

### Adding New Documents
1. Place PDFs in `PDFs/` folder
2. Run `python pdf_processor.py`
3. Run `python graph_processor.py`

### Testing Changes
```bash
# Quick validation (3 questions)
python tests/test_ragas_setup.py

# Full benchmark (20 questions)  
python benchmark/ragas_benchmark.py
```

### Modifying Evaluation
- Edit `benchmark/benchmark.csv` to add/modify test questions
- Adjust metrics in `benchmark/ragas_benchmark.py`
- Update entity types in `graph_processor.py`

## 📚 Documentation

- **Benchmark Details**: See `benchmark/BENCHMARK_README.md`
- **Schema Inspection**: Run `python tests/check_schema.py`