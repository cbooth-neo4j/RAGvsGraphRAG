# RAG vs GraphRAG vs Text2Cypher Comparison

A comprehensive evaluation framework comparing three RAG approaches using the RAGAS evaluation framework.

## 🎯 Project Overview

This project implements and benchmarks three RAG approaches:

1. **ChromaDB RAG**: Vector similarity search using embeddings
2. **GraphRAG**: Graph-Enchanced-Vector Search 
3. **Text2Cypher**: Natural language to Cypher query translation

All approaches are evaluated using RAGAS framework with automated visualizations.

For more information on approach types, see: 
https://neo4j.com/blog/developer/graphrag-field-guide-rag-patterns/

## 📁 Project Structure

```
RAGvsGraphRAG/
├── 📂 benchmark/           # Benchmark scripts and data
│   ├── ragas_benchmark.py  # Main RAGAS evaluation script
│   ├── visualizations.py   # Automated chart generation
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

## 🔍 Key Components

### RAG Implementations (`RAGvsGraphRAG.py`)

#### ChromaDB RAG
- Vector similarity search using OpenAI embeddings
- Simple document chunk retrieval

#### GraphRAG
- [GraphRAG Field Guide](https://neo4j.com/blog/developer/graphrag-field-guide-rag-patterns/)
- Neo4j Graph-Enchanced Vector Search
- Enhanced context through entity traversal
- Safe graph traversal preventing chunk cross-contamination

#### Text2Cypher
- [Text2Cypher Blog Post](https://neo4j.com/blog/developer/effortless-rag-text2cypherretriever/)
- [Text2Cypher Implementation](https://github.com/neo4j/rag-evaluation/blob/main/scripts/run_experiment_from_config_file.py)

- Natural language to Cypher query translation
- Direct graph database querying
- Few-shot examples for query generation

### Benchmark Framework (`benchmark/`)
- Three-way RAGAS evaluation using GPT-4o-mini as evaluator
- Measures Context Recall, Faithfulness, and Factual Correctness
- Automated professional visualizations (bar charts, heatmaps, pie charts)
- Organized results in `benchmark_outputs/` folder


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

### Implementing into your own data & modifying Evaluation
- Edit `benchmark/benchmark.csv` to add/modify test questions
- Adjust metrics in `benchmark/ragas_benchmark.py`
- Update nodes, relationships and entity types in `graph_processor.py`
- Modify cypher_retriever in `RAGvsGraphRAG.py`

## 📚 Documentation

- **Benchmark Details**: See `benchmark/BENCHMARK_README.md`
- **Schema Inspection**: Run `python tests/check_schema.py`