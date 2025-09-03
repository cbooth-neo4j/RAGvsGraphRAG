# Data Processors

This module handles data ingestion and processing into the Neo4j knowledge graph.

## 🚀 Quick Start

```bash
# From project root:
# Process PDFs from your documents
python data_processors/process_data.py --pdfs

# Process RAGBench dataset
python data_processors/process_data.py --ragbench --preset nano

# See all options
python data_processors/process_data.py --help
```

## 📁 Components

### **Main CLI**
- **`process_data.py`** - Unified CLI for all data processing

### **Graph Processing** 
- **`build_graph/`** - Enhanced graph processor with research-based entity discovery
  - See [build_graph/README.md](build_graph/README.md) for technical details
- **`graph_processor.py`** - Original processor (legacy, use build_graph instead)
- **`build_graph/advanced_processing.py`** - Community detection and summarization (integrated)

### **Vector Processing**
- **`chroma_processor.py`** - ChromaDB vector processing

## 🔄 Workflow

1. **Choose Data Source**:
   - `--pdfs` for your PDF documents
   - `--ragbench` for benchmark datasets

2. **Select Options**:
   - `--preset` for RAGBench presets (nano, micro, small, etc.)
   - `--domain` for domain-specific entity discovery
   - `--no-enhanced` to use basic processing

3. **Process Data**:
   - Creates Neo4j knowledge graph
   - Generates embeddings
   - Discovers entities and relationships

4. **Ready for Evaluation**:
   - Use with `benchmark/ragas_benchmark.py`
   - Test with retrievers in `retrievers/`

## 📊 Data Sources

### **PDF Documents** (`--pdfs`)
- Place PDFs in `PDFs/` folder
- Extracts text and tables
- Creates document chunks and entities
- Builds relationships between entities

### **RAGBench Datasets** (`--ragbench`)
- Research-grade benchmark datasets
- Multiple domains: financial, medical, academic
- Presets from nano (10 docs) to full (60K docs)
- Includes Q&A pairs for evaluation

## 🎯 Entity Discovery

### **Enhanced Discovery** (Default)
- Research-based corpus sampling
- Domain-aware entity extraction
- TF-IDF clustering for diversity
- Stratified document selection

### **Basic Discovery** (`--no-enhanced`)
- Simple pattern-based extraction
- Faster processing
- Good for quick tests

See [build_graph/README.md](build_graph/README.md) for technical details.

## ⚙️ Configuration

### **Neo4j Setup**
- Requires Neo4j running on `bolt://localhost:7687`
- Default credentials: `neo4j/password`
- Creates schema automatically

### **OpenAI API**
- Set `OPENAI_API_KEY` environment variable
- Uses `gpt-4o-mini` for entity extraction
- Uses `text-embedding-3-small` for embeddings

## 🔗 Integration

### **With Benchmarking**
```bash
# Process data
python data_processors/process_data.py --ragbench --preset nano

# Run evaluation
python benchmark/ragas_benchmark.py --all
```

### **With Retrievers**
```bash
# Process data
python data_processors/process_data.py --pdfs

# Test retrieval
python -m retrievers.chroma_retriever
python -m retrievers.graph_rag_retriever
```

## 📈 Performance

### **Processing Times** (Approximate)
- **Nano** (10 docs): 2-5 minutes
- **Micro** (50 docs): 10-20 minutes  
- **Small** (500 docs): 1-3 hours
- **PDF processing**: ~30 seconds per document

### **Resource Usage**
- **RAM**: 2-8GB depending on dataset size
- **Storage**: See preset details with `--list-presets`
- **API Costs**: $2-$3000 depending on dataset

## 🛠️ Troubleshooting

### **Common Issues**
- **Neo4j not running**: Start Neo4j service
- **OpenAI API key**: Set environment variable
- **PDF processing fails**: Install `PyPDF2` dependency
- **Enhanced features fail**: Install `scikit-learn` for clustering

### **Performance Tuning**
- Use `--no-enhanced` for faster processing
- Use `--no-resolution` to skip entity resolution
- Process smaller presets first to test setup
