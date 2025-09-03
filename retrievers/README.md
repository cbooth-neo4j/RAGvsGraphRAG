# Retrievers

Collection of retrieval approaches for querying the knowledge graph and vector stores.

## 🚀 Quick Start

```bash
# Test ChromaDB retrieval
python -m retrievers.chroma_retriever

# Test GraphRAG retrieval  
python -m retrievers.graph_rag_retriever

# Test Text2Cypher
python -m retrievers.text2cypher_retriever
```

## 🔍 Retrieval Approaches

### **Vector-Based Retrieval**

#### **ChromaDB RAG** (`chroma_retriever.py`)
- **Purpose**: Traditional semantic similarity search
- **Method**: Embed query → find similar document chunks
- **Best for**: General semantic questions, document similarity
- **Speed**: Fast (< 100ms)

#### **Neo4j Vector** (`neo4j_vector_retriever.py`)  
- **Purpose**: Vector search within graph database
- **Method**: Neo4j vector index for similarity search
- **Best for**: Combining vector search with graph relationships
- **Speed**: Medium (100-500ms)

### **Graph-Based Retrieval**

#### **GraphRAG** (`graph_rag_retriever.py`)
- **Purpose**: Multi-hop graph traversal for context
- **Method**: Entity extraction → graph traversal → context assembly
- **Best for**: Complex questions requiring multiple entities/relationships
- **Speed**: Medium (200-800ms)

#### **Advanced GraphRAG** (`advanced_graphrag_retriever.py`)
- **Purpose**: Enhanced GraphRAG with community detection
- **Method**: GraphRAG + community summaries + element summaries
- **Best for**: Complex analytical questions, domain expertise
- **Speed**: Slower (500-2000ms)

#### **Text2Cypher** (`text2cypher_retriever.py`)
- **Purpose**: Natural language to database queries
- **Method**: LLM converts question → Cypher query → execute
- **Best for**: Structured data questions, specific entity queries
- **Speed**: Medium (300-1000ms)

#### **Hybrid Cypher** (`hybrid_cypher_retriever.py`)
- **Purpose**: Combined vector + graph approach
- **Method**: Vector similarity + graph neighborhood exploration
- **Best for**: Balanced performance across question types
- **Speed**: Medium (400-1200ms)

### **Experimental Approaches**

#### **DRIFT GraphRAG** (`drift_graphrag_retriever.py`)
- **Purpose**: Dynamic reasoning with iterative fact-finding
- **Method**: Multi-step reasoning with context refinement
- **Best for**: Complex analytical questions requiring reasoning
- **Speed**: Slowest (1000-5000ms)

## 📊 Performance Characteristics

| Retriever | Speed | Accuracy | Complexity | Best Use Case |
|-----------|-------|----------|------------|---------------|
| ChromaDB | ⚡⚡⚡ | 📊📊 | 🔧 | Semantic similarity |
| Neo4j Vector | ⚡⚡ | 📊📊📊 | 🔧🔧 | Graph + vectors |
| GraphRAG | ⚡⚡ | 📊📊📊 | 🔧🔧🔧 | Multi-entity questions |
| Advanced GraphRAG | ⚡ | 📊📊📊📊 | 🔧🔧🔧🔧 | Complex analysis |
| Text2Cypher | ⚡⚡ | 📊📊📊📊 | 🔧🔧 | Structured queries |
| Hybrid Cypher | ⚡⚡ | 📊📊📊📊 | 🔧🔧🔧 | Balanced approach |
| DRIFT | ⚡ | 📊📊📊📊📊 | 🔧🔧🔧🔧🔧 | Complex reasoning |

## 🔧 Usage Patterns

### **Development & Testing**
```python
# Import and test a retriever
from retrievers.graph_rag_retriever import GraphRAGRetriever

retriever = GraphRAGRetriever()
result = retriever.retrieve("What companies are mentioned?")
print(result)
```

### **Benchmark Integration**
```bash
# Retrievers are automatically tested by benchmark system
python benchmark/ragas_benchmark.py --all
```

### **Custom Integration**
```python
# Use in your own applications
from retrievers.chroma_retriever import ChromaRetriever

retriever = ChromaRetriever()
response = retriever.query("Your question here")
```

## ⚙️ Configuration

### **Required Setup**
1. **Neo4j Database** - Running on `bolt://localhost:7687`
2. **ChromaDB** - Vector store (created automatically)
3. **OpenAI API** - Set `OPENAI_API_KEY` environment variable
4. **Processed Data** - Run data processors first

### **Dependencies**
```bash
# Core dependencies (in requirements.txt)
pip install neo4j chromadb langchain-openai

# Optional for advanced features
pip install scikit-learn  # For enhanced processing
```

## 🎯 Choosing the Right Retriever

### **For Simple Semantic Questions**
→ Use **ChromaDB RAG**
- "What is machine learning?"
- "Find documents about AI"

### **For Entity-Specific Questions**  
→ Use **Text2Cypher** or **Neo4j Vector**
- "Who works at Acme Corp?"
- "What projects involve John Smith?"

### **For Complex Multi-Entity Questions**
→ Use **GraphRAG** or **Advanced GraphRAG**
- "How are companies A and B related through their employees?"
- "What's the relationship between project X and technology Y?"

### **For Analytical Questions**
→ Use **Advanced GraphRAG** or **DRIFT**
- "Analyze the competitive landscape in this industry"
- "What are the key trends across these documents?"

### **For Balanced Performance**
→ Use **Hybrid Cypher**
- Good performance across various question types
- Combines benefits of vector and graph approaches

## 🔄 Integration Workflow

### **1. Data Processing**
```bash
# Process your data first
python data_processors/process_data.py --pdfs
# or
python data_processors/process_data.py --ragbench --preset nano
```

### **2. Test Retrievers**
```bash
# Test individual retrievers
python -m retrievers.chroma_retriever
python -m retrievers.graph_rag_retriever
```

### **3. Benchmark Evaluation**
```bash
# Compare all retrievers
python benchmark/ragas_benchmark.py --all
```

### **4. Production Use**
```python
# Use best-performing retriever in your application
from retrievers.hybrid_cypher_retriever import HybridCypherRetriever
retriever = HybridCypherRetriever()
```

## 🧪 DRIFT Modules

The `drift_modules/` folder contains experimental components for the DRIFT (Dynamic Reasoning with Iterative Fact-finding and Thinking) approach:

- **`drift_action.py`** - Action execution and planning
- **`drift_context.py`** - Context management and memory
- **`drift_primer.py`** - Initial query processing
- **`drift_search.py`** - Multi-step search strategies
- **`drift_state.py`** - State management across reasoning steps

## 🛠️ Troubleshooting

### **Common Issues**
- **Neo4j connection failed**: Start Neo4j service
- **ChromaDB not found**: Process data first with data processors
- **Empty results**: Check if knowledge graph has data
- **Slow performance**: Use simpler retrievers for development

### **Performance Optimization**
- Use **ChromaDB** for development and quick tests
- Use **GraphRAG** only when you need multi-hop reasoning
- Monitor **API usage** with complex retrievers
- Cache results for repeated queries

### **Debugging**
- Check Neo4j browser at `http://localhost:7474`
- Verify data exists: `MATCH (n) RETURN count(n)`
- Test with simple questions first
- Enable verbose logging in retriever code

## 🚀 Extending Retrievers

### **Adding New Retrievers**
1. Create new file in `retrievers/`
2. Implement standard interface (query method)
3. Add to benchmark system in `ragas_benchmark.py`
4. Update this README

### **Custom Retrieval Logic**
- Combine multiple approaches
- Add domain-specific processing
- Implement custom ranking algorithms
- Add result filtering and post-processing
