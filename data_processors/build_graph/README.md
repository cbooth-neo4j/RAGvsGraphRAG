# Graph Building Components

This module contains the refactored graph processing system, organized into logical components for better maintainability and extensibility.

## 📁 Structure

```
data_processors/build_graph/
├── __init__.py              # Module exports
├── README.md               # This file
├── entity_discovery.py    # Enhanced entity discovery & sampling
├── text_processing.py     # PDF extraction, chunking, embeddings
├── graph_operations.py    # Neo4j operations & entity resolution
└── main_processor.py      # Main orchestrator class
```

## 🚀 Key Features

### Enhanced Entity Discovery
- **Research-based corpus sampling** with TF-IDF clustering (optional sklearn)
- **Multi-strategy sampling**: diversity, pattern-based, and strategic random
- **Domain-aware entity discovery** with hints (financial, medical, legal, etc.)
- **Stratified document selection** for representative coverage
- **Backward compatible** with original discovery methods

### Advanced Processing (Always Enabled)
- **Element Summarization**: AI-generated summaries for entities and relationships
- **Community Detection**: Hierarchical clustering using Leiden algorithm
- **Community Summarization**: AI-generated community reports with importance ratings
- **Cost Estimation**: Transparent cost estimation with user confirmation
- **Automatic Integration**: Advanced features run automatically after basic processing

### Modular Architecture
- **Mixin-based design** for clean separation of concerns
- **Optional dependencies** - graceful fallbacks when libraries missing
- **Full backward compatibility** with existing `CustomGraphProcessor`
- **Easy to extend** with new capabilities

### Robust Text Processing
- **PDF extraction** with table support (Camelot/Tabula)
- **Intelligent chunking** with RecursiveCharacterTextSplitter
- **Batch embeddings** for efficiency
- **RAGBench document support** with enhanced sampling

### Intelligent Relationship Discovery
- **LLM-guided RELATED_TO relationships**: Smart relationship extraction using LLM analysis
- **Evidence-based connections**: Only creates relationships with clear textual evidence
- **Confidence scoring**: Each relationship includes confidence and evidence metadata
- **Proximity fallback**: Simple proximity-based relationships for large entity sets
- **No relationship explosion**: Avoids creating relationships between every entity pair

## 📊 Research-Based Improvements

### Corpus Sampling Strategy
Based on 2024 research in ontology discovery:

1. **Document Characterization**
   - Complexity scoring (word length, capitalization, entity density)
   - Length analysis for stratification
   - Domain indicator detection

2. **Stratified Sampling**
   - Creates strata based on complexity and length
   - Ensures representative coverage across document types
   - Balances entity-rich content with diversity

3. **Multi-Strategy Text Sampling**
   - **40% Diversity**: TF-IDF clustering for semantic diversity
   - **30% Patterns**: Enhanced pattern extraction (entities, definitions, lists)
   - **30% Random**: Strategic random sampling from mid-document content

4. **Quality Metrics**
   - Diversity scoring
   - Compression ratios
   - Document coverage statistics

### Entity Discovery Enhancements

- **Domain-specific patterns** for different fields
- **Frequency-based filtering** to focus on important entities
- **Context-aware validation** using LLM analysis
- **Hierarchical entity type consideration**

## 🔧 Usage

### Basic Usage (Backward Compatible)
```python
from data_processors.build_graph import CustomGraphProcessor

processor = CustomGraphProcessor()

# Process PDF documents
result = processor.process_document("document.pdf")

# Process text directly
result = processor.process_text_document(text, "doc_name")

# Process directory
summary = processor.process_directory("pdf_folder/")

processor.close()
```

### Relationship Strategy Configuration
```python
# Default: Smart strategy (semantic + proximity + co-occurrence)
processor = CustomGraphProcessor(relationship_strategy="smart")

# Only semantic relationships (WORKS_FOR, LOCATED_IN, etc.)
processor = CustomGraphProcessor(relationship_strategy="semantic")

# Only proximity relationships (CO_OCCURS based on text distance)
processor = CustomGraphProcessor(relationship_strategy="proximity")

# No explicit entity relationships (rely on chunk connections)
processor = CustomGraphProcessor(relationship_strategy="implicit")
```

#### Relationship Strategy Details:
- **"smart"** (default): LLM-guided RELATED_TO relationships with evidence + proximity fallback for large entity sets
- **"semantic"**: Same as smart - LLM-guided meaningful relationships only
- **"proximity"**: Simple proximity-based RELATED_TO relationships for adjacent entities
- **"implicit"**: No direct entity relationships - entities connect only through shared chunks

### RAGBench Integration
```python
from benchmark.ragbench import RAGBenchIngester

ingester = RAGBenchIngester(processor_type='basic')
result = ingester.run_preset('nano')
```

### Enhanced Entity Discovery
```python
# Use enhanced discovery with domain hint
result = processor.process_ragbench_documents(
    texts=texts,
    sources=sources,
    use_enhanced_discovery=True,
    domain_hint='financial'
)
```

## 🧩 Component Details

### EntityDiscoveryMixin
- **Corpus sampling**: `_sample_corpus_text_enhanced()`
- **Label discovery**: `discover_labels_for_text_enhanced()`
- **CLI approval**: `_approve_labels_cli()`
- **Caching**: Schema caching for performance

### TextProcessingMixin
- **PDF extraction**: `extract_text_from_pdf()`
- **Table extraction**: `extract_tables()` (optional Camelot/Tabula)
- **Text chunking**: `chunk_text()`
- **Embeddings**: `create_embedding()`, `create_embeddings_batch()`

### GraphOperationsMixin
- **Schema setup**: `setup_database_schema()`
- **Node creation**: `create_document_node()`, `create_chunk_nodes()`, `create_entity_nodes()`
- **Smart relationships**: `create_entity_relationships_dynamic()` with semantic + proximity discovery
- **Entity resolution**: `perform_entity_resolution()`
- **Similarity**: `create_chunk_similarity_relationships()`
- **Relationship discovery**: `_discover_semantic_relationships()`, `_discover_proximity_relationships()`

## 🔄 Migration from Original

The refactored system maintains **100% backward compatibility**. Existing code will work unchanged:

```python
# This still works exactly as before
from data_processors.build_graph import CustomGraphProcessor
# Now automatically uses: from data_processors.build_graph import CustomGraphProcessor
```

## 📈 Performance & Dependencies

### Required Dependencies
- `neo4j` - Graph database driver
- `langchain` - Text splitting and LLM integration
- `langchain-openai` - OpenAI embeddings and chat
- `numpy` - Numerical operations
- `pandas` - Data manipulation

### Optional Dependencies
- `scikit-learn` - Enhanced TF-IDF clustering (graceful fallback without)
- `PyPDF2` - PDF text extraction (error if PDF processing attempted without)
- `camelot-py` - Advanced table extraction
- `tabula-py` - Fallback table extraction

### Performance Features
- **Batch embedding creation** for efficiency
- **Corpus-wide entity discovery** with caching
- **Stratified sampling** reduces processing time while maintaining quality
- **Optional advanced features** don't impact basic functionality

## 🎯 Benefits

1. **Clean Architecture**: Logical separation of concerns
2. **Research-Based**: Incorporates 2024 best practices for entity discovery
3. **Backward Compatible**: No breaking changes to existing code
4. **Extensible**: Easy to add new capabilities
5. **Robust**: Graceful handling of missing dependencies
6. **Performance**: Optimized sampling and batch operations
7. **Maintainable**: Smaller, focused files instead of monolithic processor

## 🚀 Future Enhancements

The modular structure makes it easy to add:
- New sampling strategies
- Additional entity types
- Domain-specific processors
- Alternative embedding models
- Enhanced relationship discovery
- Graph algorithms integration

---

This refactoring maintains the power and functionality of the original `CustomGraphProcessor` while providing a clean, extensible foundation for future enhancements based on the latest research in knowledge graph construction and entity discovery.
