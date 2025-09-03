# Model Configuration System

This package provides a centralized configuration system for switching between OpenAI API and Ollama local models across the entire RAG vs GraphRAG project.

## Features

- **Unified Interface**: Single configuration system for all LLM and embedding models
- **Multiple Providers**: Support for OpenAI API and Ollama local models
- **Easy Switching**: Change models via environment variables or programmatic configuration
- **Backward Compatibility**: Existing code continues to work without changes
- **Automatic Fallbacks**: Graceful degradation when models are unavailable

## Supported Models

### LLM Models

**OpenAI:**
- `gpt-4o-mini` (default)
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

**Ollama:**
- `gemma2:12b` (recommended)
- `llama3:8b`
- `mistral:7b`

### Embedding Models

**OpenAI:**
- `text-embedding-3-small` (default, 1536 dimensions)
- `text-embedding-3-large` (3072 dimensions)
- `text-embedding-ada-002` (1536 dimensions)

**Ollama:**
- `nomic-embed-text` (768 dimensions)

## Quick Start

### 1. Environment Configuration

Copy `.env_example` to `.env` and configure your preferred models:

```bash
# Use OpenAI models (default)
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your_api_key_here

# Or use Ollama local models
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
LLM_MODEL=gemma2:12b
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

### 2. Using the Configuration

```python
from config import get_llm, get_embeddings

# Get models with current configuration
llm = get_llm()
embeddings = get_embeddings()

# Use models as usual
response = llm.invoke("Hello, world!")
embedding = embeddings.embed_query("test query")
```

### 3. Programmatic Configuration

```python
from config import ModelConfig, ModelProvider, LLMModel, EmbeddingModel, set_model_config

# Create custom configuration
config = ModelConfig(
    llm_provider=ModelProvider.OLLAMA,
    embedding_provider=ModelProvider.OLLAMA,
    llm_model=LLMModel.OLLAMA_GEMMA2_12B,
    embedding_model=EmbeddingModel.OLLAMA_NOMIC_TEXT_EMBED,
    temperature=0.1,
    ollama_base_url="http://localhost:11434"
)

# Apply configuration
set_model_config(config)

# Now all model factories will use this configuration
llm = get_llm()
embeddings = get_embeddings()
```

## Advanced Usage

### Mixed Providers

You can use different providers for LLM and embeddings:

```python
config = ModelConfig(
    llm_provider=ModelProvider.OPENAI,      # Use OpenAI for LLM
    embedding_provider=ModelProvider.OLLAMA,  # Use Ollama for embeddings
    llm_model=LLMModel.OPENAI_GPT_4O_MINI,
    embedding_model=EmbeddingModel.OLLAMA_NOMIC_TEXT_EMBED
)
```

### Retriever Integration

All retrievers now support the new configuration system:

```python
from retrievers.chroma_retriever import create_chroma_retriever
from retrievers.hybrid_cypher_retriever import create_hybrid_cypher_retriever

# Create retrievers with custom configuration
chroma_retriever = create_chroma_retriever(model_config=config)
hybrid_retriever = create_hybrid_cypher_retriever(model_config=config)
```

### Data Processor Integration

Graph processors also support the new configuration:

```python
from data_processors.graph_processor import CustomGraphProcessor

# Create processor with custom configuration
processor = CustomGraphProcessor(model_config=config)
```

## Setup Requirements

### OpenAI Setup

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Set `OPENAI_API_KEY` in your `.env` file
3. Configure your preferred OpenAI models

### Ollama Setup

1. Install Ollama: https://ollama.ai/download
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull required models:
   ```bash
   ollama pull gemma2:12b
   ollama pull nomic-embed-text
   ```
4. Configure Ollama models in your `.env` file

## Testing

Run the comprehensive test suite to verify your setup:

```bash
python tests/test_model_switching.py
```

This will test:
- OpenAI model initialization and usage
- Ollama model initialization and usage
- Mixed provider configurations
- Environment variable loading
- Retriever integration

## Configuration Options

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `LLM_PROVIDER` | LLM provider | `openai` | `openai`, `ollama` |
| `EMBEDDING_PROVIDER` | Embedding provider | `openai` | `openai`, `ollama` |
| `LLM_MODEL` | LLM model name | `gpt-4o-mini` | See supported models above |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` | See supported models above |
| `MODEL_TEMPERATURE` | LLM temperature | `0.0` | `0.0` to `2.0` |
| `MODEL_SEED` | Random seed for reproducibility | `42` | Any integer |
| `MAX_TOKENS` | Maximum tokens for LLM responses | None | Any positive integer |
| `OPENAI_API_KEY` | OpenAI API key | None | Your API key |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` | Any valid URL |

### Legacy Compatibility

The following environment variables are still supported for backward compatibility:
- `OPENAI_MODEL_NAME` (maps to `LLM_MODEL`)
- `LLM` (maps to `LLM_MODEL`)
- `EMBEDDINGS_MODEL` (maps to `EMBEDDING_MODEL`)

## Troubleshooting

### Common Issues

1. **Ollama connection errors**: Ensure Ollama server is running (`ollama serve`)
2. **Model not found**: Pull the required model (`ollama pull model_name`)
3. **OpenAI API errors**: Check your API key and billing status
4. **Dimension mismatch**: Neo4j vector indexes expect specific dimensions (usually 1536)

### Vector Index Compatibility

When switching embedding models, be aware of dimension compatibility:
- OpenAI `text-embedding-3-small`: 1536 dimensions
- OpenAI `text-embedding-3-large`: 3072 dimensions
- Ollama `nomic-embed-text`: 768 dimensions

If you change embedding models, you may need to recreate your Neo4j vector indexes.

### Performance Considerations

- **OpenAI**: Fast inference, requires internet connection and API costs
- **Ollama**: Slower inference (depends on hardware), no API costs, works offline
- **Mixed setup**: Use OpenAI for LLM (faster) and Ollama for embeddings (cheaper)

## API Reference

See the individual module documentation:
- [`model_config.py`](model_config.py) - Configuration classes and enums
- [`model_factory.py`](model_factory.py) - Factory classes for creating model instances
