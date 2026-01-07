"""
Configuration Package

Centralized model configuration and factory classes for RAG vs GraphRAG project.
Supports switching between OpenAI API and Ollama local models.
"""

from .model_config import (
    ModelProvider,
    EmbeddingModel, 
    LLMModel,
    ModelConfig,
    get_model_config,
    set_model_config,
    reset_model_config
)

from .model_factory import (
    LLMFactory,
    EmbeddingFactory,
    get_llm,
    get_embeddings,
    get_neo4j_llm,
    get_neo4j_embeddings,
    get_text2cypher_llm,
    get_text2cypher_langchain_llm,
    get_agentic_text2cypher_llm
)

__all__ = [
    # Configuration classes and enums
    'ModelProvider',
    'EmbeddingModel',
    'LLMModel', 
    'ModelConfig',
    
    # Configuration management
    'get_model_config',
    'set_model_config',
    'reset_model_config',
    
    # Factory classes
    'LLMFactory',
    'EmbeddingFactory',
    
    # Convenience functions
    'get_llm',
    'get_embeddings',
    'get_neo4j_llm',
    'get_neo4j_embeddings',
    'get_text2cypher_llm',
    'get_text2cypher_langchain_llm',
    'get_agentic_text2cypher_llm'
]
