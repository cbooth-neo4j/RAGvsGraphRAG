"""
Centralized Model Configuration System

This module provides a unified interface for switching between OpenAI API and Ollama local models
for both LLM and embedding models across the entire RAG vs GraphRAG project.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from dotenv import load_dotenv

from utils.graph_rag_logger import setup_logging, get_logger

# Load environment variables
load_dotenv()
setup_logging()
logger = get_logger(__name__)

class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    VERTEXAI = "vertexai"

class EmbeddingModel(Enum):
    """Supported embedding models"""
    # OpenAI models
    OPENAI_TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    OPENAI_TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    OPENAI_TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    VERTEXAI_TEXT_EMBEDDING_005 = "text-embedding-005"
    
    # Ollama models
    OLLAMA_NOMIC_TEXT_EMBED = "nomic-embed-text"

class LLMModel(Enum):
    """Supported LLM models"""
    # OpenAI models
    OPENAI_GPT_4O_MINI = "gpt-4o-mini"
    OPENAI_GPT_41_MINI = "gpt-4.1-mini"

    
    # Ollama models - Add new models as needed
    OLLAMA_QWEN3_8B = "qwen3:8b"
    OLLAMA_GEMMA3_7B = "gemma3:1b"
    OLLAMA_GEMMA3_12B = "gemma3:12b"
    OLLAMA_LLAMA3_1_8B = "llama3.1:8b"

    # VertexAI models
    VERTEXAI_25_PRO = "gemini-2.5-pro"


@dataclass
class ModelConfig:
    """Configuration for model settings"""
    # Provider selection
    #llm_provider: ModelProvider = ModelProvider.OPENAI
    llm_provider: ModelProvider = ModelProvider.VERTEXAI
    #embedding_provider: ModelProvider = ModelProvider.OPENAI
    embedding_provider: ModelProvider = ModelProvider.VERTEXAI
    
    # Model selection
    # llm_model: LLMModel = LLMModel.OPENAI_GPT_4O_MINI
    llm_model: LLMModel = LLMModel.VERTEXAI_25_PRO
    # embedding_model: EmbeddingModel = EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_SMALL
    embedding_model: EmbeddingModel = EmbeddingModel.VERTEXAI_TEXT_EMBEDDING_005

    # Model parameters
    temperature: float = 0.0
    seed: Optional[int] = 42
    max_tokens: Optional[int] = None
    
    # Ollama specific settings
    ollama_base_url: str = "http://localhost:11434"
    
    # OpenAI specific settings
    openai_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Initialize configuration from environment variables"""
        # Load from environment variables
        # self.llm_provider = ModelProvider(os.getenv('LLM_PROVIDER', 'openai'))
        self.llm_provider = ModelProvider(os.getenv('LLM_PROVIDER', 'vertexai'))
        # self.embedding_provider = ModelProvider(os.getenv('EMBEDDING_PROVIDER', 'openai'))
        self.embedding_provider = ModelProvider(os.getenv('EMBEDDING_PROVIDER', 'vertexai'))
        
        # Load model names
        # llm_model_name = os.getenv('LLM_MODEL', os.getenv('LLM_FALLBACK_MODEL', 'llama3.1:8b'))
        llm_model_name = os.getenv('LLM_MODEL', os.getenv('LLM_FALLBACK_MODEL', 'gemini-2.5-pro'))
        # embedding_model_name = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        embedding_model_name = os.getenv('EMBEDDING_MODEL', 'text-embedding-005')
        
        # Map model names to enums
        self.llm_model = self._get_llm_model_enum(llm_model_name)
        self.embedding_model = self._get_embedding_model_enum(embedding_model_name)
        
        # Load other parameters
        self.temperature = float(os.getenv('MODEL_TEMPERATURE', '0.0'))
        self.seed = int(os.getenv('MODEL_SEED', '42')) if os.getenv('MODEL_SEED') else 42
        self.max_tokens = int(os.getenv('MAX_TOKENS')) if os.getenv('MAX_TOKENS') else None
        
        # Load provider-specific settings
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
    
    def _get_llm_model_enum(self, model_name: str) -> LLMModel:
        """Convert model name string to LLMModel enum"""
        model_mapping = {
            # OpenAI models
            'gpt-4o-mini': LLMModel.OPENAI_GPT_4O_MINI,
            'gpt-4.1-mini': LLMModel.OPENAI_GPT_41_MINI,
            # Ollama models
            'qwen3:8b': LLMModel.OLLAMA_QWEN3_8B,
            'gemma3:1b': LLMModel.OLLAMA_GEMMA3_7B,
            'gemma3:12b': LLMModel.OLLAMA_GEMMA3_12B,
            'llama3.1:8b': LLMModel.OLLAMA_LLAMA3_1_8B,
            # Vertex AI
            'gemini-2.5-pro': LLMModel.VERTEXAI_25_PRO
        }
        # Use configurable fallback instead of hardcoded default
        # fallback_model = os.getenv('LLM_FALLBACK_MODEL', 'llama3.1:8b')
        fallback_model = os.getenv('LLM_FALLBACK_MODEL', 'gemini-2.5-pro')
        # fallback_enum = model_mapping.get(fallback_model, LLMModel.OLLAMA_LLAMA3_1_8B)
        fallback_enum = model_mapping.get(fallback_model, LLMModel.VERTEXAI_25_PRO)
        config =  model_mapping.get(model_name, fallback_enum)
        logger.debug(f"Config: {config}")
        return config
    
    def _get_embedding_model_enum(self, model_name: str) -> EmbeddingModel:
        """Convert model name string to EmbeddingModel enum"""
        model_mapping = {
            'text-embedding-3-small': EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_SMALL,
            'text-embedding-3-large': EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE,
            'text-embedding-ada-002': EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002,
            'nomic-embed-text': EmbeddingModel.OLLAMA_NOMIC_TEXT_EMBED,
            'nomic-text-embed': EmbeddingModel.OLLAMA_NOMIC_TEXT_EMBED,  # Alias
            'text-embedding-005' : EmbeddingModel.VERTEXAI_TEXT_EMBEDDING_005
        }
        return model_mapping.get(model_name, EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_SMALL)
    
    @property
    def embedding_dimensions(self) -> int:
        """Get the embedding dimensions for the selected model
        
        Priority:
        1. EMBEDDING_DIMENSION env var (manual override)
        2. Model-specific default dimensions
        3. Fallback to 768
        """
        # Check for manual override first
        env_dimension = os.getenv('EMBEDDING_DIMENSION')
        if env_dimension:
            try:
                dim = int(env_dimension)
                logger.info(f"Using manually configured embedding dimension: {dim} (from EMBEDDING_DIMENSION env var)")
                return dim
            except ValueError:
                logger.warning(f"Invalid EMBEDDING_DIMENSION value '{env_dimension}', using model default")
        
        # Use model-specific defaults
        dimension_mapping = {
            EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_SMALL: 1536,
            EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE: 3072,
            EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002: 1536,
            EmbeddingModel.OLLAMA_NOMIC_TEXT_EMBED: 768,
            EmbeddingModel.VERTEXAI_TEXT_EMBEDDING_005: 768
        }
        
        default_dim = dimension_mapping.get(self.embedding_model, 768)
        logger.info(f"Embedding model: {self.embedding_model.value}, dimensions: {default_dim}")
        return default_dim
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters as dictionary"""
        params = {
            "temperature": self.temperature,
        }
        if self.seed is not None:
            params["seed"] = self.seed
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

# Global configuration instance
_config = None

def get_model_config() -> ModelConfig:
    """Get the global model configuration instance"""
    global _config
    if _config is None:
        _config = ModelConfig()
    return _config

def set_model_config(config: ModelConfig):
    """Set the global model configuration instance"""
    global _config
    _config = config

def reset_model_config():
    """Reset the global model configuration to reload from environment"""
    global _config
    _config = None
