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

# Load environment variables
load_dotenv(override=True)

class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"

# Simple model storage - just store the string values
# The actual validation happens at the API level (OpenAI/Ollama)


@dataclass
class ModelConfig:
    """Configuration for model settings"""
    # Provider selection
    llm_provider: ModelProvider = ModelProvider.OPENAI
    embedding_provider: ModelProvider = ModelProvider.OPENAI
    
    # Model selection - just store model names as strings (no defaults - must be set in .env)
    llm_model: str = None
    embedding_model: str = None
    
    # Model parameters  
    seed: Optional[int] = 42
    max_tokens: Optional[int] = None
    
    # Ollama specific settings
    ollama_base_url: str = "http://localhost:11434"
    
    # OpenAI specific settings
    openai_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Initialize configuration from environment variables"""
        # Load providers - REQUIRED, no defaults
        llm_provider_env = os.getenv('LLM_PROVIDER')
        embedding_provider_env = os.getenv('EMBEDDING_PROVIDER')
        
        if not llm_provider_env:
            raise ValueError("LLM_PROVIDER must be set in environment variables. Choose 'openai' or 'ollama'")
        if not embedding_provider_env:
            raise ValueError("EMBEDDING_PROVIDER must be set in environment variables. Choose 'openai' or 'ollama'")
            
        self.llm_provider = ModelProvider(llm_provider_env)
        self.embedding_provider = ModelProvider(embedding_provider_env)
        
        # Load model names - REQUIRED, no defaults
        llm_model_name = os.getenv('LLM_MODEL')
        embedding_model_name = os.getenv('EMBEDDING_MODEL')
        
        if not llm_model_name:
            raise ValueError("LLM_MODEL must be set in environment variables")
        if not embedding_model_name:
            raise ValueError("EMBEDDING_MODEL must be set in environment variables")
        
        # Just store the model names directly - let the APIs validate them
        self.llm_model = llm_model_name
        self.embedding_model = embedding_model_name
        
        # Load other parameters (let models use their default temperatures)
        self.seed = int(os.getenv('MODEL_SEED', '42')) if os.getenv('MODEL_SEED') else 42
        self.max_tokens = int(os.getenv('MAX_TOKENS')) if os.getenv('MAX_TOKENS') else None
        
        # Load provider-specific settings
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
    
    
    
    
    @property
    def embedding_dimensions(self) -> int:
        """Get the embedding dimensions for the selected model"""
        # Common dimension mappings - fallback to 1536 if unknown
        dimension_mapping = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536,
            'nomic-embed-text': 768,
            'nomic-text-embed': 768,
        }
        return dimension_mapping.get(self.embedding_model, 1536)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters as dictionary - let models use their default temperatures"""
        params = {}
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
