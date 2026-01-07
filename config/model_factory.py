"""
Model Factory Classes

This module provides factory classes to create LLM and embedding model instances
based on the centralized configuration, supporting both OpenAI and Ollama providers.
"""

from typing import Union, Any, Dict
import warnings
import os

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Optional VertexAI imports
try:
    from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
    from genai_common.core.init_vertexai import init_vertexai
    from google.oauth2.credentials import Credentials
    from utils.custom_neo4j_embeddings import CustomVertexAIEmbeddings
    from utils.llms import get_vertex_llm, get_new_token, vertex_env, token_roller, get_vertex_embeddings
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False
    ChatVertexAI = None
    VertexAIEmbeddings = None
    warnings.warn("VertexAI dependencies not available. VertexAI provider will not work.")

from utils.graph_rag_logger import setup_logging, get_logger

from dotenv import load_dotenv

load_dotenv()

# Remove SSL_CERT_FILE if VertexAI is not being used
# This prevents SSL issues with OpenAI/Ollama clients when VertexAI deps aren't available
if not VERTEXAI_AVAILABLE:
    os.environ.pop('SSL_CERT_FILE', None)

setup_logging()
logger = get_logger(__name__)

# Neo4j GraphRAG imports (for backward compatibility)
try:
    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Neo4jOpenAIEmbeddings
    NEO4J_GRAPHRAG_AVAILABLE = True
except ImportError:
    NEO4J_GRAPHRAG_AVAILABLE = False
    warnings.warn("neo4j_graphrag not available, some backward compatibility features may not work")

from .model_config import ModelConfig, ModelProvider, get_model_config

class LLMFactory:
    """Factory for creating LLM instances"""
    
    @staticmethod
    def create_llm(config: ModelConfig = None, **kwargs) -> Union[ChatOpenAI, ChatOllama, ChatVertexAI]:
        """
        Create an LLM instance based on configuration
        
        Args:
            config: Model configuration (uses global config if None)
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            LLM instance (ChatOpenAI or ChatOllama)
        """
        if config is None:
            config = get_model_config()
        
        # Merge configuration parameters with kwargs
        model_params = config.get_model_params()
        model_params.update(kwargs)
        
        if config.llm_provider == ModelProvider.OPENAI:
            # Ensure SSL_CERT_FILE is not set for OpenAI to avoid SSL issues
            os.environ.pop('SSL_CERT_FILE', None)
            return ChatOpenAI(
                model=config.llm_model.value,
                openai_api_key=config.openai_api_key,
                **model_params
            )
        elif config.llm_provider == ModelProvider.OLLAMA:
            # Add progressive timeout configuration for Ollama
            base_timeout = int(os.getenv('OLLAMA_REQUEST_TIMEOUT', '300'))
            
            # Use longer timeout for RAGAS evaluation context
            if 'RAGAS' in os.environ.get('EVALUATION_CONTEXT', ''):
                timeout = base_timeout * 2  # Double timeout for RAGAS
                print(f"   🔧 Using extended timeout for RAGAS: {timeout}s")
            else:
                timeout = base_timeout
            
            ollama_params = {
                'model': config.llm_model.value,
                'base_url': config.ollama_base_url,
                'timeout': timeout,
                'keep_alive': os.getenv('OLLAMA_KEEP_ALIVE', '10m'),
                **model_params
            }
            return ChatOllama(**ollama_params)
        elif config.llm_provider == ModelProvider.VERTEXAI:
            if not VERTEXAI_AVAILABLE:
                raise ImportError("VertexAI dependencies not installed. Install with: pip install langchain-google-vertexai")
            logger.debug("In create_llm, returning VertexAI llm..")
            return get_vertex_llm()

        else:
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
    
    @staticmethod
    def create_neo4j_llm(config: ModelConfig = None, **kwargs) -> Any:
        """
        Create a Neo4j GraphRAG compatible LLM instance
        
        Args:
            config: Model configuration (uses global config if None)
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Neo4j GraphRAG LLM instance
        """
        logger.debug(f"Model Config Object: {config}")

        if not NEO4J_GRAPHRAG_AVAILABLE:
            raise ImportError("neo4j_graphrag is required for Neo4j LLM instances")
        
        if config is None:
            config = get_model_config()

        logger.debug(f"Model Config Object: {config}")

        if config.llm_provider == ModelProvider.OPENAI:

            model_params = config.get_model_params()
            model_params.update(kwargs)
            return OpenAILLM(
                model_name=config.llm_model.value,
                model_params=model_params
            )
        elif config.llm_provider == ModelProvider.OLLAMA:
            # For Ollama, we'll use the regular LangChain ChatOllama
            # as Neo4j GraphRAG doesn't have native Ollama support
            warnings.warn("Using LangChain ChatOllama instead of Neo4j GraphRAG LLM for Ollama")
            return LLMFactory.create_llm(config, **kwargs)
        elif config.llm_provider == ModelProvider.VERTEXAI:
            if not VERTEXAI_AVAILABLE:
                raise ImportError("VertexAI dependencies not installed. Install with: pip install langchain-google-vertexai")
            credentials = Credentials(token=None, refresh_handler=get_new_token)
            init_vertexai(vertex_env, token_roller)
            model_params = config.get_model_params()
            model_params.update(kwargs)
            from utils.neo4j_vertexai_llm_updated import CustomVertexAILLM
            return CustomVertexAILLM(
                model=config.llm_model.value,
                credentials=credentials,
                temperature=0.0,
                top_p=1,
                seed=0,
                max_output_tokens=65535,
                project="prj-gen-ai-9571",
                model_params=config.get_model_params())
        else:
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
    
    @staticmethod
    def create_text2cypher_llm(config: ModelConfig = None, **kwargs) -> Any:
        """
        Create a Neo4j GraphRAG compatible LLM instance for Text2Cypher
        
        Uses TEXT2CYPHER_MODEL and TEXT2CYPHER_PROVIDER if configured,
        otherwise falls back to the main LLM settings.
        
        Args:
            config: Model configuration (uses global config if None)
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Neo4j GraphRAG LLM instance configured for Text2Cypher
        """
        if not NEO4J_GRAPHRAG_AVAILABLE:
            raise ImportError("neo4j_graphrag is required for Neo4j LLM instances")
        
        if config is None:
            config = get_model_config()
        
        # Get effective Text2Cypher provider and model (falls back to main LLM if not specified)
        effective_provider = config.effective_text2cypher_provider
        effective_model = config.effective_text2cypher_model
        
        logger.info(f"Text2Cypher LLM - Provider: {effective_provider.value}, Model: {effective_model.value}")
        
        if effective_provider == ModelProvider.OPENAI:
            model_params = config.get_model_params()
            model_params.update(kwargs)
            return OpenAILLM(
                model_name=effective_model.value,
                model_params=model_params
            )
        elif effective_provider == ModelProvider.OLLAMA:
            # For Ollama, we'll use the regular LangChain ChatOllama
            # as Neo4j GraphRAG doesn't have native Ollama support
            warnings.warn("Using LangChain ChatOllama instead of Neo4j GraphRAG LLM for Ollama Text2Cypher")
            
            base_timeout = int(os.getenv('OLLAMA_REQUEST_TIMEOUT', '300'))
            model_params = config.get_model_params()
            model_params.update(kwargs)
            
            ollama_params = {
                'model': effective_model.value,
                'base_url': config.ollama_base_url,
                'timeout': base_timeout,
                'keep_alive': os.getenv('OLLAMA_KEEP_ALIVE', '10m'),
                **model_params
            }
            return ChatOllama(**ollama_params)
        elif effective_provider == ModelProvider.VERTEXAI:
            if not VERTEXAI_AVAILABLE:
                raise ImportError("VertexAI dependencies not installed. Install with: pip install langchain-google-vertexai")
            credentials = Credentials(token=None, refresh_handler=get_new_token)
            init_vertexai(vertex_env, token_roller)
            model_params = config.get_model_params()
            model_params.update(kwargs)
            from utils.neo4j_vertexai_llm_updated import CustomVertexAILLM
            return CustomVertexAILLM(
                model=effective_model.value,
                credentials=credentials,
                temperature=0.0,
                top_p=1,
                seed=0,
                max_output_tokens=65535,
                project="prj-gen-ai-9571",
                model_params=config.get_model_params())
        else:
            raise ValueError(f"Unsupported Text2Cypher LLM provider: {effective_provider}")
    
    @staticmethod
    def create_text2cypher_langchain_llm(config: ModelConfig = None, **kwargs) -> Union[ChatOpenAI, ChatOllama, ChatVertexAI]:
        """
        Create a LangChain-compatible LLM instance using Text2Cypher model settings
        
        This is used for verification and correction in the iterative refinement loop,
        which requires a standard LangChain LLM (not the Neo4j-specific wrapper).
        
        Uses TEXT2CYPHER_MODEL and TEXT2CYPHER_PROVIDER if configured,
        otherwise falls back to the main LLM settings.
        
        Args:
            config: Model configuration (uses global config if None)
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            LangChain LLM instance (ChatOpenAI, ChatOllama, or ChatVertexAI)
        """
        if config is None:
            config = get_model_config()
        
        # Get effective Text2Cypher provider and model (falls back to main LLM if not specified)
        effective_provider = config.effective_text2cypher_provider
        effective_model = config.effective_text2cypher_model
        
        logger.debug(f"Text2Cypher LangChain LLM - Provider: {effective_provider.value}, Model: {effective_model.value}")
        
        # Merge configuration parameters with kwargs
        model_params = config.get_model_params()
        model_params.update(kwargs)
        
        if effective_provider == ModelProvider.OPENAI:
            os.environ.pop('SSL_CERT_FILE', None)
            return ChatOpenAI(
                model=effective_model.value,
                openai_api_key=config.openai_api_key,
                **model_params
            )
        elif effective_provider == ModelProvider.OLLAMA:
            base_timeout = int(os.getenv('OLLAMA_REQUEST_TIMEOUT', '300'))
            ollama_params = {
                'model': effective_model.value,
                'base_url': config.ollama_base_url,
                'timeout': base_timeout,
                'keep_alive': os.getenv('OLLAMA_KEEP_ALIVE', '10m'),
                **model_params
            }
            return ChatOllama(**ollama_params)
        elif effective_provider == ModelProvider.VERTEXAI:
            if not VERTEXAI_AVAILABLE:
                raise ImportError("VertexAI dependencies not installed. Install with: pip install langchain-google-vertexai")
            logger.debug("Creating LangChain VertexAI LLM for Text2Cypher verification/correction")
            return get_vertex_llm()
        else:
            raise ValueError(f"Unsupported Text2Cypher LLM provider: {effective_provider}")
    
    @staticmethod
    def create_agentic_text2cypher_llm(config: ModelConfig = None, **kwargs) -> Union[ChatOpenAI, ChatOllama, ChatVertexAI]:
        """
        Create a LangChain-compatible LLM instance for Agentic Text2Cypher (Deep Agent)
        
        Uses AGENTIC_TEXT2CYPHER_MODEL and AGENTIC_TEXT2CYPHER_PROVIDER if configured,
        otherwise falls back to the main LLM settings.
        
        Special handling for thinking models (gpt-5.2, o3) - no temperature parameter.
        
        Args:
            config: Model configuration (uses global config if None)
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            LangChain LLM instance (ChatOpenAI, ChatOllama, or ChatVertexAI)
        """
        if config is None:
            config = get_model_config()
        
        # Get effective Agentic Text2Cypher provider and model
        effective_provider = config.effective_agentic_text2cypher_provider
        effective_model = config.effective_agentic_text2cypher_model
        
        logger.info(f"Agentic Text2Cypher LLM - Provider: {effective_provider.value}, Model: {effective_model.value}")
        
        # Check if this is a thinking model (special handling - no temperature)
        is_thinking = ModelConfig.is_thinking_model(effective_model)
        if is_thinking:
            logger.info(f"Using thinking model {effective_model.value} - temperature parameter will be omitted")
        
        # Build model params - exclude temperature for thinking models
        model_params = {}
        if not is_thinking:
            model_params = config.get_model_params()
        model_params.update(kwargs)
        
        # Remove temperature for thinking models even if passed in kwargs
        if is_thinking and 'temperature' in model_params:
            del model_params['temperature']
        
        if effective_provider == ModelProvider.OPENAI:
            os.environ.pop('SSL_CERT_FILE', None)
            return ChatOpenAI(
                model=effective_model.value,
                openai_api_key=config.openai_api_key,
                **model_params
            )
        elif effective_provider == ModelProvider.OLLAMA:
            base_timeout = int(os.getenv('OLLAMA_REQUEST_TIMEOUT', '300'))
            ollama_params = {
                'model': effective_model.value,
                'base_url': config.ollama_base_url,
                'timeout': base_timeout,
                'keep_alive': os.getenv('OLLAMA_KEEP_ALIVE', '10m'),
                **model_params
            }
            return ChatOllama(**ollama_params)
        elif effective_provider == ModelProvider.VERTEXAI:
            if not VERTEXAI_AVAILABLE:
                raise ImportError("VertexAI dependencies not installed. Install with: pip install langchain-google-vertexai")
            logger.debug("Creating LangChain VertexAI LLM for Agentic Text2Cypher")
            return get_vertex_llm()
        else:
            raise ValueError(f"Unsupported Agentic Text2Cypher LLM provider: {effective_provider}")

class EmbeddingFactory:
    """Factory for creating embedding model instances"""
    
    @staticmethod
    def create_embeddings(config: ModelConfig = None, **kwargs) -> Union[OpenAIEmbeddings, OllamaEmbeddings, VertexAIEmbeddings]:
        """
        Create an embedding model instance based on configuration
        
        Args:
            config: Model configuration (uses global config if None)
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Embedding model instance (OpenAIEmbeddings or OllamaEmbeddings)
        """
        if config is None:
            config = get_model_config()
        
        if config.embedding_provider == ModelProvider.OPENAI:
            return OpenAIEmbeddings(
                model=config.embedding_model.value,
                openai_api_key=config.openai_api_key,
                **kwargs
            )
        elif config.embedding_provider == ModelProvider.OLLAMA:
            # OllamaEmbeddings doesn't support request_timeout parameter
            # Timeout is handled at the HTTP client level
            ollama_params = {
                'model': config.embedding_model.value,
                'base_url': config.ollama_base_url,
                **kwargs
            }
            return OllamaEmbeddings(**ollama_params)
        elif config.embedding_provider == ModelProvider.VERTEXAI:
            if not VERTEXAI_AVAILABLE:
                raise ImportError("VertexAI dependencies not installed. Install with: pip install langchain-google-vertexai")
            model = get_vertex_embeddings()
            return model
        else:
            raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
    
    @staticmethod
    def create_neo4j_embeddings(config: ModelConfig = None, **kwargs) -> Any:
        """
        Create a Neo4j GraphRAG compatible embedding instance
        
        Args:
            config: Model configuration (uses global config if None)
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Neo4j GraphRAG embedding instance
        """
        if not NEO4J_GRAPHRAG_AVAILABLE:
            raise ImportError("neo4j_graphrag is required for Neo4j embedding instances")
        
        if config is None:
            config = get_model_config()
        
        if config.embedding_provider == ModelProvider.OPENAI:
            return Neo4jOpenAIEmbeddings(
                model=config.embedding_model.value,
                **kwargs
            )
        elif config.embedding_provider == ModelProvider.OLLAMA:
            # For Ollama, we'll use the regular LangChain OllamaEmbeddings
            # as Neo4j GraphRAG doesn't have native Ollama support
            warnings.warn("Using LangChain OllamaEmbeddings instead of Neo4j GraphRAG embeddings for Ollama")
            return EmbeddingFactory.create_embeddings(config, **kwargs)
        elif config.embedding_provider == ModelProvider.VERTEXAI:
            if not VERTEXAI_AVAILABLE:
                raise ImportError("VertexAI dependencies not installed. Install with: pip install langchain-google-vertexai")
            return CustomVertexAIEmbeddings(model_nm=config.embedding_model.value,)
        else:
            raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")

# Convenience functions for quick access
def get_llm(**kwargs) -> Union[ChatOpenAI, ChatOllama, ChatVertexAI]:
    """Get LLM instance with current configuration"""
    return LLMFactory.create_llm(**kwargs)

def get_embeddings(**kwargs) -> Union[OpenAIEmbeddings, OllamaEmbeddings, VertexAIEmbeddings]:
    """Get embedding model instance with current configuration"""
    return EmbeddingFactory.create_embeddings(**kwargs)

def get_neo4j_llm(**kwargs) -> Any:
    """Get Neo4j GraphRAG compatible LLM instance with current configuration"""
    logger.debug(f"In get_neo4j_llm with kwargs: {kwargs}")
    return LLMFactory.create_neo4j_llm(**kwargs)

def get_neo4j_embeddings(**kwargs) -> Any:
    """Get Neo4j GraphRAG compatible embedding instance with current configuration"""
    return EmbeddingFactory.create_neo4j_embeddings(**kwargs)

def get_text2cypher_llm(**kwargs) -> Any:
    """
    Get Neo4j GraphRAG compatible LLM instance for Text2Cypher
    
    Uses TEXT2CYPHER_MODEL and TEXT2CYPHER_PROVIDER if configured,
    otherwise falls back to the main LLM settings (LLM_MODEL and LLM_PROVIDER).
    """
    logger.debug(f"In get_text2cypher_llm with kwargs: {kwargs}")
    return LLMFactory.create_text2cypher_llm(**kwargs)

def get_text2cypher_langchain_llm(**kwargs) -> Union[ChatOpenAI, ChatOllama, ChatVertexAI]:
    """
    Get LangChain-compatible LLM instance using Text2Cypher model settings
    
    Used for verification and correction in the iterative refinement loop.
    Uses TEXT2CYPHER_MODEL and TEXT2CYPHER_PROVIDER if configured,
    otherwise falls back to the main LLM settings.
    """
    logger.debug(f"In get_text2cypher_langchain_llm with kwargs: {kwargs}")
    return LLMFactory.create_text2cypher_langchain_llm(**kwargs)

def get_agentic_text2cypher_llm(**kwargs) -> Union[ChatOpenAI, ChatOllama, ChatVertexAI]:
    """
    Get LangChain-compatible LLM instance for Agentic Text2Cypher (Deep Agent)
    
    Uses AGENTIC_TEXT2CYPHER_MODEL and AGENTIC_TEXT2CYPHER_PROVIDER if configured,
    otherwise falls back to the main LLM settings.
    
    Special handling for thinking models (gpt-5.2, o3) - no temperature parameter.
    """
    logger.debug(f"In get_agentic_text2cypher_llm with kwargs: {kwargs}")
    return LLMFactory.create_agentic_text2cypher_llm(**kwargs)
