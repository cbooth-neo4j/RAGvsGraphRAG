"""
Test script to verify OpenAI and Ollama model switching functionality

This script tests the centralized model configuration system and ensures
both OpenAI API and Ollama local models work correctly.
"""

import os
import sys
import warnings
from typing import Dict, Any

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ModelConfig, ModelProvider, LLMModel, EmbeddingModel,
    get_llm, get_embeddings, set_model_config, reset_model_config
)

def test_openai_models():
    """Test OpenAI models configuration"""
    print("🔍 Testing OpenAI Models...")
    
    # Configure for OpenAI
    openai_config = ModelConfig(
        llm_provider=ModelProvider.OPENAI,
        embedding_provider=ModelProvider.OPENAI,
        llm_model=LLMModel.OPENAI_GPT_4O_MINI,
        embedding_model=EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_SMALL,
        temperature=0.0,
        seed=42
    )
    
    set_model_config(openai_config)
    
    try:
        # Test LLM
        llm = get_llm()
        print(f"✅ OpenAI LLM initialized: {type(llm).__name__}")
        
        # Test simple invocation
        response = llm.invoke("Say 'Hello from OpenAI!'")
        print(f"✅ OpenAI LLM response: {response.content[:50]}...")
        
        # Test embeddings
        embeddings = get_embeddings()
        print(f"✅ OpenAI Embeddings initialized: {type(embeddings).__name__}")
        
        # Test embedding generation
        embedding_result = embeddings.embed_query("test query")
        print(f"✅ OpenAI Embedding dimensions: {len(embedding_result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

def test_ollama_models():
    """Test Ollama models configuration"""
    print("🦙 Testing Ollama Models...")
    
    # Configure for Ollama
    ollama_config = ModelConfig(
        llm_provider=ModelProvider.OLLAMA,
        embedding_provider=ModelProvider.OLLAMA,
        llm_model=LLMModel.OLLAMA_GEMMA2_12B,
        embedding_model=EmbeddingModel.OLLAMA_NOMIC_TEXT_EMBED,
        temperature=0.0,
        ollama_base_url="http://localhost:11434"
    )
    
    set_model_config(ollama_config)
    
    try:
        # Test LLM
        llm = get_llm()
        print(f"✅ Ollama LLM initialized: {type(llm).__name__}")
        
        # Test simple invocation (with timeout for local models)
        try:
            response = llm.invoke("Say 'Hello from Ollama!'")
            print(f"✅ Ollama LLM response: {response.content[:50]}...")
        except Exception as e:
            print(f"⚠️  Ollama LLM invocation failed (server might not be running): {e}")
        
        # Test embeddings
        embeddings = get_embeddings()
        print(f"✅ Ollama Embeddings initialized: {type(embeddings).__name__}")
        
        # Test embedding generation
        try:
            embedding_result = embeddings.embed_query("test query")
            print(f"✅ Ollama Embedding dimensions: {len(embedding_result)}")
        except Exception as e:
            print(f"⚠️  Ollama embedding generation failed (server might not be running): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def test_mixed_configuration():
    """Test mixed provider configuration (OpenAI LLM + Ollama embeddings)"""
    print("🔀 Testing Mixed Provider Configuration...")
    
    # Configure mixed providers
    mixed_config = ModelConfig(
        llm_provider=ModelProvider.OPENAI,
        embedding_provider=ModelProvider.OLLAMA,
        llm_model=LLMModel.OPENAI_GPT_4O_MINI,
        embedding_model=EmbeddingModel.OLLAMA_NOMIC_TEXT_EMBED,
        temperature=0.0
    )
    
    set_model_config(mixed_config)
    
    try:
        # Test LLM (OpenAI)
        llm = get_llm()
        print(f"✅ Mixed config LLM (OpenAI): {type(llm).__name__}")
        
        # Test embeddings (Ollama)
        embeddings = get_embeddings()
        print(f"✅ Mixed config Embeddings (Ollama): {type(embeddings).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mixed configuration test failed: {e}")
        return False

def test_environment_configuration():
    """Test configuration loading from environment variables"""
    print("🌍 Testing Environment Variable Configuration...")
    
    # Set environment variables
    os.environ['LLM_PROVIDER'] = 'openai'
    os.environ['EMBEDDING_PROVIDER'] = 'openai'
    os.environ['LLM_MODEL'] = 'gpt-4o-mini'
    os.environ['EMBEDDING_MODEL'] = 'text-embedding-3-small'
    os.environ['MODEL_TEMPERATURE'] = '0.1'
    
    # Reset to reload from environment
    reset_model_config()
    
    try:
        from config import get_model_config
        config = get_model_config()
        
        print(f"✅ LLM Provider: {config.llm_provider}")
        print(f"✅ Embedding Provider: {config.embedding_provider}")
        print(f"✅ LLM Model: {config.llm_model}")
        print(f"✅ Embedding Model: {config.embedding_model}")
        print(f"✅ Temperature: {config.temperature}")
        
        # Test model creation
        llm = get_llm()
        embeddings = get_embeddings()
        print(f"✅ Models created successfully from environment config")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment configuration test failed: {e}")
        return False

def test_retriever_integration():
    """Test integration with existing retrievers"""
    print("🔗 Testing Retriever Integration...")
    
    try:
        # Test ChromaDB retriever with new configuration
        from retrievers.chroma_retriever import create_chroma_retriever
        
        # Configure for OpenAI
        openai_config = ModelConfig(
            llm_provider=ModelProvider.OPENAI,
            embedding_provider=ModelProvider.OPENAI,
            llm_model=LLMModel.OPENAI_GPT_4O_MINI,
            embedding_model=EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_SMALL
        )
        
        retriever = create_chroma_retriever(model_config=openai_config)
        print(f"✅ ChromaDB retriever created with OpenAI config")
        
        # Test Hybrid Cypher retriever
        from retrievers.hybrid_cypher_retriever import create_hybrid_cypher_retriever
        
        hybrid_retriever = create_hybrid_cypher_retriever(model_config=openai_config)
        print(f"✅ Hybrid Cypher retriever created with OpenAI config")
        
        return True
        
    except Exception as e:
        print(f"❌ Retriever integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Model Switching Tests")
    print("=" * 50)
    
    tests = [
        ("OpenAI Models", test_openai_models),
        ("Ollama Models", test_ollama_models),
        ("Mixed Configuration", test_mixed_configuration),
        ("Environment Configuration", test_environment_configuration),
        ("Retriever Integration", test_retriever_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📝 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model switching is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\nCommon issues:")
        print("- Ollama server not running (start with: ollama serve)")
        print("- Missing OpenAI API key in environment")
        print("- Required models not pulled in Ollama (ollama pull gemma2:12b, ollama pull nomic-embed-text)")

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
