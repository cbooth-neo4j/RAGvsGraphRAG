"""
Native Ollama Integration
Direct integration with Ollama API without LangChain wrappers
"""

import ollama
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import time

@dataclass
class NativeOllamaConfig:
    """Configuration for native Ollama integration"""
    llm_model: str = "qwen3:8b"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.0
    base_url: str = "http://localhost:11434"

class NativeOllamaLLM:
    """Native Ollama LLM without LangChain wrapper"""
    
    def __init__(self, model: str = "qwen3:8b", temperature: float = 0.0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.client = ollama.Client(host=kwargs.get('base_url', 'http://localhost:11434'))
        self.keep_alive = kwargs.get('keep_alive', '10m')  # Keep model loaded for 10 minutes
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Generate response using native Ollama API"""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                keep_alive=self.keep_alive,
                options={
                    'temperature': kwargs.get('temperature', self.temperature),
                    'seed': kwargs.get('seed', 42),
                    **kwargs
                }
            )
            return response['response']
        except Exception as e:
            raise Exception(f"Native Ollama LLM error: {e}")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with full metadata"""
        return self.client.generate(
            model=self.model,
            prompt=prompt,
            keep_alive=self.keep_alive,
            options={
                'temperature': kwargs.get('temperature', self.temperature),
                'seed': kwargs.get('seed', 42),
                **kwargs
            }
        )

class NativeOllamaEmbeddings:
    """Native Ollama Embeddings without LangChain wrapper"""
    
    def __init__(self, model: str = "nomic-embed-text", **kwargs):
        self.model = model
        self.client = ollama.Client(host=kwargs.get('base_url', 'http://localhost:11434'))
        self.keep_alive = kwargs.get('keep_alive', '10m')  # Keep model loaded for 10 minutes
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a single query using the newer embed API"""
        try:
            response = self.client.embed(
                model=self.model, 
                input=text, 
                keep_alive=self.keep_alive
            )
            return response['embeddings'][0]  # Single embedding
        except Exception as e:
            raise Exception(f"Native Ollama embeddings error: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents using batch processing"""
        try:
            response = self.client.embed(
                model=self.model, 
                input=texts, 
                keep_alive=self.keep_alive
            )
            return response['embeddings']  # Multiple embeddings
        except Exception as e:
            # Fallback to individual calls if batch fails
            return [self.embed_query(text) for text in texts]

def create_native_ollama_llm(config: Optional[NativeOllamaConfig] = None) -> NativeOllamaLLM:
    """Create native Ollama LLM instance"""
    if config is None:
        config = NativeOllamaConfig()
    
    return NativeOllamaLLM(
        model=config.llm_model,
        temperature=config.temperature,
        base_url=config.base_url
    )

def create_native_ollama_embeddings(config: Optional[NativeOllamaConfig] = None) -> NativeOllamaEmbeddings:
    """Create native Ollama embeddings instance"""
    if config is None:
        config = NativeOllamaConfig()
    
    return NativeOllamaEmbeddings(
        model=config.embedding_model,
        base_url=config.base_url
    )

def benchmark_native_vs_langchain():
    """Compare performance of native Ollama vs LangChain wrapper"""
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    
    test_prompt = "What is artificial intelligence?"
    test_text = "This is a test document for embedding generation."
    
    results = {}
    
    # Initialize both (exclude initialization from timing)
    print("Initializing models...")
    native_llm = NativeOllamaLLM()
    native_embeddings = NativeOllamaEmbeddings()
    langchain_llm = ChatOllama(model="qwen3:8b", temperature=0)
    langchain_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Warm up both models with a quick call
    print("Warming up models...")
    native_llm.invoke("Hi")
    langchain_llm.invoke("Hi")
    native_embeddings.embed_query("test")
    langchain_embeddings.embed_query("test")
    
    print("Running benchmarks...")
    
    # Test native LLM
    start_time = time.time()
    native_response = native_llm.invoke(test_prompt)
    native_llm_time = time.time() - start_time
    results['native_llm_time'] = native_llm_time
    results['native_llm_response_length'] = len(native_response)
    
    # Test LangChain LLM
    start_time = time.time()
    langchain_response = langchain_llm.invoke(test_prompt)
    langchain_llm_time = time.time() - start_time
    results['langchain_llm_time'] = langchain_llm_time
    results['langchain_llm_response_length'] = len(str(langchain_response.content))
    
    # Test native embeddings
    start_time = time.time()
    native_embedding = native_embeddings.embed_query(test_text)
    native_embed_time = time.time() - start_time
    results['native_embed_time'] = native_embed_time
    results['native_embed_dimensions'] = len(native_embedding)
    
    # Test LangChain embeddings  
    start_time = time.time()
    langchain_embedding = langchain_embeddings.embed_query(test_text)
    langchain_embed_time = time.time() - start_time
    results['langchain_embed_time'] = langchain_embed_time
    results['langchain_embed_dimensions'] = len(langchain_embedding)
    
    return results

if __name__ == "__main__":
    # Quick test
    print("🚀 Testing Native Ollama Integration...")
    
    # Test LLM
    llm = create_native_ollama_llm()
    response = llm.invoke("Hello, how are you?")
    print(f"✅ Native LLM Response: {response[:100]}...")
    
    # Test Embeddings
    embeddings = create_native_ollama_embeddings()
    embedding = embeddings.embed_query("Test document")
    print(f"✅ Native Embeddings: {len(embedding)} dimensions")
    
    # Benchmark
    print("\n📊 Benchmarking Native vs LangChain...")
    results = benchmark_native_vs_langchain()
    
    print(f"Native LLM Time: {results['native_llm_time']:.3f}s")
    print(f"LangChain LLM Time: {results['langchain_llm_time']:.3f}s")
    print(f"Native Embeddings Time: {results['native_embed_time']:.3f}s")
    print(f"LangChain Embeddings Time: {results['langchain_embed_time']:.3f}s")
    
    llm_speedup = results['langchain_llm_time'] / results['native_llm_time']
    embed_speedup = results['langchain_embed_time'] / results['native_embed_time']
    
    print(f"\n🏆 Performance:")
    print(f"Native LLM is {llm_speedup:.2f}x {'faster' if llm_speedup > 1 else 'slower'}")
    print(f"Native Embeddings is {embed_speedup:.2f}x {'faster' if embed_speedup > 1 else 'slower'}")
