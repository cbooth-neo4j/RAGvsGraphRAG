#!/usr/bin/env python3
"""
Quick test to verify OpenAI API key is working
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_openai_api_key():
    """Test if OpenAI API key is configured and working"""
    print("🔑 Testing OpenAI API Key Configuration")
    print("=" * 50)
    
    # Load environment variables (override system env vars)
    load_dotenv(override=True)
    
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("💡 Set it in your .env file or export OPENAI_API_KEY=your_key_here")
        return False
    
    print(f"✅ OPENAI_API_KEY found: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    
    # Test basic OpenAI connection
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client created successfully")
        
        # Test a simple completion
        print("🧪 Testing simple completion...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello World' in exactly two words."}],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ OpenAI API response: '{result}'")
        
        # Test embeddings
        print("🧪 Testing embeddings...")
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test embedding"
        )
        
        embedding_vector = embedding_response.data[0].embedding
        print(f"✅ Embeddings working: {len(embedding_vector)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        print("💡 Common issues:")
        print("   - Invalid API key")
        print("   - No credits remaining")
        print("   - Network connectivity issues")
        print("   - Rate limiting")
        return False

def test_langchain_openai():
    """Test OpenAI integration with LangChain (used by RAGAS)"""
    print("\n🔗 Testing LangChain OpenAI Integration")
    print("=" * 50)
    
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        # Test LangChain ChatOpenAI
        print("🧪 Testing LangChain ChatOpenAI...")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=20
        )
        
        response = llm.invoke("What is 2+2? Answer in one word.")
        print(f"✅ LangChain LLM response: '{response.content.strip()}'")
        
        # Test LangChain Embeddings
        print("🧪 Testing LangChain Embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        embedding_vector = embeddings.embed_query("test query")
        print(f"✅ LangChain Embeddings working: {len(embedding_vector)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain OpenAI test failed: {e}")
        return False

def test_model_factory_openai():
    """Test our centralized model factory with OpenAI"""
    print("\n🏭 Testing Model Factory with OpenAI")
    print("=" * 50)
    
    try:
        # Temporarily set environment to use OpenAI
        original_provider = os.getenv('LLM_PROVIDER')
        original_model = os.getenv('LLM_MODEL')
        
        os.environ['LLM_PROVIDER'] = 'openai'
        os.environ['LLM_MODEL'] = 'gpt-4o-mini'
        os.environ['EMBEDDING_PROVIDER'] = 'openai'
        os.environ['EMBEDDING_MODEL'] = 'text-embedding-3-small'
        
        from config.model_factory import get_llm, get_embeddings
        from config.model_config import get_model_config
        
        # Test model config
        config = get_model_config()
        print(f"✅ Model config loaded: {config.llm_provider.value}/{config.llm_model.value}")
        
        # Test LLM factory
        print("🧪 Testing LLM factory...")
        llm = get_llm(temperature=0, max_tokens=10)
        response = llm.invoke("Say 'test'")
        print(f"✅ Model factory LLM: '{response.content.strip()}'")
        
        # Test embeddings factory
        print("🧪 Testing embeddings factory...")
        embeddings = get_embeddings()
        embedding_vector = embeddings.embed_query("factory test")
        print(f"✅ Model factory embeddings: {len(embedding_vector)} dimensions")
        
        # Restore original settings
        if original_provider:
            os.environ['LLM_PROVIDER'] = original_provider
        if original_model:
            os.environ['LLM_MODEL'] = original_model
            
        return True
        
    except Exception as e:
        print(f"❌ Model factory test failed: {e}")
        return False

def main():
    """Run all OpenAI API tests"""
    print("🚀 OpenAI API Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic OpenAI API", test_openai_api_key),
        ("LangChain Integration", test_langchain_openai),
        ("Model Factory", test_model_factory_openai)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🏆 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your OpenAI API is ready for RAGAS benchmarking.")
        print("\n💡 To use OpenAI for benchmarking, update your .env file:")
        print("   LLM_PROVIDER=openai")
        print("   LLM_MODEL=gpt-4o-mini")
        print("   EMBEDDING_PROVIDER=openai")
        print("   EMBEDDING_MODEL=text-embedding-3-small")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

