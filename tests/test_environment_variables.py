#!/usr/bin/env python3
"""
Test Environment Variables Configuration

This test displays all loaded environment variables to verify configuration.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

def test_environment_variables():
    """Display all loaded environment variables"""
    
    print("🔧 Environment Variables Test")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Define expected environment variables with descriptions
    expected_vars = {
        # Model Provider Configuration
        'LLM_PROVIDER': 'Language model provider (openai, ollama)',
        'EMBEDDING_PROVIDER': 'Embedding model provider (openai, ollama)',
        'LLM_MODEL': 'Specific LLM model to use',
        'LLM_FALLBACK_MODEL': 'Fallback LLM model if primary fails',
        'EMBEDDING_MODEL': 'Specific embedding model to use',
        
        # Model Parameters
        'MODEL_TEMPERATURE': 'Temperature for model responses',
        'MODEL_SEED': 'Random seed for reproducible results',
        'MAX_TOKENS': 'Maximum tokens for model responses',
        
        # OpenAI Configuration
        'OPENAI_API_KEY': 'OpenAI API key',
        
        # Ollama Configuration
        'OLLAMA_BASE_URL': 'Ollama server base URL',
        'OLLAMA_REQUEST_TIMEOUT': 'Request timeout for Ollama',
        'OLLAMA_KEEP_ALIVE': 'Keep alive time for Ollama models',
        
        # RAGAS Evaluation Configuration
        'RAGAS_MAX_RETRIES': 'Maximum retries for RAGAS timeout errors',
        'RAGAS_METRIC_TIMEOUT': 'Timeout per RAGAS metric',
        'RAGAS_OVERALL_TIMEOUT': 'Overall RAGAS evaluation timeout',
        'RAGAS_MAX_WORKERS': 'Maximum workers for RAGAS',
        'RAGAS_DISABLE_PARALLEL': 'Disable parallel processing in RAGAS',
        'EVALUATION_CONTEXT': 'Evaluation context flag',
        
        # Neo4j Configuration
        'NEO4J_URI': 'Neo4j database URI',
        'NEO4J_USERNAME': 'Neo4j username',
        'NEO4J_PASSWORD': 'Neo4j password',
    }
    
    print("📋 Configuration Status:")
    print()
    
    # Track loaded and missing variables
    loaded_vars = {}
    missing_vars = []
    
    for var_name, description in expected_vars.items():
        value = os.environ.get(var_name)
        
        if value is not None:
            # Mask sensitive values
            if any(sensitive in var_name.lower() for sensitive in ['password', 'key', 'secret']):
                display_value = '***' if value else 'Not set'
            else:
                display_value = value
            
            loaded_vars[var_name] = display_value
            status = "✅"
        else:
            missing_vars.append(var_name)
            display_value = "Not set"
            status = "❌"
        
        print(f"{status} {var_name:<25} = {display_value}")
        print(f"   └─ {description}")
        print()
    
    # Summary
    print("📊 Summary:")
    print(f"   ✅ Loaded variables: {len(loaded_vars)}")
    print(f"   ❌ Missing variables: {len(missing_vars)}")
    print(f"   📈 Configuration completeness: {len(loaded_vars)}/{len(expected_vars)} ({len(loaded_vars)/len(expected_vars)*100:.1f}%)")
    
    if missing_vars:
        print()
        print("⚠️  Missing variables:")
        for var in missing_vars:
            print(f"   - {var}")
    
    # Additional environment info
    print()
    print("🌍 Additional Environment Info:")
    print(f"   Python version: {sys.version}")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   .env file exists: {os.path.exists('.env')}")
    
    # Check for .env files
    env_files = ['.env', 'env.example', '.env_example']
    print()
    print("📄 Environment files:")
    for env_file in env_files:
        exists = os.path.exists(env_file)
        status = "✅" if exists else "❌"
        print(f"   {status} {env_file}")
    
    return loaded_vars, missing_vars

def test_model_configuration():
    """Test model configuration specifically"""
    
    print("\n" + "=" * 60)
    print("🤖 Model Configuration Test")
    print("=" * 60)
    
    try:
        # Import configuration modules
        from config import get_model_config, ModelProvider
        
        # Get current model config
        config = get_model_config()
        
        print("📋 Current Model Configuration:")
        print(f"   LLM Provider: {getattr(config, 'llm_provider', 'Not set')}")
        print(f"   Embedding Provider: {getattr(config, 'embedding_provider', 'Not set')}")
        print(f"   LLM Model: {getattr(config, 'llm_model', 'Not set')}")
        print(f"   Embedding Model: {getattr(config, 'embedding_model', 'Not set')}")
        print(f"   Temperature: {getattr(config, 'temperature', 'Not set')}")
        print(f"   Seed: {getattr(config, 'seed', 'Not set')}")
        
        # Test model providers
        print("\n🔧 Available Model Providers:")
        for provider in ModelProvider:
            print(f"   - {provider.value}")
        
        print("\n✅ Model configuration loaded successfully")
        
    except Exception as e:
        print(f"\n❌ Error loading model configuration: {e}")
        import traceback
        traceback.print_exc()

def test_neo4j_configuration():
    """Test Neo4j configuration specifically"""
    
    print("\n" + "=" * 60)
    print("🗄️  Neo4j Configuration Test")
    print("=" * 60)
    
    neo4j_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    neo4j_config = {}
    
    for var in neo4j_vars:
        value = os.environ.get(var)
        if var == 'NEO4J_PASSWORD':
            display_value = '***' if value else 'Not set'
        else:
            display_value = value or 'Not set'
        
        neo4j_config[var] = value
        status = "✅" if value else "❌"
        print(f"   {status} {var}: {display_value}")
    
    # Test Neo4j connection if configured
    if all(neo4j_config.values()):
        print("\n🔗 Testing Neo4j connection...")
        try:
            import neo4j
            
            with neo4j.GraphDatabase.driver(
                neo4j_config['NEO4J_URI'], 
                auth=(neo4j_config['NEO4J_USERNAME'], neo4j_config['NEO4J_PASSWORD'])
            ) as driver:
                result = driver.execute_query("RETURN 1 as test", database_="neo4j")
                print("   ✅ Neo4j connection successful")
                
                # Get basic stats
                result = driver.execute_query("MATCH (n) RETURN labels(n) as labels, count(n) as count", database_="neo4j")
                print("\n📊 Database contents:")
                for record in result.records:
                    labels = record['labels']
                    count = record['count']
                    label_str = ':'.join(labels) if labels else 'No labels'
                    print(f"   - {label_str}: {count} nodes")
                
        except Exception as e:
            print(f"   ❌ Neo4j connection failed: {e}")
    else:
        print("\n⚠️  Neo4j configuration incomplete - skipping connection test")

if __name__ == "__main__":
    print("🚀 Starting Environment Variables Test")
    print()
    
    # Run tests
    loaded_vars, missing_vars = test_environment_variables()
    test_model_configuration()
    test_neo4j_configuration()
    
    print("\n" + "=" * 60)
    print("🏁 Environment Variables Test Complete")
    print("=" * 60)
    
    # Exit with appropriate code
    if missing_vars:
        print("⚠️  Some environment variables are missing. Check your .env configuration.")
        sys.exit(1)
    else:
        print("✅ All expected environment variables are configured!")
        sys.exit(0)
