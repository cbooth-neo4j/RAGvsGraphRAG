#!/usr/bin/env python3
"""
Configuration Verification Script

This script helps verify your embedding configuration and checks for dimension
mismatches between your configuration and Neo4j indexes.
"""

import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load environment variables
load_dotenv(override=True)

from config.model_config import get_model_config

def check_neo4j_connection():
    """Check if Neo4j is accessible"""
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not neo4j_password:
        print("❌ NEO4J_PASSWORD not set in .env")
        return None
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.verify_connectivity()
        print(f"✅ Neo4j connection successful: {neo4j_uri}")
        return driver
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return None

def check_vector_indexes(driver):
    """Check Neo4j vector indexes and their dimensions"""
    try:
        with driver.session() as session:
            result = session.run("SHOW INDEXES WHERE type = 'VECTOR'")
            indexes = list(result)
            
            if not indexes:
                print("⚠️  No vector indexes found in Neo4j")
                print("   Run: python scripts/setup_neo4j_indexes.py")
                return []
            
            print("\n📊 Vector Indexes in Neo4j:")
            index_info = []
            expected_indexes = {'chunk_embeddings', 'entity_embeddings', 'document_embeddings'}
            found_indexes = set()
            
            for record in indexes:
                name = record.get('name', 'N/A')
                state = record.get('state', 'N/A')
                labels = record.get('labelsOrTypes', [])
                options = record.get('options', {})
                
                dimensions = None
                if options:
                    # Try different ways to extract dimensions
                    if isinstance(options, dict):
                        index_config = options.get('indexConfig', {})
                        if isinstance(index_config, dict):
                            dimensions = index_config.get('vector.dimensions')
                    
                    # If still None, try parsing from the options string representation
                    if dimensions is None:
                        options_str = str(options)
                        if 'vector.dimensions' in options_str:
                            import re
                            match = re.search(r'vector\.dimensions[\'"]?\s*:\s*(\d+)', options_str)
                            if match:
                                dimensions = match.group(1)
                
                print(f"  • {name} (on {labels})")
                print(f"    State: {state}")
                if dimensions:
                    print(f"    Dimensions: {dimensions}")
                    index_info.append({'name': name, 'dimensions': int(dimensions), 'labels': labels})
                else:
                    print(f"    Dimensions: Unable to determine")
                
                if name in expected_indexes:
                    found_indexes.add(name)
            
            # Check for missing expected indexes
            missing = expected_indexes - found_indexes
            if missing:
                print(f"\n  ⚠️  Missing expected indexes: {', '.join(missing)}")
                print(f"     Run: python scripts/setup_neo4j_indexes.py")
            
            return index_info
    except Exception as e:
        print(f"❌ Error checking indexes: {e}")
        return []

def verify_configuration():
    """Main verification function"""
    print("=" * 60)
    print("🔍 RAG vs GraphRAG Configuration Verification")
    print("=" * 60)
    
    # Check model configuration
    print("\n1️⃣  Checking Model Configuration...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("\n   Create .env file from template:")
        print("   1. Copy .env_example to .env:")
        print("      cp .env_example .env")
        print("   2. Edit .env and configure your settings")
        print("   3. Run this script again\n")
        return
    
    try:
        config = get_model_config()
        
        print(f"   LLM Provider: {config.llm_provider.value}")
        print(f"   LLM Model: {config.llm_model.value}")
        print(f"   Embedding Provider: {config.embedding_provider.value}")
        print(f"   Embedding Model: {config.embedding_model.value}")
        print(f"   Embedding Dimensions: {config.embedding_dimensions}")
        
        # Check if manual override is set
        if os.getenv('EMBEDDING_DIMENSION'):
            print(f"   ⚠️  Manual dimension override: {os.getenv('EMBEDDING_DIMENSION')}")
        
        config_dims = config.embedding_dimensions
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("\n   Please check your .env file configuration:")
        print("   - LLM_PROVIDER should be: openai, ollama, or vertexai")
        print("   - EMBEDDING_PROVIDER should be: openai, ollama, or vertexai")
        print("   - Make sure all required values are set")
        print("\n   See .env_example for reference\n")
        return
    
    # Check Neo4j connection
    print("\n2️⃣  Checking Neo4j Connection...")
    driver = check_neo4j_connection()
    
    if not driver:
        print("\n❌ Cannot verify indexes without Neo4j connection")
        print("   Please configure Neo4j credentials in .env")
        return
    
    # Check vector indexes
    print("\n3️⃣  Checking Vector Indexes...")
    indexes = check_vector_indexes(driver)
    
    # Verify dimension compatibility
    print("\n4️⃣  Dimension Compatibility Check...")
    if not indexes:
        print("⚠️  No vector indexes to verify")
        print("   This is normal if you haven't ingested data yet")
        print("   Run: python scripts/setup_neo4j_indexes.py")
    else:
        all_match = True
        for index in indexes:
            if index['dimensions'] == config_dims:
                print(f"✅ {index['name']}: {index['dimensions']} dims (matches config)")
            else:
                print(f"❌ {index['name']}: {index['dimensions']} dims (config expects {config_dims})")
                all_match = False
        
        if all_match:
            print("\n✅ All indexes match your embedding configuration!")
        else:
            print("\n❌ DIMENSION MISMATCH DETECTED!")
            print("\n   Solutions:")
            print("   A) Update indexes to match your config:")
            print("      python scripts/setup_neo4j_indexes.py")
            print("\n   B) Change your embedding model in .env to match indexes")
            print(f"      (Set EMBEDDING_DIMENSION={indexes[0]['dimensions']} or change EMBEDDING_MODEL)")
            print("\n   C) Re-ingest data with your current embedding model")
            print("      python ingest.py --source pdf --quantity 10 --lean")
    
    # Check for common data
    print("\n5️⃣  Checking Data in Neo4j...")
    try:
        with driver.session() as session:
            # Check for chunks
            chunk_result = session.run("""
                MATCH (c:Chunk)
                WHERE c.embedding IS NOT NULL
                RETURN count(c) as count
            """)
            chunk_count = chunk_result.single()['count']
            
            # Check for entities
            entity_result = session.run("""
                MATCH (e:__Entity__)
                WHERE e.embedding IS NOT NULL
                RETURN count(e) as count
            """)
            entity_count = entity_result.single()['count']
            
            print(f"   Chunks with embeddings: {chunk_count}")
            print(f"   Entities with embeddings: {entity_count}")
            
            if chunk_count == 0 and entity_count == 0:
                print("\n⚠️  No data found in Neo4j")
                print("   Run data ingestion:")
                print("   python ingest.py --source pdf --quantity 10 --lean")
            else:
                print("\n✅ Data found in Neo4j")
                
    except Exception as e:
        print(f"⚠️  Could not check data: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    print(f"Configuration: {config.embedding_model.value} ({config_dims} dims)")
    if indexes:
        if all_match:
            print("Status: ✅ Ready to run benchmarks")
        else:
            print("Status: ❌ Dimension mismatch - fix required")
    else:
        print("Status: ⚠️  No indexes yet - run setup_neo4j_indexes.py")
    
    print("\n📖 For more information:")
    print("   docs/EMBEDDING_DIMENSIONS.md")
    print("=" * 60)
    
    driver.close()

if __name__ == "__main__":
    verify_configuration()

