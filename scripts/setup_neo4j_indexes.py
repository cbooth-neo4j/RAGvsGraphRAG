#!/usr/bin/env python3
"""
Neo4j Index Setup Script

This script creates the required indexes for the RAG vs GraphRAG benchmark system.
It creates both vector and fulltext indexes needed by the various retrievers.
"""

import os
import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from neo4j import GraphDatabase
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv(override=True)

# Import model configuration to get embedding dimensions
from config.model_config import get_model_config

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

def setup_neo4j_indexes():
    """Create all required Neo4j indexes for the benchmark system"""
    
    # Get embedding dimensions from model configuration
    model_config = get_model_config()
    embedding_dim = model_config.embedding_dimensions
    
    print(f"📐 Detected embedding dimensions: {embedding_dim}")
    print(f"   Provider: {model_config.embedding_provider.value}")
    print(f"   Model: {model_config.embedding_model.value}")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            print("🔧 Setting up Neo4j indexes...")
            
            # Drop existing vector indexes if they exist (to handle dimension changes)
            print("  🗑️  Dropping existing vector indexes...")
            drop_queries = [
                "DROP INDEX entity_embeddings IF EXISTS",
                "DROP INDEX chunk_embeddings IF EXISTS",
                "DROP INDEX document_embeddings IF EXISTS",
                "DROP INDEX embedding IF EXISTS",  # Old unified index name
                "DROP INDEX entity_embeddings_old IF EXISTS"  # Legacy index
            ]
            
            for query in drop_queries:
                try:
                    session.run(query)
                except Exception as e:
                    print(f"    ⚠️  {e}")
            
            # Create fulltext indexes
            print("  📝 Creating fulltext indexes...")
            fulltext_queries = [
                """
                CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS
                FOR (e:__Entity__) ON EACH [e.name, e.description]
                OPTIONS {
                  indexConfig: {
                    `fulltext.analyzer`: 'standard-no-stop-words',
                    `fulltext.eventually_consistent`: true
                  }
                }
                """,
                """
                CREATE FULLTEXT INDEX chunk_fulltext_idx IF NOT EXISTS  
                FOR (c:Chunk) ON EACH [c.text, c.content]
                OPTIONS {
                  indexConfig: {
                    `fulltext.analyzer`: 'standard-no-stop-words',
                    `fulltext.eventually_consistent`: true
                  }
                }
                """
            ]
            
            for query in fulltext_queries:
                try:
                    session.run(query)
                    print(f"    ✅ Created fulltext index")
                except Exception as e:
                    print(f"    ❌ Error creating fulltext index: {e}")
            
            # Check if we have entities with embeddings before creating vector indexes
            entity_count = session.run("""
                MATCH (e:__Entity__) 
                WHERE e.embedding IS NOT NULL 
                RETURN count(e) as count
            """).single()
            
            chunk_count = session.run("""
                MATCH (c:Chunk) 
                WHERE c.embedding IS NOT NULL 
                RETURN count(c) as count
            """).single()
            
            # Create separate vector indexes for each node type
            print(f"  🎯 Creating vector indexes with {embedding_dim} dimensions...")
            
            if entity_count and entity_count['count'] > 0:
                print(f"  🔢 Found {entity_count['count']} entities with embeddings")
                
                # Entity embeddings index
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                        FOR (e:__Entity__) ON e.embedding
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
                    print(f"    ✅ Created entity_embeddings index ({embedding_dim} dimensions)")
                except Exception as e:
                    print(f"    ❌ Error creating entity_embeddings index: {e}")
            else:
                print("  ⚠️  No entities with embeddings found, skipping entity vector index")
            
            if chunk_count and chunk_count['count'] > 0:
                print(f"  🔢 Found {chunk_count['count']} chunks with embeddings")
                
                # Chunk embeddings index
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                        FOR (c:Chunk) ON c.embedding
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
                    print(f"    ✅ Created chunk_embeddings index ({embedding_dim} dimensions)")
                except Exception as e:
                    print(f"    ❌ Error creating chunk_embeddings index: {e}")
            else:
                print("  ⚠️  No chunks with embeddings found, skipping chunk vector index")
            
            # Document embeddings index (optional, if documents have embeddings)
            doc_count = session.run("""
                MATCH (d:Document) 
                WHERE d.embedding IS NOT NULL 
                RETURN count(d) as count
            """).single()
            
            if doc_count and doc_count['count'] > 0:
                print(f"  🔢 Found {doc_count['count']} documents with embeddings")
                
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
                        FOR (d:Document) ON d.embedding
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
                    print(f"    ✅ Created document_embeddings index ({embedding_dim} dimensions)")
                except Exception as e:
                    print(f"    ❌ Error creating document_embeddings index: {e}")
            
            # Show current indexes
            print("\n📊 Current indexes:")
            result = session.run("SHOW INDEXES")
            for record in result:
                index_name = record.get('name', 'N/A')
                index_type = record.get('type', 'N/A') 
                state = record.get('state', 'N/A')
                print(f"  • {index_name} ({index_type}) - {state}")
                
        print("\n✅ Neo4j index setup complete!")
        
    except Exception as e:
        print(f"❌ Error setting up indexes: {e}")
        sys.exit(1)
    finally:
        driver.close()

if __name__ == "__main__":
    setup_neo4j_indexes()
