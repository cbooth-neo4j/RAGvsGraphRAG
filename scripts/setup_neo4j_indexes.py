#!/usr/bin/env python3
"""
Neo4j Index Setup Script

This script creates the required indexes for the RAG vs GraphRAG benchmark system.
It creates both vector and fulltext indexes needed by the various retrievers.
"""

import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv(override=True)

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

def setup_neo4j_indexes():
    """Create all required Neo4j indexes for the benchmark system"""
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            print("🔧 Setting up Neo4j indexes...")
            
            # Drop existing indexes if they exist (to handle schema changes)
            print("  🗑️  Dropping existing indexes...")
            drop_queries = [
                "DROP INDEX entity_fulltext_idx IF EXISTS",
                "DROP INDEX embedding IF EXISTS", 
                "DROP INDEX chunk_fulltext_idx IF EXISTS"
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
            
            if entity_count and entity_count['count'] > 0:
                print(f"  🔢 Found {entity_count['count']} entities with embeddings")
                print("  🎯 Creating vector indexes...")
                
                # Create vector indexes with appropriate dimensions
                vector_queries = [
                    """
                    CREATE VECTOR INDEX embedding IF NOT EXISTS
                    FOR (n:__Entity__|Document|Chunk) ON n.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                    """,
                ]
                
                for query in vector_queries:
                    try:
                        session.run(query)
                        print(f"    ✅ Created unified embedding vector index")
                    except Exception as e:
                        print(f"    ❌ Error creating entity vector index: {e}")
            else:
                print("  ⚠️  No entities with embeddings found, skipping entity vector index")
            
            if chunk_count and chunk_count['count'] > 0:
                print(f"  🔢 Found {chunk_count['count']} chunks with embeddings")
                
                # Note: Using single unified embedding index instead of separate chunk index
                print("    ✅ Using unified embedding index for all node types")
            else:
                print("  ⚠️  No chunks with embeddings found, skipping chunk vector index")
            
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
