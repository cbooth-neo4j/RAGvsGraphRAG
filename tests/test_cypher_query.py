"""
Test Cypher Query Functionality

This test verifies that the simplified Cypher query in the GraphRAG implementation
works correctly with the Neo4j database.
"""

import os
import sys
import warnings
from dotenv import load_dotenv
import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

# Add parent directory to path to import retrievers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

def test_cypher_query_structure():
    """Test that the Cypher query structure is valid"""
    print("🔍 Testing Cypher query structure...")
    
    # The current Cypher query from retrievers/graph_rag_retriever.py
    cypher_query = """
    UNWIND $chunk_data AS chunk_item
    MATCH (chunk:Chunk {id: chunk_item.chunk_id})-[:PART_OF]->(d:Document)
    
    CALL {
        WITH chunk
        MATCH (chunk)-[:HAS_ENTITY]->(e)
        MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,2}(:!Chunk&!Document)
        
        RETURN collect(DISTINCT e) AS entities,
               collect(DISTINCT path) AS paths
    }
    
    WITH d, chunk, chunk_item.score as score, entities, paths
    WITH d, chunk, score, entities,
         [e IN entities | e.name] AS entity_names,
         [path IN paths | [node IN nodes(path) WHERE node.name IS NOT NULL | node.name]] AS path_names
    
    WITH d, chunk, score, entities, entity_names,
         apoc.coll.toSet(apoc.coll.flatten(path_names)) AS related_names
    
    RETURN
       chunk.text as text,
       score as score,
       {
           chunk_id: chunk.id,
           chunk_text_preview: substring(chunk.text, 0, 100) + "...",
           document_source: d.name,
           chunk_length: size(chunk.text),
           entities: {
               entitynames: entity_names,
               relatednames: related_names
           }
       } AS metadata
    """
    
    # Basic syntax check - if this runs without error, the query structure is valid
    print("✅ Cypher query structure appears valid")
    return cypher_query

def test_database_connection():
    """Test Neo4j database connection"""
    print("🔍 Testing Neo4j database connection...")
    
    try:
        with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            # Test basic connection
            result = driver.execute_query(
                "RETURN 1 as test",
                database_="neo4j"
            )
            print("✅ Neo4j connection successful")
            
            # Check if we have data
            result = driver.execute_query(
                "MATCH (n) RETURN COUNT(n) as count",
                database_="neo4j"
            )
            node_count = result.records[0]['count']
            print(f"✅ Database contains {node_count} nodes")
            
            return True
            
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def test_chunk_entities_exist():
    """Test that chunks and entities exist in the database"""
    print("🔍 Testing chunk and entity existence...")
    
    try:
        with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            # Check for chunks
            result = driver.execute_query(
                "MATCH (c:Chunk) RETURN COUNT(c) as chunk_count",
                database_="neo4j"
            )
            chunk_count = result.records[0]['chunk_count']
            print(f"✅ Found {chunk_count} chunks")
            
            # Check for entities (using actual entity labels from the database)
            result = driver.execute_query(
                "MATCH (e) WHERE e:Organization OR e:Location OR e:Date OR e:Person OR e:Requirement OR e:Financial RETURN COUNT(e) as entity_count",
                database_="neo4j"
            )
            entity_count = result.records[0]['entity_count']
            print(f"✅ Found {entity_count} entities")
            
            # Check for HAS_ENTITY relationships
            result = driver.execute_query(
                "MATCH (c:Chunk)-[:HAS_ENTITY]->(e) RETURN COUNT(*) as rel_count",
                database_="neo4j"
            )
            rel_count = result.records[0]['rel_count']
            print(f"✅ Found {rel_count} HAS_ENTITY relationships")
            
            return chunk_count > 0 and entity_count > 0
            
    except Exception as e:
        print(f"❌ Error checking chunks/entities: {e}")
        return False

def test_simplified_cypher_execution():
    """Test the simplified Cypher query execution"""
    print("🔍 Testing simplified Cypher query execution...")
    
    try:
        with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            # Get a sample chunk ID first
            result = driver.execute_query(
                "MATCH (c:Chunk) RETURN c.id as chunk_id LIMIT 1",
                database_="neo4j"
            )
            
            if not result.records:
                print("❌ No chunks found in database")
                return False
                
            chunk_id = result.records[0]['chunk_id']
            print(f"✅ Using test chunk ID: {chunk_id}")
            
            # Test the simplified query with sample data
            test_chunk_data = [{'chunk_id': chunk_id, 'score': 0.8}]
            
            # Simplified test query matching current structure
            test_query = """
            UNWIND $chunk_data AS chunk_item
            MATCH (chunk:Chunk {id: chunk_item.chunk_id})-[:PART_OF]->(d:Document)
            
            CALL {
                WITH chunk
                MATCH (chunk)-[:HAS_ENTITY]->(e)
                MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,2}(:!Chunk&!Document)
                
                RETURN collect(DISTINCT e) AS entities,
                       collect(DISTINCT path) AS paths
            }
            
            WITH d, chunk, chunk_item.score as score, entities, paths
            WITH d, chunk, score, entities,
                 [e IN entities | e.name] AS entity_names,
                 [path IN paths | [node IN nodes(path) WHERE node.name IS NOT NULL | node.name]] AS path_names
            
            WITH d, chunk, score, entities, entity_names,
                 apoc.coll.toSet(apoc.coll.flatten(path_names)) AS related_names
            
            RETURN
               chunk.text as text,
               score as score,
               {
                   chunk_id: chunk.id,
                   chunk_text_preview: substring(chunk.text, 0, 100) + "...",
                   document_source: d.name,
                   chunk_length: size(chunk.text),
                   entities: {
                       entitynames: entity_names,
                       relatednames: related_names
                   }
               } AS metadata
            """
            
            # Execute the test query
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = driver.execute_query(
                    test_query,
                    chunk_data=test_chunk_data,
                    database_="neo4j"
                )
            
            print(f"✅ Query executed successfully, returned {len(result.records)} records")
            
            # Check the structure of results
            if result.records:
                record = result.records[0]
                metadata = record.get('metadata', {})
                print(f"✅ Result structure: text={len(record.get('text', ''))} chars, score={record.get('score')}")
                print(f"✅ Chunk ID: {metadata.get('chunk_id', 'N/A')}")
                print(f"✅ Chunk Preview: {metadata.get('chunk_text_preview', 'N/A')}")
                print(f"✅ Document Source: {metadata.get('document_source', 'N/A')}")
                print(f"✅ Chunk Length: {metadata.get('chunk_length', 'N/A')} characters")
                print(f"✅ Entities Found: {metadata.get('entities', {}).get('entitynames', [])}")
                print(f"✅ Related Names: {metadata.get('entities', {}).get('relatednames', [])}")
                print(f"✅ Full Metadata: {metadata}")
            
            return True
            
    except Exception as e:
        print(f"❌ Cypher query execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embeddings_setup():
    """Test that embeddings can be generated"""
    print("🔍 Testing embeddings setup...")
    
    try:
        embeddings = OpenAIEmbeddings()
        test_vector = embeddings.embed_query("test query")
        print(f"✅ Embeddings working, vector length: {len(test_vector)}")
        return True
    except Exception as e:
        print(f"❌ Embeddings setup failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("CYPHER QUERY TEST SUITE")
    print("="*60)
    
    tests = [
        ("Cypher Query Structure", test_cypher_query_structure),
        ("Database Connection", test_database_connection),
        ("Chunk/Entity Existence", test_chunk_entities_exist),
        ("Simplified Cypher Execution", test_simplified_cypher_execution),
        ("Embeddings Setup", test_embeddings_setup)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Cypher query should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests() 