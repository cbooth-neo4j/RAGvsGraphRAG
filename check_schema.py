import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

def check_schema():
    """Check the actual Neo4j database schema"""
    try:
        # Create Neo4j graph connection
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            enhanced_schema=True
        )
        
        # Refresh and print schema
        graph.refresh_schema()
        print("="*60)
        print("ACTUAL NEO4J DATABASE SCHEMA")
        print("="*60)
        print(graph.schema)
        print("="*60)
        
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")

if __name__ == "__main__":
    check_schema() 