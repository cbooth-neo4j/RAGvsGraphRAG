"""
Neo4j Vector-Only Retriever

This retriever performs pure vector similarity search using Neo4j's vector index
without any graph traversal or entity processing. It provides a baseline vector
search approach for comparison with GraphRAG methods.


"""

import os
import warnings
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Neo4j and GraphRAG imports
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

# Import centralized configuration
from config import get_model_config, get_neo4j_embeddings, get_neo4j_llm, ModelProvider

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
NEO4J_DB = os.environ.get('CLIENT_NEO4J_DATABASE')

INDEX_NAME = "chunk_embedding"

# Initialize embeddings and LLM using centralized configuration
SEED = 42
embeddings = get_neo4j_embeddings()
llm = get_neo4j_llm()


class Neo4jVectorRetriever:
    """Neo4j Vector-Only Retriever for pure vector similarity search"""
    
    def __init__(self, 
                 driver=None,
                 index_name: str = INDEX_NAME,
                 embedder=None,
                 llm_model=None):
        """
        Initialize the Neo4j Vector Retriever
        
        Args:
            driver: Neo4j driver instance (optional, will create if not provided)
            index_name: Name of the vector index to use
            embedder: Embedding model (optional, will use default if not provided)
            llm_model: LLM model (optional, will use default if not provided)
        """
        self.driver = driver or GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DB)
        self.index_name = index_name
        self.embedder = embedder or embeddings
        self.llm_model = llm_model or llm
        
        # Initialize the vector retriever
        self.vector_retriever = VectorRetriever(
            driver=self.driver,
            index_name=self.index_name,
            embedder=self.embedder
        )
        
        # Initialize the RAG pipeline
        self.rag_pipeline = GraphRAG(
            retriever=self.vector_retriever,
            llm=self.llm_model
        )
    
    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Perform vector search and generate response
        
        Args:
            query: Search query
            top_k: Number of top results to retrieve
            
        Returns:
            Dictionary with response and retrieval details
        """
        try:
            print(f"🔍 Executing Neo4j Vector search for: {query}")
            
            # Perform vector search with RAG pipeline
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                response = self.rag_pipeline.search(
                    query_text=query, 
                    retriever_config={"top_k": top_k}
                )
            
            # Get the vector search results for metadata
            vector_results = self.vector_retriever.search(query_text=query, top_k=top_k)
            
            # Format retrieval details
            retrieval_details = []
            if hasattr(vector_results, 'items') and vector_results.items:
                for i, item in enumerate(vector_results.items, 1):
                    # Extract content and metadata from vector result
                    if hasattr(item, 'content'):
                        content = item.content
                        score = getattr(item, 'score', 0.0)
                        
                        # Parse content if it's a string representation
                        if isinstance(content, str):
                            content_text = content
                        else:
                            # Handle structured content
                            content_text = str(content)
                        
                        retrieval_details.append({
                            'content': content_text,
                            'score': score,
                            'source': 'Neo4j Vector Search',
                            'rank': i,
                            'metadata': {
                                'search_type': 'vector_similarity',
                                'index_used': self.index_name
                            }
                        })
            
            print(f"✅ Neo4j Vector search completed: {len(retrieval_details)} results")
            
            # Log detailed results
            for i, detail in enumerate(retrieval_details, 1):
                print(f"  📄 Result {i}:")
                print(f"    - Score: {detail.get('score', 'N/A')}")
                print(f"    - Content preview: {detail.get('content', '')[:100]}...")
            
            return {
                'method': 'Neo4j Vector Search + LLM',
                'query': query,
                'final_answer': response.answer if hasattr(response, 'answer') else str(response),
                'retrieved_chunks': len(retrieval_details),
                'retrieval_details': retrieval_details,
                'performance_metrics': {
                    'search_type': 'pure_vector',
                    'index_used': self.index_name,
                    'retrieved_chunks': len(retrieval_details)
                }
            }
            
        except Exception as e:
            print(f"❌ Error in Neo4j Vector search: {e}")
            import traceback
            traceback.print_exc()
            return {
                'method': 'Neo4j Vector Search + LLM',
                'query': query,
                'final_answer': f"Error during Neo4j Vector search: {e}",
                'retrieved_chunks': 0,
                'retrieval_details': [],
                'performance_metrics': {
                    'search_type': 'pure_vector',
                    'error': str(e)
                }
            }
    
    def close(self):
        """Close the Neo4j driver connection"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()


def create_neo4j_vector_retriever(driver=None, index_name: str = INDEX_NAME, **kwargs) -> Neo4jVectorRetriever:
    """
    Factory function to create a Neo4j Vector Retriever
    
    Args:
        driver: Optional Neo4j driver instance
        index_name: Vector index name to use
        **kwargs: Additional arguments for retriever configuration
        
    Returns:
        Neo4jVectorRetriever instance
    """
    return Neo4jVectorRetriever(
        driver=driver,
        index_name=index_name,
        **kwargs
    )


def query_neo4j_vector_rag(query: str, k: int = 5, **kwargs) -> Dict[str, Any]:
    """
    Main interface function for Neo4j Vector RAG queries
    
    Args:
        query: The search query
        k: Number of top results to retrieve
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary with response and retrieval details
    """
    retriever = None
    try:
        # Create retriever instance
        retriever = create_neo4j_vector_retriever(**kwargs)
        
        # Perform search
        result = retriever.search(query, top_k=k)
        
        return result
        
    except Exception as e:
        return {
            'method': 'Neo4j Vector Search + LLM',
            'query': query,
            'final_answer': f"Error initializing Neo4j Vector retriever: {e}",
            'retrieved_chunks': 0,
            'retrieval_details': [],
            'performance_metrics': {
                'search_type': 'pure_vector',
                'error': str(e)
            }
        }
    finally:
        if retriever:
            retriever.close()





if __name__ == "__main__":
    # Test the retriever
    test_query = "What are the main vendor requirements?"
    print(f"Testing Neo4j Vector Retriever with query: '{test_query}'")
    result = query_neo4j_vector_rag(test_query, k=3)
    print(f"Result: {result.get('method', 'Unknown')}")
    print(f"Answer: {result.get('final_answer', 'No answer')[:200]}...") 