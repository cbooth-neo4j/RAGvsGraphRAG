"""
GraphRAG Retriever - Graph-Enhanced Vector Search

This module implements basic GraphRAG using Neo4j with vector search
and entity relationship traversal for enhanced context.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any
import neo4j
from neo4j_graphrag.retrievers import VectorRetriever
import warnings
import json
import ast


# Import centralized configuration
from config import get_model_config, get_neo4j_embeddings, get_neo4j_llm, ModelProvider
from utils.graph_rag_logger import setup_logging, get_logger

# Load environment variables
load_dotenv()

setup_logging()
logger = get_logger(__name__)

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
NEO4J_DATABASE = os.environ.get('CLIENT_NEO4J_DATABASE')

INDEX_NAME = "chunk_embeddings"  # Match the actual chunk vector index in Neo4j

# Initialize components with centralized configuration
SEED = 42
# Note: embeddings and llm are initialized in the class to avoid module-level initialization issues

class GraphRAGRetriever:
    """GraphRAG retriever with Neo4j vector search and entity traversal"""
    
    def __init__(self):
        from config.model_config import get_model_config
        
        self.embeddings = get_neo4j_embeddings()
        self.llm = get_neo4j_llm()
        self.neo4j_uri = NEO4J_URI
        self.neo4j_user = NEO4J_USER
        self.neo4j_password = NEO4J_PASSWORD
        self.neo4j_db = NEO4J_DATABASE
        self.index_name = INDEX_NAME
        
        # Get expected embedding dimensions
        model_config = get_model_config()
        self.embedding_dim = model_config.embedding_dimensions
        
        logger.info(f'GraphRAGRetriever initialized:')
        logger.info(f'  - Index: {INDEX_NAME}')
        logger.info(f'  - Database: {self.neo4j_db}')
        logger.info(f'  - Embedding Provider: {model_config.embedding_provider.value}')
        logger.info(f'  - Embedding Model: {model_config.embedding_model.value}')
        logger.info(f'  - Expected Dimensions: {self.embedding_dim}')
        
        # Validate dimensions match what's in Neo4j
        self._validate_dimensions()
    
    def _validate_dimensions(self):
        """Validate that embedding dimensions match Neo4j vector index configuration"""
        try:
            with neo4j.GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password), database=self.neo4j_db) as driver:
                result = driver.execute_query(
                    "SHOW INDEXES WHERE name = $index_name",
                    index_name=self.index_name,
                    database_=self.neo4j_db
                )
                
                if result.records:
                    index_info = result.records[0]
                    # Try to extract dimensions from index options
                    options = index_info.get('options', {})
                    if options and 'indexConfig' in options:
                        index_dim = options['indexConfig'].get('vector.dimensions')
                        if index_dim and int(index_dim) != self.embedding_dim:
                            logger.error(f"❌ DIMENSION MISMATCH!")
                            logger.error(f"   Neo4j index '{self.index_name}' has {index_dim} dimensions")
                            logger.error(f"   But embedding model produces {self.embedding_dim} dimensions")
                            logger.error(f"   Fix: Run 'python scripts/setup_neo4j_indexes.py' to recreate indexes")
                            print(f"⚠️  WARNING: Dimension mismatch detected!")
                            print(f"   Neo4j: {index_dim} dims, Model: {self.embedding_dim} dims")
                        else:
                            logger.info(f"✅ Dimension validation passed: {self.embedding_dim} dimensions")
                else:
                    logger.warning(f"⚠️  Index '{self.index_name}' not found in Neo4j")
                    print(f"⚠️  Vector index '{self.index_name}' not found. Run 'python scripts/setup_neo4j_indexes.py'")
        except Exception as e:
            logger.warning(f"Could not validate dimensions: {e}")
    
    def search(self, query: str, k: int = 3, answer_style: str = "ragas") -> Dict[str, Any]:
        """Neo4j GraphRAG query with LLM response generation
        
        Args:
            query: The search query
            k: Number of chunks to retrieve
            answer_style: Response format - "hotpotqa" for short exact answers, "ragas" for verbose answers
        """
        self._answer_style = answer_style
        
        with neo4j.GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password), database=self.neo4j_db) as driver:
            print(f"🔍 Executing GraphRAG query for: {query}")
            logger.info(f"🔍 Executing GraphRAG query for: {query}")

            # Query parameters for the simplified pattern (we don't actually use query_vector in our Cypher)
            query_params = {}
            
            try:
                # First perform vector search to get candidate chunks with fallback
                try:
                    # Try vector search first
                    vector_retriever = VectorRetriever(
                        driver=driver,
                        index_name=self.index_name,
                        embedder=self.embeddings
                    )
                    
                    # Get top chunks from vector search
                    vector_results = vector_retriever.search(query_text=query, top_k=5)
                    
                except Exception as vector_error:
                    import traceback
                    logger.error(f'Failed to search via vector retrieve: {traceback.print_exc()}')
                    print(f"⚠️ Vector search failed: {vector_error}")
                    print(f"🔄 Falling back to direct chunk retrieval...")
                    
                    # Fallback: get chunks directly without vector search
                    fallback_query = """
                    MATCH (chunk:Chunk)-[:PART_OF]->(d:Document)
                    RETURN chunk.text as text, chunk.id as chunk_id, d.name as document_source
                    LIMIT 5
                    """
                    
                    fallback_result = driver.execute_query(fallback_query, database_= self.neo4j_db)
                    
                    # Create a mock vector_results structure
                    class MockResult:
                        def __init__(self, content, chunk_id):
                            self.content = {'id': chunk_id, 'text': content}
                            self.score = 0.8  # Default score
                    
                    class MockResults:
                        def __init__(self, items):
                            self.items = items
                    
                    mock_items = []
                    for record in fallback_result.records[:5]:
                        content = record.get('text', '')
                        chunk_id = record.get('chunk_id', '')
                        mock_items.append(MockResult(content, chunk_id))
                    
                    vector_results = MockResults(mock_items) 
                logger.debug(f'Vector Results: {vector_results}')
                if not vector_results:
                    return {
                        'method': 'GraphRAG + LLM',
                        'query': query,
                        'final_answer': 'No results found in the knowledge base.',
                        'retrieved_chunks': 0,
                        'retrieval_details': []
                    }
                
                # Extract chunk nodes and scores from vector results
                chunks_with_scores = []
                
                for result in vector_results.items[:3]:  # Use top 3 chunks with safe entity traversal
                    # Each item is a RetrieverResultItem with content and metadata
                    if hasattr(result, 'content'):
                        content = result.content
                        # Parse the content to get chunk info and score  
                        try:
                            # Content might be a JSON string or dict-like string
                            if isinstance(content, str) and content.startswith('{'):
                                try:
                                    # First try with ast.literal_eval for single quotes (safer than eval)
                                    chunk_data = ast.literal_eval(content)
                                    chunk_id = chunk_data.get('id', '')
                                    score = getattr(result, 'score', 0.8)  # Get score from result object
                                    chunks_with_scores.append((chunk_id, score))
                                except:
                                    try:
                                        # If that fails, try regular JSON parsing
                                        chunk_data = json.loads(content)
                                        chunk_id = chunk_data.get('id', '')
                                        score = getattr(result, 'score', 0.8)
                                        chunks_with_scores.append((chunk_id, score))
                                    except:
                                        # Extract ID using regex as fallback
                                        import re
                                        match = re.search(r"'id':\s*'([^']+)'", content)
                                        if match:
                                            chunk_id = match.group(1)
                                            score = getattr(result, 'score', 0.8)
                                            chunks_with_scores.append((chunk_id, score))
                                        else:
                                            raise Exception("Could not extract chunk ID")
                            elif isinstance(content, dict):
                                # Content is already a dictionary
                                chunk_id = content.get('id', '')
                                score = getattr(result, 'score', 0.8)
                                chunks_with_scores.append((chunk_id, score))
                            else:
                                # Content might be the chunk data directly
                                chunk_id = getattr(content, 'id', str(content)[:50])
                                score = getattr(result, 'score', 0.8)
                                chunks_with_scores.append((chunk_id, score))
                        except Exception as e:
                            # Fallback parsing if all parsing fails
                            chunk_id = str(content)[:50]  # Limit length
                            score = getattr(result, 'score', 0.8)
                            chunks_with_scores.append((chunk_id, score))
                
                if not chunks_with_scores:
                    return {
                        'method': 'GraphRAG + LLM',
                        'query': query,
                        'final_answer': 'Error processing search results.',
                        'retrieved_chunks': 0,
                        'retrieval_details': []
                    }
                    
                # Build Cypher query to find these specific chunks and apply complex logic
                chunk_ids = [chunk_id for chunk_id, _ in chunks_with_scores]
                chunk_scores = {chunk_id: score for chunk_id, score in chunks_with_scores}
                
                # Simplified Cypher query that avoids complex path matching
                cypher_query = """
                UNWIND $chunk_data AS chunk_item
                MATCH (chunk:Chunk {id: chunk_item.chunk_id})-[:PART_OF]->(d:Document)
                
                // Get entities directly connected to this chunk
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
                WITH d, chunk, chunk_item.score as score, collect(DISTINCT e.name) as entity_names
                
                // Get related entities through simple one-hop relationships (limit to avoid performance issues)
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)-[r]-(related)
                WHERE NOT related:Chunk AND NOT related:Document
                WITH d, chunk, score, entity_names, 
                     collect(DISTINCT related.name)[0..10] as related_names  // Limit to 10 to avoid performance issues
                
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
                
                # Prepare chunk data for the query
                chunk_data = [{'chunk_id': cid, 'score': chunk_scores[cid]} for cid in chunk_ids]
                
                # Execute with parameters (suppress warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = driver.execute_query(
                        cypher_query,
                        chunk_data=chunk_data,
                        **query_params,
                        database_= self.neo4j_db
                    )
                
                # Format results for LLM processing
                enhanced_chunks = []
                retrieval_details = []
                
                for record in result.records:
                    content = record.get('text', '')
                    score = record.get('score', 0.0)
                    metadata = record.get('metadata', {})
                    
                    enhanced_chunks.append(content)
                    retrieval_details.append({
                        'content': content,
                        'score': score,
                        'metadata': metadata
                    })
                
                print(f"✅ GraphRAG retrieved {len(enhanced_chunks)} chunks")
                
                # Log detailed chunk information
                for i, detail in enumerate(retrieval_details, 1):
                    metadata = detail.get('metadata', {})
                    print(f"  📄 Chunk {i}:")
                    print(f"    - ID: {metadata.get('chunk_id', 'N/A')}")
                    print(f"    - Preview: {metadata.get('chunk_text_preview', 'N/A')}")
                    print(f"    - Document: {metadata.get('document_source', 'N/A')}")
                    print(f"    - Score: {detail.get('score', 'N/A')}")
                    print(f"    - Entities: {metadata.get('entities', {}).get('entitynames', [])}")
                
                if not enhanced_chunks:
                    return {
                        'method': 'GraphRAG + LLM',
                        'query': query,
                        'final_answer': 'No results found after entity processing.',
                        'retrieved_chunks': 0,
                        'retrieval_details': []
                    }
                
                # Prepare context for LLM with retrieved chunks
                context_parts = []
                for i, chunk in enumerate(enhanced_chunks, 1):
                    context_parts.append(f"Document {i}:\n{chunk}")
                
                context = "\n\n".join(context_parts)
                
                # Generate LLM response with GraphRAG context - use answer_style for prompt
                if getattr(self, '_answer_style', 'ragas') == "hotpotqa":
                    # HotpotQA benchmark requires short, exact answers
                    prompt = f"""Answer the question using ONLY the retrieved documents below.

Question: {query}

Retrieved Documents:
{context}

CRITICAL INSTRUCTIONS FOR HOTPOTQA BENCHMARK:
- For yes/no questions: Answer ONLY "yes" or "no" (lowercase, no punctuation)
- For entity questions: Answer ONLY with the entity name (e.g., "Scott Derrickson", "1923", "United States")
- For comparison questions asking "same" or "both": Answer "yes" or "no"
- NO explanations, NO reasoning, NO sentences - just the bare answer
- If you cannot determine the answer from the documents, respond with "unknown"

Answer:"""
                else:
                    # RAGAS/real-world mode: verbose, comprehensive answers
                    prompt = f"""Based on the following retrieved documents, provide a factual answer to the question.

Question: {query}

Retrieved Documents:
{context}

Instructions:
1. Base your answer strictly on the information in the retrieved documents
2. You may combine information from multiple documents if they are related
3. If the documents don't contain enough information to answer the question completely, state this clearly
4. Do not make inferences beyond what is explicitly stated in the documents
5. When combining information from multiple documents, ensure accuracy and avoid mixing unrelated facts

Please provide a factual, well-structured response."""

                try:
                    llm_response = self.llm.invoke(prompt)
                    final_answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                except Exception as e:
                    final_answer = f"Error generating LLM response: {e}"
                
                return {
                    'method': 'GraphRAG + LLM',
                    'query': query,
                    'final_answer': final_answer,
                    'retrieved_chunks': len(enhanced_chunks),
                    'retrieval_details': retrieval_details
                }
                
            except Exception as e:
                return {
                    'method': 'GraphRAG + LLM',
                    'query': query,
                    'final_answer': f"Error with GraphRAG processing: {e}",
                    'retrieved_chunks': 0,
                    'retrieval_details': []
                }


# Factory function for easy instantiation
def create_graphrag_retriever() -> GraphRAGRetriever:
    """Create a GraphRAG retriever instance"""
    return GraphRAGRetriever()




# Main interface function for integration with benchmark system
def query_graphrag(query: str, k: int = 3, answer_style: str = "ragas", **kwargs) -> Dict[str, Any]:
    """GraphRAG retrieval with LLM response generation
    
    Args:
        query: The search query
        k: Number of chunks to retrieve
        answer_style: Response format - "hotpotqa" for short exact answers, "ragas" for verbose answers
        **kwargs: Additional configuration options (for compatibility)
    
    Returns:
        Dictionary with response and retrieval details
    """
    logger.debug(f'In query_graphrag with top-k: {k} & query: {query}')
    try:
        retriever = create_graphrag_retriever()
        result = retriever.search(query, k, answer_style=answer_style)
        
        # Format response for benchmark compatibility
        return {
            'final_answer': result['final_answer'],
            'retrieval_details': [
                {
                    'content': detail['content'],
                    'metadata': detail['metadata'],
                    'score': detail['score']
                } for detail in result['retrieval_details']
            ],
            'method': 'graphrag',
            'performance_metrics': {
                'retrieved_chunks': result['retrieved_chunks'],
                'completion_time': 0,  # Not measured in this simple implementation
                'llm_calls': 1,
                'prompt_tokens': 0,    # Not measured in this simple implementation
                'output_tokens': 0,    # Not measured in this simple implementation
                'total_tokens': 0
            }
        }
        
    except Exception as e:
        print(f"Error in GraphRAG retrieval: {e}")
        import traceback
        traceback.print_exc()
        return {
            'final_answer': f"Error during GraphRAG retrieval: {str(e)}",
            'retrieval_details': [],
            'method': 'graphrag_error',
            'performance_metrics': {
                'retrieved_chunks': 0,
                'completion_time': 0,
                'llm_calls': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }