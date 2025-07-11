"""
GraphRAG Retriever - Graph-Enhanced Vector Search

This module implements basic GraphRAG using Neo4j with vector search
and entity relationship traversal for enhanced context.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any
import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
import warnings
import json
import ast

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
INDEX_NAME = "chunk_embedding"  # Match the index name from graph processor

# Initialize components with deterministic settings
SEED = 42
embeddings = OpenAIEmbeddings()
llm = OpenAILLM(
    model_name="gpt-4o-mini", 
    model_params={
        "temperature": 0,
        "seed": SEED
    }
)

class GraphRAGRetriever:
    """GraphRAG retriever with Neo4j vector search and entity traversal"""
    
    def __init__(self):
        self.embeddings = embeddings
        self.llm = llm
        self.neo4j_uri = NEO4J_URI
        self.neo4j_user = NEO4J_USER
        self.neo4j_password = NEO4J_PASSWORD
        self.index_name = INDEX_NAME
    
    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Neo4j GraphRAG query with LLM response generation"""
        
        with neo4j.GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)) as driver:
            print(f"🔍 Executing GraphRAG query for: {query}")
            
            # Generate query vector
            query_vector = self.embeddings.embed_query(query)
            
            # Query parameters for the simplified pattern
            query_params = {
                'query_vector': query_vector
            }
            
            try:
                # First perform vector search to get candidate chunks
                vector_retriever = VectorRetriever(
                    driver=driver,
                    index_name=self.index_name,
                    embedder=self.embeddings
                )
                
                # Get top chunks from vector search
                vector_results = vector_retriever.search(query_text=query, top_k=5) 
                
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
                                    # First try regular JSON parsing
                                    chunk_data = json.loads(content)
                                except:
                                    # If that fails, try with ast.literal_eval for single quotes
                                    chunk_data = ast.literal_eval(content)
                                
                                chunk_id = chunk_data.get('id', '')
                                score = getattr(result, 'score', 0.8)  # Get score from result object
                                chunks_with_scores.append((chunk_id, score))
                            else:
                                # Content might be the chunk data directly
                                chunk_id = getattr(content, 'id', str(content))
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
                
                # Simplified Cypher query following the specified structure
                cypher_query = """
                UNWIND $chunk_data AS chunk_item
                MATCH (chunk:Chunk {id: chunk_item.chunk_id})-[:PART_OF]->(d:Document)
                
                CALL (chunk) {
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
                
                # Prepare chunk data for the query
                chunk_data = [{'chunk_id': cid, 'score': chunk_scores[cid]} for cid in chunk_ids]
                
                # Execute with parameters (suppress warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = driver.execute_query(
                        cypher_query,
                        chunk_data=chunk_data,
                        **query_params,
                        database_="neo4j"
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
                
                # Generate LLM response with GraphRAG context
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
def query_graphrag(query: str, k: int = 3, **kwargs) -> Dict[str, Any]:
    """
            GraphRAG retrieval with LLM response generation
    
    Args:
        query: The search query
        k: Number of chunks to retrieve
        **kwargs: Additional configuration options (for compatibility)
    
    Returns:
        Dictionary with response and retrieval details
    """
    try:
        retriever = create_graphrag_retriever()
        result = retriever.search(query, k)
        
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