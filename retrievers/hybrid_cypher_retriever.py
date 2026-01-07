"""
Hybrid Cypher Retriever - Vector + Full-text Search with Graph Traversal

This module implements a HybridCypherRetriever using Neo4j's built-in HybridCypherRetriever
that combines vector similarity search with full-text search and performs 1-hop graph 
traversal for enhanced context retrieval.
Supports both OpenAI API and Ollama local models.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import neo4j
from neo4j_graphrag.retrievers import HybridCypherRetriever
import warnings

# Import centralized configuration
from config import get_model_config, get_neo4j_embeddings, get_neo4j_llm
from utils.graph_rag_logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
CLIENT_NEO4J_DATABASE = os.environ.get('CLIENT_NEO4J_DATABASE')


# Index names for entity-focused hybrid search (Local Entity HybridCypherRetriever)
VECTOR_INDEX_NAME = "entity_embeddings"  # Vector index on __Entity__.embedding
FULLTEXT_INDEX_NAME = "entity_fulltext_idx"  # Full-text index on __Entity__.name, description

class HybridCypherRAGRetriever:
    """
    Local Entity HybridCypherRetriever implementation using Neo4j GraphRAG pattern.
    
    This retriever implements the Local Entity pattern:
    1. Hybrid search on entities (:__Entity__) using vector + full-text indexes
    2. Graph expansion to find:
       - Top co-mentioned chunks (:Chunk) via HAS_ENTITY relationships
       - Community summaries (:__Community__) via IN_COMMUNITY relationships  
       - Entity relationships and outside neighbors
    3. Returns structured context with entities, chunks, communities, and relationships
    """

    def __init__(self, model_config: Optional[Any] = None):
        self.config = model_config or get_model_config()
        
        # Initialize models based on configuration
        try:
            self.embeddings = get_neo4j_embeddings()
            self.llm = get_neo4j_llm()
        except Exception as e:
            warnings.warn(f"Could not initialize configured models, falling back to defaults: {e}")
            logger.error(f"Failed to create embedding model or Neo4J Wrapper...")
            # Fallback imports for backward compatibility
            #from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
            #from neo4j_graphrag.llm import OpenAILLM
            #self.embeddings = OpenAIEmbeddings()
            #self.llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
        
        self.neo4j_uri = NEO4J_URI
        self.neo4j_user = NEO4J_USER
        self.neo4j_password = NEO4J_PASSWORD
        self.neo4j_db = CLIENT_NEO4J_DATABASE
        self.vector_index_name = VECTOR_INDEX_NAME
        self.fulltext_index_name = FULLTEXT_INDEX_NAME
        
        # Complete Local Entity HybridCypherRetriever query
        # This performs hybrid search on entities, then expands context via graph traversal
        self.retrieval_query = """
        // Hybrid search on entities (vector + fulltext)
        CALL () {
            // Vector search on entity embeddings
            CALL db.index.vector.queryNodes($vector_index_name, $top_k, $query_vector) 
            YIELD node AS vector_node, score AS vector_score
            RETURN vector_node AS result_node, vector_score AS search_score
            UNION
            // Fulltext search on entity names and descriptions  
            CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text)
            YIELD node AS ft_node, score AS ft_score
            RETURN ft_node AS result_node, ft_score AS search_score
        }
        // Combine and deduplicate results
        WITH result_node AS node, max(search_score) AS score
        ORDER BY score DESC
        LIMIT $top_k
        
        // Collect hybrid search results
        WITH collect(node) AS nodes,
             avg(score) AS avg_score,
             collect({id: elementId(node), score: score}) AS metadata
        
        // Graph expansion for Local Entity pattern
        RETURN avg_score AS score, nodes, metadata,

            // Find top co-mentioned chunks
            collect {
                UNWIND nodes AS n
                MATCH (n)<-[:HAS_ENTITY]-(c:Chunk)
                WITH c, count(distinct n) AS freq
                RETURN c
                ORDER BY freq DESC
                LIMIT 3
            } AS chunks,

            // Find relevant communities (including hierarchical traversal for DRIFT-style context)
            collect {
                UNWIND nodes AS n
                // Get level 0 communities directly connected to entities
                OPTIONAL MATCH (n)-[:IN_COMMUNITY]->(level0:__Community__)
                WHERE level0 IS NOT NULL
                
                // Also traverse up the hierarchy to get higher-level summaries (DRIFT pattern)
                OPTIONAL MATCH (level0)-[:PARENT_COMMUNITY*0..2]->(higher:__Community__)
                WHERE higher.summary IS NOT NULL
                
                // Combine all communities from the hierarchy
                WITH collect(DISTINCT level0) + collect(DISTINCT higher) AS all_comms
                UNWIND all_comms AS comm
                WITH DISTINCT comm,
                     coalesce(comm.community_rank, 0) AS rank, 
                     coalesce(comm.weight, 1.0) AS weight,
                     comm.level AS level
                WHERE comm IS NOT NULL
                // Prefer higher-level communities for broader context, then by rank
                RETURN comm
                ORDER BY level DESC, rank DESC, weight DESC
                LIMIT 5
            } AS communities,

            // Find relationships between found entities
            collect {
                UNWIND nodes AS n
                UNWIND nodes AS m
                MATCH (n)-[r]->(m)
                WHERE n <> m
                RETURN DISTINCT r
            } AS rels,

            // Find outside entities (1-hop neighbors not in initial results)
            collect {
                UNWIND nodes AS n
                MATCH (n)-[r]-(outside_entity:__Entity__)
                WHERE NOT outside_entity IN nodes
                WITH outside_entity, collect(distinct r) AS rels, count(*) AS freq
                ORDER BY freq DESC
                LIMIT 10
                WITH collect(outside_entity) AS outsideNodes, apoc.coll.flatten(collect(rels)) AS rels
                RETURN { nodes: outsideNodes, rels: rels }
            } AS outside
        """

    def _sanitize_fulltext_query(self, query: str) -> str:
        """Sanitize query for Lucene full-text search by removing/replacing problematic characters"""
        import re
        
        # Replace problematic patterns that cause Lucene parsing errors
        sanitized = query
        
        # Replace forward slashes with spaces (common in technical terms like ERP/TMS)
        sanitized = sanitized.replace('/', ' ')
        
        # Remove other Lucene special characters that cause parsing issues
        # Keep it simple - just remove the most problematic ones
        special_chars = ['\\', '+', '-', '&&', '||', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':']
        for char in special_chars:
            sanitized = sanitized.replace(char, ' ')
        
        # Clean up multiple spaces
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized

    def search(self, query: str, k: int = 3, answer_style: str = "ragas") -> Dict[str, Any]:
        """
        Perform Local Entity hybrid search with graph expansion.

        This method:
        1. Performs hybrid search on entities using vector + full-text indexes
        2. Expands context using the Local Entity retrieval pattern
        3. Returns structured results with entities, chunks, communities, and relationships

        Args:
            query: The search query (entity-focused queries work best)
            k: Number of entity results to retrieve
            answer_style: Response format - "hotpotqa" for short exact answers (EM/F1 benchmarks),
                         "ragas" for verbose real-world answers (default)

        Returns:
            Dictionary with search results and metadata including:
            - entities: Found entities and their relationships
            - chunks: Top co-mentioned text chunks
            - communities: Relevant community summaries
            - outside_entities: Related entities not in the initial search
        """

        with neo4j.GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password), database=self.neo4j_db) as driver:
            print(f"🔍 Executing Hybrid Cypher GraphRAG query for: {query}")

            try:
                # Custom result formatter for Local Entity HybridCypherRetriever
                def local_entity_formatter(record):
                    """Format Local Entity retrieval results for compatibility"""
                    from neo4j_graphrag.types import RetrieverResultItem
                    
                    # Extract the structured data from the record
                    try:
                        data = record.data()
                        
                        # Create a simplified content representation
                        chunks = data.get('chunks', [])
                        chunk_texts = []
                        for chunk in chunks:
                            try:
                                if isinstance(chunk, dict) and 'text' in chunk:
                                    chunk_texts.append(chunk['text'])
                                elif hasattr(chunk, 'get') and callable(chunk.get):
                                    text = chunk.get('text', str(chunk))
                                    chunk_texts.append(text)
                                elif hasattr(chunk, 'text'):
                                    chunk_texts.append(chunk.text)
                                else:
                                    chunk_texts.append(str(chunk))
                            except Exception:
                                chunk_texts.append(str(chunk))
                        
                        # Safely extract metadata with proper type checking
                        nodes = data.get('nodes', [])
                        communities = data.get('communities', [])
                        rels = data.get('rels', [])
                        outside = data.get('outside', {})
                        
                        # Build content with chunks AND community summaries for richer context
                        content_parts = []
                        
                        # Add chunk texts
                        if chunk_texts:
                            content_parts.append("=== Retrieved Text Chunks ===")
                            content_parts.extend(chunk_texts)
                        
                        # Add community summaries (hierarchical context for DRIFT-style retrieval)
                        if communities:
                            community_summaries = []
                            for comm in communities:
                                try:
                                    if isinstance(comm, dict):
                                        summary = comm.get('summary', '')
                                        title = comm.get('title', '')
                                        level = comm.get('level', 0)
                                    elif hasattr(comm, 'get') and callable(comm.get):
                                        summary = comm.get('summary', '')
                                        title = comm.get('title', '')
                                        level = comm.get('level', 0)
                                    else:
                                        summary = getattr(comm, 'summary', '') if hasattr(comm, 'summary') else ''
                                        title = getattr(comm, 'title', '') if hasattr(comm, 'title') else ''
                                        level = getattr(comm, 'level', 0) if hasattr(comm, 'level') else 0
                                    
                                    if summary:
                                        level_label = f"[Level {level}]" if level else ""
                                        community_summaries.append(f"{level_label} {title}: {summary}" if title else f"{level_label} {summary}")
                                except Exception:
                                    continue
                            
                            if community_summaries:
                                content_parts.append("\n=== Community Context (Hierarchical Summaries) ===")
                                content_parts.extend(community_summaries)
                        
                        content = "\n\n".join(content_parts) if content_parts else "No content available"
                        
                        # Handle outside entities safely
                        outside_nodes = []
                        if isinstance(outside, dict):
                            outside_nodes = outside.get('nodes', [])
                        elif isinstance(outside, list):
                            outside_nodes = outside
                        
                        # Create compatible metadata (must be a dict)
                        metadata = {
                            'score': data.get('score', 0.0),
                            'entities': len(nodes) if isinstance(nodes, list) else 0,
                            'chunks': len(chunks) if isinstance(chunks, list) else 0,
                            'communities': len(communities) if isinstance(communities, list) else 0,
                            'relationships': len(rels) if isinstance(rels, list) else 0,
                            'outside_entities': len(outside_nodes) if isinstance(outside_nodes, list) else 0
                        }
                        
                        return RetrieverResultItem(content=content, metadata=metadata)
                        
                    except Exception as e:
                        # Fallback to simple string representation
                        return RetrieverResultItem(
                            content=str(record), 
                            metadata={'error': str(e), 'fallback': True}
                        )

                # Initialize the hybrid cypher retriever with custom formatter
                retriever = HybridCypherRetriever(
                    driver=driver,
                    vector_index_name=self.vector_index_name,
                    fulltext_index_name=self.fulltext_index_name,
                    embedder=self.embeddings,
                    retrieval_query=self.retrieval_query,
                    result_formatter=local_entity_formatter
                )

                # Sanitize query for full-text search and perform the hybrid search
                sanitized_query = self._sanitize_fulltext_query(query)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search_results = retriever.search(query_text=sanitized_query, top_k=k)

                if not search_results or not hasattr(search_results, 'items') or not search_results.items:
                    return {
                        'method': 'Hybrid Cypher GraphRAG',
                        'query': query,
                        'final_answer': 'No results found in the knowledge base.',
                        'retrieved_chunks': 0,
                        'retrieval_details': []
                    }

                # Process Local Entity HybridCypherRetriever results (using custom formatter)
                enhanced_chunks = []
                retrieval_details = []

                for item in search_results.items:
                    if hasattr(item, 'content'):
                        content = item.content
                        score = getattr(item, 'score', 0.8)
                        metadata = getattr(item, 'metadata', {})

                        # Content is already formatted by our custom formatter
                        enhanced_chunks.append(content)
                        
                        # Use the metadata from our custom formatter
                        retrieval_details.append({
                            'content': content,
                            'score': metadata.get('score', score),
                            'metadata': {
                                'retrieval_type': 'local_entity_hybrid',
                                'entities': metadata.get('entities', []),
                                'chunk_count': metadata.get('chunks', 0),
                                'communities': metadata.get('communities', 0),
                                'relationships': metadata.get('relationships', 0),
                                'outside_entities': metadata.get('outside_entities', 0),
                                'preview': content[:200] + "..." if len(content) > 200 else content,
                                'fallback': metadata.get('fallback', False),
                                'error': metadata.get('error', None)
                            }
                        })

                print(f"✅ Hybrid Cypher GraphRAG retrieved {len(enhanced_chunks)} chunks")

                # Log detailed information for Local Entity HybridCypherRetriever
                for i, detail in enumerate(retrieval_details, 1):
                    metadata = detail.get('metadata', {})
                    print(f"  📄 Result {i} ({metadata.get('retrieval_type', 'unknown')}):")
                    print(f"    - Score: {detail.get('score', 'N/A')}")
                    print(f"    - Entities: {metadata.get('entities', 0)} found")
                    print(f"    - Outside Entities: {metadata.get('outside_entities', 0)}")
                    print(f"    - Communities: {metadata.get('communities', 0)}")
                    print(f"    - Chunks: {metadata.get('chunks', 0)}")
                    print(f"    - Relationships: {metadata.get('relationships', 0)}")
                    if metadata.get('error'):
                        print(f"    - Error: {metadata.get('error')}")
                    print(f"    - Preview: {metadata.get('preview', 'N/A')}")

                if not enhanced_chunks:
                    return {
                        'method': 'Hybrid Cypher GraphRAG',
                        'query': query,
                        'final_answer': 'No results found after processing.',
                        'retrieved_chunks': 0,
                        'retrieval_details': []
                    }

                # Prepare context for LLM
                context_parts = []
                for i, chunk in enumerate(enhanced_chunks, 1):
                    context_parts.append(f"Document {i}:\n{chunk}")

                context = "\n\n".join(context_parts)

                # Generate LLM response with Local Entity hybrid context
                # Note: answer_style is now handled by post-processing in the benchmark layer
                prompt = f"""Answer the question based on the retrieved context below.

Question: {query}

Retrieved Context:
{context}

Instructions:
- Base your answer on the retrieved context
- Be factual and specific - cite entities, names, dates, and facts from the context
- If the context is insufficient to answer, state that clearly

Answer:"""

                try:
                    llm_response = self.llm.invoke(prompt)
                    final_answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                except Exception as e:
                    final_answer = f"Error generating LLM response: {e}"

                return {
                    'method': 'Hybrid Cypher GraphRAG',
                    'query': query,
                    'final_answer': final_answer,
                    'retrieved_chunks': len(enhanced_chunks),
                    'retrieval_details': retrieval_details
                }

            except Exception as e:
                print(f"Error in hybrid cypher search: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'method': 'Hybrid Cypher GraphRAG',
                    'query': query,
                    'final_answer': f"Error with hybrid cypher processing: {e}",
                    'retrieved_chunks': 0,
                    'retrieval_details': []
                }


# Factory function for easy instantiation
def create_hybrid_cypher_retriever(model_config: Optional[Any] = None) -> HybridCypherRAGRetriever:
    """Create a Hybrid Cypher RAG retriever instance with configurable models"""
    return HybridCypherRAGRetriever(model_config)


# Main interface function for integration with benchmark system
def query_hybrid_cypher_rag(query: str, k: int = 3, answer_style: str = "ragas", **kwargs) -> Dict[str, Any]:
    """
    Hybrid Cypher GraphRAG retrieval with LLM response generation

    Args:
        query: The search query
        k: Number of chunks to retrieve
        answer_style: Response format - "hotpotqa" for short exact answers (EM/F1 benchmarks),
                     "ragas" for verbose real-world answers (default)
        **kwargs: Additional configuration options (for compatibility)

    Returns:
        Dictionary with response and retrieval details
    """
    try:
        retriever = create_hybrid_cypher_retriever()
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
            'method': 'hybrid_cypher_graphrag',
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
        print(f"Error in Hybrid Cypher GraphRAG retrieval: {e}")
        import traceback
        traceback.print_exc()
        return {
            'final_answer': f"Error during Hybrid Cypher GraphRAG retrieval: {str(e)}",
            'retrieval_details': [],
            'method': 'hybrid_cypher_graphrag_error',
            'performance_metrics': {
                'retrieved_chunks': 0,
                'completion_time': 0,
                'llm_calls': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }