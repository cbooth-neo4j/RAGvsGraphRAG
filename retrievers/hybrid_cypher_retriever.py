"""
Hybrid Cypher Retriever - Vector + Full-text Search with Graph Traversal

This module implements a HybridCypherRetriever using Neo4j's built-in HybridCypherRetriever
that combines vector similarity search with full-text search and performs 1-hop graph 
traversal for enhanced context retrieval.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import HybridCypherRetriever
import warnings

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

# Index names (matching existing indexes from schema check)
VECTOR_INDEX_NAME = "chunk_embedding"  # Vector index on Chunk.embedding
FULLTEXT_INDEX_NAME = "chunk_text_fulltext"  # Full-text index on Chunk.text

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

class HybridCypherRAGRetriever:
    """
    Hybrid retriever using Neo4j's HybridCypherRetriever with 1-hop graph traversal.
    Combines vector search, full-text search, and entity relationship traversal.
    """

    def __init__(self):
        self.embeddings = embeddings
        self.llm = llm
        self.neo4j_uri = NEO4J_URI
        self.neo4j_user = NEO4J_USER
        self.neo4j_password = NEO4J_PASSWORD
        self.vector_index_name = VECTOR_INDEX_NAME
        self.fulltext_index_name = FULLTEXT_INDEX_NAME
        
        # Define the 1-hop retrieval query similar to graph_rag_retriever.py
        # According to Neo4j docs, the retrieval query should return the enhanced content
        # The 'node' variable refers to the chunk found by hybrid search
        self.retrieval_query = """
        MATCH (node)-[:PART_OF]->(d:Document)
        OPTIONAL MATCH (node)-[:HAS_ENTITY]->(e)
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(related_entity)
        WITH node, d, 
             collect(DISTINCT e.name) as direct_entities,
             collect(DISTINCT related_entity.name) as related_entities,
             collect(DISTINCT type(r)) as relationship_types
        RETURN node.text + 
               CASE WHEN size(direct_entities) > 0 
                    THEN '\\n\\nEntities: ' + apoc.text.join(direct_entities, ', ')
                    ELSE '' END +
               CASE WHEN size(related_entities) > 0 
                    THEN '\\n\\nRelated: ' + apoc.text.join(related_entities, ', ')
                    ELSE '' END
        """

    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Perform hybrid search with vector similarity, full-text, and 1-hop graph traversal.

        Args:
            query: The search query
            k: Number of results to retrieve

        Returns:
            Dictionary with search results and metadata
        """

        with neo4j.GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)) as driver:
            print(f"🔍 Executing Hybrid Cypher GraphRAG query for: {query}")

            try:
                # Initialize the hybrid cypher retriever with our custom retrieval query
                retriever = HybridCypherRetriever(
                    driver=driver,
                    vector_index_name=self.vector_index_name,
                    fulltext_index_name=self.fulltext_index_name,
                    embedder=self.embeddings,
                    retrieval_query=self.retrieval_query
                )

                # Perform the hybrid search
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search_results = retriever.search(query_text=query, top_k=k)

                if not search_results or not hasattr(search_results, 'items') or not search_results.items:
                    return {
                        'method': 'Hybrid Cypher GraphRAG',
                        'query': query,
                        'final_answer': 'No results found in the knowledge base.',
                        'retrieved_chunks': 0,
                        'retrieval_details': []
                    }

                # Process and format results
                enhanced_chunks = []
                retrieval_details = []

                for item in search_results.items:
                    if hasattr(item, 'content'):
                        content = item.content
                        score = getattr(item, 'score', 0.8)

                        # Extract the actual text content from the Neo4j result
                        chunk_text = str(content)
                        
                        # The content is a string representation of a Neo4j Record with a complex expression
                        # The actual result is after the '=' sign
                        if chunk_text.startswith("<Record") and "=" in chunk_text:
                            # Find the equals sign and extract everything after it until the closing >
                            equals_pos = chunk_text.find("='")
                            if equals_pos > 0:
                                start_pos = equals_pos + 2  # Skip ='
                                end_pos = chunk_text.rfind("'>")
                                if end_pos > start_pos:
                                    chunk_text = chunk_text[start_pos:end_pos]
                                    # Unescape the text
                                    chunk_text = chunk_text.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
                        
                        # Extract entity information from the enhanced chunk text
                        direct_entities = []
                        related_entities = []
                        base_text = chunk_text
                        
                        # Parse entities from the enhanced text
                        if "\n\nEntities: " in chunk_text:
                            parts = chunk_text.split("\n\nEntities: ")
                            base_text = parts[0]
                            entity_part = parts[1]
                            
                            if "\n\nRelated: " in entity_part:
                                entity_parts = entity_part.split("\n\nRelated: ")
                                direct_entities = [e.strip() for e in entity_parts[0].split(", ") if e.strip()]
                                related_entities = [e.strip() for e in entity_parts[1].split(", ") if e.strip()]
                            else:
                                direct_entities = [e.strip() for e in entity_part.split(", ") if e.strip()]
                        
                        enhanced_chunks.append(base_text)  # Use base text without entity annotations for LLM
                        
                        # Try to get metadata from the item if available
                        metadata = {}
                        if hasattr(item, 'metadata') and item.metadata:
                            metadata = item.metadata
                        
                        retrieval_details.append({
                            'content': base_text,
                            'score': score,
                            'metadata': {
                                'chunk_id': metadata.get('chunk_id', 'unknown'),
                                'document_name': metadata.get('document_name', 'N/A'),
                                'preview': base_text[:100] + "..." if len(base_text) > 100 else base_text,
                                'direct_entities': direct_entities,
                                'related_entities': related_entities,
                                'relationship_types': metadata.get('relationship_types', []),
                                'total_entities': len(direct_entities) + len(related_entities),
                                'enhanced_content': chunk_text  # Keep the full enhanced content for reference
                            }
                        })

                print(f"✅ Hybrid Cypher GraphRAG retrieved {len(enhanced_chunks)} chunks")

                # Log detailed information
                for i, detail in enumerate(retrieval_details, 1):
                    metadata = detail.get('metadata', {})
                    print(f"  📄 Chunk {i}:")
                    print(f"    - ID: {metadata.get('chunk_id', 'N/A')}")
                    print(f"    - Document: {metadata.get('document_name', 'N/A')}")
                    print(f"    - Preview: {metadata.get('preview', 'N/A')}")
                    print(f"    - Direct Entities: {metadata.get('direct_entities', [])}")
                    print(f"    - Related Entities: {metadata.get('related_entities', [])}")
                    print(f"    - Score: {detail.get('score', 'N/A')}")

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

                # Generate LLM response with hybrid context
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
def create_hybrid_cypher_retriever() -> HybridCypherRAGRetriever:
    """Create a Hybrid Cypher RAG retriever instance"""
    return HybridCypherRAGRetriever()


# Main interface function for integration with benchmark system
def query_hybrid_cypher_rag(query: str, k: int = 3, **kwargs) -> Dict[str, Any]:
    """
    Hybrid Cypher GraphRAG retrieval with LLM response generation

    Args:
        query: The search query
        k: Number of chunks to retrieve
        **kwargs: Additional configuration options (for compatibility)

    Returns:
        Dictionary with response and retrieval details
    """
    try:
        retriever = create_hybrid_cypher_retriever()
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