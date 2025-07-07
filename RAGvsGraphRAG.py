"""
RAG vs GraphRAG Comparison

This script compares traditional RAG using ChromaDB with advanced GraphRAG using Neo4j
for question answering on RFP documents. The GraphRAG implementation uses a complex
multi-hop query with embedding-based entity intelligence.

Objectives:
* Compare traditional RAG using ChromaDB with complex GraphRAG using Neo4j
* Demonstrate embedding-aware entity traversal and intelligent path selection
* Provide interactive query interface with rich graph context
* Visualize and compare results with detailed metadata
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, Any
import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from langchain_neo4j import Neo4jGraph
from langchain_chroma import Chroma
import warnings
import logging

# Load environment variables
load_dotenv()

# Comprehensive telemetry and logging suppression
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_DB_TELEMETRY"] = "False"

# Suppress various logging that might show telemetry
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")

# ChromaDB configuration
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "rfp_docs"

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
INDEX_NAME = "chunk_embedding"  # Match the index name from custom_graph_processor.py

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings()
llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})

# %% [markdown]
# ## Verify Database Connections
# Let's check if our databases are properly connected and contain data

# %%
# Check ChromaDB (with telemetry suppressed)
try:
    import chromadb
    chromadb.telemetry.telemetry_client = None
except:
    pass

vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)
print(f"ChromaDB collection size: {vectorstore._collection.count()}")

# %%
# Check Neo4j
with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
    result = driver.execute_query(
        "MATCH (n) RETURN COUNT(n) as count",
        database_="neo4j"
    )
    print(f"Neo4j node count: {result.records[0]['count']}")

# %% [markdown]
# ## Define Query Functions with LLM Response Generation

# %% [markdown]
# ### 1. ChromaDB Query Function with LLM Response

# %%
def query_chroma_with_llm(query: str, k: int = 5) -> Dict[str, Any]:
    """Query ChromaDB and generate LLM response"""
    # Initialize ChromaDB
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # Perform similarity search
    docs = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    
    # Prepare context from retrieved documents
    context_parts = []
    retrieval_info = []
    
    for i, (doc, score) in enumerate(docs, 1):
        context_parts.append(f"Document {i}:\n{doc.page_content}")
        retrieval_info.append({
            'content': doc.page_content,
            'metadata': doc.metadata,
            'similarity_score': score
        })
    
    context = "\n\n".join(context_parts)
    
    # Generate LLM response
    prompt = f"""Based on the following retrieved documents, please provide a comprehensive answer to the question.

Question: {query}

Retrieved Documents:
{context}

Please provide a well-structured, informative response based on the retrieved information. If the documents don't contain enough information to fully answer the question, please indicate what information is missing."""

    try:
        llm_response = llm.invoke(prompt)
        final_answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
    except Exception as e:
        final_answer = f"Error generating LLM response: {e}"
    
    return {
        'method': 'ChromaDB + LLM',
        'query': query,
        'final_answer': final_answer,
        'retrieved_chunks': len(docs),
        'retrieval_details': retrieval_info
    }

# %% [markdown]
# ### 2. Neo4j GraphRAG Query Function with LLM Response

# %%
def query_neo4j_with_llm(query: str, k: int = 5) -> Dict[str, Any]:
    """Neo4j GraphRAG query with LLM response generation"""
    
    with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        print(f"🔍 Executing GraphRAG query for: {query}")
        
        # Generate query vector
        query_vector = embeddings.embed_query(query)
        
        # Query parameters for the complex query
        query_params = {
            'query_vector': query_vector,
            'no_of_entities': 10,  # Limit number of entities to process
            'embedding_match_min': 0.7,  # Minimum similarity threshold
            'embedding_match_max': 0.95,  # Maximum similarity threshold  
            'entity_limit_minmax_case': 50,  # Path limit for medium similarity
            'entity_limit_max_case': 100   # Path limit for high similarity
        }
        
        try:
            # First perform vector search to get candidate chunks
            from neo4j_graphrag.retrievers import VectorRetriever
            vector_retriever = VectorRetriever(
                driver=driver,
                index_name=INDEX_NAME,
                embedder=embeddings
            )
            
            # Get top chunks from vector search
            vector_results = vector_retriever.search(query_text=query, top_k=k*2)  # Get more candidates
            
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
            
            for result in vector_results.items[:k]:  # Access items from RetrieverResult
                # Each item is a RetrieverResultItem with content and metadata
                if hasattr(result, 'content'):
                    content = result.content
                    # Parse the content to get chunk info and score  
                    import json
                    try:
                        # Content might be a JSON string or dict-like string
                        if isinstance(content, str) and content.startswith('{'):
                            try:
                                # First try regular JSON parsing
                                chunk_data = json.loads(content)
                            except:
                                # If that fails, try with ast.literal_eval for single quotes
                                import ast
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
            
            # Updated Cypher query with proper CALL syntax (no deprecation warnings)
            cypher_query = """
            UNWIND $chunk_data AS chunk_item
            MATCH (chunk:Chunk {id: chunk_item.chunk_id})
            
            // Find the document of the chunk
            MATCH (chunk)-[:PART_OF]->(d:Document)
            
            // Get entities from chunks
            OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
            WITH d, chunk, chunk_item.score as score, e, $query_vector as query_vector,
                 $no_of_entities as no_of_entities,
                 $embedding_match_min as embedding_match_min,
                 $embedding_match_max as embedding_match_max,
                 $entity_limit_minmax_case as entity_limit_minmax_case,
                 $entity_limit_max_case as entity_limit_max_case
            
            // Apply entity frequency filtering and embedding-based traversal
            WITH d, chunk, score, 
                 collect(DISTINCT e) as chunk_entities,
                 query_vector, no_of_entities, embedding_match_min, embedding_match_max,
                 entity_limit_minmax_case, entity_limit_max_case
            
            // Get related entities based on embedding similarity
            UNWIND chunk_entities as entity
            WITH d, chunk, score, entity, query_vector, 
                 embedding_match_min, embedding_match_max,
                 entity_limit_minmax_case, entity_limit_max_case
            WHERE entity IS NOT NULL
            
            // Apply conditional traversal based on entity embedding similarity
            CALL (entity, query_vector, embedding_match_min, embedding_match_max) {
                WITH entity, query_vector, embedding_match_min, embedding_match_max
                OPTIONAL MATCH path1 = (entity)-[:RELATES_TO*0..1]-(related:Organization|Location|Date|Requirement|Person|Financial)
                WHERE entity.embedding IS NULL OR 
                      (embedding_match_min <= vector.similarity.cosine(query_vector, entity.embedding) 
                       AND vector.similarity.cosine(query_vector, entity.embedding) <= embedding_match_max)
                RETURN collect(path1) as paths_medium
            }
            CALL (entity, query_vector, embedding_match_max) {
                WITH entity, query_vector, embedding_match_max
                OPTIONAL MATCH path2 = (entity)-[:RELATES_TO*0..2]-(related:Organization|Location|Date|Requirement|Person|Financial)
                WHERE entity.embedding IS NOT NULL AND 
                      vector.similarity.cosine(query_vector, entity.embedding) > embedding_match_max
                RETURN collect(path2) as paths_high
            }
            CALL (entity, query_vector, embedding_match_min, embedding_match_max) {
                WITH entity, query_vector, embedding_match_min, embedding_match_max
                MATCH path3 = (entity)
                WHERE NOT (entity.embedding IS NULL OR 
                          (embedding_match_min <= vector.similarity.cosine(query_vector, entity.embedding) 
                           AND vector.similarity.cosine(query_vector, entity.embedding) <= embedding_match_max))
                      AND NOT (entity.embedding IS NOT NULL AND 
                              vector.similarity.cosine(query_vector, entity.embedding) > embedding_match_max)
                RETURN collect(path3) as paths_basic
            }
            
            // Collect and aggregate results
            WITH d, chunk, score, 
                 collect(DISTINCT entity.name) as entity_names,
                 apoc.coll.flatten(collect(paths_medium + paths_high + paths_basic)) as all_paths
            WITH d, chunk, score, entity_names,
                 [p IN all_paths WHERE p IS NOT NULL | nodes(p)] as path_nodes,
                 [p IN all_paths WHERE p IS NOT NULL | relationships(p)] as path_rels
            
            // Build enhanced content with entity context
            WITH d, 
                 avg(score) as avg_score,
                 chunk.text + 
                 CASE WHEN size(entity_names) > 0 THEN "\\nEntities: " + apoc.text.join(entity_names, ", ") ELSE "" END +
                 CASE WHEN size(path_nodes) > 0 THEN "\\nRelated Context: " + apoc.text.join([n IN apoc.coll.flatten(path_nodes) WHERE n.name IS NOT NULL | n.name], ", ") ELSE "" END
                 as enhanced_text,
                 collect({id: chunk.id, score: score}) as chunk_details,
                 apoc.coll.flatten(path_nodes) as all_entities,
                 apoc.coll.flatten(path_rels) as all_relationships
            
            RETURN
               enhanced_text as text,
               avg_score as score,
               {
                   length: size(enhanced_text),
                   source: d.name,
                   chunkdetails: chunk_details,
                   entities: {
                       entityids: [n IN all_entities WHERE n.name IS NOT NULL | elementId(n)],
                       relationshipids: [r IN all_relationships | elementId(r)]
                   }
               } as metadata
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
            
            print(f"✅ GraphRAG retrieved {len(enhanced_chunks)} enhanced chunks")
            
            if not enhanced_chunks:
                return {
                    'method': 'GraphRAG + LLM',
                    'query': query,
                    'final_answer': 'No enhanced results found after entity processing.',
                    'retrieved_chunks': 0,
                    'retrieval_details': []
                }
            
            # Prepare context for LLM with enhanced chunks
            context_parts = []
            for i, chunk in enumerate(enhanced_chunks, 1):
                context_parts.append(f"Enhanced Document {i}:\n{chunk}")
            
            context = "\n\n".join(context_parts)
            
            # Generate LLM response with GraphRAG context
            prompt = f"""Based on the following enhanced documents retrieved through GraphRAG 
            (which includes entity relationships and contextual information), 
            please provide a comprehensive answer to the question.

Question: {query}

Enhanced Retrieved Documents with Entity Context:
{context}

Please provide a well-structured, informative response based on the enhanced retrieval information. 
Take advantage of the entity relationships and additional context provided by the GraphRAG system."""

            try:
                llm_response = llm.invoke(prompt)
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

# %% [markdown]
# ### 3. Simple Comparison Function

# %%
def compare_both_approaches(query: str, k: int = 5):
    """Compare ChromaDB vs GraphRAG approaches side by side"""
    print(f"\n{'='*80}")
    print(f"🔍 QUERY: {query}")
    print(f"{'='*80}")
    
    # Get results from both approaches
    print("\n🔵 ChromaDB + LLM Result:")
    print("-" * 50)
    chroma_result = query_chroma_with_llm(query, k)
    print(f"Retrieved Chunks: {chroma_result['retrieved_chunks']}")
    print(f"\n{chroma_result['final_answer']}")
    
    print(f"\n🟢 GraphRAG + LLM Result:")
    print("-" * 50)
    graphrag_result = query_neo4j_with_llm(query, k)
    print(f"Retrieved Chunks: {graphrag_result['retrieved_chunks']}")
    print(f"\n{graphrag_result['final_answer']}")
    
    print(f"\n{'='*80}")
    print("✅ COMPARISON COMPLETE")
    print(f"{'='*80}")
    
    return {
        'chroma_result': chroma_result,
        'graphrag_result': graphrag_result
    }

# Create aliases for backward compatibility
query_chroma = query_chroma_with_llm
query_neo4j = query_neo4j_with_llm
query_neo4j_enhanced = query_neo4j_with_llm

# %% [markdown]
# ## Test the Comparison

# %%
# Example usage - remove this if you don't want auto-execution
if __name__ == "__main__":
    test_query = "What are the main vendor requirements?"
    results = compare_both_approaches(test_query, k=3)
