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
from typing import Dict, Any
import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
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
def query_chroma_with_llm(query: str, k: int = 3) -> Dict[str, Any]:
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
def query_neo4j_with_llm(query: str, k: int = 3) -> Dict[str, Any]:
    """Neo4j GraphRAG query with LLM response generation"""
    
    with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        print(f"🔍 Executing GraphRAG query for: {query}")
        
        # Generate query vector
        query_vector = embeddings.embed_query(query)
        
        # Query parameters for the safe Neo4j pattern
        query_params = {
            'query_vector': query_vector,
            'embedding_match_min': 0.4,  # Lower threshold like Neo4j example (was 0.7)
            'embedding_match_max': 0.9   # Lower threshold like Neo4j example (was 0.95)
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
            vector_results = vector_retriever.search(query_text=query, top_k=5)  # Get candidates, but only use top 1
            
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
            
            # Updated Cypher query with Neo4j safe pattern to prevent chunk cross-contamination
            cypher_query = """
            UNWIND $chunk_data AS chunk_item
            MATCH (chunk:Chunk {id: chunk_item.chunk_id})
            
            // Find the document of the chunk
            MATCH (chunk)-[:PART_OF]->(d:Document)
            
            // Get entities from chunks with frequency prioritization (Neo4j pattern)
            CALL {
                WITH chunk
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
                WITH e, count(*) AS numChunks
                WHERE e IS NOT NULL
                ORDER BY numChunks DESC
                LIMIT 20  // Limit entities like Neo4j example
                
                WITH e, $query_vector as query_vector,
                     $embedding_match_min as embedding_match_min,
                     $embedding_match_max as embedding_match_max
                
                // Apply Neo4j safe traversal pattern based on embedding similarity
                WITH
                CASE
                    // Low/medium similarity: 1-hop traversal with safety constraints
                    WHEN e.embedding IS NULL OR 
                         (embedding_match_min <= vector.similarity.cosine(query_vector, e.embedding) 
                          AND vector.similarity.cosine(query_vector, e.embedding) <= embedding_match_max) THEN
                        collect {
                            OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,1}(:!Chunk&!Document)
                            RETURN path LIMIT 10
                        }
                    // High similarity: 2-hop traversal with safety constraints  
                    WHEN e.embedding IS NOT NULL AND 
                         vector.similarity.cosine(query_vector, e.embedding) > embedding_match_max THEN
                        collect {
                            OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,2}(:!Chunk&!Document)
                            RETURN path LIMIT 20
                        }
                    // Fallback: just the entity itself
                    ELSE
                        collect {
                            MATCH path=(e)
                            RETURN path
                        }
                END AS paths, e
                
                // Flatten and deduplicate paths
                WITH apoc.coll.toSet(apoc.coll.flatten(collect(DISTINCT paths))) AS paths,
                     collect(DISTINCT e) AS entities
                
                // Extract nodes and relationships safely
                RETURN
                    collect {
                        UNWIND paths AS p
                        UNWIND relationships(p) AS r
                        RETURN DISTINCT r
                    } AS rels,
                    collect {
                        UNWIND paths AS p  
                        UNWIND nodes(p) AS n
                        RETURN DISTINCT n
                    } AS nodes,
                    entities
            }
            
            // Build enhanced content with entity context (structured like Neo4j example)
            WITH d, chunk, chunk_item.score as score, entities, nodes, rels,
                 [e IN entities | e.name] AS entity_names,
                 [n IN nodes WHERE n.name IS NOT NULL | n.name] AS related_names
            
            // Structured output similar to Neo4j example - clear separation of content types
            WITH d,
                 score,
                 "Text Content:\\n" + chunk.text + 
                 CASE WHEN size(entity_names) > 0 THEN "\\n\\nEntities:\\n" + apoc.text.join(entity_names, ", ") ELSE "" END +
                 CASE WHEN size(related_names) > 0 THEN "\\n\\nRelated Context:\\n" + apoc.text.join(related_names, ", ") ELSE "" END
                 as enhanced_text,
                 collect({id: chunk.id, score: score}) as chunk_details,
                 [n IN nodes WHERE n.name IS NOT NULL | elementId(n)] AS entity_ids,
                 [r IN rels | elementId(r)] AS rel_ids
            
            RETURN
               enhanced_text as text,
               score as score,
               {
                   length: size(enhanced_text),
                   source: d.name,
                   chunkdetails: chunk_details,
                   entities: {
                       entityids: entity_ids,
                       relationshipids: rel_ids
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
            prompt = f"""Based on the following retrieved documents, provide a factual answer using ONLY the information in the Text Content sections. The Entities and Related Context sections provide additional context but should not be treated as factual sources.

Question: {query}

Retrieved Documents:
{context}

Instructions:
1. Base your answer strictly on the Text Content sections of the documents
2. You may combine information from multiple Text Content sections if they are related
3. Use the Entities sections to understand key topics mentioned
4. Use the Related Context sections only for background understanding
5. If the Text Content sections don't contain enough information to answer the question completely, state this clearly
6. Do not make inferences beyond what is explicitly stated in the Text Content sections
7. When combining information from multiple documents, ensure accuracy and avoid mixing unrelated facts

Please provide a factual, well-structured response."""

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
