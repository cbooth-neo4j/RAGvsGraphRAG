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
SEED = 42
embeddings = OpenAIEmbeddings()
llm = OpenAILLM(
    model_name="gpt-4o-mini", 
    model_params={
        "temperature": 0,
        "seed": SEED
    }
)

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
def query_chroma_with_llm(query: str, k: int = 1) -> Dict[str, Any]:
    """Query ChromaDB and generate LLM response"""
    # Initialize ChromaDB
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # Perform similarity search
    docs = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    
    # Add deterministic sorting for consistent results
    # Sort by: 1) similarity score (desc), 2) content hash (for ties), 3) metadata hash
    docs = sorted(docs, key=lambda x: (
        -x[1],  # Negative similarity score for descending order
        hash(x[0].page_content),  # Content hash for deterministic tie-breaking
        hash(str(sorted(x[0].metadata.items())))  # Metadata hash for additional stability
    ))
    
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
        
        # Query parameters for the simplified pattern
        query_params = {
            'query_vector': query_vector
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
            
            # Simplified Cypher query following the specified structure
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
# ### 3. Neo4j Text2Cypher Query Function with LLM Response

# %%
def query_neo4j_text2cypher(query: str) -> Dict[str, Any]:
    """Neo4j Text2Cypher query with natural language to Cypher conversion"""
    
    # Dynamically extract the Neo4j schema from the actual database
    from langchain_neo4j import Neo4jGraph
    
    # Create Neo4j graph connection to extract schema
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        enhanced_schema=True
    )
    
    # Refresh and get the current schema
    graph.refresh_schema()
    neo4j_schema = graph.schema
    print(f"🔍 Text2Cypher - Schema extracted: {len(neo4j_schema)} characters")

    # Use GPT-4-turbo for better Cypher query generation
    text2cypher_llm = OpenAILLM(
        model_name="gpt-4.1-mini", 
        model_params={
            "temperature": 0,
            "seed": SEED
        }
    )
    print(f"🔍 Text2Cypher - Using GPT-4o-mini")

    # Few-shot examples - ALWAYS use chunk text search for specific company/data questions
    examples = [
        "USER INPUT: 'What city is NovaGrid Energy Corporation headquartered in?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'NovaGrid' RETURN c.text LIMIT 10",
        "USER INPUT: 'What year is AlTahadi Aviation Group scheduled to take its inaugural flight?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'AlTahadi' RETURN c.text LIMIT 10", 
        "USER INPUT: 'Where is AtlasVentures headquartered?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'AtlasVentures' RETURN c.text LIMIT 10",
        "USER INPUT: 'What is the revenue of NovaGrid?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'NovaGrid' RETURN c.text LIMIT 10",
        "USER INPUT: 'How many Boeing aircraft does AlTahadi have?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'Boeing' OR c.text CONTAINS 'aircraft' RETURN c.text LIMIT 10",
        "USER INPUT: 'Which system must integrate with SAP Concur?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'SAP Concur' OR c.text CONTAINS 'integration' RETURN c.text LIMIT 10",
        "USER INPUT: 'What jobs will be created by 2030?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS '2030' OR c.text CONTAINS 'jobs' RETURN c.text LIMIT 10",
        "USER INPUT: 'Which RFP mentions virtual accounts?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'virtual account' OR c.text CONTAINS 'Virtual Account' RETURN c.text LIMIT 10",
        "USER INPUT: 'When are presentations scheduled?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'presentation' OR c.text CONTAINS 'August' RETURN c.text LIMIT 10",
        "USER INPUT: 'What is AtlasVentures proposal deadline?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'AtlasVentures' OR c.text CONTAINS 'deadline' RETURN c.text LIMIT 10"
    ]
    
    with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        print(f"🔍 Executing Text2Cypher query for: {query}")
        
        try:
            from neo4j_graphrag.retrievers import Text2CypherRetriever
            
            # Initialize the Text2Cypher retriever with GPT-4-turbo
            retriever = Text2CypherRetriever(
                driver=driver,
                llm=text2cypher_llm,  # Use the more powerful model
                neo4j_schema=neo4j_schema,
                examples=examples,
                neo4j_database="neo4j"
            )
            
            print(f"🔍 Text2Cypher retriever initialized successfully")
            
            # Generate Cypher query and execute it
            search_results = retriever.search(query_text=query)
            
            # Try to access the generated Cypher query for debugging
            if hasattr(search_results, 'metadata') and search_results.metadata:
                if 'cypher_query' in search_results.metadata:
                    print(f"🔍 Generated Cypher: {search_results.metadata['cypher_query']}")
                elif 'cypher' in search_results.metadata:
                    print(f"🔍 Generated Cypher: {search_results.metadata['cypher']}")
                else:
                    print(f"🔍 Metadata keys: {list(search_results.metadata.keys())}")
                    print(f"🔍 Full metadata: {search_results.metadata}")
            
            # Check if we can access the retriever's last query
            if hasattr(retriever, '_last_cypher_query'):
                print(f"🔍 Last Cypher query: {retriever._last_cypher_query}")
            
            print(f"🔍 Text2Cypher search completed, type: {type(search_results)}")
            
            if hasattr(search_results, 'items'):
                items = search_results.items
                print(f"🔍 Found {len(items)} items in search_results.items")
            else:
                items = search_results if isinstance(search_results, list) else [search_results]
                print(f"🔍 Using direct results, type: {type(items)}, length: {len(items) if hasattr(items, '__len__') else 'unknown'}")
            
            print(f"✅ Text2Cypher retrieved {len(items) if hasattr(items, '__len__') else 'unknown'} results")
            
            # Format results for LLM processing
            retrieval_details = []
            context_parts = []
            
            for i, item in enumerate(items, 1):
                print(f"🔍 Processing item {i}: {type(item)}")
                
                if hasattr(item, 'content'):
                    content = item.content
                    print(f"🔍 Item {i} content: {str(content)[:100]}...")
                elif isinstance(item, dict):
                    content = str(item)
                    print(f"🔍 Item {i} dict: {str(content)[:100]}...")
                else:
                    content = str(item)
                    print(f"🔍 Item {i} string: {str(content)[:100]}...")
                
                context_parts.append(f"Result {i}:\n{content}")
                retrieval_details.append({
                    'content': content,
                    'source': 'Text2Cypher Query',
                    'type': 'cypher_result'
                })
            
            if not context_parts:
                print("⚠️ No context parts found!")
                return {
                    'method': 'Text2Cypher + LLM',
                    'query': query,
                    'final_answer': 'No results found using Text2Cypher approach.',
                    'retrieved_chunks': 0,
                    'retrieval_details': []
                }
            
            context = "\n\n".join(context_parts)
            print(f"🔍 Combined context length: {len(context)} characters")
            
            # Generate LLM response with Text2Cypher results
            prompt = f"""Based on the following query results from a knowledge graph database, provide a comprehensive answer to the question.

Question: {query}

Query Results:
{context}

Instructions:
1. Use the information from the query results to answer the question directly
2. If the results contain the exact answer, state it clearly
3. If the results are partial or need interpretation, explain what you found
4. If the results don't contain enough information to answer the question, state this clearly
5. Be factual and only use information present in the results
6. Format your response clearly and concisely

Please provide a factual, well-structured response."""

            try:
                llm_response = text2cypher_llm.invoke(prompt)  # Use GPT-4-turbo for final answer too
                final_answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                print(f"🔍 LLM response generated: {len(final_answer)} characters")
            except Exception as e:
                final_answer = f"Error generating LLM response: {e}"
                print(f"❌ LLM error: {e}")
            
            return {
                'method': 'Text2Cypher + LLM',
                'query': query,
                'final_answer': final_answer,
                'retrieved_chunks': len(items) if hasattr(items, '__len__') else 0,
                'retrieval_details': retrieval_details
            }
            
        except Exception as e:
            print(f"❌ Text2Cypher error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'method': 'Text2Cypher + LLM',
                'query': query,
                'final_answer': f"Error with Text2Cypher processing: {e}",
                'retrieved_chunks': 0,
                'retrieval_details': []
            }

# %% [markdown]
# ### 4. Three-Way Comparison Function

# %%
def compare_all_approaches(query: str, k: int = 5):
    """Compare ChromaDB vs GraphRAG vs Text2Cypher approaches side by side"""
    print(f"\n{'='*80}")
    print(f"🔍 QUERY: {query}")
    print(f"{'='*80}")
    
    # Get results from all three approaches
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
    
    print(f"\n🟡 Text2Cypher + LLM Result:")
    print("-" * 50)
    text2cypher_result = query_neo4j_text2cypher(query)
    print(f"Retrieved Results: {text2cypher_result['retrieved_chunks']}")
    print(f"\n{text2cypher_result['final_answer']}")
    
    print(f"\n{'='*80}")
    print("✅ THREE-WAY COMPARISON COMPLETE")
    print(f"{'='*80}")
    
    return {
        'chroma_result': chroma_result,
        'graphrag_result': graphrag_result,
        'text2cypher_result': text2cypher_result
    }


# Create aliases for backward compatibility
query_chroma = query_chroma_with_llm
query_neo4j = query_neo4j_with_llm
query_neo4j_enhanced = query_neo4j_with_llm
query_text2cypher = query_neo4j_text2cypher

# %% [markdown]
# ## Test the Comparison

# %%
# Example usage - remove this if you don't want auto-execution
if __name__ == "__main__":
    test_query = "What are the main vendor requirements?"
    results = compare_all_approaches(test_query, k=1)
