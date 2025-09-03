"""
Text2Cypher Retriever - Natural Language to Cypher Query Translation

This module implements Text2Cypher using Neo4j with natural language
to Cypher query translation for direct graph database querying.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any
import neo4j
from neo4j_graphrag.retrievers import Text2CypherRetriever
from langchain_neo4j import Neo4jGraph

# Import centralized configuration
from config import get_model_config, get_neo4j_embeddings, get_neo4j_llm, ModelProvider

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

# Initialize components with centralized configuration
SEED = 42
embeddings = get_neo4j_embeddings()

# Use centralized LLM configuration for Cypher query generation
text2cypher_llm = get_neo4j_llm()

class Text2CypherRAGRetriever:
    """Text2Cypher retriever with natural language to Cypher conversion"""
    
    def __init__(self):
        self.embeddings = embeddings
        self.llm = text2cypher_llm
        self.neo4j_uri = NEO4J_URI
        self.neo4j_user = NEO4J_USER
        self.neo4j_password = NEO4J_PASSWORD
        
        # Few-shot examples - ALWAYS use chunk text search for specific company/data questions
        self.examples = [
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
        
        # Extract Neo4j schema
        self.neo4j_schema = self._extract_schema()
    
    def _extract_schema(self) -> str:
        """Dynamically extract the Neo4j schema from the actual database"""
        try:
            # Create Neo4j graph connection to extract schema
            graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
                enhanced_schema=True
            )
            
            # Refresh and get the current schema
            graph.refresh_schema()
            schema = graph.schema
            print(f"🔍 Text2Cypher - Schema extracted: {len(schema)} characters")
            return schema
        except Exception as e:
            print(f"Error extracting schema: {e}")
            return "Schema extraction failed"
    
    def search(self, query: str) -> Dict[str, Any]:
        """Neo4j Text2Cypher query with natural language to Cypher conversion"""
        
        with neo4j.GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)) as driver:
            print(f"🔍 Executing Text2Cypher query for: {query}")
            
            try:
                # Initialize the Text2Cypher retriever
                retriever = Text2CypherRetriever(
                    driver=driver,
                    llm=self.llm,
                    neo4j_schema=self.neo4j_schema,
                    examples=self.examples,
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
                    llm_response = self.llm.invoke(prompt)
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


# Factory function for easy instantiation
def create_text2cypher_retriever() -> Text2CypherRAGRetriever:
    """Create a Text2Cypher retriever instance"""
    return Text2CypherRAGRetriever()





# Main interface function for integration with benchmark system
def query_text2cypher_rag(query: str, **kwargs) -> Dict[str, Any]:
    """
    Text2Cypher RAG retrieval with natural language to Cypher conversion
    
    Args:
        query: The search query
        **kwargs: Additional configuration options (for compatibility)
    
    Returns:
        Dictionary with response and retrieval details
    """
    try:
        retriever = create_text2cypher_retriever()
        result = retriever.search(query)
        
        # Format response for benchmark compatibility
        return {
            'final_answer': result['final_answer'],
            'retrieval_details': [
                {
                    'content': detail['content'],
                    'metadata': {'source': detail['source'], 'type': detail['type']}
                } for detail in result['retrieval_details']
            ],
            'method': 'text2cypher_rag',
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
        print(f"Error in Text2Cypher retrieval: {e}")
        import traceback
        traceback.print_exc()
        return {
            'final_answer': f"Error during Text2Cypher retrieval: {str(e)}",
            'retrieval_details': [],
            'method': 'text2cypher_rag_error',
            'performance_metrics': {
                'retrieved_chunks': 0,
                'completion_time': 0,
                'llm_calls': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        } 