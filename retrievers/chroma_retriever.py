"""
ChromaDB Retriever - Traditional Vector Similarity Search

This module implements traditional RAG using ChromaDB with configurable embeddings
for vector similarity search and LLM response generation.
Supports both OpenAI API and Ollama local models.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from langchain_chroma import Chroma
import warnings
import logging

# Import centralized configuration
from config import get_model_config, get_embeddings, get_neo4j_llm, ModelProvider

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

class ChromaRetriever:
    """Traditional ChromaDB vector similarity search retriever with configurable models"""
    
    def __init__(self, persist_directory: str = PERSIST_DIRECTORY, collection_name: str = COLLECTION_NAME, 
                 model_config: Optional[Any] = None):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.config = model_config or get_model_config()
        
        # Initialize models based on configuration
        self.embeddings = get_embeddings()
        
        # Use Neo4j GraphRAG LLM for OpenAI, regular LLM for Ollama
        try:
            if self.config.llm_provider == ModelProvider.OPENAI:
                self.llm = get_neo4j_llm()
            else:
                # For Ollama, use regular LangChain LLM
                from config import get_llm
                self.llm = get_llm()
        except Exception as e:
            warnings.warn(f"Could not initialize configured LLM, falling back to default: {e}")
            # Fallback to regular LLM
            from config import get_llm
            self.llm = get_llm()
        
        # Suppress ChromaDB telemetry
        try:
            import chromadb
            chromadb.telemetry.telemetry_client = None
        except:
            pass
        
        # Initialize vectorstore
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
    
    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Query ChromaDB and generate LLM response"""
        
        # Perform similarity search
        docs = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        
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
            # Handle different LLM response formats
            if self.config.llm_provider == ModelProvider.OPENAI and hasattr(self.llm, 'invoke'):
                # Neo4j GraphRAG LLM
                llm_response = self.llm.invoke(prompt)
                final_answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            else:
                # Regular LangChain LLM (including Ollama)
                llm_response = self.llm.invoke(prompt)
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
    
    def get_collection_size(self) -> int:
        """Get the number of documents in the ChromaDB collection"""
        return self.vectorstore._collection.count()


# Factory function for easy instantiation
def create_chroma_retriever(persist_directory: str = PERSIST_DIRECTORY, collection_name: str = COLLECTION_NAME, 
                          model_config: Optional[Any] = None) -> ChromaRetriever:
    """Create a ChromaDB retriever instance with configurable models"""
    return ChromaRetriever(persist_directory, collection_name, model_config)



# Main interface function for integration with benchmark system
def query_chroma_rag(query: str, k: int = 3, **kwargs) -> Dict[str, Any]:
    """
    ChromaDB RAG retrieval with LLM response generation
    
    Args:
        query: The search query
        k: Number of similar documents to retrieve
        **kwargs: Additional configuration options (for compatibility)
    
    Returns:
        Dictionary with response and retrieval details
    """
    try:
        retriever = create_chroma_retriever()
        result = retriever.search(query, k)
        
        # Format response for benchmark compatibility
        return {
            'final_answer': result['final_answer'],
            'retrieval_details': [
                {
                    'content': detail['content'],
                    'metadata': detail['metadata'],
                    'similarity_score': detail['similarity_score']
                } for detail in result['retrieval_details']
            ],
            'method': 'chroma_rag',
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
        print(f"Error in ChromaDB retrieval: {e}")
        return {
            'final_answer': f"Error during ChromaDB retrieval: {str(e)}",
            'retrieval_details': [],
            'method': 'chroma_rag_error',
            'performance_metrics': {
                'retrieved_chunks': 0,
                'completion_time': 0,
                'llm_calls': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        } 