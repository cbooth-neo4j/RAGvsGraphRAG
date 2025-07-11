"""
RAG Retrievers Module

This module provides easy access to all retriever implementations:
- ChromaDB RAG: Traditional vector similarity search
- GraphRAG: Basic Neo4j graph-enhanced vector search  
- Advanced GraphRAG: Intelligent global/local routing with community detection
- DRIFT GraphRAG: Microsoft's iterative refinement with action graphs
- Text2Cypher RAG: Natural language to Cypher query translation
- Neo4j Vector RAG: Pure vector similarity search using Neo4j vector index

Each retriever can be imported directly or accessed through the universal interface.
"""

# Core imports for all retrievers
try:
    from .chroma_retriever import query_chroma_rag, create_chroma_retriever, ChromaRetriever
    from .graph_rag_retriever import query_graphrag, create_graphrag_retriever, GraphRAGRetriever 
    from .text2cypher_retriever import query_text2cypher_rag, create_text2cypher_retriever, Text2CypherRAGRetriever
    from .neo4j_vector_retriever import query_neo4j_vector_rag, create_neo4j_vector_retriever, Neo4jVectorRetriever
    
    # Advanced retrievers with conditional imports
    try:
        from .advanced_graphrag_retriever import (
            query_advanced_graphrag, 
            create_advanced_graphrag_retriever,
            GraphRAGHybridRetriever
        )
        ADVANCED_GRAPHRAG_AVAILABLE = True
    except ImportError as e:
        print(f"Advanced GraphRAG not available: {e}")
        ADVANCED_GRAPHRAG_AVAILABLE = False
        query_advanced_graphrag = None
        create_advanced_graphrag_retriever = None
        GraphRAGHybridRetriever = None
    
    try:
        from .drift_graphrag_retriever import query_drift_graphrag, create_drift_retriever, DriftGraphRAGRetriever
        DRIFT_GRAPHRAG_AVAILABLE = True
    except ImportError as e:
        print(f"DRIFT GraphRAG not available: {e}")
        DRIFT_GRAPHRAG_AVAILABLE = False
        query_drift_graphrag = None
        create_drift_retriever = None
        DriftGraphRAGRetriever = None

except ImportError as e:
    print(f"Error importing retrievers: {e}")
    # Provide fallback None values
    query_chroma_rag = None
    query_graphrag = None
    query_advanced_graphrag = None
    query_drift_graphrag = None
    query_text2cypher_rag = None
    query_neo4j_vector_rag = None
    ADVANCED_GRAPHRAG_AVAILABLE = False
    DRIFT_GRAPHRAG_AVAILABLE = False

# Central registry of all available retrievers
AVAILABLE_RETRIEVERS = {
    'chroma': {
        'function': query_chroma_rag,
        'name': 'ChromaDB RAG',
        'description': 'Traditional vector similarity search',
        'available': query_chroma_rag is not None
    },
    'graphrag': {
        'function': query_graphrag,
        'name': 'GraphRAG',
        'description': 'Basic Neo4j graph-enhanced vector search',
        'available': query_graphrag is not None
    },
    'advanced_graphrag': {
        'function': query_advanced_graphrag,
        'name': 'Advanced GraphRAG',
        'description': 'Intelligent global/local routing with community detection',
        'available': ADVANCED_GRAPHRAG_AVAILABLE
    },
    'drift_graphrag': {
        'function': query_drift_graphrag,
        'name': 'DRIFT GraphRAG',
        'description': 'Microsoft\'s iterative refinement with action graphs',
        'available': DRIFT_GRAPHRAG_AVAILABLE
    },
    'text2cypher': {
        'function': query_text2cypher_rag,
        'name': 'Text2Cypher RAG',
        'description': 'Natural language to Cypher query translation',
        'available': query_text2cypher_rag is not None
    },
    'neo4j_vector': {
        'function': query_neo4j_vector_rag,
        'name': 'Neo4j Vector RAG',
        'description': 'Pure vector similarity search using Neo4j vector index',
        'available': query_neo4j_vector_rag is not None
    }
}


def get_available_retrievers():
    """Get list of available retrievers"""
    return {k: v for k, v in AVAILABLE_RETRIEVERS.items() if v['available']}


def get_retriever_function(approach: str):
    """Get retriever function by approach name"""
    if approach in AVAILABLE_RETRIEVERS:
        retriever_info = AVAILABLE_RETRIEVERS[approach]
        if retriever_info['available']:
            return retriever_info['function']
        else:
            raise ImportError(f"Retriever '{approach}' is not available. Check dependencies.")
    else:
        raise ValueError(f"Unknown retriever approach: {approach}. Available: {list(AVAILABLE_RETRIEVERS.keys())}")


def query_retriever(approach: str, query: str, **kwargs):
    """
    Universal retriever interface
    
    Args:
        approach: Retriever type ('chroma', 'graphrag', 'advanced_graphrag', 'drift_graphrag', 'text2cypher', 'neo4j_vector')
        query: The search query
        **kwargs: Additional parameters for the specific retriever
    
    Returns:
        Dictionary with response and retrieval details
    """
    retriever_func = get_retriever_function(approach)
    return retriever_func(query, **kwargs)


def test_all_retrievers(test_query: str = "What are the main requirements mentioned in the documents?"):
    """
    Test all available retrievers with a sample query
    
    Args:
        test_query: Query to test with
    
    Returns:
        Dictionary of results from each available retriever
    """
    results = {}
    available = get_available_retrievers()
    
    print(f"Testing {len(available)} available retrievers with query: '{test_query}'")
    print("=" * 60)
    
    for approach, info in available.items():
        print(f"\n🧪 Testing {info['name']}...")
        try:
            result = query_retriever(approach, test_query)
            results[approach] = result
            print(f"✅ {info['name']}: Success")
            print(f"   Method: {result.get('method', 'Unknown')}")
            print(f"   Retrieved: {result.get('performance_metrics', {}).get('retrieved_chunks', 'Unknown')} items")
            answer_preview = result.get('final_answer', '')[:100] + "..." if len(result.get('final_answer', '')) > 100 else result.get('final_answer', '')
            print(f"   Answer: {answer_preview}")
        except Exception as e:
            print(f"❌ {info['name']}: Failed - {e}")
            results[approach] = {'error': str(e)}
    
    print("\n" + "=" * 60)
    print(f"✅ Testing completed. {len([r for r in results.values() if 'error' not in r])} successful, {len([r for r in results.values() if 'error' in r])} failed.")
    
    return results


# Export all main functions and classes
__all__ = [
    # Main interface functions
    'query_chroma_rag',
    'query_graphrag', 
    'query_advanced_graphrag',
    'query_drift_graphrag',
    'query_text2cypher_rag',
    'query_neo4j_vector_rag',
    
    # Factory functions
    'create_chroma_retriever',
    'create_graphrag_retriever',
    'create_text2cypher_retriever',
    'create_neo4j_vector_retriever',
    'create_advanced_graphrag_retriever',
    
    # Retriever classes
    'ChromaRetriever',
    'GraphRAGRetriever',
    'Text2CypherRAGRetriever',
    'Neo4jVectorRetriever',
    
    # Utility functions
    'get_available_retrievers',
    'get_retriever_function',
    'query_retriever',
    'test_all_retrievers',
    
    # Registry
    'AVAILABLE_RETRIEVERS'
]

# Conditional exports for advanced retrievers
if ADVANCED_GRAPHRAG_AVAILABLE:
    __all__.extend(['create_advanced_graphrag_retriever', 'GraphRAGHybridRetriever'])

if DRIFT_GRAPHRAG_AVAILABLE:
    __all__.extend(['create_drift_retriever', 'DriftGraphRAGRetriever'])

print(f"🔗 Retrievers module loaded. Available approaches: {list(get_available_retrievers().keys())}") 