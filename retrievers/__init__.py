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
    from .text2cypher_retriever import (
        query_text2cypher_rag, 
        create_text2cypher_retriever, 
        Text2CypherRAGRetriever,
        # Cypher verification classes (inlined in text2cypher_retriever)
        CypherVerificationPipeline,
        SyntaxVerifier,
        ExecutionVerifier,
        LLMVerifier,
        VerificationResult,
        create_verification_pipeline,
        # Cypher correction classes (inlined in text2cypher_retriever)
        CypherCorrectionPipeline,
        RuleBasedCorrector,
        LLMCorrector,
        CorrectionResult,
        create_correction_pipeline
    )
    try:
        from .hybrid_cypher_retriever import (
            query_hybrid_cypher_rag,
            create_hybrid_cypher_retriever,
            HybridCypherRAGRetriever
        )
    except Exception as e:
        import traceback
        print("================FAILED===============")
        print(traceback.print_exc())
        print("================FAILED===============")

    from .neo4j_vector_retriever import query_neo4j_vector_rag, create_neo4j_vector_retriever, Neo4jVectorRetriever
    
    # Advanced retrievers with conditional imports
    try:
        from .advanced_graphrag_retriever import (
            query_advanced_graphrag_sync as query_advanced_graphrag, 
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
        from .drift_graphrag_retriever import query_drift_graphrag_sync as query_drift_graphrag, create_drift_retriever, DriftGraphRAGRetriever
        DRIFT_GRAPHRAG_AVAILABLE = True
    except ImportError as e:
        print(f"DRIFT GraphRAG not available: {e}")
        DRIFT_GRAPHRAG_AVAILABLE = False
        query_drift_graphrag = None
        create_drift_retriever = None
        DriftGraphRAGRetriever = None

except ImportError as e:
    import traceback
    print("+"*30)
    print(traceback.print_exc())
    print("+"*35)
    print(f"Error importing retrievers: {e}")
    # Provide fallback None values
    query_chroma_rag = None
    query_graphrag = None
    query_advanced_graphrag = None
    query_drift_graphrag = None
    query_text2cypher_rag = None
    query_neo4j_vector_rag = None
    query_hybrid_cypher_rag = None
    ADVANCED_GRAPHRAG_AVAILABLE = False
    DRIFT_GRAPHRAG_AVAILABLE = False
    # Agentic and tools will be tried separately below
    AGENTIC_TEXT2CYPHER_AVAILABLE = False
    query_agentic_text2cypher_rag = None
    create_agentic_text2cypher_retriever = None
    AgenticText2CypherRetriever = None
    # Neo4j Agent Tools fallbacks
    Neo4jAgentTools = None
    neo4j_get_schema = None
    neo4j_read_cypher = None
    neo4j_list_gds = None
    AGENT_TOOLS = []
    AGENT_TOOLS_MINIMAL = []

# Central registry of all available retrievers
# Keys use hyphens for CLI consistency (e.g., --retrievers advanced-graphrag)
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
    'advanced-graphrag': {
        'function': query_advanced_graphrag,
        'name': 'Advanced GraphRAG',
        'description': 'Intelligent global/local routing with community detection',
        'available': ADVANCED_GRAPHRAG_AVAILABLE
    },
    'drift-graphrag': {
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
    'neo4j-vector': {
        'function': query_neo4j_vector_rag,
        'name': 'Neo4j Vector RAG',
        'description': 'Pure vector similarity search using Neo4j vector index',
        'available': query_neo4j_vector_rag is not None
    },
    'hybrid-cypher': {
        'function': query_hybrid_cypher_rag,
        'name': 'Hybrid Cypher RAG',
        'description': 'Hybrid (vector + fulltext) with generic neighborhood expansion',
        'available': query_hybrid_cypher_rag is not None
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
        print(f"\nTesting {info['name']}...")
        try:
            result = query_retriever(approach, test_query)
            results[approach] = result
            print(f"OK {info['name']}: Success")
            print(f"   Method: {result.get('method', 'Unknown')}")
            print(f"   Retrieved: {result.get('performance_metrics', {}).get('retrieved_chunks', 'Unknown')} items")
            answer_preview = result.get('final_answer', '')[:100] + "..." if len(result.get('final_answer', '')) > 100 else result.get('final_answer', '')
            print(f"   Answer: {answer_preview}")
        except Exception as e:
            print(f"FAILED {info['name']}: Failed - {e}")
            results[approach] = {'error': str(e)}
    
    print("\n" + "=" * 60)
    print(f"Testing completed. {len([r for r in results.values() if 'error' not in r])} successful, {len([r for r in results.values() if 'error' in r])} failed.")
    
    return results


# Agentic Text2Cypher (Deep Agent based) - imported from subpackage
# Must be imported BEFORE __all__ and conditional checks that use AGENTIC_TEXT2CYPHER_AVAILABLE
try:
    from .agentic_text2cypher import (
        query_agentic_text2cypher_rag,
        create_agentic_text2cypher_retriever,
        AgenticText2CypherRetriever,
        Neo4jAgentTools,
        neo4j_get_schema,
        neo4j_read_cypher,
        neo4j_list_gds,
        AGENT_TOOLS,
        AGENT_TOOLS_MINIMAL,
        DEEP_AGENTS_AVAILABLE
    )
    AGENTIC_TEXT2CYPHER_AVAILABLE = DEEP_AGENTS_AVAILABLE
    # Add to registry after successful import
    if AGENTIC_TEXT2CYPHER_AVAILABLE:
        AVAILABLE_RETRIEVERS['agentic-text2cypher'] = {
            'function': query_agentic_text2cypher_rag,
            'name': 'Agentic Text2Cypher RAG',
            'description': 'Deep Agent-powered adaptive graph exploration',
            'available': True
        }
except ImportError as e:
    print(f"Agentic Text2Cypher not available: {e}")
    AGENTIC_TEXT2CYPHER_AVAILABLE = False
    DEEP_AGENTS_AVAILABLE = False
    query_agentic_text2cypher_rag = None
    create_agentic_text2cypher_retriever = None
    AgenticText2CypherRetriever = None
    Neo4jAgentTools = None
    neo4j_get_schema = None
    neo4j_read_cypher = None
    neo4j_list_gds = None
    AGENT_TOOLS = []
    AGENT_TOOLS_MINIMAL = []

# Export all main functions and classes
__all__ = [
    # Main interface functions
    'query_chroma_rag',
    'query_graphrag', 
    'query_advanced_graphrag',
    'query_drift_graphrag',
    'query_text2cypher_rag',
    'query_agentic_text2cypher_rag',
    'query_neo4j_vector_rag',
    
    # Factory functions
    'create_chroma_retriever',
    'create_graphrag_retriever',
    'create_text2cypher_retriever',
    'create_agentic_text2cypher_retriever',
    'create_neo4j_vector_retriever',
    'create_advanced_graphrag_retriever',
    
    # Retriever classes
    'ChromaRetriever',
    'GraphRAGRetriever',
    'Text2CypherRAGRetriever',
    'AgenticText2CypherRetriever',
    'Neo4jVectorRetriever',
    'HybridCypherRAGRetriever',
    
    # Neo4j Agent Tools
    'Neo4jAgentTools',
    'neo4j_get_schema',
    'neo4j_read_cypher',
    'neo4j_list_gds',
    'AGENT_TOOLS',
    'AGENT_TOOLS_MINIMAL',
    
    # Cypher verification classes
    'CypherVerificationPipeline',
    'SyntaxVerifier',
    'ExecutionVerifier',
    'LLMVerifier',
    'VerificationResult',
    'create_verification_pipeline',
    
    # Cypher correction classes
    'CypherCorrectionPipeline',
    'RuleBasedCorrector',
    'LLMCorrector',
    'CorrectionResult',
    'create_correction_pipeline',
    
    # Utility functions
    'get_available_retrievers',
    'get_retriever_function',
    'query_retriever',
    'test_all_retrievers',
    
    # Registry
    'AVAILABLE_RETRIEVERS',
    
    # Availability flags
    'AGENTIC_TEXT2CYPHER_AVAILABLE',
    'DEEP_AGENTS_AVAILABLE'
]

# Conditional exports for advanced retrievers
if ADVANCED_GRAPHRAG_AVAILABLE:
    __all__.extend(['create_advanced_graphrag_retriever', 'GraphRAGHybridRetriever'])

if DRIFT_GRAPHRAG_AVAILABLE:
    __all__.extend(['create_drift_retriever', 'DriftGraphRAGRetriever'])

if AGENTIC_TEXT2CYPHER_AVAILABLE:
    __all__.extend(['AgenticText2CypherRetriever'])

print(f"Retrievers module loaded. Available approaches: {list(get_available_retrievers().keys())}") 