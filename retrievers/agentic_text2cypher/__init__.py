"""
Agentic Text2Cypher Retriever Package

This package provides a Deep Agent-powered approach to querying Neo4j
using adaptive, multi-step graph exploration.

Components:
- retriever: Main AgenticText2CypherRetriever class
- tools: Neo4j agent tools (get_schema, read_cypher, list_gds)
"""

from .retriever import (
    AgenticText2CypherRetriever,
    AgenticSearchResult,
    create_agentic_text2cypher_retriever,
    query_agentic_text2cypher_rag,
    GRAPH_EXPLORATION_SYSTEM_PROMPT,
    DEEP_AGENTS_AVAILABLE
)

from .tools import (
    Neo4jAgentTools,
    neo4j_get_schema,
    neo4j_read_cypher,
    neo4j_list_gds,
    AGENT_TOOLS,
    AGENT_TOOLS_MINIMAL
)

__all__ = [
    # Retriever
    'AgenticText2CypherRetriever',
    'AgenticSearchResult',
    'create_agentic_text2cypher_retriever',
    'query_agentic_text2cypher_rag',
    'GRAPH_EXPLORATION_SYSTEM_PROMPT',
    'DEEP_AGENTS_AVAILABLE',
    
    # Tools
    'Neo4jAgentTools',
    'neo4j_get_schema',
    'neo4j_read_cypher',
    'neo4j_list_gds',
    'AGENT_TOOLS',
    'AGENT_TOOLS_MINIMAL'
]

