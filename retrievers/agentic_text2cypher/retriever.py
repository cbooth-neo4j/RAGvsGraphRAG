"""
Agentic Text2Cypher Retriever - Deep Agent-powered Graph Exploration

This module implements an adaptive, agent-based approach to querying Neo4j
using Deep Agents (LangGraph under the hood). Unlike the fixed Text2Cypher
pipeline, this agent can:

- Inspect the schema before querying
- Try multiple query strategies
- Examine and interpret results
- Iterate until it finds the answer
- Handle failures gracefully

Features:
- Uses GPT-5.2 (or configured thinking model) for reasoning
- Neo4j tools: get_schema, read_cypher
- Built-in planning with write_todos
- Context management with file tools
- Configurable via AGENTIC_TEXT2CYPHER_MODEL env var
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import time

# Deep Agents
try:
    from deepagents import create_deep_agent
    DEEP_AGENTS_AVAILABLE = True
except ImportError:
    DEEP_AGENTS_AVAILABLE = False
    create_deep_agent = None

# Local imports
from config import get_model_config, get_agentic_text2cypher_llm
from .tools import (
    neo4j_get_schema,
    neo4j_read_cypher,
    AGENT_TOOLS_MINIMAL
)
from utils.graph_rag_logger import get_logger

load_dotenv()
logger = get_logger(__name__)


# System prompt teaching the agent about our graph schema and query strategies
GRAPH_EXPLORATION_SYSTEM_PROMPT = """You are an expert graph database researcher with deep knowledge of Neo4j and Cypher.
Your task is to answer questions by exploring a knowledge graph database.

## YOUR TOOLS

You have access to Neo4j database tools:

### `neo4j_get_schema`
Returns the database schema including:
- Node labels (entity types) and their counts
- Properties available on each node type
- Relationship types connecting nodes

ALWAYS call this FIRST to understand what's in the database.

### `neo4j_read_cypher`
Executes read-only Cypher queries. Use this to:
- Find entities by name or property
- Traverse relationships
- Aggregate and analyze data

## DATABASE SCHEMA PATTERNS

This knowledge graph has the following structure:

### Entity Nodes
The database contains various entity types (PERSON, ORGANIZATION, LOCATION, FILM, WORK, etc.)
Each entity typically has:
- `name`: The entity's name (use CONTAINS for fuzzy matching)
- `ai_summary`: **KEY PROPERTY** - AI-generated summary often containing the answer
- `description`: Detailed description of the entity
- `embedding`: Vector embedding for similarity search

### Relationships
All entity-to-entity relationships use the type `RELATED_TO` with an `evidence` property:
```cypher
(entity1)-[:RELATED_TO {evidence: "describes the relationship"}]->(entity2)
```

### Document Structure
- `Document` nodes: Source documents
- `Chunk` nodes: Text segments from documents
- `Chunk.text`: Raw text content (fallback for direct text search)

## QUERY STRATEGIES (in order of preference)

### Strategy 1: Entity Lookup with ai_summary (FASTEST)
```cypher
MATCH (e:__Entity__) 
WHERE e.name CONTAINS 'SearchTerm' 
RETURN e.name, e.ai_summary, e.description 
LIMIT 5
```
The `ai_summary` property often contains the answer directly!

### Strategy 2: Relationship Traversal with Evidence
```cypher
MATCH (e1:__Entity__)-[r:RELATED_TO]->(e2:__Entity__)
WHERE e1.name CONTAINS 'SearchTerm'
RETURN e1.name, r.evidence, e2.name
LIMIT 10
```
The `evidence` property describes HOW entities are related.

### Strategy 3: Two-Hop Exploration
```cypher
MATCH (e1:__Entity__)-[:RELATED_TO*1..2]-(e2:__Entity__)
WHERE e1.name CONTAINS 'SearchTerm'
RETURN DISTINCT e1.name, e2.name, e2.ai_summary
LIMIT 10
```

### Strategy 4: Chunk Text Search (FALLBACK)
```cypher
MATCH (c:Chunk)
WHERE c.text CONTAINS 'specific phrase'
RETURN c.text
LIMIT 5
```
Use this when entity search doesn't find results.

## IMPORTANT TIPS

1. **ALWAYS START** with neo4j_get_schema() to see what's available
2. **USE CONTAINS** for name matching - exact matches often fail
3. **CHECK ai_summary FIRST** - it often has the answer
4. **LIMIT RESULTS** to avoid overwhelming output
5. **TRY MULTIPLE STRATEGIES** if the first doesn't work
6. **For dates/numbers** - search in ai_summary or chunk text
7. **For locations** - entities may be typed as LOCATION or GPE
8. **Case sensitivity** - Neo4j is case-sensitive, use toLower() if needed

## RESPONSE FORMAT

**CRITICAL: Your final response must be EXTREMELY CONCISE.**

HotpotQA-style benchmarks expect answers in this exact format:
- For yes/no questions: respond with just `yes` or `no` (lowercase)
- For "who" questions: respond with just the person/entity name
- For "what" questions: respond with just the specific fact or entity
- For "which" questions: respond with just the selection
- For comparison questions ("who is older/taller/etc"): respond with just the name

**Examples of correct final answers:**
- Q: "Were X and Y of the same nationality?" → `yes` or `no`
- Q: "Who directed the film X?" → `Tim Burton`
- Q: "What position did X hold?" → `Chief of Protocol`
- Q: "Who is older, X or Y?" → `Terry Richardson`

**DO NOT include:**
- Explanations or reasoning in your final answer
- Source citations or confidence levels
- Full sentences like "Yes, they are both American"
- Phrases like "The answer is..." or "Based on my research..."

Your thinking and exploration can be detailed, but your FINAL ANSWER must be just the bare fact.

If you cannot find the answer, respond with: `unknown`
"""


@dataclass 
class AgenticSearchResult:
    """Result from an agentic search operation"""
    question: str
    answer: str
    iterations: int
    queries_executed: List[str]
    tool_calls: int
    search_time_seconds: float
    success: bool
    error: Optional[str] = None


class AgenticText2CypherRetriever:
    """
    Deep Agent-powered graph retriever.
    
    Uses an LLM agent with Neo4j tools to adaptively explore
    the knowledge graph and answer questions.
    """
    
    def __init__(self, model: str = None, provider: str = None):
        """
        Initialize the agentic retriever.
        
        Args:
            model: Override the configured model (e.g., 'gpt-5.2')
            provider: Override the configured provider (e.g., 'openai')
        """
        if not DEEP_AGENTS_AVAILABLE:
            raise ImportError(
                "Deep Agents not installed. Install with: pip install deepagents"
            )
        
        self.config = get_model_config()
        
        # Get model configuration
        effective_model = self.config.effective_agentic_text2cypher_model
        effective_provider = self.config.effective_agentic_text2cypher_provider
        
        logger.info(f"Agentic Text2Cypher - Provider: {effective_provider.value}, "
                   f"Model: {effective_model.value}")
        
        # Check if using thinking model
        is_thinking = self.config.is_thinking_model(effective_model)
        if is_thinking:
            logger.info("Using thinking model - extended reasoning enabled")
        
        # Create the LLM
        self.llm = get_agentic_text2cypher_llm()
        
        # Create the Deep Agent
        self.agent = create_deep_agent(
            model=self.llm,
            tools=AGENT_TOOLS_MINIMAL,
            system_prompt=GRAPH_EXPLORATION_SYSTEM_PROMPT
        )
        
        logger.info("Agentic Text2Cypher retriever initialized")
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        Execute an agentic search to answer the query.
        
        The agent will:
        1. Plan its approach
        2. Inspect the schema
        3. Execute Cypher queries
        4. Interpret results
        5. Iterate until answer found or give up
        
        Args:
            query: The natural language question
            
        Returns:
            Dictionary with answer and search details
        """
        start_time = time.time()
        logger.info(f"Agentic search for: {query}")
        
        try:
            # Invoke the agent
            result = self.agent.invoke({
                "messages": [
                    {"role": "user", "content": query}
                ]
            })
            
            # Extract the final answer from agent response
            final_answer = ""
            tool_calls = 0
            queries_executed = []
            
            if "messages" in result:
                # Get the last assistant message
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'content') and msg.content:
                        if hasattr(msg, 'type') and msg.type == 'ai':
                            final_answer = msg.content
                            break
                        elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                            final_answer = msg.get('content', '')
                            break
                
                # Count tool calls and extract queries
                for msg in result["messages"]:
                    if hasattr(msg, 'tool_calls'):
                        tool_calls += len(msg.tool_calls)
                        for tc in msg.tool_calls:
                            if tc.get('name') == 'neo4j_read_cypher':
                                args = tc.get('args', {})
                                if 'query' in args:
                                    queries_executed.append(args['query'])
            
            search_time = time.time() - start_time
            
            return {
                'method': 'Agentic Text2Cypher',
                'query': query,
                'final_answer': final_answer,
                'retrieved_chunks': len(queries_executed),  # Approximate
                'retrieval_details': [
                    {'content': q, 'source': 'Cypher Query', 'type': 'cypher'}
                    for q in queries_executed
                ],
                'tool_calls': tool_calls,
                'queries_executed': queries_executed,
                'search_time_seconds': search_time,
                'success': bool(final_answer)
            }
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Agentic search error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'method': 'Agentic Text2Cypher',
                'query': query,
                'final_answer': f"Error during agentic search: {str(e)}",
                'retrieved_chunks': 0,
                'retrieval_details': [],
                'tool_calls': 0,
                'queries_executed': [],
                'search_time_seconds': search_time,
                'success': False,
                'error': str(e)
            }


# Factory function
def create_agentic_text2cypher_retriever(
    model: str = None,
    provider: str = None
) -> AgenticText2CypherRetriever:
    """
    Create an Agentic Text2Cypher retriever instance.
    
    Args:
        model: Override model (defaults to AGENTIC_TEXT2CYPHER_MODEL or LLM_MODEL)
        provider: Override provider (defaults to AGENTIC_TEXT2CYPHER_PROVIDER or LLM_PROVIDER)
    
    Returns:
        AgenticText2CypherRetriever instance
    """
    return AgenticText2CypherRetriever(model=model, provider=provider)


# Main interface function for benchmark system
def query_agentic_text2cypher_rag(query: str, **kwargs) -> Dict[str, Any]:
    """
    Agentic Text2Cypher RAG retrieval using Deep Agents.
    
    This is an adaptive, multi-step approach where an LLM agent
    explores the graph database to find answers.
    
    Args:
        query: The search query
        **kwargs: Additional configuration options
    
    Returns:
        Dictionary with response and retrieval details
    """
    try:
        retriever = create_agentic_text2cypher_retriever()
        result = retriever.search(query)
        
        # Format for benchmark compatibility
        return {
            'final_answer': result['final_answer'],
            'retrieval_details': [
                {
                    'content': detail['content'],
                    'metadata': {'source': detail['source'], 'type': detail['type']}
                } for detail in result['retrieval_details']
            ],
            'method': 'agentic_text2cypher_rag',
            'performance_metrics': {
                'retrieved_chunks': result['retrieved_chunks'],
                'completion_time': result['search_time_seconds'],
                'llm_calls': result['tool_calls'],
                'prompt_tokens': 0,  # Not tracked
                'output_tokens': 0,  # Not tracked
                'total_tokens': 0,
                'queries_executed': len(result['queries_executed']),
                'success': result['success']
            }
        }
        
    except ImportError as e:
        logger.error(f"Deep Agents not available: {e}")
        return {
            'final_answer': "Agentic Text2Cypher requires Deep Agents. Install with: pip install deepagents",
            'retrieval_details': [],
            'method': 'agentic_text2cypher_rag_error',
            'performance_metrics': {
                'retrieved_chunks': 0,
                'completion_time': 0,
                'llm_calls': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'queries_executed': 0,
                'success': False
            }
        }
    except Exception as e:
        logger.error(f"Error in Agentic Text2Cypher retrieval: {e}")
        import traceback
        traceback.print_exc()
        return {
            'final_answer': f"Error during Agentic Text2Cypher retrieval: {str(e)}",
            'retrieval_details': [],
            'method': 'agentic_text2cypher_rag_error',
            'performance_metrics': {
                'retrieved_chunks': 0,
                'completion_time': 0,
                'llm_calls': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'queries_executed': 0,
                'success': False
            }
        }

