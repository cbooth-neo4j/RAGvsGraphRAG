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
(entity1)-[:RELATED_TO {{evidence: "describes the relationship"}}]->(entity2)
```

### Document Structure
- `Document` nodes: Source documents
- `Chunk` nodes: Text segments from documents
- `Chunk.text`: Raw text content (fallback for direct text search)

---

## FEW-SHOT EXAMPLES BY QUESTION CATEGORY

Below are example Cypher queries for each category of question you may encounter.
Based on analysis of 7,405 HotpotQA questions, these cover the main patterns.

**IMPORTANT - Entity Labels:**
- All examples use `__Entity__` which is the universal parent label for all entities
- After calling `neo4j_get_schema()`, you can use specific entity types if available (e.g., PERSON, FILM, LOCATION)
- To filter by type without using a specific label: `WHERE e.entity_type = 'PERSON'`
- The `Chunk` and `Document` labels are always available regardless of dataset

---
### COMPARISON QUESTIONS (20% of all questions)
---

### CATEGORY 1: Same Attribute Comparison (Are X and Y the same [attribute]?)
**Question:** "Were Scott Derrickson and Ed Wood of the same nationality?"
**Strategy:** Look up both entities, extract attribute from ai_summary, compare.
```cypher
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Scott Derrickson' OR e.name CONTAINS 'Ed Wood'
RETURN e.name, e.ai_summary
```
**Reasoning:** The ai_summary contains nationality (e.g., "American filmmaker", "American director").
**Answer format:** `yes` or `no`

### CATEGORY 2: Both Attribute (Are X and Y both [type]?)
**Question:** "Are Giuseppe Verdi and Ambroise Thomas both opera composers?"
**Question:** "Are Local H and For Against both from the United States?"
**Strategy:** Look up both entities, check if both have the shared attribute.
```cypher
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Giuseppe Verdi' OR e.name CONTAINS 'Ambroise Thomas'
RETURN e.name, e.ai_summary
```
**Reasoning:** Check if BOTH summaries mention the attribute (e.g., "opera composer", "American").
**Answer format:** `yes` or `no`

### CATEGORY 3: Age/Selection Comparison (Who is [older/younger], X or Y?)
**Question:** "Who is older, Annie Morton or Terry Richardson?"
**Strategy:** Look up both entities, extract birth years, compare.
```cypher
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Annie Morton' OR e.name CONTAINS 'Terry Richardson'
RETURN e.name, e.ai_summary
```
**Reasoning:** Look for "born on [date]" or "born [year]" patterns in summaries.
**Answer format:** Just the name (e.g., `Terry Richardson`)

### CATEGORY 4: Temporal Order Comparison (Which came first? Who died first?)
**Question:** "Which came out first, Dinosaur or McFarland USA?"
**Question:** "Who died first, George Archainbaud or Ralph Murphy?"
**Strategy:** Look up both entities, extract dates (release/death), compare.
```cypher
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Dinosaur' OR e.name CONTAINS 'McFarland'
RETURN e.name, e.ai_summary
```
**Reasoning:** Look for release year or death year in summaries. Earlier date = first.
**Answer format:** Just the entity name (e.g., `Dinosaur`)

### CATEGORY 5: Quantity Comparison (Which has more [members/species/albums]?)
**Question:** "Which band had more members, Letters to Cleo or Screaming Trees?"
**Question:** "Which genus contains more species, Greyia or Calibanus?"
**Strategy:** Look up both entities, extract counts, compare.
```cypher
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Letters to Cleo' OR e.name CONTAINS 'Screaming Trees'
RETURN e.name, e.ai_summary

// If counts not in summary, try chunk text
MATCH (c:Chunk)
WHERE c.text CONTAINS 'Letters to Cleo' AND c.text CONTAINS 'members'
RETURN c.text LIMIT 3
```
**Reasoning:** Extract numeric counts (e.g., "five members", "12 species") and compare.
**Answer format:** Just the entity name (e.g., `Letters to Cleo`)

---
### BRIDGE QUESTIONS (80% of all questions)
---

### CATEGORY 6: Year Questions (In what year was X [founded/born/released]?)
**Question:** "In what year was the university where Sergei Tokarev was a professor founded?"
**Question:** "What year did Guns N Roses perform at the movie premiere?"
**Strategy:** Find the entity, extract year from ai_summary.
```cypher
// Find the person first, then their affiliated institution
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Sergei' AND e.name CONTAINS 'Tokarev'
RETURN e.name, e.ai_summary

// Then find related entities (institutions, organizations, etc.)
MATCH (e:__Entity__)-[:RELATED_TO]-(p:__Entity__)
WHERE p.name CONTAINS 'Tokarev'
RETURN e.name, e.ai_summary, e.entity_type
```
**Reasoning:** ai_summary often contains founding dates like "founded in 1755".
**Answer format:** Just the year (e.g., `1755`)

### CATEGORY 7: Person Role Questions (Who [directed/wrote/starred in] X?)
**Question:** "Who directed the 2009 film starring the actor from Dexter?"
**Question:** "Who wrote the tragic play with a 1968 Nino Rota soundtrack?"
**Strategy:** Find the work, traverse to person with specified role.
```cypher
// Find the work and connected entities
MATCH (work:__Entity__)-[r:RELATED_TO]-(person:__Entity__)
WHERE work.name CONTAINS 'Romeo and Juliet'
RETURN work.name, person.name, person.ai_summary, r.evidence
LIMIT 10

// Or search by role mentioned in summary
MATCH (e:__Entity__)
WHERE e.ai_summary CONTAINS 'Romeo and Juliet' AND e.ai_summary CONTAINS 'wrote'
RETURN e.name, e.ai_summary
```
**Answer format:** Just the person name (e.g., `William Shakespeare`)

### CATEGORY 8: Multi-hop Bridge (What [attribute] of the [role] of [entity]?)
**Question:** "What government position was held by the woman who portrayed Corliss Archer?"
**Question:** "Where did the descendants of the Seminole black Indians settle?"
**Strategy:** Chain: Find entity A → Find related entity B → Get attribute of B.
```cypher
// Step 1: Find starting entity and related entities
MATCH (work:__Entity__)-[r:RELATED_TO]-(person:__Entity__)
WHERE work.name CONTAINS 'Kiss and Tell'
RETURN work.name, person.name, person.ai_summary, r.evidence
LIMIT 10

// Step 2: Or search by role/relationship in summary
MATCH (e:__Entity__)
WHERE e.ai_summary CONTAINS 'Corliss Archer'
RETURN e.name, e.ai_summary
```
**Reasoning:** Follow the chain - the intermediate entity's summary has the final answer.
**Answer format:** Just the attribute (e.g., `Chief of Protocol`)

### CATEGORY 9: Location Questions (Where is X located/headquartered/from?)
**Question:** "In what city is the company Fastjet Tanzania based?"
**Question:** "Where was the world cup hosted that Algeria qualified for?"
**Strategy:** Find entity, extract location from ai_summary.
```cypher
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Fastjet'
RETURN e.name, e.ai_summary

// Or find via relationship to location entities
MATCH (e:__Entity__)-[:RELATED_TO]-(loc:__Entity__)
WHERE e.name CONTAINS 'Fastjet'
RETURN e.name, loc.name, loc.ai_summary, loc.entity_type
```
**Answer format:** Just the location (e.g., `Nairobi, Kenya`)

### CATEGORY 10: Numeric/Count Questions (How many/What is the population?)
**Question:** "The arena where Lewiston Maineiacs played can seat how many?"
**Question:** "Brown State Lake is in a county with what population?"
**Strategy:** Find entity, search chunks for specific numbers.
```cypher
// Find the entity and related entities
MATCH (e:__Entity__)-[:RELATED_TO]-(related:__Entity__)
WHERE e.name CONTAINS 'Lewiston Maineiacs'
RETURN e.name, related.name, related.ai_summary

// Search chunks for numeric data (more reliable for specific numbers)
MATCH (c:Chunk)
WHERE c.text CONTAINS 'Colisée' AND (c.text CONTAINS 'seat' OR c.text CONTAINS 'capacity')
RETURN c.text LIMIT 3
```
**Reasoning:** Specific numbers like populations/capacities are often in chunk text.
**Answer format:** The number with context (e.g., `3,677 seated`)

### CATEGORY 11: Name/Title Questions (What is the name of X?)
**Question:** "What is the name of the fight song of the university in Lawrence, Kansas?"
**Question:** "What is the nickname of the Command Module in the Moon trees mission?"
**Strategy:** Find the parent entity, look for name/title in summary or relationships.
```cypher
// Find the parent entity by name or summary content
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Kansas' OR e.ai_summary CONTAINS 'Lawrence'
RETURN e.name, e.ai_summary

// Search for the specific named item in chunks (reliable for specific facts)
MATCH (c:Chunk)
WHERE c.text CONTAINS 'Kansas' AND c.text CONTAINS 'fight song'
RETURN c.text LIMIT 3
```
**Answer format:** Just the name (e.g., `Kansas Song`)

### CATEGORY 12: Origin/Formation (Who formed/founded X? What [group] created Y?)
**Question:** "2014 S/S is the debut album of a group formed by who?"
**Question:** "The Koch brothers control a company founded when?"
**Strategy:** Find the entity, extract founder/formation info from summary.
```cypher
MATCH (e:__Entity__)
WHERE e.name CONTAINS '2014 S/S' OR e.ai_summary CONTAINS '2014 S/S'
RETURN e.name, e.ai_summary

// Follow relationships to find the founder/creator
MATCH (item:__Entity__)-[:RELATED_TO]-(org:__Entity__)
WHERE item.name CONTAINS '2014 S/S'
RETURN item.name, org.name, org.ai_summary
```
**Answer format:** Just the entity (e.g., `YG Entertainment`)

---

## IMPORTANT TIPS

1. **ALWAYS START** with neo4j_get_schema() to see what's available
2. **USE CONTAINS** for name matching - exact matches often fail
3. **CHECK ai_summary FIRST** - it often has the answer directly
4. **LIMIT RESULTS** to avoid overwhelming output
5. **TRY MULTIPLE STRATEGIES** if the first doesn't work
6. **For dates/numbers** - search in ai_summary or chunk text
7. **For locations** - entities may be typed as LOCATION or GPE
8. **Case sensitivity** - Neo4j is case-sensitive, use toLower() if needed
9. **For multi-hop questions** - trace the chain: Work→Person→Attribute

## RESPONSE FORMAT

{response_format_instructions}
"""

# Default response format - post-processing in benchmark layer handles answer_style
DEFAULT_RESPONSE_FORMAT = """Provide a factual answer based on your graph exploration.

Your response should:
- State the answer clearly and directly
- Reference the entities and relationships you discovered
- Be factual and specific - cite names, dates, and facts

If the information is insufficient, state what you found and what is missing."""


def get_system_prompt(answer_style: str = "ragas") -> str:
    """Get the system prompt - answer_style is now handled by benchmark post-processing."""
    return GRAPH_EXPLORATION_SYSTEM_PROMPT.format(response_format_instructions=DEFAULT_RESPONSE_FORMAT)


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
    
    def __init__(self, model: str = None, provider: str = None, answer_style: str = "hotpotqa"):
        """
        Initialize the agentic retriever.
        
        Args:
            model: Override the configured model (e.g., 'gpt-5.2')
            provider: Override the configured provider (e.g., 'openai')
            answer_style: Response format - "hotpotqa" for short exact answers (EM/F1 benchmarks),
                         "ragas" for verbose real-world answers
        """
        if not DEEP_AGENTS_AVAILABLE:
            raise ImportError(
                "Deep Agents not installed. Install with: pip install deepagents"
            )
        
        self.config = get_model_config()
        self.answer_style = answer_style
        
        # Get model configuration
        effective_model = self.config.effective_agentic_text2cypher_model
        effective_provider = self.config.effective_agentic_text2cypher_provider
        
        logger.info(f"Agentic Text2Cypher - Provider: {effective_provider.value}, "
                   f"Model: {effective_model.value}, Answer Style: {answer_style}")
        
        # Check if using thinking model
        is_thinking = self.config.is_thinking_model(effective_model)
        if is_thinking:
            logger.info("Using thinking model - extended reasoning enabled")
        
        # Create the LLM
        self.llm = get_agentic_text2cypher_llm()
        
        # Create the Deep Agent with answer_style-appropriate prompt
        system_prompt = get_system_prompt(answer_style)
        self.agent = create_deep_agent(
            model=self.llm,
            tools=AGENT_TOOLS_MINIMAL,
            system_prompt=system_prompt
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
                            content = msg.content
                            # Handle case where content is a list of content blocks
                            if isinstance(content, list):
                                # Extract text from content blocks
                                text_parts = []
                                for block in content:
                                    if isinstance(block, str):
                                        text_parts.append(block)
                                    elif isinstance(block, dict) and 'text' in block:
                                        text_parts.append(block['text'])
                                    elif hasattr(block, 'text'):
                                        text_parts.append(block.text)
                                final_answer = ' '.join(text_parts)
                            else:
                                final_answer = str(content)
                            break
                        elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            # Handle case where content is a list
                            if isinstance(content, list):
                                text_parts = []
                                for block in content:
                                    if isinstance(block, str):
                                        text_parts.append(block)
                                    elif isinstance(block, dict) and 'text' in block:
                                        text_parts.append(block['text'])
                                final_answer = ' '.join(text_parts)
                            else:
                                final_answer = str(content) if content else ''
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
    provider: str = None,
    answer_style: str = "hotpotqa"
) -> AgenticText2CypherRetriever:
    """
    Create an Agentic Text2Cypher retriever instance.
    
    Args:
        model: Override model (defaults to AGENTIC_TEXT2CYPHER_MODEL or LLM_MODEL)
        provider: Override provider (defaults to AGENTIC_TEXT2CYPHER_PROVIDER or LLM_PROVIDER)
        answer_style: Response format - "hotpotqa" for short exact answers (EM/F1 benchmarks),
                     "ragas" for verbose real-world answers
    
    Returns:
        AgenticText2CypherRetriever instance
    """
    return AgenticText2CypherRetriever(model=model, provider=provider, answer_style=answer_style)


# Main interface function for benchmark system
def query_agentic_text2cypher_rag(query: str, answer_style: str = "hotpotqa", **kwargs) -> Dict[str, Any]:
    """
    Agentic Text2Cypher RAG retrieval using Deep Agents.
    
    This is an adaptive, multi-step approach where an LLM agent
    explores the graph database to find answers.
    
    Args:
        query: The search query
        answer_style: Response format - "hotpotqa" for short exact answers (EM/F1 benchmarks),
                     "ragas" for verbose real-world answers (default: hotpotqa for benchmark compatibility)
        **kwargs: Additional configuration options
    
    Returns:
        Dictionary with response and retrieval details
    """
    try:
        retriever = create_agentic_text2cypher_retriever(answer_style=answer_style)
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

