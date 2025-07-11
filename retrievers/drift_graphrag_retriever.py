"""
DRIFT GraphRAG Retriever Implementation

This module implements Microsoft's DRIFT (Dynamic Iterative Refinement) approach
for GraphRAG retrieval, featuring:

1. Query decomposition with primer using community reports
2. Action graph management with follow-up questions  
3. Iterative refinement over multiple search depths
4. Dynamic exploration based on intermediate results
5. Multi-action synthesis for comprehensive answers

Based on Microsoft's DRIFT implementation but adapted for our graph structure.
"""

import os
import asyncio
import json
import time
import logging
import secrets
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from tqdm.asyncio import tqdm as atqdm

# Core dependencies
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import neo4j
import networkx as nx
import numpy as np
from dotenv import load_dotenv

# Import our graph processor and advanced retriever
from data_processors import AdvancedGraphProcessor
from .advanced_graphrag_retriever import GraphRAGLocalRetriever, SearchResult

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class DRIFTConfig:
    """Configuration for DRIFT search"""
    n_depth: int = 3                           # Maximum search depth
    drift_k_followups: int = 3                 # Max follow-ups per iteration
    primer_folds: int = 2                      # Number of primer folds for parallel processing
    local_search_max_tokens: int = 8000        # Token limit for local search
    reduce_max_tokens: int = 2000              # Token limit for final synthesis
    min_relevance_score: int = 20              # Minimum score for including results
    temperature: float = 0.1                  # LLM temperature
    max_concurrent: int = 3                    # Max concurrent LLM calls

class DriftAction:
    """
    Represents a search action with query, answer, and follow-up actions.
    Forms nodes in the action graph.
    """
    
    def __init__(
        self,
        query: str,
        answer: Optional[str] = None,
        follow_ups: Optional[List["DriftAction"]] = None,
    ):
        self.query = query
        self.answer = answer
        self.score: Optional[float] = None
        self.follow_ups: List[DriftAction] = follow_ups or []
        self.metadata: Dict[str, Any] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "completion_time": 0.0,
            "context_data": None
        }
    
    @property
    def is_complete(self) -> bool:
        """Check if the action has been executed (has an answer)"""
        return self.answer is not None
    
    async def execute_search(self, search_engine: GraphRAGLocalRetriever, global_query: str) -> "DriftAction":
        """Execute search using local search engine and update action with results"""
        if self.is_complete:
            logger.warning(f"Action already complete: {self.query[:50]}...")
            return self
        
        start_time = time.time()
        
        try:
            # Execute local search with drift context
            result = await search_engine.search(
                query=self.query,
                max_entities=10,
                max_chunks=5,
                max_communities=3,
                drift_query=global_query  # Pass global context
            )
            
            # Parse structured response if available
            try:
                if result.response.startswith('{') and result.response.endswith('}'):
                    response_data = json.loads(result.response)
                    self.answer = response_data.get("response", result.response)
                    self.score = float(response_data.get("score", 50))
                    self.follow_ups = [
                        DriftAction(query=fq) for fq in response_data.get("follow_up_queries", [])
                    ]
                else:
                    # Fallback for non-JSON responses
                    self.answer = result.response
                    self.score = 50.0  # Default score
                    self.follow_ups = []
            except json.JSONDecodeError:
                self.answer = result.response
                self.score = 50.0
                self.follow_ups = []
            
            # Update metadata
            self.metadata.update({
                "llm_calls": result.llm_calls,
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens,
                "completion_time": time.time() - start_time,
                "context_data": result.context_data
            })
            
        except Exception as e:
            logger.error(f"Error executing search for query '{self.query[:50]}...': {e}")
            self.answer = f"Error during search: {str(e)}"
            self.score = 0.0
            self.follow_ups = []
            self.metadata["completion_time"] = time.time() - start_time
        
        return self
    
    def serialize(self, include_follow_ups: bool = True) -> Dict[str, Any]:
        """Serialize action to dictionary"""
        data = {
            "query": self.query,
            "answer": self.answer,
            "score": self.score,
            "metadata": self.metadata,
        }
        if include_follow_ups:
            data["follow_ups"] = [action.serialize() for action in self.follow_ups]
        return data
    
    @classmethod
    def from_primer_response(cls, query: str, response_data: Dict[str, Any]) -> "DriftAction":
        """Create DriftAction from primer response"""
        action = cls(
            query=query,
            answer=response_data.get("intermediate_answer"),
            follow_ups=[
                cls(query=fq) for fq in response_data.get("follow_up_queries", [])
            ]
        )
        action.score = response_data.get("score", 50.0)
        return action
    
    def __hash__(self) -> int:
        """Make action hashable for use in networkx graphs"""
        return hash(self.query)
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on query"""
        if not isinstance(other, DriftAction):
            return False
        return self.query == other.query

class QueryState:
    """Manages the action graph and query state"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
    
    def add_action(self, action: DriftAction, metadata: Optional[Dict[str, Any]] = None):
        """Add action to the graph"""
        self.graph.add_node(action, **(metadata or {}))
    
    def relate_actions(self, parent: DriftAction, child: DriftAction, weight: float = 1.0):
        """Create relationship between actions"""
        self.graph.add_edge(parent, child, weight=weight)
    
    def add_all_follow_ups(self, action: DriftAction, follow_ups: List[DriftAction], weight: float = 1.0):
        """Add all follow-up actions and link them to parent"""
        for follow_up in follow_ups:
            self.add_action(follow_up)
            self.relate_actions(action, follow_up, weight)
    
    def find_incomplete_actions(self) -> List[DriftAction]:
        """Find all actions that haven't been executed yet"""
        return [node for node in self.graph.nodes if not node.is_complete]
    
    def rank_incomplete_actions(self) -> List[DriftAction]:
        """Rank incomplete actions (currently random, could add scoring)"""
        incomplete = self.find_incomplete_actions()
        random.shuffle(incomplete)  # Simple randomization
        return incomplete
    
    def serialize(self, include_context: bool = True) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any], str]]:
        """Serialize the entire query state"""
        # Create node-to-ID mapping
        node_to_id = {node: idx for idx, node in enumerate(self.graph.nodes())}
        
        # Serialize nodes
        nodes = [
            {
                **node.serialize(include_follow_ups=False),
                "id": node_to_id[node],
                **self.graph.nodes[node],
            }
            for node in self.graph.nodes()
        ]
        
        # Serialize edges
        edges = [
            {
                "source": node_to_id[u],
                "target": node_to_id[v],
                "weight": edge_data.get("weight", 1.0),
            }
            for u, v, edge_data in self.graph.edges(data=True)
        ]
        
        serialized = {"nodes": nodes, "edges": edges}
        
        if include_context:
            context_data = {
                node["query"]: node["metadata"].get("context_data", {})
                for node in nodes
                if node["metadata"].get("context_data")
            }
            context_text = json.dumps(context_data, indent=2)
            return serialized, context_data, context_text
        
        return serialized
    
    def action_token_count(self) -> Dict[str, int]:
        """Calculate total token usage across all actions"""
        total_calls = total_prompt = total_output = 0
        
        for action in self.graph.nodes:
            total_calls += action.metadata.get("llm_calls", 0)
            total_prompt += action.metadata.get("prompt_tokens", 0)
            total_output += action.metadata.get("output_tokens", 0)
        
        return {
            "llm_calls": total_calls,
            "prompt_tokens": total_prompt,
            "output_tokens": total_output
        }

class PrimerQueryProcessor:
    """Processes initial query using community reports to generate follow-up actions"""
    
    def __init__(self, graph_processor: AdvancedGraphProcessor, config: DRIFTConfig):
        self.graph = graph_processor
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=config.temperature)
        self.embeddings = OpenAIEmbeddings()
        
        # Primer prompt for query decomposition
        self.primer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research analyst. Your task is to analyze a user query and create a comprehensive research plan using the provided community reports as context.

Based on the user query and community reports, generate:
1. An intermediate answer based on what you can determine from the community reports
2. A set of specific follow-up queries that would help provide a more complete answer
3. A relevance score (0-100) for how well the community reports address the query

Return your response as a JSON object with this structure:
{{
    "intermediate_answer": "What you can determine from the community reports...",
    "follow_up_queries": [
        "Specific question 1 that would help answer the original query",
        "Specific question 2 about details not covered",
        "Specific question 3 to explore related aspects"
    ],
    "score": 85
}}

Guidelines:
- Generate 2-5 specific, actionable follow-up queries
- Each follow-up should seek different types of information
- Focus on gaps in the intermediate answer
- Follow-ups should be answerable by the knowledge base

Community Reports:
{community_reports}"""),
            ("human", "User Query: {query}\n\nPlease analyze this query and generate a research plan.")
        ])
    
    async def expand_query(self, query: str) -> Tuple[str, Dict[str, int]]:
        """Expand query using HyDE approach with community report template"""
        try:
            # Get a random community report as template
            with self.graph.driver.session() as session:
                community_data = session.run("""
                    MATCH (c:__Community__)
                    WHERE c.summary IS NOT NULL
                    RETURN c.summary as summary
                    ORDER BY rand()
                    LIMIT 1
                """).single()
                
                if not community_data:
                    return query, {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0}
                
                template = community_data["summary"]
            
            # Create hypothetical answer
            expansion_prompt = f"""Create a hypothetical answer to the following query: {query}

Format it to follow the structure of the template below:
{template}

Ensure that the hypothetical answer does not reference new named entities that are not present in the original query."""
            
            response = await self.llm.ainvoke(expansion_prompt)
            expanded_query = response.content.strip()
            
            # Rough token counting
            prompt_tokens = len(expansion_prompt) // 4
            output_tokens = len(expanded_query) // 4
            
            return expanded_query, {
                "llm_calls": 1,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens
            }
            
        except Exception as e:
            logger.error(f"Error in query expansion: {e}")
            return query, {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0}
    
    async def get_relevant_communities(self, query: str, max_communities: int = 10) -> List[Dict[str, Any]]:
        """Get most relevant communities for the query"""
        try:
            # Expand query first
            expanded_query, _ = await self.expand_query(query)
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(expanded_query)
            
            with self.graph.driver.session() as session:
                # Get communities with summaries
                communities_data = session.run("""
                    MATCH (c:__Community__)
                    WHERE c.summary IS NOT NULL
                    RETURN c.id as community_id,
                           c.level as level,
                           c.summary as summary,
                           c.community_rank as rank
                    ORDER BY c.level, coalesce(c.community_rank, 1) DESC
                """).data()
                
                if not communities_data:
                    return []
                
                # Calculate similarities
                summaries = [comm['summary'] for comm in communities_data]
                summary_embeddings = self.embeddings.embed_documents(summaries)
                
                # Compute cosine similarities
                similarities = []
                query_norm = np.linalg.norm(query_embedding)
                
                for embedding in summary_embeddings:
                    if query_norm > 0 and np.linalg.norm(embedding) > 0:
                        similarity = np.dot(query_embedding, embedding) / (
                            query_norm * np.linalg.norm(embedding)
                        )
                        similarities.append(similarity)
                    else:
                        similarities.append(0.0)
                
                # Add similarities and sort
                for i, comm in enumerate(communities_data):
                    comm['similarity'] = similarities[i]
                
                # Sort by similarity and return top-k
                communities_data.sort(key=lambda x: x['similarity'], reverse=True)
                return communities_data[:max_communities]
                
        except Exception as e:
            logger.error(f"Error getting relevant communities: {e}")
            return []
    
    async def decompose_query(self, query: str, communities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decompose query into actions using community reports"""
        try:
            # Prepare community reports text
            community_reports = "\n\n".join([
                f"Community {comm['community_id']} (Level {comm['level']}):\n{comm['summary']}"
                for comm in communities
            ])
            
            # Format and send prompt
            messages = self.primer_prompt.format_messages(
                query=query,
                community_reports=community_reports
            )
            
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()
            
            # Parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            parsed_response = json.loads(response_text)
            
            # Validate response structure
            required_keys = ["intermediate_answer", "follow_up_queries", "score"]
            for key in required_keys:
                if key not in parsed_response:
                    logger.warning(f"Missing key '{key}' in primer response")
                    parsed_response[key] = [] if key == "follow_up_queries" else "No information available" if key == "intermediate_answer" else 0
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error in query decomposition: {e}")
            return {
                "intermediate_answer": f"Error in analysis: {str(e)}",
                "follow_up_queries": [],
                "score": 0
            }

class DriftGraphRAGRetriever:
    """Main DRIFT GraphRAG retriever implementing iterative refinement"""
    
    def __init__(self, graph_processor: AdvancedGraphProcessor, config: Optional[DRIFTConfig] = None):
        self.graph = graph_processor
        self.config = config or DRIFTConfig()
        self.primer = PrimerQueryProcessor(graph_processor, self.config)
        self.local_search = GraphRAGLocalRetriever(graph_processor)
        self.query_state = QueryState()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=self.config.temperature)
        
        # Reduce phase prompt
        self.reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a master research synthesizer. Your task is to create a comprehensive, well-structured answer by combining insights from multiple research actions.

You have been provided with results from various research queries related to the original question. Your goal is to synthesize these into a single, coherent, and comprehensive response.

Guidelines:
1. Combine information from all research actions
2. Organize the information logically and coherently
3. Prioritize higher-scored research results
4. Remove redundancy while preserving important details
5. Ensure the final answer directly addresses the original query
6. Use clear, professional language
7. Include specific details and examples where available

Research Results:
{research_results}"""),
            ("human", "Original Query: {query}\n\nPlease synthesize all research results into a comprehensive answer.")
        ])
    
    async def search(
        self,
        query: str,
        max_communities: int = 8,
        reduce: bool = True,
        **kwargs
    ) -> SearchResult:
        """Execute DRIFT search with iterative refinement"""
        
        start_time = time.time()
        total_llm_calls = 0
        total_prompt_tokens = 0
        total_output_tokens = 0
        
        try:
            print(f"🌀 Starting DRIFT search for: {query[:60]}...")
            
            # Phase 1: Primer - decompose query using communities
            print("📋 Phase 1: Query decomposition (Primer)")
            relevant_communities = await self.primer.get_relevant_communities(query, max_communities)
            
            if not relevant_communities:
                return SearchResult(
                    response="No relevant communities found for analysis.",
                    context_data={},
                    context_text="",
                    completion_time=time.time() - start_time,
                    llm_calls=0,
                    prompt_tokens=0,
                    output_tokens=0,
                    method="drift_graphrag"
                )
            
            primer_response = await self.primer.decompose_query(query, relevant_communities)
            
            # Create initial action from primer
            initial_action = DriftAction.from_primer_response(query, primer_response)
            self.query_state.add_action(initial_action)
            self.query_state.add_all_follow_ups(initial_action, initial_action.follow_ups)
            
            print(f"✅ Primer completed: {len(initial_action.follow_ups)} follow-up actions generated")
            
            # Phase 2: Iterative refinement
            print(f"🔄 Phase 2: Iterative refinement (max depth: {self.config.n_depth})")
            
            for depth in range(self.config.n_depth):
                incomplete_actions = self.query_state.rank_incomplete_actions()
                
                if not incomplete_actions:
                    print(f"  Depth {depth}: No more actions to process")
                    break
                
                # Select top-k actions for this iteration
                actions_to_process = incomplete_actions[:self.config.drift_k_followups]
                print(f"  Depth {depth}: Processing {len(actions_to_process)} actions")
                
                # Execute actions in parallel with concurrency limit
                semaphore = asyncio.Semaphore(self.config.max_concurrent)
                
                async def execute_with_semaphore(action):
                    async with semaphore:
                        return await action.execute_search(self.local_search, query)
                
                # Process actions
                tasks = [execute_with_semaphore(action) for action in actions_to_process]
                completed_actions = await atqdm.gather(*tasks, desc=f"Depth {depth}")
                
                # Update query state with results
                for action in completed_actions:
                    self.query_state.add_action(action)  # Update existing node
                    if action.follow_ups:
                        self.query_state.add_all_follow_ups(action, action.follow_ups)
                
                print(f"  ✅ Depth {depth} completed: {len(completed_actions)} actions processed")
            
            # Phase 3: Synthesis
            print("🔄 Phase 3: Final synthesis")
            
            # Calculate token usage from all actions
            token_counts = self.query_state.action_token_count()
            total_llm_calls += token_counts["llm_calls"]
            total_prompt_tokens += token_counts["prompt_tokens"]
            total_output_tokens += token_counts["output_tokens"]
            
            # Serialize state
            serialized_state, context_data, context_text = self.query_state.serialize(include_context=True)
            
            if reduce:
                final_response = await self._reduce_responses(query, serialized_state)
                # Add reduction tokens (rough estimate)
                total_llm_calls += 1
                total_prompt_tokens += len(str(serialized_state)) // 4
                total_output_tokens += len(final_response) // 4
            else:
                final_response = self._format_raw_responses(serialized_state)
            
            print(f"✅ DRIFT search completed in {time.time() - start_time:.2f}s")
            
            return SearchResult(
                response=final_response,
                context_data=context_data,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=total_llm_calls,
                prompt_tokens=total_prompt_tokens,
                output_tokens=total_output_tokens,
                method="drift_graphrag"
            )
            
        except Exception as e:
            logger.error(f"Error in DRIFT search: {e}")
            import traceback
            traceback.print_exc()
            
            return SearchResult(
                response=f"Error during DRIFT search: {str(e)}",
                context_data={},
                context_text="",
                completion_time=time.time() - start_time,
                llm_calls=total_llm_calls,
                prompt_tokens=total_prompt_tokens,
                output_tokens=total_output_tokens,
                method="drift_graphrag_error"
            )
    
    async def _reduce_responses(self, query: str, serialized_state: Dict[str, Any]) -> str:
        """Reduce all action responses into a final comprehensive answer"""
        try:
            # Extract completed actions with answers
            completed_actions = [
                node for node in serialized_state.get("nodes", [])
                if node.get("answer") and node.get("score", 0) >= self.config.min_relevance_score
            ]
            
            if not completed_actions:
                return "No relevant information found to answer the query."
            
            # Sort by score and format for reduction
            completed_actions.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            research_results = []
            for i, action in enumerate(completed_actions[:10]):  # Limit to top 10 results
                result_text = f"Research Action {i+1}:\n"
                result_text += f"Query: {action['query']}\n"
                result_text += f"Score: {action.get('score', 0)}\n"
                result_text += f"Answer: {action['answer']}\n"
                research_results.append(result_text)
            
            combined_results = "\n\n".join(research_results)
            
            # Generate final synthesis
            messages = self.reduce_prompt.format_messages(
                research_results=combined_results,
                query=query
            )
            
            response = await self.llm.ainvoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in response reduction: {e}")
            return f"Error synthesizing results: {str(e)}"
    
    def _format_raw_responses(self, serialized_state: Dict[str, Any]) -> str:
        """Format raw responses without reduction"""
        completed_actions = [
            node for node in serialized_state.get("nodes", [])
            if node.get("answer")
        ]
        
        if not completed_actions:
            return "No responses generated."
        
        formatted_responses = []
        for i, action in enumerate(completed_actions, 1):
            formatted_responses.append(
                f"{i}. Query: {action['query']}\n"
                f"   Answer: {action['answer']}\n"
                f"   Score: {action.get('score', 'N/A')}"
            )
        
        return "\n\n".join(formatted_responses)

# Main integration function for benchmark
async def query_drift_graphrag(query: str, **kwargs) -> Dict[str, Any]:
    """
    DRIFT GraphRAG retrieval for benchmark integration
    
    Args:
        query: The search query
        **kwargs: Additional configuration options
    
    Returns:
        Dictionary with response and retrieval details
    """
    
    # Initialize graph processor
    from data_processors import AdvancedGraphProcessor
    processor = AdvancedGraphProcessor()
    
    try:
        # Configure DRIFT
        config = DRIFTConfig(
            n_depth=kwargs.get('depth', 3),
            drift_k_followups=kwargs.get('k_followups', 3),
            max_concurrent=kwargs.get('max_concurrent', 3)
        )
        
        # Create DRIFT retriever
        retriever = DriftGraphRAGRetriever(processor, config)
        
        # Perform search
        result = await retriever.search(query, **kwargs)
        
        # Format response for benchmark compatibility
        return {
            'final_answer': result.response,
            'retrieval_details': [
                {
                    'content': result.context_text,
                    'metadata': result.context_data,
                    'method': result.method,
                    'completion_time': result.completion_time,
                    'llm_calls': result.llm_calls,
                    'tokens_used': result.prompt_tokens + result.output_tokens
                }
            ],
            'method': result.method,
            'performance_metrics': {
                'completion_time': result.completion_time,
                'llm_calls': result.llm_calls,
                'prompt_tokens': result.prompt_tokens,
                'output_tokens': result.output_tokens,
                'total_tokens': result.prompt_tokens + result.output_tokens
            }
        }
        
    except Exception as e:
        print(f"Error in DRIFT GraphRAG retrieval: {e}")
        import traceback
        traceback.print_exc()
        return {
            'final_answer': f"Error during DRIFT GraphRAG retrieval: {str(e)}",
            'retrieval_details': [],
            'method': 'drift_graphrag_error',
            'performance_metrics': {
                'completion_time': 0,
                'llm_calls': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }
    finally:
        processor.close()

# Factory function for easy instantiation
def create_drift_retriever() -> DriftGraphRAGRetriever:
    """Create a DRIFT GraphRAG retriever instance"""
    from data_processors import AdvancedGraphProcessor
    processor = AdvancedGraphProcessor()
    config = DRIFTConfig()
    return DriftGraphRAGRetriever(processor, config)

if __name__ == "__main__":
    # Test the DRIFT retriever
    import asyncio
    
    async def test_drift():
        """Test function"""
        test_query = "What are the main digital capabilities mentioned across all RFPs?"
        
        print(f"Testing DRIFT with query: {test_query}")
        result = await query_drift_graphrag(test_query, depth=2, k_followups=2)
        
        print(f"Response length: {len(result['final_answer'])}")
        print(f"Method: {result['method']}")
        print(f"LLM calls: {result['performance_metrics']['llm_calls']}")
        print(f"Completion time: {result['performance_metrics']['completion_time']:.2f}s")
        print(f"\nResponse: {result['final_answer'][:300]}...")
    
    # Uncomment to test
    # asyncio.run(test_drift()) 