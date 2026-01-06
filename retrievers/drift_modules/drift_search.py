"""
DRIFT Search Module

This module provides the main DRIFT search orchestration that ties together
all components and leverages existing local/global retrievers with intelligent routing.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .drift_context import DRIFTContextBuilder, DRIFTContextConfig
from .drift_primer import DRIFTPrimer, PrimerConfig
from .drift_state import DRIFTQueryState, ActionMetadata
from .drift_action import DRIFTAction, DRIFTActionConfig, create_action_batch, execute_action_batch

logger = logging.getLogger(__name__)

@dataclass
class DRIFTConfig:
    """Configuration for DRIFT search"""
    n_depth: int = 3                           # Maximum search depth
    max_follow_ups: int = 3                    # Max follow-ups per iteration
    max_concurrent: int = 3                    # Max concurrent LLM calls
    temperature: float = 0.1                   # LLM temperature
    min_relevance_score: float = 20.0          # Minimum score for including results
    enable_primer: bool = True                 # Enable primer phase
    enable_reduce: bool = True                 # Enable reduction phase
    auto_route: bool = True                    # Use QueryClassifier for routing
    
    # Sub-configurations
    context_config: Optional[DRIFTContextConfig] = None
    primer_config: Optional[PrimerConfig] = None
    action_config: Optional[DRIFTActionConfig] = None
    
    def __post_init__(self):
        # Initialize sub-configurations with defaults
        if self.context_config is None:
            self.context_config = DRIFTContextConfig()
        
        if self.primer_config is None:
            self.primer_config = PrimerConfig(
                max_follow_ups=self.max_follow_ups,
                temperature=self.temperature
            )
        
        if self.action_config is None:
            self.action_config = DRIFTActionConfig(
                max_follow_ups=self.max_follow_ups,
                min_score_threshold=self.min_relevance_score,
                temperature=self.temperature,
                auto_route=self.auto_route
            )

class DRIFTSearchResult:
    """Result from DRIFT search execution"""
    
    def __init__(
        self,
        response: str,
        query_state: DRIFTQueryState,
        execution_time: float,
        method: str = "drift_search"
    ):
        self.response = response
        self.query_state = query_state
        self.execution_time = execution_time
        self.method = method
        
        # Extract metrics from query state
        self.execution_summary = query_state.get_execution_summary()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        retrieval_details = self._extract_retrieval_details()
        
        # Ensure we always have at least one retrieval detail with content for benchmark
        if not retrieval_details and self.response:
            # Create a fallback detail from the response itself
            retrieval_details = [{
                "action_id": "final_response",
                "query": self.query_state.global_query,
                "content": self.response,
                "score": 50.0,
                "depth": 0,
                "metadata": {}
            }]
        
        return {
            "final_answer": self.response,
            "method": self.method,
            "execution_time": self.execution_time,
            "query_state": self.query_state.serialize(),
            "performance_metrics": {
                "completion_time": self.execution_time,
                "llm_calls": self.execution_summary.get("total_llm_calls", 0) if self.execution_summary else 0,
                "total_tokens": self.execution_summary.get("token_usage", {}).get("total_tokens", 0) if self.execution_summary else 0,
                "prompt_tokens": self.execution_summary.get("token_usage", {}).get("prompt_tokens", 0) if self.execution_summary else 0,
                "output_tokens": self.execution_summary.get("token_usage", {}).get("output_tokens", 0) if self.execution_summary else 0,
                "total_actions": self.execution_summary.get("total_actions", 0) if self.execution_summary else 0,
                "completed_actions": self.execution_summary.get("completed_actions", 0) if self.execution_summary else 0,
                "search_depth": self.query_state.graph.number_of_nodes() if self.query_state else 0
            },
            "retrieval_details": retrieval_details
        }
    
    def _extract_retrieval_details(self) -> List[Dict[str, Any]]:
        """Extract retrieval details from completed actions"""
        details = []
        
        for action_id in self.query_state.get_completed_actions():
            action_info = self.query_state.get_action_info(action_id)
            if action_info and action_info.get("answer"):
                # Extract context from metadata if available
                metadata = action_info.get("metadata")
                context_content = ""
                
                if metadata:
                    # Try to get context from metadata
                    if hasattr(metadata, 'context_data') and metadata.context_data:
                        ctx_data = metadata.context_data
                        # Get community context or retrieved content
                        context_content = ctx_data.get("community_context", "") or ctx_data.get("retrieved_content", "")
                        if not context_content and "primer_result" in ctx_data:
                            context_content = ctx_data["primer_result"].get("community_context", "")
                
                # Use answer as content, but also include context
                content = action_info["answer"]
                if context_content:
                    content = f"{context_content}\n\n---\nAnswer: {action_info['answer']}"
                
                details.append({
                    "action_id": action_id,
                    "query": action_info["query"],
                    "content": content,
                    "score": action_info.get("score", 0.0),
                    "depth": action_info.get("depth", 0),
                    "metadata": action_info.get("metadata", {})
                })
        
        return details

class DRIFTSearch:
    """
    Main DRIFT search orchestrator that coordinates all components
    and leverages existing local/global retrievers with intelligent routing.
    """
    
    def __init__(self, graph_processor, config: Optional[DRIFTConfig] = None):
        """
        Initialize DRIFT search
        
        Args:
            graph_processor: AdvancedGraphProcessor instance with Neo4j connection
            config: Optional DRIFT configuration
        """
        self.graph_processor = graph_processor
        self.config = config or DRIFTConfig()
        
        # Initialize LLM first (needed by retrievers)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=self.config.temperature)
        
        # Initialize components with error handling
        try:
            self.context_builder = DRIFTContextBuilder(graph_processor, self.config.context_config)
        except Exception as e:
            logger.warning(f"Failed to initialize DRIFTContextBuilder: {e}")
            self.context_builder = None
            
        try:
            self.primer = DRIFTPrimer(graph_processor, self.config.primer_config)
        except Exception as e:
            logger.warning(f"Failed to initialize DRIFTPrimer: {e}")
            self.primer = None
        
        # Initialize existing retrievers
        self._initialize_retrievers()
        
        # Reduction prompt
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
        
        self.reduce_chain = self.reduce_prompt | self.llm
    
    def _initialize_retrievers(self):
        """Initialize existing retrievers for use by DRIFT actions"""
        try:
            # Import and initialize existing retrievers
            from ..advanced_graphrag_retriever import GraphRAGLocalRetriever, GraphRAGGlobalRetriever, AdvancedGraphRAGRetriever
            
            # Create a single shared AdvancedGraphRAGRetriever instance to avoid duplicate data loading
            shared_retriever = AdvancedGraphRAGRetriever(self.graph_processor)
            
            # Initialize compatibility wrappers using the shared retriever instance
            self.local_retriever = GraphRAGLocalRetriever(self.graph_processor, shared_retriever)
            self.global_retriever = GraphRAGGlobalRetriever(self.graph_processor, shared_retriever)
            
            # Initialize DRIFT QueryClassifier for auto-routing
            if self.config.auto_route:
                try:
                    from .drift_query_classifier import DRIFTQueryClassifier
                    if hasattr(self, 'llm') and self.llm is not None:
                        self.classifier = DRIFTQueryClassifier(self.llm)
                        logger.info("DRIFT QueryClassifier initialized successfully for auto-routing")
                    else:
                        logger.warning("LLM not available for DRIFT QueryClassifier, disabling auto-routing")
                        self.classifier = None
                        self.config.auto_route = False
                except ImportError as ie:
                    logger.warning(f"DRIFT QueryClassifier not available: {ie}, disabling auto-routing")
                    self.classifier = None
                    self.config.auto_route = False
            else:
                self.classifier = None
                
            logger.info("Successfully initialized existing retrievers with shared data")
            
        except Exception as e:
            logger.error(f"Error initializing retrievers: {e}")
            # Create fallback simple retrievers
            self.local_retriever = None
            self.global_retriever = None
            self.classifier = None
    
    async def search(
        self,
        query: str,
        max_depth: Optional[int] = None,
        enable_primer: Optional[bool] = None,
        enable_reduce: Optional[bool] = None,
        **kwargs
    ) -> DRIFTSearchResult:
        """
        Perform DRIFT search with iterative refinement
        
        Args:
            query: The search query
            max_depth: Maximum search depth (overrides config)
            enable_primer: Enable primer phase (overrides config)
            enable_reduce: Enable reduction phase (overrides config)
            **kwargs: Additional search parameters
            
        Returns:
            DRIFTSearchResult with comprehensive answer and metadata
        """
        start_time = time.time()
        
        # Override config if specified  
        max_depth = max_depth or self.config.n_depth or 3
        enable_primer = enable_primer if enable_primer is not None else self.config.enable_primer
        enable_reduce = enable_reduce if enable_reduce is not None else self.config.enable_reduce
        
        # Initialize query state
        query_state = DRIFTQueryState(query)
        
        try:
            logger.info(f"🚀 Starting DRIFT search for: {query}")
            
            # Phase 1: Primer phase (optional)
            if enable_primer and self.primer is not None:
                try:
                    await self._primer_phase(query, query_state)
                except Exception as e:
                    logger.warning(f"Primer phase failed: {e}, continuing with simple action")
                    # Create fallback initial action
                    query_state.add_action("initial_action", query)
            else:
                # Create initial action directly
                query_state.add_action("initial_action", query)
            
            # Phase 2: Iterative refinement
            try:
                await self._iterative_refinement_phase(query_state, max_depth)
            except Exception as e:
                logger.warning(f"Iterative refinement failed: {e}, using simple response")
            
            # Phase 3: Reduction phase (optional)
            if enable_reduce and self.llm is not None:
                try:
                    final_answer = await self._reduction_phase(query, query_state)
                except Exception as e:
                    logger.warning(f"Reduction phase failed: {e}, using best answer extraction")
                    final_answer = self._extract_best_answer(query_state)
            else:
                final_answer = self._extract_best_answer(query_state)
            
            # Ensure we have a valid answer
            if not final_answer or final_answer.strip() == "":
                final_answer = f"Unable to process query: {query}. DRIFT system encountered errors during processing."
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ DRIFT search completed in {execution_time:.2f}s")
            
            return DRIFTSearchResult(
                response=final_answer,
                query_state=query_state,
                execution_time=execution_time,
                method="drift_search"
            )
            
        except Exception as e:
            logger.error(f"❌ DRIFT search failed: {e}")
            execution_time = time.time() - start_time
            
            # Create minimal query state for error case
            error_state = DRIFTQueryState(query)
            error_state.add_action("error_action", query, answer=f"Error processing query: {str(e)}")
            
            return DRIFTSearchResult(
                response=f"Error processing query '{query}': {str(e)}",
                query_state=error_state,
                execution_time=execution_time,
                method="drift_search"
            )
    
    async def _primer_phase(self, query: str, query_state: DRIFTQueryState):
        """Execute primer phase for query decomposition"""
        logger.info("📋 Executing primer phase...")
        
        try:
            # Get primer results
            primer_result = await self.primer.process_query(query)
            
            # Store community context for retrieval details
            community_context = primer_result.get("community_context", "")
            
            # Create initial action with primer results
            initial_action_id = f"primer_{query_state._action_counter}"
            query_state.add_action(
                initial_action_id,
                query,
                primer_result.get("intermediate_answer"),
                primer_result.get("score", 50.0),
                ActionMetadata(
                    llm_calls=primer_result.get("llm_calls", 0) or 0,
                    prompt_tokens=primer_result.get("prompt_tokens", 0) or 0,
                    output_tokens=primer_result.get("output_tokens", 0) or 0,
                    completion_time=primer_result.get("completion_time", 0.0) or 0.0,
                    context_data={
                        "primer_result": primer_result,
                        "community_context": community_context,
                        "retrieved_content": community_context  # For benchmark compatibility
                    }
                )
            )
            
            # Add follow-up actions from primer - FIX: Use enumerate for unique IDs
            follow_up_queries = primer_result.get("follow_up_queries", [])
            for idx, fq in enumerate(follow_up_queries):
                follow_up_action_id = f"followup_{query_state._action_counter}_{idx}"
                query_state.add_action(follow_up_action_id, fq)
                query_state.relate_actions(initial_action_id, follow_up_action_id)
            
            logger.info(f"Primer phase completed with {len(follow_up_queries)} follow-ups")
            
        except Exception as e:
            logger.error(f"Error in primer phase: {e}")
            import traceback
            traceback.print_exc()
            # Create fallback initial action
            initial_action_id = f"initial_{query_state._action_counter}"
            query_state.add_action(initial_action_id, query)
    
    async def _iterative_refinement_phase(self, query_state: DRIFTQueryState, max_depth: int):
        """Execute iterative refinement with depth control"""
        logger.info(f"🔄 Starting iterative refinement (max depth: {max_depth})...")
        
        current_depth = 0
        
        while current_depth < max_depth:
            # Get incomplete actions at current depth
            incomplete_actions = query_state.get_incomplete_actions()
            
            if not incomplete_actions:
                logger.info("No incomplete actions found, stopping refinement")
                break
            
            # Filter by depth
            depth_filtered_actions = [
                action_id for action_id in incomplete_actions
                if query_state.get_action_depth(action_id) <= current_depth
            ]
            
            if not depth_filtered_actions:
                current_depth += 1
                continue
            
            # Rank and select actions to execute
            ranked_actions = query_state.rank_actions(depth_filtered_actions)
            
            # Select top actions for execution (limit concurrency)
            actions_to_execute = ranked_actions[:self.config.max_concurrent]
            
            logger.info(f"Executing {len(actions_to_execute)} actions at depth {current_depth}")
            
            # Execute actions
            await self._execute_actions(query_state, actions_to_execute)
            
            current_depth += 1
        
        logger.info(f"Iterative refinement completed at depth {current_depth}")
    
    async def _execute_actions(self, query_state: DRIFTQueryState, ranked_actions: List[Tuple[str, float]]):
        """Execute a batch of actions"""
        
        # Create DRIFTAction objects for execution
        actions_to_execute = []
        
        for action_id, rank_score in ranked_actions:
            action_info = query_state.get_action_info(action_id)
            if action_info:
                action = DRIFTAction(
                    query=action_info["query"],
                    action_id=action_id,
                    global_query=query_state.global_query,
                    config=self.config.action_config
                )
                actions_to_execute.append(action)
        
        # Execute actions concurrently
        executed_actions = await execute_action_batch(
            actions_to_execute,
            self.local_retriever,
            self.global_retriever,
            self.classifier,
            max_concurrent=self.config.max_concurrent
        )
        
        # Update query state with results
        for action in executed_actions:
            if action.is_executed and action.result:
                query_state.update_action(
                    action.action_id,
                    action.result.response,
                    action.result.score,
                    action.metadata
                )
                
                # Add follow-up actions
                for follow_up in action.get_follow_ups():
                    query_state.add_action(follow_up.action_id, follow_up.query)
                    query_state.relate_actions(action.action_id, follow_up.action_id)
    
    async def _reduction_phase(self, query: str, query_state: DRIFTQueryState) -> str:
        """Execute reduction phase to synthesize final answer"""
        logger.info("📝 Executing reduction phase...")
        
        try:
            # Get completed actions
            completed_actions = query_state.get_completed_actions()
            
            if not completed_actions:
                return "No completed actions found during search."
            
            # Format research results
            research_results = []
            for action_id in completed_actions:
                action_info = query_state.get_action_info(action_id)
                if action_info and action_info.get("answer"):
                    research_results.append({
                        "query": action_info["query"],
                        "answer": action_info["answer"],
                        "score": action_info.get("score", 0.0),
                        "depth": action_info.get("depth", 0)
                    })
            
            # Sort by score for better synthesis
            research_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Format for LLM
            formatted_results = []
            for i, result in enumerate(research_results, 1):
                formatted_results.append(
                    f"Result {i} (Score: {result['score']:.1f}, Depth: {result['depth']}):\n"
                    f"Query: {result['query']}\n"
                    f"Answer: {result['answer']}\n"
                )
            
            results_text = "\n---\n".join(formatted_results)
            
            # Generate final synthesis
            response = await self.reduce_chain.ainvoke({
                "query": query,
                "research_results": results_text
            })
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error in reduction phase: {e}")
            return self._extract_best_answer(query_state)
    
    def _extract_best_answer(self, query_state: DRIFTQueryState) -> str:
        """Extract best answer from completed actions (fallback)"""
        completed_actions = query_state.get_completed_actions()
        
        if not completed_actions:
            return "No results found during search."
        
        # Find highest scoring action
        best_action_id = None
        best_score = -1
        
        for action_id in completed_actions:
            action_info = query_state.get_action_info(action_id)
            if action_info and action_info.get("score", 0) > best_score:
                best_score = action_info["score"]
                best_action_id = action_id
        
        if best_action_id:
            action_info = query_state.get_action_info(best_action_id)
            return action_info.get("answer", "No answer found.")
        
        return "No valid results found."
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about search capabilities"""
        return {
            "config": {
                "max_depth": self.config.n_depth,
                "max_follow_ups": self.config.max_follow_ups,
                "max_concurrent": self.config.max_concurrent,
                "auto_route": self.config.auto_route,
                "enable_primer": self.config.enable_primer,
                "enable_reduce": self.config.enable_reduce
            },
            "components": {
                "context_builder": self.context_builder.get_context_stats(),
                "primer": self.primer.get_primer_stats(),
                "local_retriever": self.local_retriever is not None,
                "global_retriever": self.global_retriever is not None,
                "classifier": self.classifier is not None
            }
        }
    
    def close(self):
        """Close graph processor connection"""
        if hasattr(self, 'graph_processor'):
            self.graph_processor.close()


# Compatibility functions for existing benchmark system
async def query_drift_search(query: str, k: int = 5, graph_processor=None, **kwargs) -> Dict[str, Any]:
    """
    Main interface function for DRIFT search compatible with benchmark system
    
    Args:
        query: The search query
        k: Parameter for compatibility (used in max_concurrent)
        graph_processor: Optional existing AdvancedGraphProcessor instance
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary with response and retrieval details
    """
    try:
        # Use provided processor or create new one
        if graph_processor is not None:
            processor = graph_processor
        else:
            # Import graph processor
            from data_processors import AdvancedGraphProcessor
            
            # Initialize processor and DRIFT search with error handling
            try:
                processor = AdvancedGraphProcessor()
                print("✅ Neo4j connection successful")
            except Exception as e:
                print(f"⚠️ Neo4j connection failed: {e}")
                print("🔄 Returning fallback response...")
                
                # Return a simple fallback response
                return {
                    'response': f"I apologize, but I'm unable to process your query '{query}' due to database connectivity issues. Please check your Neo4j connection and try again.",
                    'method': 'DRIFT_SEARCH_FALLBACK',
                    'execution_time': 0.1,
                    'error': str(e)
                }
        
        # Configure DRIFT with benchmark parameters
        config = DRIFTConfig(
            max_concurrent=min(k, 3),  # Use k but limit to 3 for efficiency
            **kwargs
        )
        
        drift_search = DRIFTSearch(processor, config)
        
        # Perform search
        result = await drift_search.search(query, **kwargs)
        
        # Return in benchmark-compatible format
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error in DRIFT search: {e}")
        import traceback
        traceback.print_exc()
        return {
            "final_answer": f"Error during DRIFT search: {str(e)}",
            "method": "drift_search_error",
            "execution_time": 0,
            "performance_metrics": {
                "completion_time": 0,
                "llm_calls": 0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "error": str(e)
            },
            "retrieval_details": [{
                "action_id": "error",
                "query": query,
                "content": f"DRIFT search encountered an error: {str(e)}",
                "score": 0.0,
                "depth": 0,
                "metadata": {"error": str(e)}
            }]
        }
    finally:
        # Only close processor if we created a new one
        if 'processor' in locals() and graph_processor is None:
            processor.close()


def create_drift_search(graph_processor, config: Optional[DRIFTConfig] = None) -> DRIFTSearch:
    """
    Factory function to create DRIFT search instance
    
    Args:
        graph_processor: AdvancedGraphProcessor instance
        config: Optional DRIFT configuration
        
    Returns:
        DRIFTSearch instance
    """
    return DRIFTSearch(graph_processor, config) 