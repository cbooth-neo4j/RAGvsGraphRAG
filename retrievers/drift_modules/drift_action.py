"""
DRIFT Action Module

This module provides individual search actions that can be executed
with adaptive routing using existing local/global retrievers.
"""

import asyncio
import json
import time
import secrets
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .drift_state import ActionMetadata

logger = logging.getLogger(__name__)

class ActionResult(BaseModel):
    """Structured result from action execution"""
    response: str = Field(description="The response/answer from the action")
    follow_up_queries: List[str] = Field(description="Generated follow-up queries")
    score: float = Field(description="Relevance score (0-100)")
    search_type: str = Field(description="Type of search performed (local/global)")
    context_summary: str = Field(description="Brief summary of context used")

@dataclass
class DRIFTActionConfig:
    """Configuration for DRIFT actions"""
    max_follow_ups: int = 3
    min_score_threshold: float = 20.0
    temperature: float = 0.1
    auto_route: bool = True  # Use existing QueryClassifier
    default_search_type: str = "local"  # local, global, or hybrid

class DRIFTAction:
    """
    Individual search action that can be executed with adaptive routing.
    Uses existing GraphRAGLocalRetriever and GraphRAGGlobalRetriever.
    """
    
    def __init__(
        self,
        query: str,
        action_id: Optional[str] = None,
        global_query: Optional[str] = None,
        config: Optional[DRIFTActionConfig] = None
    ):
        """
        Initialize DRIFT action
        
        Args:
            query: The specific query for this action
            action_id: Optional unique identifier
            global_query: The original global query for context
            config: Optional configuration
        """
        self.query = query
        self.action_id = action_id or self._generate_action_id()
        self.global_query = global_query or query
        self.config = config or DRIFTActionConfig()
        
        # Execution state
        self.is_executed = False
        self.result: Optional[ActionResult] = None
        self.metadata: Optional[ActionMetadata] = None
        self.follow_up_actions: List["DRIFTAction"] = []
        
        # For structured output processing
        self.action_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant conducting focused analysis for a specific query.

Your task is to:
1. Analyze the search results and provide a comprehensive answer
2. Generate follow-up queries that would help provide more complete information
3. Assign a relevance score (0-100) based on how well the results address the query
4. Provide a brief context summary

Guidelines for follow-up queries:
- Focus on gaps in the current information
- Ask about specific details, relationships, or aspects not fully covered
- Keep queries specific and actionable
- Limit to {max_follow_ups} most important follow-ups

Return a JSON object with the following structure:
{{
    "response": "Comprehensive answer based on search results",
    "follow_up_queries": ["Query 1", "Query 2", "Query 3"],
    "score": 85,
    "search_type": "local",
    "context_summary": "Brief summary of context used"
}}"""),
            ("human", """Query: {query}
Global Context: {global_query}

Search Results:
{search_results}

Please analyze and provide structured response.""")
        ])
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=self.config.temperature)
        self.action_chain = self.action_prompt | self.llm.with_structured_output(ActionResult)
    
    def _generate_action_id(self) -> str:
        """Generate unique action ID"""
        return f"action_{secrets.token_hex(8)}"
    
    async def execute(
        self,
        local_retriever,
        global_retriever=None,
        classifier=None,
        **kwargs
    ) -> "DRIFTAction":
        """
        Execute the action using appropriate retriever
        
        Args:
            local_retriever: GraphRAGLocalRetriever instance
            global_retriever: Optional GraphRAGGlobalRetriever instance
            classifier: Optional QueryClassifier for auto-routing
            **kwargs: Additional search parameters
            
        Returns:
            Self with execution results
        """
        if self.is_executed:
            logger.warning(f"Action {self.action_id} already executed")
            return self
        
        start_time = time.time()
        
        try:
            # Step 1: Determine search type
            search_type = await self._determine_search_type(classifier)
            
            # Step 2: Execute search
            search_result = await self._execute_search(
                search_type, local_retriever, global_retriever, **kwargs
            )
            
            # Step 3: Process results with LLM
            self.result = await self._process_search_results(search_result, search_type)
            
            # Step 4: Generate follow-up actions
            self.follow_up_actions = [
                DRIFTAction(
                    query=fq,
                    global_query=self.global_query,
                    config=self.config
                ) for fq in self.result.follow_up_queries
            ]
            
            # Step 5: Update metadata
            completion_time = time.time() - start_time
            self.metadata = ActionMetadata(
                llm_calls=search_result.get("llm_calls", 1),  # Use actual LLM calls from search
                prompt_tokens=search_result.get("prompt_tokens", 0),  # Use actual token counts
                output_tokens=search_result.get("output_tokens", 0),  # Use actual token counts
                completion_time=completion_time,
                context_data=search_result.get("context_data", {}),
                execution_depth=0,  # Would be set by parent
                created_at=start_time,
                executed_at=time.time()
            )
            
            self.is_executed = True
            logger.info(f"Action {self.action_id} executed successfully (score: {self.result.score})")
            
        except Exception as e:
            logger.error(f"Error executing action {self.action_id}: {e}")
            self.result = ActionResult(
                response=f"Error during action execution: {str(e)}",
                follow_up_queries=[],
                score=0.0,
                search_type="error",
                context_summary="Error occurred during execution"
            )
            self.metadata = ActionMetadata(
                llm_calls=0,  # No LLM calls on error
                prompt_tokens=0,  # No tokens on error
                output_tokens=0,  # No tokens on error
                completion_time=time.time() - start_time,
                context_data={},
                execution_depth=0,
                created_at=start_time,
                executed_at=time.time()
            )
            self.is_executed = True
        
        return self
    
    async def _determine_search_type(self, classifier) -> str:
        """Determine which search type to use"""
        if not self.config.auto_route or classifier is None:
            return self.config.default_search_type
        
        try:
            # Use existing QueryClassifier to determine search type
            classification = await classifier.classify(self.query)
            return "local" if classification == "LOCAL" else "global"
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            return self.config.default_search_type
    
    async def _execute_search(
        self,
        search_type: str,
        local_retriever,
        global_retriever=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute search using appropriate retriever"""
        try:
            if search_type == "global" and global_retriever:
                logger.debug(f"Executing global search for: {self.query[:50]}...")
                result = await global_retriever.search(self.query, **kwargs)
            else:
                logger.debug(f"Executing local search for: {self.query[:50]}...")
                result = await local_retriever.search(self.query, **kwargs)
            
            # Convert SearchResult to dict format
            if hasattr(result, 'response'):
                return {
                    "response": result.response or "",
                    "context_data": result.context_data or {},
                    "context_text": result.context_text or "",
                    "completion_time": result.completion_time or 0.0,
                    "llm_calls": result.llm_calls or 0,
                    "prompt_tokens": result.prompt_tokens or 0,
                    "output_tokens": result.output_tokens or 0,
                    "search_type": search_type
                }
            else:
                # Handle dict-style result
                return {
                    "response": result.get("final_answer", ""),
                    "context_data": result.get("context_data", {}),
                    "context_text": result.get("context_text", ""),
                    "completion_time": result.get("completion_time", 0),
                    "llm_calls": result.get("llm_calls", 0),
                    "prompt_tokens": result.get("prompt_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "search_type": search_type
                }
            
        except Exception as e:
            logger.error(f"Error in search execution: {e}")
            return {
                "response": f"Error during search: {str(e)}",
                "context_data": {},
                "context_text": "",
                "completion_time": 0,
                "llm_calls": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "search_type": search_type,
                "error": str(e)
            }
    
    async def _process_search_results(
        self, 
        search_result: Dict[str, Any], 
        search_type: str
    ) -> ActionResult:
        """Process search results with LLM to generate structured output"""
        try:
            # Prepare search results for LLM
            search_results_text = f"Search Type: {search_type}\n\n"
            search_results_text += f"Response: {search_result.get('response', '')}\n\n"
            
            if search_result.get('context_text'):
                search_results_text += f"Context: {search_result['context_text'][:1000]}...\n\n"
            
            # Add metadata
            search_results_text += f"Metadata: {search_result.get('llm_calls', 0)} LLM calls, "
            search_results_text += f"{search_result.get('completion_time', 0):.2f}s completion time"
            
            # Generate structured result
            result = await self.action_chain.ainvoke({
                "query": self.query,
                "global_query": self.global_query,
                "search_results": search_results_text,
                "max_follow_ups": self.config.max_follow_ups
            })
            
            # Update search type in result
            result.search_type = search_type
            
            # Validate and clean up
            if not result.follow_up_queries:
                result.follow_up_queries = []
            
            result.follow_up_queries = result.follow_up_queries[:self.config.max_follow_ups]
            result.score = max(0, min(100, result.score))
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            return ActionResult(
                response=search_result.get('response', f"Error processing results: {str(e)}"),
                follow_up_queries=[],
                score=0.0,
                search_type=search_type,
                context_summary="Error during result processing"
            )
    
    def get_answer(self) -> str:
        """Get the answer from executed action"""
        if not self.is_executed or not self.result:
            return ""
        return self.result.response
    
    def get_score(self) -> float:
        """Get the relevance score"""
        if not self.is_executed or not self.result:
            return 0.0
        return self.result.score
    
    def get_follow_ups(self) -> List["DRIFTAction"]:
        """Get follow-up actions"""
        return self.follow_up_actions
    
    def serialize(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Serialize action to dictionary"""
        data = {
            "action_id": self.action_id,
            "query": self.query,
            "global_query": self.global_query,
            "is_executed": self.is_executed
        }
        
        if self.result:
            data["result"] = {
                "response": self.result.response,
                "follow_up_queries": self.result.follow_up_queries,
                "score": self.result.score,
                "search_type": self.result.search_type,
                "context_summary": self.result.context_summary
            }
        
        if include_metadata and self.metadata:
            data["metadata"] = {
                "llm_calls": self.metadata.llm_calls,
                "prompt_tokens": self.metadata.prompt_tokens,
                "output_tokens": self.metadata.output_tokens,
                "completion_time": self.metadata.completion_time,
                "execution_depth": self.metadata.execution_depth,
                "created_at": self.metadata.created_at,
                "executed_at": self.metadata.executed_at
            }
        
        if self.follow_up_actions:
            data["follow_up_actions"] = [
                action.serialize(include_metadata=False) for action in self.follow_up_actions
            ]
        
        return data
    
    def __str__(self) -> str:
        """String representation"""
        status = "✅" if self.is_executed else "⏳"
        score = f" (score: {self.get_score():.1f})" if self.is_executed else ""
        return f"{status} {self.action_id}: {self.query[:50]}...{score}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"DRIFTAction(id={self.action_id}, query='{self.query[:30]}...', executed={self.is_executed})"
    
    def __hash__(self) -> int:
        """Make action hashable"""
        return hash(self.action_id)
    
    def __eq__(self, other) -> bool:
        """Check equality based on action ID"""
        if not isinstance(other, DRIFTAction):
            return False
        return self.action_id == other.action_id
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DRIFTAction":
        """Create action from dictionary"""
        action = cls(
            query=data["query"],
            action_id=data["action_id"],
            global_query=data.get("global_query")
        )
        
        action.is_executed = data.get("is_executed", False)
        
        if "result" in data:
            result_data = data["result"]
            action.result = ActionResult(
                response=result_data["response"],
                follow_up_queries=result_data.get("follow_up_queries", []),
                score=result_data.get("score", 0.0),
                search_type=result_data.get("search_type", "local"),
                context_summary=result_data.get("context_summary", "")
            )
        
        if "metadata" in data:
            metadata_data = data["metadata"]
            action.metadata = ActionMetadata(
                llm_calls=metadata_data.get("llm_calls", 0),
                prompt_tokens=metadata_data.get("prompt_tokens", 0),
                output_tokens=metadata_data.get("output_tokens", 0),
                completion_time=metadata_data.get("completion_time", 0.0),
                execution_depth=metadata_data.get("execution_depth", 0),
                created_at=metadata_data.get("created_at"),
                executed_at=metadata_data.get("executed_at")
            )
        
        if "follow_up_actions" in data:
            action.follow_up_actions = [
                cls.from_dict(fup_data) for fup_data in data["follow_up_actions"]
            ]
        
        return action


# Utility functions for working with actions
def create_action_batch(queries: List[str], global_query: str, config: Optional[DRIFTActionConfig] = None) -> List[DRIFTAction]:
    """Create a batch of actions from queries"""
    return [
        DRIFTAction(query=query, global_query=global_query, config=config)
        for query in queries
    ]

async def execute_action_batch(
    actions: List[DRIFTAction],
    local_retriever,
    global_retriever=None,
    classifier=None,
    max_concurrent: int = 3,
    **kwargs
) -> List[DRIFTAction]:
    """Execute a batch of actions with concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_semaphore(action):
        async with semaphore:
            return await action.execute(local_retriever, global_retriever, classifier, **kwargs)
    
    # Execute all actions concurrently
    tasks = [execute_with_semaphore(action) for action in actions]
    executed_actions = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    results = []
    for i, result in enumerate(executed_actions):
        if isinstance(result, Exception):
            logger.error(f"Error executing action {i}: {result}")
            actions[i].result = ActionResult(
                response=f"Error: {str(result)}",
                follow_up_queries=[],
                score=0.0,
                search_type="error",
                context_summary="Error during execution"
            )
            actions[i].is_executed = True
        results.append(actions[i])
    
    return results 