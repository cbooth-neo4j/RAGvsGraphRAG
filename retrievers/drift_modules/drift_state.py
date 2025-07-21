"""
DRIFT Query State Module

This module provides query state management using NetworkX graphs
for tracking action relationships and execution flow.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ActionMetadata:
    """Metadata for tracking action execution"""
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    completion_time: float = 0.0
    context_data: Optional[Dict[str, Any]] = None
    execution_depth: int = 0
    parent_action: Optional[str] = None
    created_at: Optional[float] = None
    executed_at: Optional[float] = None

class DRIFTQueryState:
    """
    Manages the action graph and query state using NetworkX.
    Tracks relationships between actions, execution status, and metadata.
    """
    
    def __init__(self, global_query: str):
        """
        Initialize query state
        
        Args:
            global_query: The original query that initiated the DRIFT search
        """
        self.global_query = global_query
        self.graph = nx.MultiDiGraph()
        self._action_counter = 0
        self._execution_order = []
        self._completed_actions = set()
        self._failed_actions = set()
        
        # Track token usage across all actions
        self._total_tokens = {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        
        # Track LLM calls
        self._total_llm_calls = 0
        self._total_completion_time = 0.0
    
    def add_action(
        self, 
        action_id: str, 
        query: str, 
        answer: Optional[str] = None,
        score: Optional[float] = None,
        metadata: Optional[ActionMetadata] = None
    ) -> str:
        """
        Add action to the graph
        
        Args:
            action_id: Unique identifier for the action
            query: The query text for this action
            answer: Optional answer if action is completed
            score: Optional relevance score
            metadata: Optional metadata for tracking
            
        Returns:
            Action ID that was added
        """
        if action_id in self.graph.nodes:
            logger.warning(f"Action {action_id} already exists in graph")
            return action_id
        
        self._action_counter += 1
        
        # Create default metadata if not provided
        if metadata is None:
            metadata = ActionMetadata()
        
        # Add node to graph
        self.graph.add_node(
            action_id,
            query=query,
            answer=answer,
            score=score,
            metadata=metadata,
            is_complete=answer is not None,
            node_type="action"
        )
        
        logger.debug(f"Added action {action_id} to graph")
        return action_id
    
    def update_action(
        self, 
        action_id: str, 
        answer: Optional[str] = None,
        score: Optional[float] = None,
        metadata: Optional[ActionMetadata] = None
    ) -> bool:
        """
        Update existing action with results
        
        Args:
            action_id: ID of action to update
            answer: Answer from action execution
            score: Relevance score
            metadata: Execution metadata
            
        Returns:
            True if update was successful
        """
        if action_id not in self.graph.nodes:
            logger.error(f"Action {action_id} not found in graph")
            return False
        
        # Update node attributes
        if answer is not None:
            self.graph.nodes[action_id]["answer"] = answer
            self.graph.nodes[action_id]["is_complete"] = True
            self._completed_actions.add(action_id)
        
        if score is not None:
            self.graph.nodes[action_id]["score"] = score
        
        if metadata is not None:
            self.graph.nodes[action_id]["metadata"] = metadata
            
            # Update global token tracking with null checks
            prompt_tokens = metadata.prompt_tokens if metadata.prompt_tokens is not None else 0
            output_tokens = metadata.output_tokens if metadata.output_tokens is not None else 0
            llm_calls = metadata.llm_calls if metadata.llm_calls is not None else 0
            completion_time = metadata.completion_time if metadata.completion_time is not None else 0.0
            
            self._total_tokens["prompt_tokens"] += prompt_tokens
            self._total_tokens["output_tokens"] += output_tokens
            self._total_tokens["total_tokens"] += prompt_tokens + output_tokens
            
            # Update global call tracking
            self._total_llm_calls += llm_calls
            self._total_completion_time += completion_time
        
        # Track execution order
        if answer is not None and action_id not in self._execution_order:
            self._execution_order.append(action_id)
        
        logger.debug(f"Updated action {action_id}")
        return True
    
    def relate_actions(
        self, 
        parent_id: str, 
        child_id: str, 
        relationship_type: str = "follows_up",
        weight: float = 1.0
    ) -> bool:
        """
        Create relationship between actions
        
        Args:
            parent_id: ID of parent action
            child_id: ID of child action
            relationship_type: Type of relationship
            weight: Edge weight for ranking
            
        Returns:
            True if relationship was created
        """
        if parent_id not in self.graph.nodes:
            logger.error(f"Parent action {parent_id} not found")
            return False
        
        if child_id not in self.graph.nodes:
            logger.error(f"Child action {child_id} not found")
            return False
        
        # Add edge with attributes
        self.graph.add_edge(
            parent_id, 
            child_id, 
            relationship_type=relationship_type,
            weight=weight
        )
        
        logger.debug(f"Created relationship {parent_id} -> {child_id} ({relationship_type})")
        return True
    
    def get_incomplete_actions(self) -> List[str]:
        """Get all actions that haven't been executed yet"""
        incomplete = []
        for node_id in self.graph.nodes:
            if not self.graph.nodes[node_id].get("is_complete", False):
                incomplete.append(node_id)
        return incomplete
    
    def get_completed_actions(self) -> List[str]:
        """Get all completed actions"""
        return list(self._completed_actions)
    
    def rank_actions(self, action_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Rank actions based on scores and graph structure
        
        Args:
            action_ids: List of action IDs to rank
            
        Returns:
            List of (action_id, rank_score) tuples sorted by rank
        """
        if not action_ids:
            return []
        
        ranked = []
        for action_id in action_ids:
            if action_id not in self.graph.nodes:
                continue
            
            node = self.graph.nodes[action_id]
            base_score = node.get("score", 0.0)
            
            # Handle None scores properly
            if base_score is None:
                base_score = 0.0
            
            # Consider graph structure (number of predecessors/successors)
            in_degree = self.graph.in_degree(action_id)
            out_degree = self.graph.out_degree(action_id)
            
            # Simple ranking formula (can be made more sophisticated)
            rank_score = base_score + (in_degree * 5) + (out_degree * 2)
            
            ranked.append((action_id, rank_score))
        
        # Sort by rank score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def get_action_path(self, action_id: str) -> List[str]:
        """Get path from root to specified action"""
        if action_id not in self.graph.nodes:
            return []
        
        # Find path from any root node to this action
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        
        for root in roots:
            try:
                path = nx.shortest_path(self.graph, root, action_id)
                return path
            except nx.NetworkXNoPath:
                continue
        
        return [action_id]  # Return single node if no path found
    
    def get_action_depth(self, action_id: str) -> int:
        """Get depth of action in the graph"""
        path = self.get_action_path(action_id)
        return len(path) - 1 if path else 0
    
    def get_follow_up_actions(self, action_id: str) -> List[str]:
        """Get all follow-up actions for a given action"""
        if action_id not in self.graph.nodes:
            return []
        
        return list(self.graph.successors(action_id))
    
    def get_parent_actions(self, action_id: str) -> List[str]:
        """Get all parent actions for a given action"""
        if action_id not in self.graph.nodes:
            return []
        
        return list(self.graph.predecessors(action_id))
    
    def get_action_info(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get complete information about an action"""
        if action_id not in self.graph.nodes:
            return None
        
        node = self.graph.nodes[action_id]
        return {
            "id": action_id,
            "query": node.get("query", ""),
            "answer": node.get("answer"),
            "score": node.get("score"),
            "is_complete": node.get("is_complete", False),
            "metadata": node.get("metadata"),
            "depth": self.get_action_depth(action_id),
            "follow_ups": self.get_follow_up_actions(action_id),
            "parents": self.get_parent_actions(action_id)
        }
    
    def serialize(self, include_context: bool = False) -> Dict[str, Any]:
        """
        Serialize query state to dictionary
        
        Args:
            include_context: Whether to include full context data
            
        Returns:
            Dictionary representation of query state
        """
        # Serialize nodes
        nodes = {}
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            node_data = {
                "id": node_id,
                "query": node.get("query", ""),
                "answer": node.get("answer"),
                "score": node.get("score"),
                "is_complete": node.get("is_complete", False),
                "depth": self.get_action_depth(node_id)
            }
            
            # Handle metadata
            metadata = node.get("metadata")
            if metadata:
                if include_context:
                    node_data["metadata"] = asdict(metadata)
                else:
                    # Exclude context_data for size
                    node_data["metadata"] = {
                        "llm_calls": metadata.llm_calls,
                        "prompt_tokens": metadata.prompt_tokens,
                        "output_tokens": metadata.output_tokens,
                        "completion_time": metadata.completion_time,
                        "execution_depth": metadata.execution_depth
                    }
            
            nodes[node_id] = node_data
        
        # Serialize edges
        edges = []
        for edge in self.graph.edges(data=True):
            edges.append({
                "source": edge[0],
                "target": edge[1],
                "relationship_type": edge[2].get("relationship_type", "follows_up"),
                "weight": edge[2].get("weight", 1.0)
            })
        
        return {
            "global_query": self.global_query,
            "nodes": nodes,
            "edges": edges,
            "execution_order": self._execution_order,
            "completed_actions": list(self._completed_actions),
            "failed_actions": list(self._failed_actions),
            "total_tokens": self._total_tokens,
            "total_llm_calls": self._total_llm_calls,
            "total_completion_time": self._total_completion_time,
            "action_count": self._action_counter,
            "graph_stats": {
                "node_count": self.graph.number_of_nodes(),
                "edge_count": self.graph.number_of_edges(),
                "completed_count": len(self._completed_actions),
                "incomplete_count": len(self.get_incomplete_actions())
            }
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution statistics"""
        incomplete_actions = self.get_incomplete_actions()
        
        return {
            "global_query": self.global_query,
            "total_actions": self._action_counter,
            "completed_actions": len(self._completed_actions),
            "incomplete_actions": len(incomplete_actions),
            "failed_actions": len(self._failed_actions),
            "execution_order": self._execution_order,
            "total_llm_calls": self._total_llm_calls,
            "total_completion_time": self._total_completion_time,
            "token_usage": self._total_tokens,
            "graph_structure": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0.0
            }
        }
    
    def to_json(self, include_context: bool = False) -> str:
        """Convert query state to JSON string"""
        return json.dumps(self.serialize(include_context), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DRIFTQueryState":
        """Create query state from dictionary"""
        state = cls(data["global_query"])
        
        # Restore nodes
        for node_id, node_data in data["nodes"].items():
            metadata = None
            if "metadata" in node_data:
                metadata = ActionMetadata(**node_data["metadata"])
            
            state.add_action(
                node_id,
                node_data["query"],
                node_data.get("answer"),
                node_data.get("score"),
                metadata
            )
        
        # Restore edges
        for edge in data["edges"]:
            state.relate_actions(
                edge["source"],
                edge["target"],
                edge.get("relationship_type", "follows_up"),
                edge.get("weight", 1.0)
            )
        
        # Restore state
        state._execution_order = data.get("execution_order", [])
        state._completed_actions = set(data.get("completed_actions", []))
        state._failed_actions = set(data.get("failed_actions", []))
        state._total_tokens = data.get("total_tokens", {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        state._total_llm_calls = data.get("total_llm_calls", 0)
        state._total_completion_time = data.get("total_completion_time", 0.0)
        state._action_counter = data.get("action_count", 0)
        
        return state 