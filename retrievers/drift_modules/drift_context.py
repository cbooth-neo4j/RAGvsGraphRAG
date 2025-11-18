"""
DRIFT Context Builder

This module provides context building capabilities for DRIFT search,
leveraging existing community detection and Neo4j integration.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

@dataclass
class DRIFTContextConfig:
    """Configuration for DRIFT context building"""
    max_communities: int = 8
    max_context_tokens: int = 8000
    min_relevance_score: float = 0.7
    levels: List[int] = None
    
    def __post_init__(self):
        if self.levels is None:
            self.levels = [0, 1, 2]

class DRIFTContextBuilder:
    """
    Context builder for DRIFT search that leverages existing community detection
    and Neo4j integration from the advanced GraphRAG system.
    """
    
    def __init__(self, graph_processor, config: Optional[DRIFTContextConfig] = None):
        """
        Initialize context builder
        
        Args:
            graph_processor: AdvancedGraphProcessor instance with Neo4j connection
            config: Optional configuration for context building
        """
        self.graph = graph_processor
        self.driver = graph_processor.driver
        self.config = config or DRIFTContextConfig()
        self.embeddings = graph_processor.embeddings
        
        # Community selection prompt
        self.community_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at selecting relevant communities for query answering.
            
Given a query and a list of community summaries, select the most relevant communities
that would help answer the query comprehensively.

Consider:
1. Direct relevance to the query topic
2. Complementary information from different communities
3. Hierarchical relationships between communities
4. Coverage of different aspects of the query

Return a JSON object with selected community IDs and relevance scores (0-100)."""),
            ("human", """Query: {query}

Communities:
{communities}

Select the most relevant communities (max {max_communities}).""")
        ])
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    async def build_context(
        self, 
        query: str, 
        max_communities: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build context for DRIFT search using existing community detection
        
        Args:
            query: The search query
            max_communities: Maximum number of communities to select
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing context data and metadata
        """
        max_communities = max_communities or self.config.max_communities
        
        try:
            # Step 1: Get available communities from existing graph
            communities = await self._get_available_communities()
            
            if not communities:
                logger.warning("No communities found in graph")
                return {
                    "context_chunks": [],
                    "selected_communities": [],
                    "context_text": "",
                    "llm_calls": 0,
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "error": "No communities available"
                }
            
            # Step 2: Select relevant communities based on query
            selected_communities = await self._select_relevant_communities(
                query, communities, max_communities
            )
            
            # Step 3: Build context chunks from selected communities
            context_chunks = await self._build_context_chunks(selected_communities)
            
            # Step 4: Format context text
            context_text = self._format_context_text(context_chunks)
            
            return {
                "context_chunks": context_chunks,
                "selected_communities": selected_communities,
                "context_text": context_text,
                "llm_calls": 1,  # For community selection
                "prompt_tokens": 0,  # Would need to count tokens
                "output_tokens": 0,  # Would need to count tokens
                "total_communities": len(communities),
                "selected_count": len(selected_communities)
            }
            
        except Exception as e:
            logger.error(f"Error building DRIFT context: {e}")
            return {
                "context_chunks": [],
                "selected_communities": [],
                "context_text": "",
                "llm_calls": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "error": str(e)
            }
    
    async def _get_available_communities(self) -> List[Dict[str, Any]]:
        """Get available communities from the Neo4j graph"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:__Community__)
                    WHERE c.summary IS NOT NULL
                    OPTIONAL MATCH (c)<-[:IN_COMMUNITY*]-(e:__Entity__)
                    WITH c, count(DISTINCT e) as entity_count
                    RETURN 
                        c.id as community_id,
                        c.level as level,
                        c.summary as summary,
                        c.community_rank as rank,
                        entity_count
                    ORDER BY c.community_rank DESC, entity_count DESC
                    LIMIT 50
                """)
                
                communities = []
                for record in result:
                    communities.append({
                        "id": record["community_id"],
                        "level": record["level"],
                        "summary": record["summary"],
                        "rank": record["rank"] or 0,
                        "entity_count": record["entity_count"]
                    })
                
                logger.info(f"Retrieved {len(communities)} communities from graph")
                return communities
                
        except Exception as e:
            logger.error(f"Error retrieving communities: {e}")
            return []
    
    async def _select_relevant_communities(
        self, 
        query: str, 
        communities: List[Dict[str, Any]], 
        max_communities: int
    ) -> List[Dict[str, Any]]:
        """Select the most relevant communities for the query"""
        
        if not communities:
            return []
        
        # For now, use a simple ranking approach
        # In a full implementation, this would use the LLM for intelligent selection
        
        # Filter by levels specified in config
        filtered_communities = [
            c for c in communities 
            if c["level"] in self.config.levels
        ]
        
        # Sort by rank and entity count
        sorted_communities = sorted(
            filtered_communities,
            key=lambda x: (x["rank"], x["entity_count"]),
            reverse=True
        )
        
        # Return top communities
        selected = sorted_communities[:max_communities]
        
        logger.info(f"Selected {len(selected)} communities from {len(communities)} available")
        return selected
    
    async def _build_context_chunks(self, communities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build context chunks from selected communities"""
        context_chunks = []
        
        for community in communities:
            # Get additional community details if needed
            community_context = await self._get_community_context(community)
            
            context_chunks.append({
                "community_id": community["id"],
                "level": community["level"],
                "summary": community["summary"],
                "rank": community["rank"],
                "entity_count": community["entity_count"],
                "context": community_context
            })
        
        return context_chunks
    
    async def _get_community_context(self, community: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional context for a community"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:__Community__ {id: $community_id})<-[:IN_COMMUNITY*]-(e:__Entity__)
                    WITH c, collect(DISTINCT e.id) as entity_ids
                    RETURN entity_ids
                    LIMIT 1
                """, community_id=community["id"])
                
                record = result.single()
                if record:
                    return {
                        "entity_ids": record["entity_ids"][:10]  # Limit to top 10
                    }
                
        except Exception as e:
            logger.error(f"Error getting community context: {e}")
        
        return {}
    
    def _format_context_text(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks into text for LLM consumption"""
        if not context_chunks:
            return ""
        
        context_parts = []
        for chunk in context_chunks:
            context_part = f"Community {chunk['community_id']} (Level {chunk['level']}, Rank {chunk['rank']}):\n"
            context_part += f"{chunk['summary']}\n"
            
            # Add entity information if available
            if chunk.get('context', {}).get('entity_ids'):
                entity_ids = chunk['context']['entity_ids']
                context_part += f"Key entities: {', '.join(entity_ids[:5])}\n"
            
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about available context"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:__Community__)
                    WITH count(c) as total_communities
                    MATCH (c:__Community__)
                    WHERE c.summary IS NOT NULL
                    RETURN 
                        total_communities,
                        count(c) as communities_with_summaries
                """)
                
                record = result.single()
                if record:
                    return {
                        "total_communities": record["total_communities"],
                        "communities_with_summaries": record["communities_with_summaries"]
                    }
                
        except Exception as e:
            logger.error(f"Error getting context stats: {e}")
        
        return {"total_communities": 0, "communities_with_summaries": 0} 