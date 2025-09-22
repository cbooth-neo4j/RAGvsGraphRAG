"""
DRIFT Primer Module

This module provides query decomposition and primer capabilities for DRIFT search,
leveraging existing community reports and HyDE query expansion.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from config.model_factory import get_llm

logger = logging.getLogger(__name__)

class PrimerResponse(BaseModel):
    """Structured response from primer query processing"""
    intermediate_answer: str = Field(description="Initial answer based on community context")
    follow_up_queries: List[str] = Field(description="List of follow-up queries for deeper exploration")
    score: float = Field(description="Relevance score (0-100)")
    reasoning: str = Field(description="Brief reasoning for the follow-up queries")

@dataclass
class PrimerConfig:
    """Configuration for primer query processing"""
    max_communities: int = 8
    max_follow_ups: int = 3
    min_score_threshold: float = 20.0
    # Removed temperature - let models use their defaults
    use_hyde: bool = True

class DRIFTPrimer:
    """
    Query decomposition and primer processor for DRIFT search.
    Uses existing community reports for initial query processing and follow-up generation.
    """
    
    def __init__(self, graph_processor, config: Optional[PrimerConfig] = None):
        """
        Initialize primer processor
        
        Args:
            graph_processor: CustomGraphProcessor instance with Neo4j connection
            config: Optional configuration for primer processing
        """
        self.graph = graph_processor
        self.driver = graph_processor.driver
        self.config = config or PrimerConfig()
        self.embeddings = graph_processor.embeddings
        self.llm = get_llm(temperature=self.config.temperature)
        
        # HyDE query expansion prompt
        self.hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query expander. Your task is to generate a hypothetical ideal answer 
that would perfectly respond to the user's question. This hypothetical answer will be used to find similar 
content in the knowledge base.

Generate a comprehensive, detailed hypothetical answer that covers:
1. Key concepts and terminology relevant to the query
2. Specific details that would be in a perfect answer
3. Related topics and context
4. Technical or domain-specific information

Make the answer detailed and specific, as if it were extracted from authoritative documentation."""),
            ("human", "Query: {query}\n\nGenerate a hypothetical ideal answer:")
        ])
        
        # Primer query processing prompt
        self.primer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant conducting initial analysis of a query using available community summaries.

Your task is to:
1. Analyze the query and available community information
2. Provide an initial intermediate answer based on the community context
3. Generate follow-up queries that would help provide a more comprehensive answer
4. Assign a relevance score (0-100) based on how well the communities address the query

Guidelines for follow-up queries:
- Focus on gaps in the current information
- Ask about specific details, relationships, or aspects not fully covered
- Prioritize queries that would provide the most valuable additional context
- Keep queries specific and actionable
- Limit to {max_follow_ups} most important follow-ups

Return a JSON object with the following structure:
{{
    "intermediate_answer": "Initial answer based on community context",
    "follow_up_queries": ["Query 1", "Query 2", "Query 3"],
    "score": 75,
    "reasoning": "Brief explanation of why these follow-ups were chosen"
}}"""),
            ("human", """Query: {query}

Available Community Context:
{community_context}

Please analyze and provide structured response.""")
        ])
        
        # Set up structured output for primer processing
        self.primer_chain = self.primer_prompt | self.llm.with_structured_output(PrimerResponse)
    
    async def process_query(
        self, 
        query: str, 
        community_context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process query using primer approach with community context
        
        Args:
            query: The search query to process
            community_context: Optional pre-built community context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing primer results and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Expand query using HyDE if enabled
            expanded_query = query
            if self.config.use_hyde:
                expanded_query = await self._expand_query_hyde(query)
            
            # Step 2: Get community context if not provided
            if community_context is None:
                community_context = await self._get_community_context(expanded_query)
            
            # Step 3: Process with primer prompt
            primer_result = await self._process_with_primer(query, community_context)
            
            # Step 4: Format and return results
            completion_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "query": query,
                "expanded_query": expanded_query,
                "intermediate_answer": primer_result.intermediate_answer,
                "follow_up_queries": primer_result.follow_up_queries,
                "score": primer_result.score,
                "reasoning": primer_result.reasoning,
                "community_context": community_context,
                "completion_time": completion_time,
                "llm_calls": 2 if self.config.use_hyde else 1,
                "prompt_tokens": 0,  # Would need token counting
                "output_tokens": 0,  # Would need token counting
                "method": "drift_primer"
            }
            
        except Exception as e:
            logger.error(f"Error in primer processing: {e}")
            return {
                "query": query,
                "intermediate_answer": f"Error during primer processing: {str(e)}",
                "follow_up_queries": [],
                "score": 0.0,
                "reasoning": "Error occurred during processing",
                "community_context": community_context or "",
                "completion_time": asyncio.get_event_loop().time() - start_time,
                "llm_calls": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "method": "drift_primer_error",
                "error": str(e)
            }
    
    async def _expand_query_hyde(self, query: str) -> str:
        """Expand query using HyDE (Hypothetical Document Embeddings)"""
        try:
            response = await self.llm.ainvoke(
                self.hyde_prompt.format_messages(query=query)
            )
            expanded_query = response.content.strip()
            logger.debug(f"HyDE expanded query: {expanded_query[:100]}...")
            return expanded_query
        except Exception as e:
            logger.error(f"Error in HyDE expansion: {e}")
            return query
    
    async def _get_community_context(self, query: str) -> str:
        """Get relevant community context for the query"""
        try:
            # Get top-ranked communities with summaries
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
                    LIMIT $max_communities
                """, max_communities=self.config.max_communities)
                
                communities = []
                for record in result:
                    communities.append({
                        "id": record["community_id"],
                        "level": record["level"],
                        "summary": record["summary"],
                        "rank": record["rank"] or 0,
                        "entity_count": record["entity_count"]
                    })
                
                # Format community context
                context_parts = []
                for i, community in enumerate(communities, 1):
                    context_part = f"Community {i} (ID: {community['id']}, Level: {community['level']}, Rank: {community['rank']}):\n"
                    context_part += f"{community['summary']}\n"
                    context_parts.append(context_part)
                
                community_context = "\n---\n".join(context_parts)
                logger.info(f"Built community context with {len(communities)} communities")
                return community_context
                
        except Exception as e:
            logger.error(f"Error getting community context: {e}")
            return ""
    
    async def _process_with_primer(self, query: str, community_context: str) -> PrimerResponse:
        """Process query with primer prompt using community context"""
        try:
            # Invoke structured output chain
            result = await self.primer_chain.ainvoke({
                "query": query,
                "community_context": community_context,
                "max_follow_ups": self.config.max_follow_ups
            })
            
            # Validate and clean up results
            if not result.follow_up_queries:
                result.follow_up_queries = []
            
            # Limit follow-ups to configured maximum
            result.follow_up_queries = result.follow_up_queries[:self.config.max_follow_ups]
            
            # Ensure score is within valid range
            result.score = max(0, min(100, result.score))
            
            logger.info(f"Primer processing completed with score {result.score}, {len(result.follow_up_queries)} follow-ups")
            return result
            
        except Exception as e:
            logger.error(f"Error in primer processing: {e}")
            return PrimerResponse(
                intermediate_answer=f"Error during primer processing: {str(e)}",
                follow_up_queries=[],
                score=0.0,
                reasoning="Error occurred during processing"
            )
    
    async def batch_process_queries(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        tasks = [self.process_query(query, **kwargs) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing query {i}: {result}")
                processed_results.append({
                    "query": queries[i],
                    "intermediate_answer": f"Error: {str(result)}",
                    "follow_up_queries": [],
                    "score": 0.0,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_primer_stats(self) -> Dict[str, Any]:
        """Get statistics about primer processing capabilities"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:__Community__)
                    WITH count(c) as total_communities
                    MATCH (c:__Community__)
                    WHERE c.summary IS NOT NULL
                    WITH total_communities, count(c) as communities_with_summaries
                    MATCH (e:__Entity__)
                    RETURN 
                        total_communities,
                        communities_with_summaries,
                        count(e) as total_entities
                """)
                
                record = result.single()
                if record:
                    return {
                        "total_communities": record["total_communities"],
                        "communities_with_summaries": record["communities_with_summaries"],
                        "total_entities": record["total_entities"],
                        "max_communities_used": self.config.max_communities,
                        "max_follow_ups": self.config.max_follow_ups,
                        "use_hyde": self.config.use_hyde
                    }
                
        except Exception as e:
            logger.error(f"Error getting primer stats: {e}")
        
        return {
            "total_communities": 0,
            "communities_with_summaries": 0,
            "total_entities": 0,
            "max_communities_used": self.config.max_communities,
            "max_follow_ups": self.config.max_follow_ups,
            "use_hyde": self.config.use_hyde
        } 