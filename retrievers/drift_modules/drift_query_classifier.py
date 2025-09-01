"""
DRIFT Query Classifier for GraphRAG Routing

This module provides a query classifier specifically designed for the DRIFT system
to determine whether a query should use local (entity-focused) or global (community-focused) search.
"""

from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)

class DRIFTQueryClassifier:
    """
    DRIFT query classifier that determines whether to use local or global search.
    
    Local search is better for:
    - Specific entity queries (names, dates, specific facts)
    - Questions about particular relationships
    - Detailed information about specific topics
    
    Global search is better for:
    - Broad thematic questions
    - Summarization requests
    - Questions requiring synthesis across multiple topics
    - High-level overviews
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query routing classifier for a DRIFT GraphRAG system. Your task is to classify queries as either LOCAL or GLOBAL search.

LOCAL search is best for:
- Specific entity queries (names, dates, numbers, specific facts)
- Questions about particular relationships between entities
- Detailed information about specific topics or entities
- Questions that require precise, factual answers
- Examples: "What is NovaGrid's target capacity?", "Which RFP mentions AI/ML?", "Give the deadline for proposals"

GLOBAL search is best for:
- Broad thematic questions requiring synthesis
- Summarization requests across multiple topics
- High-level overviews and comparisons
- Questions requiring understanding of overall patterns or themes
- Examples: "Summarize the main themes", "Compare approaches across documents", "What are the common pain points?"

Respond with exactly one word: either "LOCAL" or "GLOBAL"."""),
            ("human", "Query: {query}")
        ])
        
        self.chain = self.classification_prompt | self.llm
    
    async def classify(self, query: str) -> Literal["LOCAL", "GLOBAL"]:
        """
        Classify a query as LOCAL or GLOBAL search.
        
        Args:
            query: The query to classify
            
        Returns:
            "LOCAL" or "GLOBAL"
        """
        try:
            response = await self.chain.ainvoke({"query": query})
            
            # Extract the classification from the response
            classification = response.content.strip().upper()
            
            # Ensure we return a valid classification
            if classification in ["LOCAL", "GLOBAL"]:
                logger.debug(f"DRIFT classified query '{query[:50]}...' as {classification}")
                return classification
            else:
                # Default to LOCAL if classification is unclear
                logger.warning(f"Unclear DRIFT classification '{classification}' for query, defaulting to LOCAL")
                return "LOCAL"
                
        except Exception as e:
            logger.error(f"Error in DRIFT query classification: {e}")
            # Default to LOCAL search on error
            return "LOCAL"
    
    def classify_sync(self, query: str) -> Literal["LOCAL", "GLOBAL"]:
        """
        Synchronous version of classify method.
        
        Args:
            query: The query to classify
            
        Returns:
            "LOCAL" or "GLOBAL"
        """
        try:
            response = self.chain.invoke({"query": query})
            
            # Extract the classification from the response
            classification = response.content.strip().upper()
            
            # Ensure we return a valid classification
            if classification in ["LOCAL", "GLOBAL"]:
                logger.debug(f"DRIFT classified query '{query[:50]}...' as {classification}")
                return classification
            else:
                # Default to LOCAL if classification is unclear
                logger.warning(f"Unclear DRIFT classification '{classification}' for query, defaulting to LOCAL")
                return "LOCAL"
                
        except Exception as e:
            logger.error(f"Error in DRIFT query classification: {e}")
            # Default to LOCAL search on error
            return "LOCAL"


def create_drift_query_classifier(llm: ChatOpenAI) -> DRIFTQueryClassifier:
    """Factory function to create a DRIFTQueryClassifier instance."""
    return DRIFTQueryClassifier(llm)


# Alias for backward compatibility
QueryClassifier = DRIFTQueryClassifier
