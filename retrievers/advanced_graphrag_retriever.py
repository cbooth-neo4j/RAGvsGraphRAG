"""
Advanced GraphRAG Retriever Implementation

This module implements a production-ready GraphRAG retrieval system with sophisticated
data models, context builders, and search patterns optimized for performance and accuracy.
"""

import os
import asyncio
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import tiktoken

# Core dependencies
import neo4j
from dotenv import load_dotenv

# Import centralized configuration
from config import get_model_config, get_embeddings, get_llm, ModelProvider

# Import our graph processor
from data_processors import CustomGraphProcessor as AdvancedGraphProcessor

# Load environment variables
load_dotenv()

# Configuration
# Configuration is now handled by centralized config system

logger = logging.getLogger(__name__)

# Advanced Data Models
@dataclass
class Entity:
    """Entity data model with comprehensive attributes and metadata."""
    id: str
    title: str
    type: str
    description: str
    human_readable_id: int
    graph_embedding: Optional[List[float]] = None
    text_unit_ids: Optional[List[str]] = None
    description_embedding: Optional[List[float]] = None
    community_ids: Optional[List[str]] = None
    rank: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class Relationship:
    """Relationship data model with weight and degree information."""
    id: str
    source: str
    target: str
    description: str
    weight: float
    human_readable_id: int
    source_degree: int
    target_degree: int
    rank: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class Community:
    """Community data model with hierarchical information and ranking."""
    id: str
    title: str
    level: int
    rank: Optional[float] = None
    rank_explanation: Optional[str] = None
    full_content: Optional[str] = None
    summary: Optional[str] = None
    weight: Optional[float] = None
    attributes: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class CommunityReport:
    """Community report data model with structured content and analysis."""
    community_id: str
    full_content: str
    level: int
    rank: Optional[float] = None
    title: Optional[str] = None
    rank_explanation: Optional[str] = None
    summary: Optional[str] = None
    findings: Optional[List[str]] = None
    full_content_json: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class TextUnit:
    """Text unit data model with embeddings and entity associations."""
    id: str
    text: str
    text_embedding: Optional[List[float]] = None
    entity_ids: Optional[List[str]] = None
    relationship_ids: Optional[List[str]] = None
    covariate_ids: Optional[Dict[str, List[str]]] = field(default_factory=dict)
    n_tokens: Optional[int] = None
    document_ids: Optional[List[str]] = None
    attributes: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class SearchResult:
    """Search result data model with comprehensive metrics and context."""
    response: str
    context_data: str | List[pd.DataFrame] | Dict[str, pd.DataFrame]
    context_text: str | List[str] | Dict[str, str]
    completion_time: float
    llm_calls: int
    prompt_tokens: int
    output_tokens: int
    llm_calls_categories: Optional[Dict[str, int]] = field(default_factory=dict)
    prompt_tokens_categories: Optional[Dict[str, int]] = field(default_factory=dict)
    output_tokens_categories: Optional[Dict[str, int]] = field(default_factory=dict)

@dataclass
class GlobalSearchResult(SearchResult):
    """Global search result with map responses."""
    map_responses: List[SearchResult] = field(default_factory=list)
    reduce_context_data: str | List[pd.DataFrame] | Dict[str, pd.DataFrame] = ""
    reduce_context_text: str | List[str] | Dict[str, str] = ""

@dataclass
class ContextBuilderResult:
    """Context builder result with structured data and token metrics."""
    context_chunks: str | List[str]
    context_records: Dict[str, pd.DataFrame]
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0


# Vector Store Abstraction for Neo4j
class Neo4jVectorStore:
    """Vector store abstraction for Neo4j with optimized similarity search capabilities."""
    
    def __init__(self, driver: neo4j.GraphDatabase.driver, embedding_model):
        self.driver = driver
        self.embedding_model = embedding_model
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self._setup_vector_index()
    
    def _setup_vector_index(self):
        """Create vector index for entities if it doesn't exist."""        
        try:
            with self.driver.session() as session:
                # First check if index exists
                result = session.run("SHOW INDEXES").data()
                index_names = [idx.get('name', '') for idx in result]
                
                if 'entity_embedding' not in index_names:
                    # Check if we have entities with embeddings
                    count_result = session.run("""
                        MATCH (e:__Entity__) 
                        WHERE e.embedding IS NOT NULL 
                        RETURN count(e) as count
                    """).single()
                    
                    if count_result and count_result['count'] > 0:
                        session.run("""
                            CREATE VECTOR INDEX entity_embedding IF NOT EXISTS 
                            FOR (e:__Entity__) ON e.embedding
                            OPTIONS {indexConfig: {
                                `vector.dimensions`: 1536,
                                `vector.similarity_function`: 'cosine'
                            }}
                        """)
                        logger.info("Created entity_embedding vector index")
                    else:
                        logger.warning("No entities with embeddings found, skipping vector index creation")
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")
            # Continue without vector index - will fall back to alternative methods
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 10,
        oversample_scaler: int = 2
    ) -> List[Tuple[Entity, float]]:
        """Search for similar entities using vector index or fallback to text matching."""
        oversample_k = k * oversample_scaler
        
        with self.driver.session() as session:
            # First try vector search if index exists
            try:
                query_embedding = self.embedding_model.embed_query(query)
                result = session.run("""
                    CALL db.index.vector.queryNodes('entity_embedding', $k, $query_embedding)
                    YIELD node, score
                    RETURN node.id as id,
                           node.name as title,
                           node.entity_type as type,
                           node.description as description,
                           node.human_readable_id as human_readable_id,
                           node.embedding as description_embedding,
                           [] as text_unit_ids,
                           CASE WHEN node.communityId IS NOT NULL THEN node.communityId ELSE [] END as community_ids,
                           coalesce(node.human_readable_id, 0) as rank,
                           score
                    ORDER BY score DESC
                    LIMIT $k
                """, query_embedding=query_embedding, k=oversample_k)
                
                entities_with_scores = []
                for record in result:
                    entity = Entity(
                        id=record["id"],
                        title=record["title"],
                        type=record["type"], 
                        description=record["description"],
                        human_readable_id=record["human_readable_id"] or 0,
                        description_embedding=record["description_embedding"],
                        text_unit_ids=record["text_unit_ids"] or [],
                        community_ids=record["community_ids"] or [],
                        rank=record["rank"]
                    )
                    entities_with_scores.append((entity, record["score"]))
                
                return entities_with_scores[:k]
                
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to text matching: {e}")
                # Fallback to text-based matching
                return self._fallback_text_search(query, k, session)
    
    def _fallback_text_search(self, query: str, k: int, session) -> List[Tuple[Entity, float]]:
        """Fallback text-based entity search when vector index is not available."""
        # Simple text matching based on entity names and descriptions
        query_lower = query.lower()
        query_tokens = query_lower.split()
        
        result = session.run("""
            MATCH (e:__Entity__)
            WITH e, 
                 toLower(coalesce(e.name, '')) as name_lower,
                 toLower(coalesce(e.description, '')) as desc_lower
            WITH e, name_lower, desc_lower,
                 // Score based on text matching
                 CASE 
                     WHEN any(token IN $query_tokens WHERE name_lower CONTAINS token) THEN 0.9
                     WHEN any(token IN $query_tokens WHERE desc_lower CONTAINS token) THEN 0.7  
                     WHEN name_lower CONTAINS $query_lower THEN 0.8
                     WHEN desc_lower CONTAINS $query_lower THEN 0.6
                     ELSE 0.1
                 END as score
            WHERE score > 0.1
            OPTIONAL MATCH (e)-[r:RELATED_TO]-()
            WITH e, score, count(r) as rel_count
            RETURN e.id as id,
                   e.name as title,
                   coalesce(e.entity_type, [l IN labels(e) WHERE l <> '__Entity__'][0]) as type,
                   e.description as description,
                   coalesce(e.human_readable_id, 0) as human_readable_id,
                   e.embedding as description_embedding,
                   [] as text_unit_ids,
                   CASE WHEN e.communityId IS NOT NULL THEN e.communityId ELSE [] END as community_ids,
                   coalesce(e.human_readable_id, rel_count) as rank,
                   score
            ORDER BY score DESC, rank DESC
            LIMIT $k
        """, query_tokens=query_tokens, query_lower=query_lower, k=k)
        
        entities_with_scores = []
        for record in result:
            entity = Entity(
                id=record["id"],
                title=record["title"] or record["id"],
                type=record["type"] or 'UNKNOWN',
                description=record["description"] or '',
                human_readable_id=record["human_readable_id"],
                description_embedding=record["description_embedding"],
                text_unit_ids=record["text_unit_ids"] or [],
                community_ids=record["community_ids"] or [],
                rank=record["rank"]
            )
            entities_with_scores.append((entity, record["score"]))
        
        return entities_with_scores

def num_tokens(text: str, token_encoder: tiktoken.Encoding) -> int:
    """Count tokens in text using the provided encoder."""
    if not text:
        return 0
    return len(token_encoder.encode(text))

def map_query_to_entities(
        query: str, 
    text_embedding_vectorstore: Neo4jVectorStore,
    text_embedder,
    all_entities_dict: Dict[str, Entity],
    k: int = 10,
    oversample_scaler: int = 2,
    include_entity_names: List[str] = None,
    exclude_entity_names: List[str] = None,
    **kwargs
) -> List[Entity]:
    """Map query to entities using vector similarity search."""
    if include_entity_names is None:
        include_entity_names = []
    if exclude_entity_names is None:
        exclude_entity_names = []
    
    # If no entities available, return empty list
    if not all_entities_dict:
        logger.warning("No entities available for mapping")
        return []
    
    # Get candidate entities from vector search
    entities_with_scores = text_embedding_vectorstore.similarity_search_with_score(
        query=query,
        k=k,
        oversample_scaler=oversample_scaler
    )
    
    # If no entities found, try to get some entities from the dictionary as fallback
    if not entities_with_scores:
        logger.warning("No entities found via vector search, using fallback selection")
        # Take first k entities from the dictionary as fallback
        fallback_entities = list(all_entities_dict.values())[:k]
        return fallback_entities
    
    # Filter by include/exclude lists
    filtered_entities = []
    for entity, score in entities_with_scores:
        if include_entity_names and entity.title not in include_entity_names:
            continue
        if exclude_entity_names and entity.title in exclude_entity_names:
            continue
        filtered_entities.append(entity)
    
    return filtered_entities[:k]


# Context Builder Base Classes
class ContextBuilder(ABC):
    """Base context builder for structured data assembly."""
    
    @abstractmethod
    def build_context(self, query: str, **kwargs) -> ContextBuilderResult:
        """Build context for the given query."""
        pass

class LocalContextBuilder(ContextBuilder):
    """Local context builder base class."""
    pass

class GlobalContextBuilder(ContextBuilder):
    """Global context builder base class."""
    pass


def count_relationships(relationships: List[Relationship], text_unit: TextUnit) -> int:
    """Count relationships associated with a text unit."""
    if not text_unit.relationship_ids:
        return 0
    
    relationship_ids_set = set(text_unit.relationship_ids)
    return sum(1 for rel in relationships if rel.id in relationship_ids_set)


def build_entity_context(
    selected_entities: List[Entity],
    token_encoder: tiktoken.Encoding,
    max_context_tokens: int = 8000,
    include_entity_rank: bool = False,
    rank_description: str = "number of relationships",
    column_delimiter: str = "|",
    context_name: str = "Entities"
) -> Tuple[str, pd.DataFrame]:
    """Build entity context with token-aware truncation and ranking."""
    if not selected_entities:
        return "", pd.DataFrame()
    
    # Create entity records
    entity_records = []
    for entity in selected_entities:
        record = {
            "id": entity.id,
            "entity": entity.title,
            "description": entity.description
        }
        if include_entity_rank and entity.rank is not None:
            record[rank_description] = entity.rank
        entity_records.append(record)
    
    # Convert to DataFrame
    entity_df = pd.DataFrame(entity_records)
    
    # Build context text
    header = f"-----{context_name}-----"
    if entity_df.empty:
        return header, entity_df
    
    # Convert DataFrame to string with proper formatting
    # Note: pandas to_string() doesn't support 'sep' parameter, so we'll format manually
    if column_delimiter == "|":
        context_text = header + "\n" + entity_df.to_string(index=False)
    else:
        # For non-default delimiters, format manually
        context_text = header + "\n" + entity_df.to_csv(index=False, sep=column_delimiter).strip()
    
    # Check token limit
    current_tokens = num_tokens(context_text, token_encoder)
    if current_tokens > max_context_tokens:
        # Truncate entities to fit token limit
        for i in range(len(selected_entities) - 1, 0, -1):
            truncated_df = entity_df.head(i)
            if column_delimiter == "|":
                truncated_text = header + "\n" + truncated_df.to_string(index=False)
            else:
                truncated_text = header + "\n" + truncated_df.to_csv(index=False, sep=column_delimiter).strip()
            if num_tokens(truncated_text, token_encoder) <= max_context_tokens:
                return truncated_text, truncated_df
    
    return context_text, entity_df


# Data Loaders for Existing Neo4j Schema
class Neo4jDataLoader:
    """Load data from existing Neo4j graph into structured data models."""
    
    def __init__(self, driver: neo4j.GraphDatabase.driver):
        self.driver = driver
    
    def load_entities(self) -> Dict[str, Entity]:
        """Load entities from Neo4j __Entity__ nodes."""
        entities = {}
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:__Entity__)
                OPTIONAL MATCH (c:Chunk)-[:HAS_ENTITY]->(e)
                WITH e, collect(DISTINCT c.id) as text_unit_ids, count(c) as mention_count
                WITH e, text_unit_ids, mention_count, 
                     CASE WHEN e.communityId IS NOT NULL THEN e.communityId ELSE [] END as community_ids
                OPTIONAL MATCH (e)-[r:RELATED_TO]-()
                WITH e, text_unit_ids, mention_count, community_ids, count(r) as relationship_count
                RETURN e.id as id,
                       e.name as title,
                       coalesce(e.entity_type, [l IN labels(e) WHERE l <> '__Entity__'][0]) as type,
                       e.description as description,
                       coalesce(e.human_readable_id, 0) as human_readable_id,
                       e.embedding as description_embedding,
                       text_unit_ids,
                       community_ids,
                       relationship_count as rank
            """).data()
            
            for record in result:
                if record['id']:
                    entity = Entity(
                        id=record['id'],
                        title=record['title'] or record['id'],
                        type=record['type'] or 'UNKNOWN',
                        description=record['description'] or '',
                        human_readable_id=record['human_readable_id'],
                        description_embedding=record['description_embedding'],
                        text_unit_ids=record['text_unit_ids'] or [],
                        community_ids=record['community_ids'] or [],
                        rank=record['rank']
                    )
                    entities[entity.id] = entity
        
        return entities
    
    def load_relationships(self) -> Dict[str, Relationship]:
        """Load relationships from Neo4j RELATED_TO relationships."""
        relationships = {}
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e1:__Entity__)-[r:RELATED_TO]->(e2:__Entity__)
                RETURN elementId(r) as id,
                       e1.name as source,
                       e2.name as target,
                       coalesce(r.evidence, 'related to') as description,
                       coalesce(r.count, r.confidence, 1.0) as weight,
                       coalesce(r.human_readable_id, 0) as human_readable_id,
                       coalesce(r.count, r.confidence, 1) as rank,
                       COUNT { (e1)-[:RELATED_TO]-() } as source_degree,
                       COUNT { (e2)-[:RELATED_TO]-() } as target_degree
            """).data()
            
            for i, record in enumerate(result):
                rel_id = record['id'] or f"rel_{i}"
                relationship = Relationship(
                    id=rel_id,
                    source=record['source'],
                    target=record['target'],
                    description=record['description'],
                    weight=float(record['weight']),
                    human_readable_id=record['human_readable_id'],
                    source_degree=record['source_degree'],
                    target_degree=record['target_degree'],
                    rank=record['rank']
                )
                relationships[rel_id] = relationship
        
        return relationships
    
    def load_text_units(self) -> Dict[str, TextUnit]:
        """Load text units from Neo4j Chunk nodes."""
        text_units = {}
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk)
                OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
                OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
                WITH c, collect(DISTINCT e.id) as entity_ids, collect(DISTINCT d.id) as document_ids
                RETURN c.id as id,
                       c.text as text,
                       c.embedding as text_embedding,
                       entity_ids,
                       document_ids,
                       coalesce(size(c.text), 0) as n_tokens
            """).data()
            
            for record in result:
                if record['id']:
                    text_unit = TextUnit(
                        id=record['id'],
                        text=record['text'] or '',
                        text_embedding=record['text_embedding'],
                        entity_ids=record['entity_ids'] or [],
                        document_ids=record['document_ids'] or [],
                        n_tokens=record['n_tokens']
                    )
                    text_units[text_unit.id] = text_unit
        
        return text_units
    
    def load_communities(self) -> List[Community]:
        """Load communities from Neo4j __Community__ nodes."""
        communities = []
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:__Community__)
                RETURN c.id as id,
                       coalesce(c.title, c.id) as title,
                       c.level as level,
                       c.community_rank as rank,
                       c.rank_explanation as rank_explanation,
                       c.summary as summary,
                       coalesce(c.weight, 1.0) as weight
                ORDER BY c.level, c.community_rank DESC
            """).data()
            
            for record in result:
                community = Community(
                    id=record['id'],
                    title=record['title'],
                    level=record['level'] or 0,
                    rank=record['rank'],
                    rank_explanation=record['rank_explanation'],
                    summary=record['summary'],
                    weight=record['weight']
                )
                communities.append(community)
        
        return communities
    
    def load_community_reports(self) -> List[CommunityReport]:
        """Load community reports from Neo4j __Community__ nodes."""
        reports = []
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:__Community__)
                WHERE c.summary IS NOT NULL
                RETURN c.id as community_id,
                       c.summary as full_content,
                       c.level as level,
                       c.community_rank as rank,
                       coalesce(c.title, c.id) as title,
                       c.rank_explanation as rank_explanation,
                       c.summary as summary
                ORDER BY c.level, c.community_rank DESC
            """).data()
            
            for record in result:
                report = CommunityReport(
                    community_id=record['community_id'],
                    full_content=record['full_content'],
                    level=record['level'] or 0,
                    rank=record['rank'],
                    title=record['title'],
                    rank_explanation=record['rank_explanation'],
                    summary=record['summary']
                )
                reports.append(report)
        
        return reports


def build_relationship_context(
    selected_entities: List[Entity],
    relationships: List[Relationship],
    token_encoder: tiktoken.Encoding,
    max_context_tokens: int = 8000,
    column_delimiter: str = "|",
    top_k_relationships: int = 10,
    include_relationship_weight: bool = False,
    relationship_ranking_attribute: str = "rank",
    context_name: str = "Relationships"
) -> Tuple[str, pd.DataFrame]:
    """Build relationship context with ranking and token-aware filtering."""
    if not selected_entities or not relationships:
        return "", pd.DataFrame()
    
    # Get entity names for filtering
    entity_names = {entity.title for entity in selected_entities}
    
    # Filter relationships that involve selected entities
    relevant_relationships = []
    for rel in relationships:
        if rel.source in entity_names or rel.target in entity_names:
            relevant_relationships.append(rel)
    
    # Sort by ranking attribute
    relevant_relationships.sort(
        key=lambda x: getattr(x, relationship_ranking_attribute, 0) or 0,
                reverse=True
            )
            
    # Take top K
    top_relationships = relevant_relationships[:top_k_relationships]
    
    # Create relationship records
    rel_records = []
    for rel in top_relationships:
        record = {
            "source": rel.source,
            "target": rel.target,
            "description": rel.description
        }
        if include_relationship_weight:
            record["weight"] = rel.weight
        rel_records.append(record)
    
    # Convert to DataFrame
    rel_df = pd.DataFrame(rel_records)
    
    # Build context text
    header = f"-----{context_name}-----"
    if rel_df.empty:
        return header, rel_df
    
    # Convert DataFrame to string with proper formatting
    if column_delimiter == "|":
        context_text = header + "\n" + rel_df.to_string(index=False)
    else:
        context_text = header + "\n" + rel_df.to_csv(index=False, sep=column_delimiter).strip()
    
    # Check token limit and truncate if necessary
    current_tokens = num_tokens(context_text, token_encoder)
    if current_tokens > max_context_tokens:
        for i in range(len(top_relationships) - 1, 0, -1):
            truncated_df = rel_df.head(i)
            if column_delimiter == "|":
                truncated_text = header + "\n" + truncated_df.to_string(index=False)
            else:
                truncated_text = header + "\n" + truncated_df.to_csv(index=False, sep=column_delimiter).strip()
            if num_tokens(truncated_text, token_encoder) <= max_context_tokens:
                return truncated_text, truncated_df
    
    return context_text, rel_df


def build_text_unit_context(
    text_units: List[TextUnit],
    token_encoder: tiktoken.Encoding,
    max_context_tokens: int = 8000,
    shuffle_data: bool = False,
    context_name: str = "Sources",
    column_delimiter: str = "|"
) -> Tuple[str, pd.DataFrame]:
    """Build text unit context with incremental token budgeting."""
    if not text_units:
        return "", pd.DataFrame()
    
    # Create text unit records
    text_records = []
    for unit in text_units:
        record = {
            "id": unit.id,
            "text": unit.text[:500] + "..." if len(unit.text) > 500 else unit.text  # Truncate long texts
        }
        text_records.append(record)
    
    # Convert to DataFrame
    text_df = pd.DataFrame(text_records)
    
    # Shuffle if requested
    if shuffle_data:
        text_df = text_df.sample(frac=1).reset_index(drop=True)
    
    # Build context text incrementally to respect token limits
    header = f"-----{context_name}-----"
    context_parts = [header]
    current_tokens = num_tokens(header, token_encoder)
    final_records = []
    
    for _, record in text_df.iterrows():
        record_text = f"Text: {record['text']}"
        record_tokens = num_tokens(record_text, token_encoder)
        
        if current_tokens + record_tokens > max_context_tokens:
            break
        
        context_parts.append(record_text)
        current_tokens += record_tokens
        final_records.append(record.to_dict())
    
    context_text = "\n".join(context_parts)
    final_df = pd.DataFrame(final_records)
    
    return context_text, final_df


def build_community_context(
    community_reports: List[CommunityReport],
    token_encoder: tiktoken.Encoding,
    use_community_summary: bool = True,
    column_delimiter: str = "|",
    shuffle_data: bool = False,
    include_community_rank: bool = False,
    min_community_rank: int = 0,
    max_context_tokens: int = 8000,
    single_batch: bool = True,
    context_name: str = "Reports",
    entities: Optional[List[Entity]] = None,
    **kwargs
) -> Union[Tuple[str, Dict[str, pd.DataFrame]], Tuple[List[str], Dict[str, pd.DataFrame]]]:
    """Build community context with batch processing and ranking support."""
    if not community_reports:
        empty_result = ("", {context_name.lower(): pd.DataFrame()})
        return empty_result if single_batch else ([""], {context_name.lower(): pd.DataFrame()})
    
    # Filter by minimum rank
    filtered_reports = [
        report for report in community_reports
        if (report.rank or 0) >= min_community_rank
    ]
    
    if not filtered_reports:
        empty_result = ("", {context_name.lower(): pd.DataFrame()})
        return empty_result if single_batch else ([""], {context_name.lower(): pd.DataFrame()})
    
    # Sort by rank
    filtered_reports.sort(key=lambda x: x.rank or 0, reverse=True)
    
    # Shuffle if requested
    if shuffle_data:
        import random
        random.shuffle(filtered_reports)
    
    # Create community records
    community_records = []
    for report in filtered_reports:
        record = {
            "id": report.community_id,
            "title": report.title or report.community_id,
            "content": report.summary if use_community_summary else report.full_content
        }
        if include_community_rank and report.rank is not None:
            record["rank"] = report.rank
        community_records.append(record)
    
    # Convert to DataFrame
    community_df = pd.DataFrame(community_records)
    
    if single_batch:
        # Build single context respecting token limits
        header = f"-----{context_name}-----"
        context_parts = [header]
        current_tokens = num_tokens(header, token_encoder)
        final_records = []
        
        for _, record in community_df.iterrows():
            record_text = f"Community: {record['title']}\nContent: {record['content']}"
            record_tokens = num_tokens(record_text, token_encoder)
            
            if current_tokens + record_tokens > max_context_tokens:
                break
            
            context_parts.append(record_text)
            current_tokens += record_tokens
            final_records.append(record.to_dict())
        
        context_text = "\n\n".join(context_parts)
        final_df = pd.DataFrame(final_records)
        
        return context_text, {context_name.lower(): final_df}
    
    else:
        # Build multiple batches for global search
        batches = []
        batch_data = []
        current_tokens = 0
        batch_records = []
        
        for _, record in community_df.iterrows():
            record_text = f"Community: {record['title']}\nContent: {record['content']}"
            record_tokens = num_tokens(record_text, token_encoder)
            
            if current_tokens + record_tokens > max_context_tokens and batch_records:
                # Finalize current batch
                batch_text = "\n\n".join([f"-----{context_name}-----"] + batch_data)
                batches.append(batch_text)
                batch_data = []
                current_tokens = 0
                batch_records = []
            
            batch_data.append(record_text)
            current_tokens += record_tokens
            batch_records.append(record.to_dict())
        
        # Add final batch if there's remaining data
        if batch_data:
            batch_text = "\n\n".join([f"-----{context_name}-----"] + batch_data)
            batches.append(batch_text)
        
        # Return all batches
        final_df = pd.DataFrame(community_records)  # Include all records in metadata
        return batches, {context_name.lower(): final_df} 


# Advanced Context Builders adapted for existing Neo4j schema
class LocalSearchMixedContext(LocalContextBuilder):
    """Local search mixed context builder with proportional token allocation."""
    
    def __init__(
        self,
        entities: Dict[str, Entity],
        entity_text_embeddings: Neo4jVectorStore,
        text_embedder,
        text_units: Optional[Dict[str, TextUnit]] = None,
        community_reports: Optional[List[CommunityReport]] = None,
        relationships: Optional[Dict[str, Relationship]] = None,
        token_encoder: Optional[tiktoken.Encoding] = None,
    ):
        self.entities = entities or {}
        self.entity_text_embeddings = entity_text_embeddings
        self.text_embedder = text_embedder
        self.text_units = text_units or {}
        self.community_reports = {
            report.community_id: report for report in (community_reports or [])
        }
        self.relationships = relationships or {}
        self.token_encoder = token_encoder or tiktoken.get_encoding("cl100k_base")
    
    def build_context(
        self, 
        query: str, 
        include_entity_names: Optional[List[str]] = None,
        exclude_entity_names: Optional[List[str]] = None,
        max_context_tokens: int = 8000,
        text_unit_prop: float = 0.5,
        community_prop: float = 0.25,
        top_k_mapped_entities: int = 10,
        top_k_relationships: int = 10,
        include_community_rank: bool = False,
        include_entity_rank: bool = False,
        rank_description: str = "number of relationships",
        include_relationship_weight: bool = False,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        use_community_summary: bool = False,
        min_community_rank: int = 0,
        community_context_name: str = "Reports",
        column_delimiter: str = "|",
        **kwargs: Dict[str, Any],
    ) -> ContextBuilderResult:
        """Build data context for local search prompt."""
        
        if include_entity_names is None:
            include_entity_names = []
        if exclude_entity_names is None:
            exclude_entity_names = []
        
        if community_prop + text_unit_prop > 1:
            raise ValueError("The sum of community_prop and text_unit_prop should not exceed 1.")
        
        # Step 1: Map user query to entities
        selected_entities = map_query_to_entities(
                query=query,
            text_embedding_vectorstore=self.entity_text_embeddings,
            text_embedder=self.text_embedder,
            all_entities_dict=self.entities,
            include_entity_names=include_entity_names,
            exclude_entity_names=exclude_entity_names,
            k=top_k_mapped_entities,
            oversample_scaler=2,
        )
        
        # If no entities found, provide a minimal context with available text units
        if not selected_entities:
            logger.warning("No entities found for query, providing minimal context")
            # Build context from available text units only
            available_text_units = list(self.text_units.values())[:10] if self.text_units else []
            text_context, text_data = build_text_unit_context(
                text_units=available_text_units,
                token_encoder=self.token_encoder,
                max_context_tokens=max_context_tokens,
                context_name="Available Sources"
            )
            
            return ContextBuilderResult(
                context_chunks=text_context if text_context else "No relevant context found.",
                context_records=text_data if isinstance(text_data, dict) and text_data else {},
            )
        
        # Step 2: Build context components
        final_context = []
        final_context_data = {}
        
        # Build community context
        community_tokens = max(int(max_context_tokens * community_prop), 0)
        community_context, community_context_data = self._build_community_context(
            selected_entities=selected_entities,
            max_context_tokens=community_tokens,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            return_candidate_context=return_candidate_context,
            context_name=community_context_name,
        )
        if community_context.strip():
            final_context.append(community_context)
            final_context_data.update(community_context_data)
        
        # Build local (entity-relationship) context
        local_prop = 1 - community_prop - text_unit_prop
        local_tokens = max(int(max_context_tokens * local_prop), 0)
        local_context, local_context_data = self._build_local_context(
            selected_entities=selected_entities,
            max_context_tokens=local_tokens,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            include_relationship_weight=include_relationship_weight,
            top_k_relationships=top_k_relationships,
            relationship_ranking_attribute=relationship_ranking_attribute,
            return_candidate_context=return_candidate_context,
            column_delimiter=column_delimiter,
        )
        if local_context.strip():
            final_context.append(local_context)
            final_context_data.update(local_context_data)
        
        # Build text unit context
        text_unit_tokens = max(int(max_context_tokens * text_unit_prop), 0)
        text_unit_context, text_unit_context_data = self._build_text_unit_context(
            selected_entities=selected_entities,
            max_context_tokens=text_unit_tokens,
            return_candidate_context=return_candidate_context,
        )
        if text_unit_context.strip():
            final_context.append(text_unit_context)
            final_context_data.update(text_unit_context_data)
        
        return ContextBuilderResult(
            context_chunks="\n\n".join(final_context),
            context_records=final_context_data,
        )
    
    def _build_community_context(
        self,
        selected_entities: List[Entity],
        max_context_tokens: int = 4000,
        use_community_summary: bool = False,
        column_delimiter: str = "|",
        include_community_rank: bool = False,
        min_community_rank: int = 0,
        return_candidate_context: bool = False,
        context_name: str = "Reports",
    ) -> Tuple[str, Dict[str, pd.DataFrame]]:
        """Add community data to the context window until it hits the max_context_tokens limit."""
        if not selected_entities or not self.community_reports:
            return "", {context_name.lower(): pd.DataFrame()}
        
        # Find communities that contain selected entities
        community_matches = {}
        for entity in selected_entities:
            if entity.community_ids:
                for community_id in entity.community_ids:
                    community_matches[community_id] = community_matches.get(community_id, 0) + 1
        
        # Get matching community reports
        selected_community_reports = []
        for community_id, match_count in community_matches.items():
            if community_id in self.community_reports:
                report = self.community_reports[community_id]
                # Add match count as temporary attribute for sorting
                report_copy = CommunityReport(
                    community_id=report.community_id,
                    full_content=report.full_content,
                    level=report.level,
                    rank=report.rank,
                    title=report.title,
                    rank_explanation=report.rank_explanation,
                    summary=report.summary,
                    attributes={"matches": match_count}
                )
                selected_community_reports.append(report_copy)
        
        # Sort by matches and rank
        selected_community_reports.sort(
            key=lambda x: (x.attributes.get("matches", 0), x.rank or 0),
            reverse=True
        )
        
        # Remove temporary attributes
        for report in selected_community_reports:
            if report.attributes:
                report.attributes.pop("matches", None)
        
        # Build context using the generic function
        context_text, context_data = build_community_context(
            community_reports=selected_community_reports,
            token_encoder=self.token_encoder,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            max_context_tokens=max_context_tokens,
            single_batch=True,
            context_name=context_name,
        )
        
        return str(context_text), context_data
    
    def _build_text_unit_context(
        self,
        selected_entities: List[Entity],
        max_context_tokens: int = 8000,
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
        context_name: str = "Sources",
    ) -> Tuple[str, Dict[str, pd.DataFrame]]:
        """Rank matching text units and add them to the context window."""
        if not selected_entities or not self.text_units:
            return "", {context_name.lower(): pd.DataFrame()}
        
        # Collect text units associated with selected entities
        selected_text_units = []
        text_unit_ids_set = set()
        
        unit_info_list = []
        relationships_list = list(self.relationships.values())
        
        for index, entity in enumerate(selected_entities):
            # Get text units for this entity
            for text_unit_id in entity.text_unit_ids or []:
                if text_unit_id not in text_unit_ids_set and text_unit_id in self.text_units:
                    text_unit = self.text_units[text_unit_id]
                    
                    # Count relationships involving this entity that are mentioned in this text unit
                    entity_relationships = [
                        rel for rel in relationships_list
                        if rel.source == entity.title or rel.target == entity.title
                    ]
                    
                    # Simple relationship count for now (could be more sophisticated)
                    num_relationships = len(entity_relationships)
                    
                    text_unit_ids_set.add(text_unit_id)
                    unit_info_list.append((text_unit, index, num_relationships))
        
        # Sort by entity order and relationship count
        unit_info_list.sort(key=lambda x: (x[1], -x[2]))
        selected_text_units = [unit[0] for unit in unit_info_list]
        
        # Build context
        context_text, context_data = build_text_unit_context(
            text_units=selected_text_units,
            token_encoder=self.token_encoder,
            max_context_tokens=max_context_tokens,
            context_name=context_name,
            column_delimiter=column_delimiter,
        )
        
        return context_text, context_data
    
    def _build_local_context(
        self,
        selected_entities: List[Entity],
        max_context_tokens: int = 8000,
        include_entity_rank: bool = False,
        rank_description: str = "relationship count",
        include_relationship_weight: bool = False,
        top_k_relationships: int = 10,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
    ) -> Tuple[str, Dict[str, pd.DataFrame]]:
        """Build data context for local search prompt combining entity/relationship tables."""
        
        # Build entity context
        entity_context, entity_context_data = build_entity_context(
            selected_entities=selected_entities,
            token_encoder=self.token_encoder,
            max_context_tokens=max_context_tokens,
            column_delimiter=column_delimiter,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            context_name="Entities",
        )
        entity_tokens = num_tokens(entity_context, self.token_encoder)
        
        # Build relationship context with remaining token budget
        remaining_tokens = max_context_tokens - entity_tokens
        relationships_list = list(self.relationships.values())
        
        relationship_context, relationship_context_data = build_relationship_context(
            selected_entities=selected_entities,
            relationships=relationships_list,
            token_encoder=self.token_encoder,
            max_context_tokens=remaining_tokens,
            column_delimiter=column_delimiter,
            top_k_relationships=top_k_relationships,
            include_relationship_weight=include_relationship_weight,
            relationship_ranking_attribute=relationship_ranking_attribute,
            context_name="Relationships",
        )
        
        # Combine contexts
        if entity_context and relationship_context:
            final_context_text = entity_context + "\n\n" + relationship_context
        elif entity_context:
            final_context_text = entity_context
        elif relationship_context:
            final_context_text = relationship_context
        else:
            final_context_text = ""
        
        final_context_data = {
            "entities": entity_context_data,
            "relationships": relationship_context_data
        }
        
        return final_context_text, final_context_data


class GlobalCommunityContext(GlobalContextBuilder):
    """Global search community context builder with batch processing."""
    
    def __init__(
        self,
        community_reports: List[CommunityReport],
        communities: List[Community],
        entities: Optional[List[Entity]] = None,
        token_encoder: Optional[tiktoken.Encoding] = None,
        random_state: int = 86,
    ):
        self.community_reports = community_reports
        self.communities = communities
        self.entities = entities
        self.token_encoder = token_encoder or tiktoken.get_encoding("cl100k_base")
        self.random_state = random_state
    
    async def build_context(
        self, 
        query: str, 
        use_community_summary: bool = True,
        column_delimiter: str = "|",
        shuffle_data: bool = True,
        include_community_rank: bool = False,
        min_community_rank: int = 0,
        community_rank_name: str = "rank",
        include_community_weight: bool = True,
        community_weight_name: str = "occurrence",
        normalize_community_weight: bool = True,
        max_context_tokens: int = 8000,
        context_name: str = "Reports",
        **kwargs: Any,
    ) -> ContextBuilderResult:
        """Prepare batches of community report data table as context data for global search."""
        
        # Build community context in batches for global search
        community_context, community_context_data = build_community_context(
            community_reports=self.community_reports,
            entities=self.entities,
            token_encoder=self.token_encoder,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            shuffle_data=shuffle_data,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            max_context_tokens=max_context_tokens,
            single_batch=False,  # Multiple batches for global search
            context_name=context_name,
        )
        
        return ContextBuilderResult(
            context_chunks=community_context,
            context_records=community_context_data,
        )


# Advanced Search Classes
class LocalSearch:
    """Local search implementation with comprehensive context building."""
    
    def __init__(
        self,
        model,
        context_builder: LocalContextBuilder,
        token_encoder: Optional[tiktoken.Encoding] = None,
        system_prompt: Optional[str] = None,
        response_type: str = "multiple paragraphs",
        model_params: Optional[Dict[str, Any]] = None,
        context_builder_params: Optional[Dict] = None,
    ):
        self.model = model
        self.context_builder = context_builder
        self.token_encoder = token_encoder or tiktoken.get_encoding("cl100k_base")
        self.system_prompt = system_prompt or LOCAL_SEARCH_SYSTEM_PROMPT
        self.response_type = response_type
        self.model_params = model_params or {}
        self.context_builder_params = context_builder_params or {}
    
    async def search(
        self,
        query: str,
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query."""
        start_time = time.time()
        search_prompt = ""
        llm_calls, prompt_tokens, output_tokens = {}, {}, {}
        
        try:
            # Build context
            context_result = self.context_builder.build_context(
                query=query,
                **kwargs,
                **self.context_builder_params,
            )
            llm_calls["build_context"] = context_result.llm_calls
            prompt_tokens["build_context"] = context_result.prompt_tokens
            output_tokens["build_context"] = context_result.output_tokens
            
            # Format search prompt
            search_prompt = self.system_prompt.format(
                context_data=context_result.context_chunks,
                response_type=self.response_type,
            )
            
            # Generate response  
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content=search_prompt),
                HumanMessage(content=query)
            ]
            
            response = await self.model.ainvoke(
                messages,
                **self.model_params,
            )
            
            full_response = response.content
            
            llm_calls["response"] = 1
            prompt_tokens["response"] = num_tokens(search_prompt, self.token_encoder)
            output_tokens["response"] = num_tokens(full_response, self.token_encoder)
            
            return SearchResult(
                response=full_response,
                context_data=context_result.context_records,
                context_text=context_result.context_chunks,
                completion_time=time.time() - start_time,
                llm_calls=sum(llm_calls.values()),
                prompt_tokens=sum(prompt_tokens.values()),
                output_tokens=sum(output_tokens.values()),
                llm_calls_categories=llm_calls,
                prompt_tokens_categories=prompt_tokens,
                output_tokens_categories=output_tokens,
            )
            
        except Exception as e:
            logger.exception("Exception in local search")
            return SearchResult(
                response=f"Error occurred during search: {str(e)}",
                context_data={},
                context_text="",
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                output_tokens=0,
            )


# System Prompts for advanced GraphRAG processing
LOCAL_SEARCH_SYSTEM_PROMPT = """
You are a helpful assistant answering questions based on the provided context data.

Use the context data below to answer the user's question. The context includes:
- Entity information
- Relationships between entities  
- Community information
- Relevant text sources

Guidelines:
1. Answer based on the provided context
2. Be specific and cite relevant entities or sources when possible
3. If the context doesn't contain sufficient information, state this clearly
4. Provide a comprehensive but concise answer
5. Use a {response_type} format for your response

Context Data:
{context_data}
"""

MAP_SYSTEM_PROMPT = """
You are a helpful assistant responding to questions about data in community reports.

Generate a response consisting of a list of key points that respond to the user's question, based on the given community report.

You should use the data provided in the community report below as the primary context for generating the response.
If you don't know the answer or if the provided community report doesn't contain sufficient information, respond with an empty list.

Each key point should be:
- A single paragraph
- Highly relevant to the user's question
- Supported by the community report data
- Include a relevance score from 0-100 (100 being most relevant)

IMPORTANT: You MUST return your response as a valid JSON object with this exact structure:
{{
    "points": [
        {{
            "description": "First key point based on the community data",
            "score": 85
        }},
        {{
            "description": "Second key point based on the community data", 
            "score": 92
        }}
    ]
}}

Do not include any text before or after the JSON. Only return valid JSON.
If you have no relevant information, return: {{"points": []}}

Community Report:
{context_data}
"""

REDUCE_SYSTEM_PROMPT = """
You are a helpful assistant that synthesizes information from multiple analysts to answer questions comprehensively.

You have received responses from multiple analysts, each focused on different aspects of the data. Your task is to synthesize these responses into a comprehensive, coherent answer that addresses the user's question.

Guidelines:
1. Combine information from all analyst responses
2. Organize the information logically
3. Remove redundancy while preserving important details
4. Ensure the final answer is well-structured and comprehensive
5. Prioritize information from higher-scored analyst responses
6. Write in a clear, professional tone
7. Use a {response_type} format for your response

Analyst Responses:
{report_data}
"""


class GlobalSearch:
    """Global search implementation with map-reduce processing."""
    
    def __init__(
        self,
        model,
        context_builder: GlobalContextBuilder,
        token_encoder: Optional[tiktoken.Encoding] = None,
        map_system_prompt: Optional[str] = None,
        reduce_system_prompt: Optional[str] = None,
        response_type: str = "multiple paragraphs",
        json_mode: bool = True,
        max_data_tokens: int = 8000,
        map_llm_params: Optional[Dict[str, Any]] = None,
        reduce_llm_params: Optional[Dict[str, Any]] = None,
        map_max_length: int = 1000,
        reduce_max_length: int = 2000,
        context_builder_params: Optional[Dict[str, Any]] = None,
        concurrent_coroutines: int = 32,
    ):
        self.model = model
        self.context_builder = context_builder
        self.token_encoder = token_encoder or tiktoken.get_encoding("cl100k_base")
        self.map_system_prompt = map_system_prompt or MAP_SYSTEM_PROMPT
        self.reduce_system_prompt = reduce_system_prompt or REDUCE_SYSTEM_PROMPT
        self.response_type = response_type
        self.max_data_tokens = max_data_tokens
        
        self.map_llm_params = map_llm_params or {}
        self.reduce_llm_params = reduce_llm_params or {}
        
        # Note: Ollama doesn't support response_format, so we skip JSON mode for Ollama
        if json_mode:
            # Check if we're using Ollama and skip response_format if so
            from config import get_model_config
            config = get_model_config()
            if config.llm_provider.value != "ollama":
                self.map_llm_params["response_format"] = {"type": "json_object"}
        else:
            self.map_llm_params.pop("response_format", None)
            
        self.map_max_length = map_max_length
        self.reduce_max_length = reduce_max_length
        self.context_builder_params = context_builder_params or {}
        
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)
    
    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> GlobalSearchResult:
        """Perform a global search using map-reduce pattern."""
        
        # Step 1: Generate answers for each batch of community short summaries
        llm_calls, prompt_tokens, output_tokens = {}, {}, {}
        start_time = time.time()
        
        # Build context batches
        context_result = await self.context_builder.build_context(
            query=query,
            **self.context_builder_params,
        )
        llm_calls["build_context"] = context_result.llm_calls
        prompt_tokens["build_context"] = context_result.prompt_tokens
        output_tokens["build_context"] = context_result.output_tokens
        
        # Map phase: process each context batch
        map_responses = await asyncio.gather(*[
            self._map_response_single_batch(
                context_data=data,
                query=query,
                max_length=self.map_max_length,
                **self.map_llm_params,
            )
            for data in context_result.context_chunks
        ])
        
        llm_calls["map"] = sum(response.llm_calls for response in map_responses)
        prompt_tokens["map"] = sum(response.prompt_tokens for response in map_responses)
        output_tokens["map"] = sum(response.output_tokens for response in map_responses)
        
        # Step 2: Reduce phase - combine intermediate answers
        reduce_response = await self._reduce_response(
            map_responses=map_responses,
            query=query,
            **self.reduce_llm_params,
        )
        llm_calls["reduce"] = reduce_response.llm_calls
        prompt_tokens["reduce"] = reduce_response.prompt_tokens
        output_tokens["reduce"] = reduce_response.output_tokens
        
        return GlobalSearchResult(
            response=reduce_response.response,
            context_data=context_result.context_records,
            context_text=context_result.context_chunks,
            map_responses=map_responses,
            reduce_context_data=reduce_response.context_data,
            reduce_context_text=reduce_response.context_text,
            completion_time=time.time() - start_time,
            llm_calls=sum(llm_calls.values()),
            prompt_tokens=sum(prompt_tokens.values()),
            output_tokens=sum(output_tokens.values()),
            llm_calls_categories=llm_calls,
            prompt_tokens_categories=prompt_tokens,
            output_tokens_categories=output_tokens,
        )
    
    async def _map_response_single_batch(
        self,
        context_data: str,
        query: str,
        max_length: int,
        **llm_kwargs,
    ) -> SearchResult:
        """Generate answer for a single chunk of community reports."""
        start_time = time.time()
        search_prompt = ""
        
        try:
            search_prompt = self.map_system_prompt.format(
                context_data=context_data, max_length=max_length
            )
            
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=search_prompt),
                HumanMessage(content=query)
            ]
            
            async with self.semaphore:
                response = await self.model.ainvoke(
                    messages, 
                    **llm_kwargs,
                )
                search_response = response.content
                logger.debug("Map response: %s", search_response)
            
            try:
                # Parse search response json
                processed_response = self._parse_search_response(search_response)
                if not processed_response or processed_response == [{"answer": "", "score": 0}]:
                    logger.warning("Warning: Empty or default response from parsing - may indicate LLM response issues")
            except (ValueError, json.JSONDecodeError, Exception) as e:
                logger.warning(f"Warning: Error parsing search response ({type(e).__name__}: {e}) - skipping this batch")
                logger.debug(f"Problematic response: {search_response[:200]}...")
                processed_response = []
            
            return SearchResult(
                response=processed_response,
                context_data=context_data,
                context_text=context_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                output_tokens=num_tokens(search_response, self.token_encoder),
            )
            
        except Exception as e:
            logger.exception(f"Exception in _map_response_single_batch: {type(e).__name__}: {e}")
            # Provide more detailed error information for debugging
            if "timeout" in str(e).lower():
                logger.error("Timeout error - consider increasing timeout settings or reducing batch size")
            elif "connection" in str(e).lower():
                logger.error("Connection error - check network connectivity and service availability")
            elif "json" in str(e).lower():
                logger.error("JSON parsing error - LLM may not be following expected response format")
            
            return SearchResult(
                response=[{"answer": "", "score": 0}],
                context_data=context_data,
                context_text=context_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                output_tokens=0,
            )
    
    def _parse_search_response(self, search_response: str) -> List[Dict[str, Any]]:
        """Parse the search response json and return a list of key points."""
        try:
            # Clean up JSON response - handle multiple formats
            original_response = search_response
            
            # Remove markdown code blocks
            if search_response.startswith('```json'):
                search_response = search_response[7:]
                if search_response.endswith('```'):
                    search_response = search_response[:-3]
            elif search_response.startswith('```'):
                search_response = search_response[3:]
                if search_response.endswith('```'):
                    search_response = search_response[:-3]
            
            search_response = search_response.strip()
            
            # Handle empty or whitespace-only responses
            if not search_response:
                logger.warning("Empty search response received")
                return [{"answer": "", "score": 0}]
            
            # Try to find JSON in the response if it contains other text
            if '{' in search_response and '}' in search_response:
                # Find the first opening brace and the matching closing brace
                start_idx = search_response.find('{')
                if start_idx != -1:
                    # Count braces to find the matching closing brace
                    brace_count = 0
                    end_idx = -1
                    for i, char in enumerate(search_response[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i
                                break
                    
                    if end_idx != -1:
                        search_response = search_response[start_idx:end_idx + 1]
            
            # Try parsing the JSON
            try:
                parsed_json = json.loads(search_response)
            except json.JSONDecodeError as e:
                # If JSON parsing fails, try to fix common issues
                logger.warning(f"Initial JSON parse failed: {e}, attempting to fix...")
                
                # Try to fix common JSON issues
                fixed_response = search_response
                
                # Remove any trailing commas
                import re
                fixed_response = re.sub(r',(\s*[}\]])', r'\1', fixed_response)
                
                # Try parsing again
                try:
                    parsed_json = json.loads(fixed_response)
                    logger.info("Successfully parsed JSON after fixes")
                except json.JSONDecodeError:
                    # If still failing, log the problematic response and return default
                    logger.error(f"Could not parse JSON response. Original: {original_response[:500]}...")
                    return [{"answer": "", "score": 0}]
            
            if not isinstance(parsed_json, dict):
                logger.warning(f"Parsed JSON is not a dict: {type(parsed_json)}")
                return [{"answer": "", "score": 0}]
            
            parsed_elements = parsed_json.get("points")
            if not parsed_elements:
                # Try alternative keys
                for key in ["results", "items", "data", "responses"]:
                    if key in parsed_json:
                        parsed_elements = parsed_json[key]
                        break
                
                if not parsed_elements:
                    logger.warning("No 'points' or alternative keys found in parsed JSON")
                    return [{"answer": "", "score": 0}]
            
            if not isinstance(parsed_elements, list):
                logger.warning(f"Parsed elements is not a list: {type(parsed_elements)}")
                return [{"answer": "", "score": 0}]
            
            results = []
            for element in parsed_elements:
                if isinstance(element, dict):
                    # Handle various field names
                    description = element.get("description") or element.get("answer") or element.get("text") or ""
                    score = element.get("score", 0)
                    
                    # Convert score to int if possible
                    try:
                        score = int(float(score))
                    except (ValueError, TypeError):
                        score = 0
                    
                    if description:  # Only add if we have some content
                        results.append({
                            "answer": str(description),
                            "score": score,
                        })
            
            if not results:
                logger.warning("No valid elements found in parsed JSON")
                return [{"answer": "", "score": 0}]
            
            return results
            
        except Exception as e:
            logger.error(f"Unexpected error parsing search response: {e}")
            logger.error(f"Response content (first 500 chars): {search_response[:500]}...")
            return [{"answer": "", "score": 0}]
    
    async def _reduce_response(
        self, 
        map_responses: List[SearchResult],
        query: str, 
        **llm_kwargs,
    ) -> SearchResult:
        """Combine all intermediate responses from single batches into a final answer."""
        text_data = ""
        search_prompt = ""
        start_time = time.time()
        
        try:
            # Collect all key points into a single list
            key_points = []
            for index, response in enumerate(map_responses):
                if not isinstance(response.response, list):
                    continue
                for element in response.response:
                    if not isinstance(element, dict):
                        continue
                    if "answer" not in element or "score" not in element:
                        continue
                    key_points.append({
                        "analyst": index,
                        "answer": element["answer"],
                        "score": element["score"],
                    })
            
            # Filter response with score = 0 and rank responses by descending order of score
            filtered_key_points = [
                point for point in key_points if point["score"] > 0
            ]
            
            if len(filtered_key_points) == 0:
                return SearchResult(
                    response="I don't have sufficient information in the knowledge base to answer this question.",
                    context_data="",
                    context_text="",
                    completion_time=time.time() - start_time,
                    llm_calls=0,
                    prompt_tokens=0,
                    output_tokens=0,
                )
            
            filtered_key_points = sorted(
                filtered_key_points,
                key=lambda x: x["score"],
                reverse=True,
            )
            
            # Prepare data for reduce phase within token limit
            data = []
            total_tokens = 0
            for point in filtered_key_points:
                formatted_response_data = [
                    f"----Analyst {point['analyst'] + 1}----",
                    f"Importance Score: {point['score']}",
                    point["answer"]
                ]
                formatted_response_text = "\n".join(formatted_response_data)
                
                if (total_tokens + num_tokens(formatted_response_text, self.token_encoder) > self.max_data_tokens):
                    break
                    
                data.append(formatted_response_text)
                total_tokens += num_tokens(formatted_response_text, self.token_encoder)
            
            text_data = "\n\n".join(data)
            
            search_prompt = self.reduce_system_prompt.format(
                report_data=text_data,
                response_type=self.response_type,
                max_length=self.reduce_max_length,
            )
            
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=search_prompt),
                HumanMessage(content=query)
            ]
            
            response = await self.model.ainvoke(
                messages,
                **llm_kwargs,
            )
            
            search_response = response.content
            
            return SearchResult(
                response=search_response,
                context_data=text_data,
                context_text=text_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                output_tokens=num_tokens(search_response, self.token_encoder),
            )
            
        except Exception:
            logger.exception("Exception in reduce_response")
            return SearchResult(
                response="",
                context_data=text_data,
                context_text=text_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                output_tokens=0,
            )


# Helper functions for Advanced GraphRAG context transformation

def _is_advanced_graphrag_format(text: str) -> bool:
    """Check if text is in Advanced GraphRAG structured format."""
    return "-----Entities-----" in text or "-----Reports-----" in text


def _extract_section(text: str, start_marker: str, end_marker: str = None) -> str:
    """Extract a section between markers."""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    
    start_idx += len(start_marker)
    
    if end_marker:
        end_idx = text.find(end_marker, start_idx)
        if end_idx != -1:
            return text[start_idx:end_idx].strip()
    
    return text[start_idx:].strip()


def _parse_entities_to_natural_text(entities_section: str) -> str:
    """Convert entities section to natural language."""
    if not entities_section:
        return ""
    
    natural_descriptions = []
    
    # Split by lines and process each entity
    lines = entities_section.split('\n')
    current_entity = None
    current_description = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a new entity (has ID pattern)
        if line.count('_') > 0 and len(line.split()) > 2:
            # Save previous entity if exists
            if current_entity and current_description:
                desc = ' '.join(current_description)
                # Clean up the entity name
                entity_name = current_entity.split('_', 1)[-1] if '_' in current_entity else current_entity
                entity_name = entity_name.replace('_', ' ')
                natural_descriptions.append(f"{entity_name}: {desc}")
            
            # Start new entity
            parts = line.split(None, 2)  # Split into at most 3 parts
            if len(parts) >= 3:
                current_entity = parts[1]  # Second part is usually the entity name
                current_description = [parts[2]]  # Rest is description
            else:
                current_entity = line
                current_description = []
        else:
            # Continuation of current description
            if current_entity:
                current_description.append(line)
    
    # Don't forget the last entity
    if current_entity and current_description:
        desc = ' '.join(current_description)
        entity_name = current_entity.split('_', 1)[-1] if '_' in current_entity else current_entity
        entity_name = entity_name.replace('_', ' ')
        natural_descriptions.append(f"{entity_name}: {desc}")
    
    return '. '.join(natural_descriptions) if natural_descriptions else ""


def _parse_reports_to_natural_text(reports_section: str) -> str:
    """Convert reports/community section to natural language."""
    if not reports_section:
        return ""
    
    summaries = []
    
    # Split by "Community:" markers
    parts = reports_section.split('Community:')
    
    for part in parts[1:]:  # Skip first empty part
        if not part.strip():
            continue
            
        lines = part.strip().split('\n')
        if len(lines) < 2:
            continue
            
        # First line has community info, look for "Content:" marker
        content_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('Content:'):
                content_start = i
                break
        
        if content_start != -1 and content_start + 1 < len(lines):
            # Extract content after "Content:" marker
            content_lines = lines[content_start + 1:]
            content = ' '.join(line.strip() for line in content_lines if line.strip())
            
            if content:
                # Clean up and truncate if too long
                if len(content) > 300:
                    content = content[:300] + "..."
                summaries.append(content)
    
    return '. '.join(summaries) if summaries else ""


def _transform_advanced_graphrag_context(text: str) -> str:
    """Transform Advanced GraphRAG structured output into RAGAS-friendly natural language."""
    natural_parts = []
    
    # Extract and transform entities section
    entities_section = _extract_section(text, "-----Entities-----", "-----Reports-----")
    if not entities_section:
        entities_section = _extract_section(text, "-----Entities-----")
    
    if entities_section:
        entity_descriptions = _parse_entities_to_natural_text(entities_section)
        if entity_descriptions:
            natural_parts.append(f"Relevant entities and services: {entity_descriptions}")
    
    # Extract and transform reports section
    reports_section = _extract_section(text, "-----Reports-----")
    if reports_section:
        community_summaries = _parse_reports_to_natural_text(reports_section)
        if community_summaries:
            natural_parts.append(f"Context summaries: {community_summaries}")
    
    # If we couldn't parse anything, return a cleaned version of the original
    if not natural_parts:
        # Basic cleanup - remove technical markers
        cleaned = text.replace("-----Entities-----", "").replace("-----Reports-----", "")
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace
        return cleaned if cleaned else text
    
    return '\n\n'.join(natural_parts)


# Helper function to normalize context text for RAGAS compatibility
def _normalize_context_text(context_text: str | List[str] | Dict[str, str]) -> str:
    """
    Normalize and transform context text to a RAGAS-friendly format.
    
    For Advanced GraphRAG structured output, this transforms:
    - "-----Entities-----" sections into natural descriptions
    - "-----Reports-----" sections into readable summaries
    - Entity IDs and technical metadata into plain text
    
    For other formats, performs basic normalization.
    """
    # Convert to string first
    if isinstance(context_text, str):
        text = context_text
    elif isinstance(context_text, list):
        text = "\n\n".join(context_text)
    elif isinstance(context_text, dict):
        text = "\n\n".join(f"{k}: {v}" for k, v in context_text.items())
    else:
        text = str(context_text)
    
    # Check if this is Advanced GraphRAG structured output
    if _is_advanced_graphrag_format(text):
        return _transform_advanced_graphrag_context(text)
    
    return text


# Integration Layer for Existing Benchmark System
class AdvancedGraphRAGRetriever:
    """Main retriever class that integrates advanced GraphRAG patterns with existing Neo4j schema."""
    
    def __init__(self, graph_processor: AdvancedGraphProcessor):
        self.graph_processor = graph_processor
        self.driver = graph_processor.driver
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        
        # Initialize data loader
        self.data_loader = Neo4jDataLoader(self.driver)
        
        # Load data from existing graph
        self._load_graph_data()
        
        # Initialize context builders
        self._initialize_context_builders()
        
        # Initialize search engines
        self._initialize_search_engines()
    
    def _load_graph_data(self):
        """Load data from existing Neo4j graph."""
        print("Loading data from Neo4j graph...")
        
        self.entities = self.data_loader.load_entities()
        self.relationships = self.data_loader.load_relationships()
        self.text_units = self.data_loader.load_text_units()
        self.communities = self.data_loader.load_communities()
        self.community_reports = self.data_loader.load_community_reports()
        
        print(f"Loaded {len(self.entities)} entities, {len(self.relationships)} relationships, "
              f"{len(self.text_units)} text units, {len(self.community_reports)} community reports")
    
    def _initialize_context_builders(self):
        """Initialize context builders."""
        # Vector store for entity embeddings
        self.vector_store = Neo4jVectorStore(self.driver, self.embeddings)
        
        # Local context builder
        self.local_context_builder = LocalSearchMixedContext(
            entities=self.entities,
            entity_text_embeddings=self.vector_store,
            text_embedder=self.embeddings,
            text_units=self.text_units,
            community_reports=self.community_reports,
            relationships=self.relationships,
        )
        
        # Global context builder
        self.global_context_builder = GlobalCommunityContext(
            community_reports=self.community_reports,
            communities=self.communities,
            entities=list(self.entities.values()),
        )
    
    def _initialize_search_engines(self):
        """Initialize search engines."""
        self.local_search = LocalSearch(
            model=self.llm,
            context_builder=self.local_context_builder,
            response_type="multiple paragraphs",
        )
        
        self.global_search = GlobalSearch(
            model=self.llm,
            context_builder=self.global_context_builder,
            response_type="multiple paragraphs",
        )
    
    async def local_search_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform local search query."""
        result = await self.local_search.search(query, **kwargs)
        
        return {
            'final_answer': result.response,
            'retrieval_details': [
                {
                    'content': _normalize_context_text(result.context_text),
                    'metadata': result.context_data,
                    'method': 'advanced_local',
                    'completion_time': result.completion_time,
                    'llm_calls': result.llm_calls,
                    'tokens_used': result.prompt_tokens + result.output_tokens
                }
            ],
            'method': 'advanced_local',
            'performance_metrics': {
                'completion_time': result.completion_time,
                'llm_calls': result.llm_calls,
                'prompt_tokens': result.prompt_tokens,
                'output_tokens': result.output_tokens,
                'total_tokens': result.prompt_tokens + result.output_tokens
            }
        }
    
    async def global_search_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform global search query."""
        result = await self.global_search.search(query, **kwargs)
        
        return {
            'final_answer': result.response,
            'retrieval_details': [
                {
                    'content': _normalize_context_text(result.context_text),
                    'metadata': result.context_data,
                    'method': 'advanced_global',
                    'completion_time': result.completion_time,
                    'llm_calls': result.llm_calls,
                    'tokens_used': result.prompt_tokens + result.output_tokens
                }
            ],
            'method': 'advanced_global',
            'performance_metrics': {
                'completion_time': result.completion_time,
                'llm_calls': result.llm_calls,
                'prompt_tokens': result.prompt_tokens,
                'output_tokens': result.output_tokens,
                'total_tokens': result.prompt_tokens + result.output_tokens,
                'map_responses': len(result.map_responses)
            }
        }
    
    def close(self):
        """Close the graph processor."""
        self.graph_processor.close()


# Lightweight retriever for benchmarking (no processor initialization overhead)
class LightweightAdvancedGraphRAGRetriever:
    """Lightweight retriever that connects directly to Neo4j without processor initialization."""
    
    def __init__(self):
        """Initialize with direct Neo4j connection."""
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j connection details
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
        
        self.driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        
        # Initialize data loader
        self.data_loader = Neo4jDataLoader(self.driver)
        
        # Load data from existing graph (lightweight)
        self._load_graph_data()
        
        # Initialize context builders
        self._initialize_context_builders()
        
        # Initialize search engines
        self._initialize_search_engines()
    
    def _load_graph_data(self):
        """Load data from existing Neo4j graph (silent)."""
        self.entities = self.data_loader.load_entities()
        self.relationships = self.data_loader.load_relationships()
        self.text_units = self.data_loader.load_text_units()
        self.communities = self.data_loader.load_communities()
        self.community_reports = self.data_loader.load_community_reports()
    
    def _initialize_context_builders(self):
        """Initialize context builders."""
        # Vector store for entity embeddings
        self.vector_store = Neo4jVectorStore(self.driver, self.embeddings)
        
        # Local context builder
        self.local_context_builder = LocalSearchMixedContext(
            entities=self.entities,
            entity_text_embeddings=self.vector_store,
            text_embedder=self.embeddings,
            text_units=self.text_units,
            community_reports=self.community_reports,
            relationships=self.relationships,
        )
        
        # Global context builder
        self.global_context_builder = GlobalCommunityContext(
            community_reports=self.community_reports,
            communities=self.communities,
            entities=list(self.entities.values()),
        )
    
    def _initialize_search_engines(self):
        """Initialize search engines."""
        self.local_search = LocalSearch(
            model=self.llm,
            context_builder=self.local_context_builder,
            response_type="multiple paragraphs",
        )
        
        self.global_search = GlobalSearch(
            model=self.llm,
            context_builder=self.global_context_builder,
            response_type="multiple paragraphs",
        )
    
    async def local_search_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform local search query."""
        result = await self.local_search.search(query, **kwargs)
        
        return {
            'final_answer': result.response,
            'retrieval_details': [
                {
                    'content': _normalize_context_text(result.context_text),
                    'metadata': result.context_data,
                    'method': 'advanced_local',
                    'completion_time': result.completion_time,
                    'llm_calls': result.llm_calls,
                    'tokens_used': result.prompt_tokens + result.output_tokens
                }
            ],
            'method': 'advanced_local',
            'performance_metrics': {
                'completion_time': result.completion_time,
                'llm_calls': result.llm_calls,
                'prompt_tokens': result.prompt_tokens,
                'output_tokens': result.output_tokens,
                'total_tokens': result.prompt_tokens + result.output_tokens
            }
        }
    
    async def global_search_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform global search query."""
        result = await self.global_search.search(query, **kwargs)
        
        return {
            'final_answer': result.response,
            'retrieval_details': [
                {
                    'content': _normalize_context_text(result.context_text),
                    'metadata': result.context_data,
                    'method': 'advanced_global',
                    'completion_time': result.completion_time,
                    'llm_calls': result.llm_calls,
                    'tokens_used': result.prompt_tokens + result.output_tokens
                }
            ],
            'method': 'advanced_global',
            'performance_metrics': {
                'completion_time': result.completion_time,
                'llm_calls': result.llm_calls,
                'prompt_tokens': result.prompt_tokens,
                'output_tokens': result.output_tokens,
                'total_tokens': result.prompt_tokens + result.output_tokens,
                'map_responses': len(result.map_responses) if hasattr(result, 'map_responses') else 0
            }
        }
    
    def close(self):
        """Close the Neo4j driver."""
        if hasattr(self, 'driver'):
            self.driver.close()


# Main integration functions for benchmark compatibility
async def query_advanced_graphrag_local(query: str, k: int = 10, **kwargs) -> Dict[str, Any]:
    """
    Advanced local GraphRAG retrieval
    
    Args:
        query: The search query
        k: Number of entities to consider
        **kwargs: Additional configuration options
    
    Returns:
        Dictionary with response and retrieval details
    """
    
    # Use lightweight connection for benchmarking - no processor initialization
    try:
        # Create a minimal retriever without processor initialization overhead
        retriever = LightweightAdvancedGraphRAGRetriever()
        
        # Perform local search
        result = await retriever.local_search_query(
            query, 
            top_k_mapped_entities=k,
            **kwargs
        )
        
        return result
        
    except Exception as e:
        print(f"Error in advanced local GraphRAG retrieval: {e}")
        import traceback
        traceback.print_exc()
        return {
            'final_answer': f"Error during advanced local GraphRAG retrieval: {str(e)}",
            'retrieval_details': [],
            'method': 'advanced_local_error',
            'performance_metrics': {
                'completion_time': 0,
                'llm_calls': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }
    finally:
        if 'retriever' in locals():
            retriever.close()


async def query_advanced_graphrag_global(query: str, k: int = 8, **kwargs) -> Dict[str, Any]:
    """
    Advanced global GraphRAG retrieval
    
    Args:
        query: The search query
        k: Number of communities to consider (passed to context builder)
        **kwargs: Additional configuration options
    
    Returns:
        Dictionary with response and retrieval details
    """
    
    # Use lightweight connection for benchmarking - no processor initialization
    try:
        # Create a minimal retriever without processor initialization overhead
        retriever = LightweightAdvancedGraphRAGRetriever()
        
        # Perform global search
        result = await retriever.global_search_query(query, **kwargs)
        
        return result
        
    except Exception as e:
        print(f"Error in advanced global GraphRAG retrieval: {e}")
        import traceback
        traceback.print_exc()
        return {
            'final_answer': f"Error during advanced global GraphRAG retrieval: {str(e)}",
            'retrieval_details': [],
            'method': 'advanced_global_error',
            'performance_metrics': {
                'completion_time': 0,
                'llm_calls': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }
    finally:
        if 'retriever' in locals():
            retriever.close()


# Test function
if __name__ == "__main__":
    import asyncio
    
    async def test_advanced_retriever():
        """Test function for the advanced GraphRAG retriever"""
        
        test_queries = [
            "What are the main themes in the documents?",  # Global query
            "What specific requirements did NovaGrid mention?",  # Local query
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing query: {query}")
            print('='*60)
            
            # Test local mode
            print("\n--- Advanced Local Search ---")
            result = await query_advanced_graphrag_local(query)
            print(f"Result: {result['final_answer'][:200]}...")
            print(f"Performance: {result['performance_metrics']}")
            
            # Test global mode
            print("\n--- Advanced Global Search ---")
            result = await query_advanced_graphrag_global(query)
            print(f"Result: {result['final_answer'][:200]}...")
            print(f"Performance: {result['performance_metrics']}")
    
    # Uncomment to test
    # asyncio.run(test_advanced_retriever()) 


# ===== COMPATIBILITY LAYER =====
# These functions maintain compatibility with existing benchmark and drift GraphRAG systems

async def query_advanced_graphrag(query: str, mode: str = "hybrid", k: int = 5, **kwargs) -> Dict[str, Any]:
    """
    Compatibility function that matches the old advanced_graphrag interface.
    Now powered by advanced GraphRAG implementation with production-ready patterns.
    
    Args:
        query: The search query
        mode: "local", "global", or "hybrid" 
        k: Number of results/entities to consider
        **kwargs: Additional configuration options
    
    Returns:
        Dictionary with response and retrieval details
    """
    
    if mode == "local":
        return await query_advanced_graphrag_local(query, k=k, **kwargs)
    elif mode == "global":
        return await query_advanced_graphrag_global(query, k=k, **kwargs)
    elif mode == "hybrid":
        # For hybrid mode, run both local and global and combine
        try:
            # Create a minimal retriever without processor initialization overhead
            retriever = LightweightAdvancedGraphRAGRetriever()
            
            # Run both local and global searches
            local_result = await retriever.local_search_query(query, top_k_mapped_entities=k, **kwargs)
            global_result = await retriever.global_search_query(query, **kwargs)
            
            # Combine responses - prioritize local but include global insights
            combined_answer = f"**Local Context Analysis:**\n{local_result['final_answer']}\n\n**Global Context Analysis:**\n{global_result['final_answer']}"
            
            # Combine retrieval details
            combined_details = local_result['retrieval_details'] + global_result['retrieval_details']
            
            # Combine performance metrics
            combined_metrics = {
                'completion_time': local_result['performance_metrics']['completion_time'] + global_result['performance_metrics']['completion_time'],
                'llm_calls': local_result['performance_metrics']['llm_calls'] + global_result['performance_metrics']['llm_calls'],
                'prompt_tokens': local_result['performance_metrics']['prompt_tokens'] + global_result['performance_metrics']['prompt_tokens'],
                'output_tokens': local_result['performance_metrics']['output_tokens'] + global_result['performance_metrics']['output_tokens'],
                'total_tokens': local_result['performance_metrics']['total_tokens'] + global_result['performance_metrics']['total_tokens']
            }
            
            return {
                'final_answer': combined_answer,
                'retrieval_details': combined_details,
                'method': 'advanced_hybrid',
                'performance_metrics': combined_metrics
            }
            
        except Exception as e:
            print(f"Error in hybrid advanced GraphRAG retrieval: {e}")
            import traceback
            traceback.print_exc()
            return {
                'final_answer': f"Error during hybrid advanced GraphRAG retrieval: {str(e)}",
                'retrieval_details': [],
                'method': 'advanced_hybrid_error',
                'performance_metrics': {
                    'completion_time': 0,
                    'llm_calls': 0,
                    'prompt_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0
                }
            }
        finally:
            if 'retriever' in locals():
                retriever.close()
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'local', 'global', or 'hybrid'")


class GraphRAGLocalRetriever:
    """
    Compatibility class that wraps the advanced GraphRAG implementation.
    This maintains compatibility with the drift GraphRAG retriever and other systems.
    """
    
    def __init__(self, graph_processor: AdvancedGraphProcessor, shared_retriever: Optional[AdvancedGraphRAGRetriever] = None):
        """Initialize with a graph processor and optional shared retriever."""
        self.graph_processor = graph_processor
        self.retriever = shared_retriever if shared_retriever is not None else AdvancedGraphRAGRetriever(graph_processor)
    
    async def retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        """Retrieve using local search."""
        return await self.retriever.local_search_query(query, **kwargs)
    
    async def search(self, query: str, **kwargs) -> SearchResult:
        """Search using advanced local search."""
        result_dict = await self.retriever.local_search_query(query, **kwargs)
        
        # Convert to SearchResult format
        return SearchResult(
            response=result_dict['final_answer'],
            context_data=result_dict['retrieval_details'][0]['metadata'] if result_dict['retrieval_details'] else {},
            context_text=result_dict['retrieval_details'][0]['content'] if result_dict['retrieval_details'] else "",
            completion_time=result_dict['performance_metrics']['completion_time'],
            llm_calls=result_dict['performance_metrics']['llm_calls'],
            prompt_tokens=result_dict['performance_metrics']['prompt_tokens'],
            output_tokens=result_dict['performance_metrics']['output_tokens']
        )
    
    def close(self):
        """Close the retriever."""
        self.retriever.close()


# Additional compatibility aliases
GraphRAGGlobalRetriever = AdvancedGraphRAGRetriever  # For global search compatibility
GraphRAGHybridRetriever = AdvancedGraphRAGRetriever  # For hybrid search compatibility


# ===== FACTORY FUNCTION FOR COMPATIBILITY =====

def create_advanced_graphrag_retriever(
    graph_processor: AdvancedGraphProcessor, 
    mode: str = "local"
) -> Union[AdvancedGraphRAGRetriever, GraphRAGLocalRetriever, "GraphRAGGlobalRetriever", "GraphRAGHybridRetriever"]:
    """
    Factory function to create advanced GraphRAG retrievers.
    This maintains compatibility with the old advanced_graphrag_retriever interface.
    
    Args:
        graph_processor: The graph processor instance
        mode: "local", "global", or "hybrid"
    
    Returns:
        Appropriate retriever instance
    """
    
    if mode == "local":
        return GraphRAGLocalRetriever(graph_processor)
    elif mode == "global":
        return AdvancedGraphRAGRetriever(graph_processor)  # Global mode
    elif mode == "hybrid":
        return AdvancedGraphRAGRetriever(graph_processor)  # Hybrid mode
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'local', 'global', or 'hybrid'")

def query_advanced_graphrag_sync(query: str, **kwargs) -> Dict[str, Any]:
    """Synchronous wrapper for query_advanced_graphrag"""
    import asyncio
    import sys
    
    def run_async():
        """Run the async function in a clean environment"""
        return asyncio.run(query_advanced_graphrag(query, **kwargs))
    
    # Always use asyncio.run in a clean way
    try:
        # Check if we're in a Jupyter notebook or similar environment
        if 'ipykernel' in sys.modules:
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(query_advanced_graphrag(query, **kwargs))
        else:
            # Standard environment - use asyncio.run
            return asyncio.run(query_advanced_graphrag(query, **kwargs))
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # We're in a running event loop, use thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result()
        else:
            raise e


# ===== ADDITIONAL COMPATIBILITY CLASSES =====

class GraphRAGGlobalRetriever:
    """
    Compatibility class for global GraphRAG retrieval.
    Wraps the advanced implementation.
    """
    
    def __init__(self, graph_processor: AdvancedGraphProcessor, shared_retriever: Optional[AdvancedGraphRAGRetriever] = None):
        """Initialize with a graph processor and optional shared retriever."""
        self.graph_processor = graph_processor
        self.retriever = shared_retriever if shared_retriever is not None else AdvancedGraphRAGRetriever(graph_processor)
    
    async def retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        """Retrieve using global search."""
        return await self.retriever.global_search_query(query, **kwargs)
    
    async def search(self, query: str, **kwargs) -> GlobalSearchResult:
        """Search using advanced global search."""
        result_dict = await self.retriever.global_search_query(query, **kwargs)
        
        # Convert to GlobalSearchResult format
        return GlobalSearchResult(
            response=result_dict['final_answer'],
            context_data=result_dict['retrieval_details'][0]['metadata'] if result_dict['retrieval_details'] else {},
            context_text=result_dict['retrieval_details'][0]['content'] if result_dict['retrieval_details'] else "",
            completion_time=result_dict['performance_metrics']['completion_time'],
            llm_calls=result_dict['performance_metrics']['llm_calls'],
            prompt_tokens=result_dict['performance_metrics']['prompt_tokens'],
            output_tokens=result_dict['performance_metrics']['output_tokens'],
            map_responses=[]  # Could be populated from map_responses if available
        )
    
    def close(self):
        """Close the retriever."""
        self.retriever.close()


class GraphRAGHybridRetriever:
    """
    Compatibility class for hybrid GraphRAG retrieval.
    Wraps the advanced implementation with both local and global search.
    """
    
    def __init__(self, graph_processor: AdvancedGraphProcessor):
        """Initialize with a graph processor."""
        self.graph_processor = graph_processor
        self.retriever = AdvancedGraphRAGRetriever(graph_processor)
    
    async def retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        """Retrieve using hybrid search (both local and global)."""
        return await query_advanced_graphrag(query, mode="hybrid", **kwargs)
    
    async def search(self, query: str, **kwargs) -> SearchResult:
        """Search using hybrid approach."""
        result_dict = await self.retrieve(query, **kwargs)
        
        # Convert to SearchResult format
        return SearchResult(
            response=result_dict['final_answer'],
            context_data=result_dict['retrieval_details'][0]['metadata'] if result_dict['retrieval_details'] else {},
            context_text=result_dict['retrieval_details'][0]['content'] if result_dict['retrieval_details'] else "",
            completion_time=result_dict['performance_metrics']['completion_time'],
            llm_calls=result_dict['performance_metrics']['llm_calls'],
            prompt_tokens=result_dict['performance_metrics']['prompt_tokens'],
            output_tokens=result_dict['performance_metrics']['output_tokens']
        )
    
    def close(self):
        """Close the retriever."""
        self.retriever.close()


# Update the aliases
GraphRAGGlobalRetriever = GraphRAGGlobalRetriever
GraphRAGHybridRetriever = GraphRAGHybridRetriever 