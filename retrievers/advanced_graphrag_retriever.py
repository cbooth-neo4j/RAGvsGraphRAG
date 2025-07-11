"""
GraphRAG Retriever Implementation

This module implements a GraphRAG retrieval system based on Microsoft's approach.

The retriever supports:
1. Global Search: Community-based map-reduce for global sensemaking
2. Local Search: Entity-enhanced mixed context retrieval  
3. Hybrid Search: Automatic mode selection based on query type
"""

import os
import asyncio
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Core dependencies
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import neo4j
from dotenv import load_dotenv

# Import our graph processor from the new data_processors module
from data_processors import AdvancedGraphProcessor

# Load environment variables
load_dotenv()

@dataclass
class SearchResult:
    """Result from a GraphRAG search operation"""
    response: str
    context_data: Dict[str, Any]
    context_text: str
    completion_time: float
    llm_calls: int
    prompt_tokens: int
    output_tokens: int
    method: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MapResponse:
    """Response from map phase of global search"""
    points: List[Dict[str, Any]]
    community_id: str
    score: float
    llm_calls: int
    prompt_tokens: int
    output_tokens: int

class QueryClassifier:
    """Classifies queries as global, local, or hybrid"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for a GraphRAG system. Classify the query as:

GLOBAL: Questions that require understanding of themes, patterns, trends, or insights across the entire dataset
- Examples: "What are the main themes?", "How do entities relate?", "What patterns emerge?", "What is the overall structure?"

LOCAL: Questions that seek specific facts, entities, events, or details
- Examples: "What did John Smith say?", "When was the contract signed?", "What is the price of X?"

Return only "GLOBAL" or "LOCAL"."""),
            ("human", "Classify this query: {query}")
        ])
    
    async def classify(self, query: str) -> str:
        """Classify query as GLOBAL or LOCAL"""
        try:
            response = await self.llm.ainvoke(
                self.classification_prompt.format_messages(query=query)
            )
            classification = response.content.strip().upper()
            return classification if classification in ["GLOBAL", "LOCAL"] else "GLOBAL"
        except Exception as e:
            print(f"Error in query classification: {e}")
            return "GLOBAL"  # Default to global

class CommunitySelector:
    """Selects relevant communities for a given query"""
    
    def __init__(self, embeddings: OpenAIEmbeddings, driver: neo4j.GraphDatabase.driver):
        self.embeddings = embeddings
        self.driver = driver
    
    async def select_communities(
        self, 
        query: str, 
        max_communities: int = 10,
        levels: List[int] = [0, 1, 2],
        min_rank: int = 1,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Select relevant communities based on query embedding similarity"""
        
        # Create query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        with self.driver.session() as session:
            # Get communities with their summaries and metadata
            communities_data = session.run("""
                MATCH (c:__Community__)
                WHERE c.level IN $levels 
                  AND c.summary IS NOT NULL
                  AND coalesce(c.community_rank, 0) >= $min_rank
                WITH c, size([(c)<-[:IN_COMMUNITY*]-(e:__Entity__) | e]) as entity_count
                RETURN c.id as community_id,
                       c.level as level,
                       c.summary as summary,
                       c.community_rank as rank,
                       entity_count,
                       c.summary_generated_at as generated_at
                ORDER BY c.level, entity_count DESC
            """, levels=levels, min_rank=min_rank).data()
            
            if not communities_data:
                print("No communities found for selection")
                return []
            
            # Calculate embeddings for community summaries (in batches for efficiency)
            summaries = [comm['summary'] for comm in communities_data]
            summary_embeddings = self.embeddings.embed_documents(summaries)
            
            # Calculate similarities
            similarities = []
            query_norm = np.linalg.norm(query_embedding)
            
            for i, summary_embedding in enumerate(summary_embeddings):
                summary_norm = np.linalg.norm(summary_embedding)
                if summary_norm > 0 and query_norm > 0:
                    similarity = np.dot(query_embedding, summary_embedding) / (query_norm * summary_norm)
                    similarities.append(similarity)
                else:
                    similarities.append(0.0)
            
            # Combine data with similarities
            for i, comm in enumerate(communities_data):
                comm['similarity'] = similarities[i]
            
            # Filter by similarity threshold and sort
            relevant_communities = [
                comm for comm in communities_data 
                if comm['similarity'] >= similarity_threshold
            ]
            
            # Sort by similarity * rank (weighted relevance)
            relevant_communities.sort(
                key=lambda x: x['similarity'] * (x.get('rank', 1) or 1),
                reverse=True
            )
            
            # Apply diversity filtering - ensure we get communities from different levels
            selected_communities = []
            level_counts = {level: 0 for level in levels}
            max_per_level = max(1, max_communities // len(levels))
            
            for comm in relevant_communities:
                level = comm['level']
                if level_counts[level] < max_per_level and len(selected_communities) < max_communities:
                    selected_communities.append(comm)
                    level_counts[level] += 1
            
            # Fill remaining slots if needed
            for comm in relevant_communities:
                if len(selected_communities) >= max_communities:
                    break
                if comm not in selected_communities:
                    selected_communities.append(comm)
            
            print(f"Selected {len(selected_communities)} communities from {len(communities_data)} available")
            return selected_communities[:max_communities]

class GraphRAGGlobalRetriever:
    """Global search implementation using community-based map-reduce"""
    
    def __init__(self, graph_processor: AdvancedGraphProcessor):
        self.graph = graph_processor
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.community_selector = CommunitySelector(self.embeddings, self.graph.driver)
        
        # Map phase prompt (based on Microsoft's implementation)
        self.map_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant responding to questions about data in community reports.

Generate a response consisting of a list of key points that respond to the user's question, based on the given community report.

You should use the data provided in the community report below as the primary context for generating the response.
If you don't know the answer or if the provided community report doesn't contain sufficient information, respond with an empty list.

Each key point should be:
- A single paragraph
- Highly relevant to the user's question
- Supported by the community report data
- Include a relevance score from 0-100 (100 being most relevant)

Return your response as a JSON object with this structure:
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

Community Report:
{community_summary}"""),
            ("human", "{query}")
        ])
        
        # Reduce phase prompt
        self.reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that synthesizes information from multiple analysts to answer questions comprehensively.

You have received responses from multiple analysts, each focused on different aspects of the data. Your task is to synthesize these responses into a comprehensive, coherent answer that addresses the user's question.

Guidelines:
1. Combine information from all analyst responses
2. Organize the information logically
3. Remove redundancy while preserving important details
4. Ensure the final answer is well-structured and comprehensive
5. Prioritize information from higher-scored analyst responses
6. Write in a clear, professional tone

Analyst Responses:
{analyst_responses}"""),
            ("human", "Based on the analyst responses above, provide a comprehensive answer to: {query}")
        ])
        
    async def search(
        self, 
        query: str, 
        max_communities: int = 8,
        levels: List[int] = [0, 1, 2],
        max_context_tokens: int = 8000,
        min_relevance_score: int = 30,
        **kwargs
    ) -> SearchResult:
        """Perform global search using community-based map-reduce"""
        
        start_time = time.time()
        total_llm_calls = 0
        total_prompt_tokens = 0
        total_output_tokens = 0
        
        try:
            # Step 1: Select relevant communities
            print(f"🔍 Selecting communities for query: {query[:60]}...")
            communities = await self.community_selector.select_communities(
                query=query,
                max_communities=max_communities,
                levels=levels
            )
            
            if not communities:
                return SearchResult(
                    response="I don't have sufficient information in the knowledge base to answer this question.",
                    context_data={},
                    context_text="",
                    completion_time=time.time() - start_time,
                    llm_calls=0,
                    prompt_tokens=0,
                    output_tokens=0,
                    method="graphrag_global"
                )
            
            # Step 2: Map phase - parallel processing of communities
            print(f"📊 Processing {len(communities)} communities in map phase...")
            map_responses = await self._map_phase(query, communities)
            
            # Aggregate token counts
            for response in map_responses:
                total_llm_calls += response.llm_calls
                total_prompt_tokens += response.prompt_tokens
                total_output_tokens += response.output_tokens
            
            # Step 3: Filter and rank responses
            all_points = []
            for response in map_responses:
                for point in response.points:
                    if point.get('score', 0) >= min_relevance_score:
                        all_points.append({
                            'description': point['description'],
                            'score': point['score'],
                            'community_id': response.community_id
                        })
            
            # Sort by score
            all_points.sort(key=lambda x: x['score'], reverse=True)
            
            if not all_points:
                return SearchResult(
                    response="No relevant information found in the knowledge base for this question.",
                    context_data={"communities": communities, "map_responses": map_responses},
                    context_text=str(communities),
                    completion_time=time.time() - start_time,
                    llm_calls=total_llm_calls,
                    prompt_tokens=total_prompt_tokens,
                    output_tokens=total_output_tokens,
                    method="graphrag_global"
                )
            
            # Step 4: Reduce phase - synthesize final answer
            print(f"🔄 Synthesizing final answer from {len(all_points)} key points...")
            final_response, reduce_tokens = await self._reduce_phase(query, all_points, max_context_tokens)
            
            total_llm_calls += 1
            total_prompt_tokens += reduce_tokens['prompt_tokens']
            total_output_tokens += reduce_tokens['output_tokens']
            
            return SearchResult(
                response=final_response,
                context_data={
                    "communities": communities,
                    "map_responses": [r.__dict__ for r in map_responses],
                    "key_points": all_points
                },
                context_text=f"Used {len(communities)} communities and {len(all_points)} key points",
                completion_time=time.time() - start_time,
                llm_calls=total_llm_calls,
                prompt_tokens=total_prompt_tokens,
                output_tokens=total_output_tokens,
                method="graphrag_global"
            )
            
        except Exception as e:
            print(f"Error in global search: {e}")
            return SearchResult(
                response=f"Error occurred during search: {str(e)}",
                context_data={},
                context_text="",
                completion_time=time.time() - start_time,
                llm_calls=total_llm_calls,
                prompt_tokens=total_prompt_tokens,
                output_tokens=total_output_tokens,
                method="graphrag_global"
            )
    
    async def _map_phase(self, query: str, communities: List[Dict[str, Any]]) -> List[MapResponse]:
        """Execute map phase - parallel processing of community summaries"""
        
        async def process_community(community: Dict[str, Any]) -> MapResponse:
            """Process a single community"""
            try:
                # Format prompt
                messages = self.map_prompt.format_messages(
                    community_summary=community['summary'],
                    query=query
                )
                
                # Calculate prompt tokens (approximate)
                prompt_text = str(messages)
                prompt_tokens = len(prompt_text) // 4  # Rough token estimate
                
                # Call LLM
                response = await self.llm.ainvoke(messages)
                response_text = response.content.strip()
                
                # Calculate output tokens
                output_tokens = len(response_text) // 4  # Rough token estimate
                
                # Parse JSON response
                try:
                    if response_text.startswith('```json'):
                        response_text = response_text[7:-3]
                    elif response_text.startswith('```'):
                        response_text = response_text[3:-3]
                    
                    parsed_response = json.loads(response_text)
                    points = parsed_response.get('points', [])
                    
                    # Validate points structure
                    valid_points = []
                    for point in points:
                        if isinstance(point, dict) and 'description' in point and 'score' in point:
                            valid_points.append({
                                'description': point['description'],
                                'score': int(point.get('score', 0))
                            })
                    
                    return MapResponse(
                        points=valid_points,
                        community_id=community['community_id'],
                        score=sum(p['score'] for p in valid_points) / len(valid_points) if valid_points else 0,
                        llm_calls=1,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens
                    )
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parse error for community {community['community_id']}: {e}")
                    return MapResponse(
                        points=[],
                        community_id=community['community_id'],
                        score=0,
                        llm_calls=1,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens
                    )
                    
            except Exception as e:
                print(f"Error processing community {community['community_id']}: {e}")
                return MapResponse(
                    points=[],
                    community_id=community['community_id'],
                    score=0,
                    llm_calls=1,
                    prompt_tokens=0,
                    output_tokens=0
                )
        
        # Process communities in parallel (with concurrency limit)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def bounded_process(community):
            async with semaphore:
                return await process_community(community)
        
        # Execute all tasks
        tasks = [bounded_process(community) for community in communities]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = [r for r in responses if isinstance(r, MapResponse)]
        
        print(f"✅ Map phase completed: {len(valid_responses)}/{len(communities)} communities processed")
        return valid_responses
    
    async def _reduce_phase(
        self, 
        query: str, 
        key_points: List[Dict[str, Any]], 
        max_context_tokens: int = 8000
    ) -> Tuple[str, Dict[str, int]]:
        """Execute reduce phase - synthesize final answer from key points"""
        
        # Prepare analyst responses for the reduce prompt
        analyst_responses = []
        token_count = 0
        
        for i, point in enumerate(key_points):
            analyst_response = f"----Analyst {i + 1}----\n"
            analyst_response += f"Importance Score: {point['score']}\n"
            analyst_response += f"{point['description']}\n"
            
            # Rough token estimation (4 chars per token)
            response_tokens = len(analyst_response) // 4
            
            if token_count + response_tokens > max_context_tokens:
                break
                
            analyst_responses.append(analyst_response)
            token_count += response_tokens
        
        # Combine all analyst responses
        combined_responses = "\n\n".join(analyst_responses)
        
        # Format reduce prompt
        messages = self.reduce_prompt.format_messages(
            analyst_responses=combined_responses,
            query=query
        )
        
        # Calculate token counts
        prompt_text = str(messages)
        prompt_tokens = len(prompt_text) // 4
        
        # Generate final response
        response = await self.llm.ainvoke(messages)
        final_answer = response.content.strip()
        
        output_tokens = len(final_answer) // 4
        
        return final_answer, {
            'prompt_tokens': prompt_tokens,
            'output_tokens': output_tokens
        }

class GraphRAGLocalRetriever:
    """Local search implementation using entity-enhanced mixed context"""
    
    def __init__(self, graph_processor: AdvancedGraphProcessor):
        self.graph = graph_processor
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Local search prompt
        self.local_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering questions based on the provided context data.

Use the context data below to answer the user's question. The context includes:
- Entity information (organizations, people, locations, requirements, etc.)
- Relationships between entities  
- Community information
- Relevant text chunks from documents

Guidelines:
1. Answer based on the provided context
2. Be specific and cite relevant entities or sources when possible
3. If the context doesn't contain sufficient information, state this clearly
4. Provide a comprehensive but concise answer

Context Data:
{context_data}"""),
            ("human", "{query}")
        ])
    
    async def search(
        self,
        query: str,
        max_entities: int = 10,
        max_chunks: int = 5,
        max_communities: int = 3,
        **kwargs
    ) -> SearchResult:
        """Perform local search using entity-enhanced mixed context"""
        
        start_time = time.time()
        
        try:
            # Step 1: Find relevant entities via embedding similarity
            print(f"🔍 Finding relevant entities for: {query[:60]}...")
            relevant_entities = await self._find_relevant_entities(query, max_entities)
            
            # Step 2: Get related chunks and communities
            print(f"📄 Building context from {len(relevant_entities)} entities...")
            context_data = await self._build_mixed_context(
                query, 
                relevant_entities, 
                max_chunks, 
                max_communities
            )
            
            # Step 3: Generate response using mixed context
            print(f"🤖 Generating response with mixed context...")
            response, token_counts = await self._generate_response(query, context_data)
            
            return SearchResult(
                response=response,
                context_data={
                    "relevant_entities": relevant_entities,
                    "context_summary": context_data[:500] + "..." if len(context_data) > 500 else context_data
                },
                context_text=context_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=token_counts['prompt_tokens'],
                output_tokens=token_counts['output_tokens'],
                method="graphrag_local"
            )
            
        except Exception as e:
            print(f"Error in local search: {e}")
            return SearchResult(
                response=f"Error occurred during search: {str(e)}",
                context_data={},
                context_text="",
                completion_time=time.time() - start_time,
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0,
                method="graphrag_local"
            )
    
    async def _find_relevant_entities(self, query: str, max_entities: int) -> List[Dict[str, Any]]:
        """Find entities most relevant to the query using embedding similarity"""
        
        with self.graph.driver.session() as session:
            # Get all entities with their descriptions and embeddings
            entities_data = session.run("""
                MATCH (e:__Entity__)
                WHERE e.description IS NOT NULL AND e.embedding IS NOT NULL
                RETURN e.id as entity_id,
                       e.name as name,
                       e.entity_type as type,
                       e.description as description,
                       e.embedding as embedding
            """).data()
            
            if not entities_data:
                return []
            
            # Create query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate similarities
            similarities = []
            for entity in entities_data:
                entity_embedding = entity['embedding']
                if entity_embedding:
                    # Calculate cosine similarity
                    query_norm = np.linalg.norm(query_embedding)
                    entity_norm = np.linalg.norm(entity_embedding)
                    
                    if query_norm > 0 and entity_norm > 0:
                        similarity = np.dot(query_embedding, entity_embedding) / (query_norm * entity_norm)
                        similarities.append((entity, similarity))
            
            # Sort by similarity and return top entities
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            relevant_entities = []
            for entity, similarity in similarities[:max_entities]:
                entity_data = dict(entity)
                entity_data['similarity'] = similarity
                relevant_entities.append(entity_data)
            
            return relevant_entities
    
    async def _build_mixed_context(
        self, 
        query: str, 
        entities: List[Dict[str, Any]], 
        max_chunks: int, 
        max_communities: int
    ) -> str:
        """Build mixed context from entities, chunks, and communities"""
        
        context_parts = []
        
        with self.graph.driver.session() as session:
            # 1. Entity context
            if entities:
                entity_context = "ENTITIES:\n"
                for entity in entities[:5]:  # Limit to top 5 entities
                    entity_context += f"- {entity['type']}: {entity['name']}\n"
                    entity_context += f"  Description: {entity['description']}\n"
                    entity_context += f"  Relevance: {entity['similarity']:.3f}\n\n"
                context_parts.append(entity_context)
            
            # 2. Relationship context
            if entities:
                entity_names = [e['name'] for e in entities[:3]]  # Top 3 entities
                relationships_data = session.run("""
                    MATCH (e1:__Entity__)-[r:RELATES_TO]->(e2:__Entity__)
                    WHERE e1.name IN $entity_names OR e2.name IN $entity_names
                    RETURN e1.name as source, e2.name as target, r.co_occurrences as strength
                    ORDER BY r.co_occurrences DESC
                    LIMIT $max_relationships
                """, entity_names=entity_names, max_relationships=10).data()
                
                if relationships_data:
                    relationship_context = "RELATIONSHIPS:\n"
                    for rel in relationships_data:
                        relationship_context += f"- {rel['source']} → {rel['target']} (strength: {rel['strength']})\n"
                    context_parts.append(relationship_context)
            
            # 3. Community context
            if entities:
                entity_names = [e['name'] for e in entities[:3]]
                communities_data = session.run("""
                    MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c:__Community__)
                    WHERE e.name IN $entity_names AND c.summary IS NOT NULL
                    RETURN DISTINCT c.id as community_id, c.summary as summary, c.level as level
                    ORDER BY c.level
                    LIMIT $max_communities
                """, entity_names=entity_names, max_communities=max_communities).data()
                
                if communities_data:
                    community_context = "RELATED COMMUNITIES:\n"
                    for comm in communities_data:
                        community_context += f"- Community {comm['community_id']} (Level {comm['level']}):\n"
                        community_context += f"  {comm['summary'][:300]}...\n\n"
                    context_parts.append(community_context)
            
            # 4. Text chunks context
            if entities:
                entity_names = [e['name'] for e in entities[:3]]
                chunks_data = session.run("""
                    MATCH (e:__Entity__)<-[:HAS_ENTITY]-(c:Chunk)
                    WHERE e.name IN $entity_names
                    RETURN DISTINCT c.text as text, c.source as source, c.index as chunk_index
                    ORDER BY c.index
                    LIMIT $max_chunks
                """, entity_names=entity_names, max_chunks=max_chunks).data()
                
                if chunks_data:
                    chunks_context = "RELEVANT TEXT CHUNKS:\n"
                    for i, chunk in enumerate(chunks_data, 1):
                        chunks_context += f"{i}. Source: {chunk['source']}\n"
                        chunks_context += f"   {chunk['text'][:400]}...\n\n"
                    context_parts.append(chunks_context)
        
        return "\n".join(context_parts)
    
    async def _generate_response(self, query: str, context_data: str) -> Tuple[str, Dict[str, int]]:
        """Generate response using the mixed context"""
        
        # Format prompt
        messages = self.local_prompt.format_messages(
            context_data=context_data,
            query=query
        )
        
        # Calculate token counts
        prompt_text = str(messages)
        prompt_tokens = len(prompt_text) // 4
        
        # Generate response
        response = await self.llm.ainvoke(messages)
        answer = response.content.strip()
        
        output_tokens = len(answer) // 4
        
        return answer, {
            'prompt_tokens': prompt_tokens,
            'output_tokens': output_tokens
        }

class GraphRAGHybridRetriever:
    """Hybrid retriever that automatically selects between global and local search"""
    
    def __init__(self, graph_processor: AdvancedGraphProcessor):
        self.graph = graph_processor
        self.classifier = QueryClassifier(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        self.global_retriever = GraphRAGGlobalRetriever(graph_processor)
        self.local_retriever = GraphRAGLocalRetriever(graph_processor)
    
    async def search(self, query: str, **kwargs) -> SearchResult:
        """Automatically route query to appropriate search method"""
        
        # Classify query
        query_type = await self.classifier.classify(query)
        
        print(f"🎯 Query classified as: {query_type}")
        
        if query_type == "GLOBAL":
            result = await self.global_retriever.search(query, **kwargs)
            result.method = "graphrag_hybrid_global"
        else:
            result = await self.local_retriever.search(query, **kwargs)
            result.method = "graphrag_hybrid_local"
        
        return result

# Main interface function for integration with benchmark
def create_advanced_graphrag_retriever(
    graph_processor: AdvancedGraphProcessor, 
    mode: str = "global"
) -> Union[GraphRAGGlobalRetriever, GraphRAGLocalRetriever, GraphRAGHybridRetriever]:
    """Factory function to create appropriate retriever"""
    
    if mode == "global":
        return GraphRAGGlobalRetriever(graph_processor)
    elif mode == "local":
        return GraphRAGLocalRetriever(graph_processor)
    elif mode == "hybrid":
        return GraphRAGHybridRetriever(graph_processor)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'global', 'local', or 'hybrid'")

# Main integration function for benchmark system
async def query_advanced_graphrag(query: str, mode: str = "hybrid", k: int = 5, **kwargs) -> Dict[str, Any]:
    """
    Advanced GraphRAG retrieval with intelligent mode selection
    
    Args:
        query: The search query
        mode: 'global', 'local', or 'hybrid' (default: 'hybrid' for automatic selection)
        k: Number of communities/entities to consider
        **kwargs: Additional configuration options
    
    Returns:
        Dictionary with response and retrieval details
    """
    
    # Initialize graph processor
    from data_processors import AdvancedGraphProcessor
    processor = AdvancedGraphProcessor()
    
    try:
        # Create hybrid retriever (handles all modes intelligently)
        retriever = GraphRAGHybridRetriever(processor)
        
        # Override mode if explicitly specified
        if mode in ["global", "local"]:
            if mode == "global":
                retriever = GraphRAGGlobalRetriever(processor)
            else:
                retriever = GraphRAGLocalRetriever(processor)
        
        # Perform search
        result = await retriever.search(query, max_communities=k, **kwargs)
        
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
        print(f"Error in Advanced GraphRAG retrieval: {e}")
        import traceback
        traceback.print_exc()
        return {
            'final_answer': f"Error during Advanced GraphRAG retrieval: {str(e)}",
            'retrieval_details': [],
            'method': f'advanced_graphrag_{mode}_error',
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



if __name__ == "__main__":
    # Test the retriever
    import asyncio
    
    async def test_retriever():
        """Test function for the GraphRAG retriever"""
        
        # Test queries
        test_queries = [
            "What are the main themes in the documents?",  # Global query
            "What specific requirements did NovaGrid mention?",  # Local query
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing query: {query}")
            print('='*60)
            
            # Test global mode
            result = await query_advanced_graphrag(query, mode="global")
            print(f"Global mode result: {result['final_answer'][:200]}...")
            
            # Test hybrid mode
            result = await query_advanced_graphrag(query, mode="hybrid")
            print(f"Hybrid mode result: {result['final_answer'][:200]}...")
    
    # Uncomment to test
    # asyncio.run(test_retriever()) 