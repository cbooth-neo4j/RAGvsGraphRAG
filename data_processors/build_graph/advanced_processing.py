"""
Advanced Processing Mixin for Element Summarization and Community Detection

This mixin adds advanced capabilities to the CustomGraphProcessor:
1. Element summarization for enhanced entity descriptions
2. Community detection and summarization for hierarchical graph structure
3. Cost estimation and user confirmation
Supports configurable LLM models.
"""

import os
import sys
import json
from typing import List, Dict, Tuple, Any, Optional
from graphdatascience import GraphDataScience
from pydantic import BaseModel, Field 
from langchain_core.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass
from math import ceil
import pandas as pd
import numpy as np
from langchain_core.output_parsers import StrOutputParser

# Import centralized configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_llm

@dataclass
class ElementSummary:
    """Data class for element summaries"""
    name: str
    type: str  # 'entity' or 'relationship'
    description: str
    element_id: str
    
class EntitySummary(BaseModel):
    """Structured output for entity summary generation"""
    summary: str = Field(description="A comprehensive summary of the entity based on its relationships and mentions")

class CommunityReport(BaseModel):
    """Structured output for community report generation"""
    title: str = Field(description="A descriptive title for this community")
    summary: str = Field(description="A comprehensive summary of the community's main themes and relationships")
    rating: float = Field(description="A numeric rating of the community's importance (0-10)")

class AdvancedProcessingMixin:
    """Mixin providing advanced graph processing capabilities with configurable models"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.element_summarization_enabled = True
        self.community_summarization_enabled = True
        # Use configurable LLM
        self.llm = get_llm()
    
    # ==================== ELEMENT SUMMARIZATION ====================
    
    def batch_summarize_entities(self, entity_summaries: Dict[str, ElementSummary], max_workers: int = 3) -> Dict[str, str]:
        """
        Process entity summaries in batches using multithreading
        """
        if not self.element_summarization_enabled:
            return {}
            
        summaries = {}
        
        # Create batches
        items = list(entity_summaries.items())
        batch_size = 10
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        print(f"📝 Processing {len(items)} entity summaries in {len(batches)} batches...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_entity_batch, batch): batch_idx 
                for batch_idx, batch in enumerate(batches)
            }
            
            # Process completed batches
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Entity summarization"):
                batch_idx = future_to_batch[future]
                try:
                    batch_summaries = future.result()
                    summaries.update(batch_summaries)
                except Exception as e:
                    print(f"⚠️ Error processing batch {batch_idx}: {e}")
        
        return summaries
    
    def _process_entity_batch(self, batch: List[Tuple[str, ElementSummary]]) -> Dict[str, str]:
        """Process a batch of entities for summarization"""
        batch_summaries = {}
        
        for entity_name, element_summary in batch:
            try:
                summary = self._generate_entity_summary(element_summary)
                if summary:
                    batch_summaries[entity_name] = summary
            except Exception as e:
                print(f"⚠️ Error summarizing entity {entity_name}: {e}")
        
        return batch_summaries
    
    def _generate_entity_summary(self, element_summary: ElementSummary) -> Optional[str]:
        """Generate a summary for a single entity using LLM"""
        if not element_summary.description.strip():
            return None
            
        prompt = ChatPromptTemplate.from_template("""
        You are an expert analyst tasked with creating comprehensive summaries of entities based on their relationships and context.
        
        Entity: {entity_name}
        Type: {entity_type}
        Context: {description}
        
        Create a concise but comprehensive summary of this entity that captures:
        1. What this entity is/represents
        2. Its key relationships and connections
        3. Its significance in the overall context
        
        Keep the summary factual, informative, and under 200 words.
        """)
        
        try:
            chain = prompt | self.llm.with_structured_output(EntitySummary)
            result = chain.invoke({
                "entity_name": element_summary.name,
                "entity_type": element_summary.type,
                "description": element_summary.description
            })
            return result.summary
        except Exception as e:
            print(f"⚠️ LLM error for entity {element_summary.name}: {e}")
            return None
    
    def perform_element_summarization(self, summarize_entities: bool = True, summarize_relationships: bool = False):
        """
        Perform element summarization on entities and relationships
        """
        if not self.element_summarization_enabled:
            print("⚠️ Element summarization is disabled")
            return
            
        print("🔍 Starting element summarization...")
        
        if summarize_entities:
            # Get entities and their context
            entity_query = """
            MATCH (e:__Entity__)
            OPTIONAL MATCH (e)-[r]-(connected)
            WITH e, 
                 collect(DISTINCT type(r) + ': ' + coalesce(connected.name, connected.text, connected.id, 'unknown')) as relationships,
                 collect(DISTINCT coalesce(connected.name, connected.text, connected.id)) as connected_entities
            RETURN e.name as entity_name,
                   e.description as current_description,
                   relationships,
                   connected_entities,
                   elementId(e) as element_id
            """
            
            results = list(self.driver.execute_query(entity_query).records)
            print(f"📊 Found {len(results)} entities to summarize")
            
            # Prepare summaries
            entity_summaries = {}
            for record in results:
                entity_name = record['entity_name']
                relationships = record['relationships'] or []
                connected_entities = record['connected_entities'] or []
                current_desc = record['current_description'] or ""
                
                # Build context description
                context_parts = []
                if current_desc:
                    context_parts.append(f"Current description: {current_desc}")
                if relationships:
                    context_parts.append(f"Relationships: {', '.join(relationships[:10])}")  # Limit to avoid token overflow
                if connected_entities:
                    context_parts.append(f"Connected to: {', '.join(connected_entities[:10])}")
                
                context = " | ".join(context_parts)
                
                entity_summaries[entity_name] = ElementSummary(
                    name=entity_name,
                    type="entity",
                    description=context,
                    element_id=record['element_id']
                )
            
            # Generate summaries
            summaries = self.batch_summarize_entities(entity_summaries)
            
            # Update database
            if summaries:
                print(f"💾 Updating {len(summaries)} entity descriptions...")
                for entity_name, summary in summaries.items():
                    update_query = """
                    MATCH (e:Entity {name: $entity_name})
                    SET e.ai_summary = $summary,
                        e.summary_generated_at = datetime()
                    """
                    self.driver.execute_query(update_query, entity_name=entity_name, summary=summary)
                
                print(f"✅ Updated {len(summaries)} entity summaries")
    
    # ==================== COMMUNITY DETECTION ====================
    
    def retry_community_detection_only(self, max_levels: int = 3):
        """
        Retry just the community detection step without running the full pipeline.
        Useful for debugging Leiden parameters after data is already processed.
        """
        print("🔄 Retrying community detection on existing graph...")
        
        # Check if we have entities
        with self.driver.session() as session:
            result = session.run("MATCH (e:__Entity__) RETURN count(e) as count")
            entity_count = result.single()['count']
            
            if entity_count == 0:
                print("❌ No entities found. Run data processing first.")
                return
                
            print(f"📊 Found {entity_count} entities to cluster")
            
            # Clear existing communities first
            print("🗑️ Clearing existing communities...")
            session.run("MATCH (c:__Community__) DETACH DELETE c")
        
        # Run community detection
        self.perform_community_detection(max_levels)
    
    def retry_community_summaries_only(self, max_levels: int = 3):
        """
        Retry just the community summarization step.
        Useful when communities exist but summaries failed.
        """
        print("🔄 Retrying community summarization...")
        
        # Check if we have communities
        with self.driver.session() as session:
            result = session.run("MATCH (c:__Community__) RETURN count(c) as count")
            community_count = result.single()['count']
            
            if community_count == 0:
                print("❌ No communities found. Run community detection first.")
                print("💡 Try: processor.retry_community_detection_only()")
                return
                
            print(f"📊 Found {community_count} communities to summarize")
        
        # Run community summarization
        if self.community_summarization_enabled:
            summaries_created = self.generate_community_summaries_hierarchical(max_levels)
            print(f"✅ Created {summaries_created} community summaries")
        else:
            print("⚠️ Community summarization is disabled")
            print("💡 Enable with: processor.community_summarization_enabled = True")
    
    def perform_community_detection(self, max_levels: int = 3):
        """
        Perform community detection using GDS Leiden algorithm (Neo4j LLM Graph Builder approach).
        Let GDS handle the heavy lifting with default parameters - no custom min_community_size.
        """
        print("🏘️ Starting hierarchical community detection...")
        
        # Ensure GDS is available
        if not hasattr(self, 'gds') or self.gds is None:
            self.gds = GraphDataScience(self.driver)
        
        # Create graph projection for community detection
        projection_name = "entity_relationships"
        
        # Drop existing projection if it exists
        try:
            self.gds.graph.drop(projection_name)
        except:
            pass
        
        # Create new projection - focus on entities only for proper community detection
        try:
            G, _ = self.gds.graph.project(
                projection_name,
                "__Entity__",  # Only entities for community detection
                {
                    "RELATED_TO": {"orientation": "UNDIRECTED"}  # Use the actual relationship type that exists
                }
            )
            print(f"📊 Created graph projection with {G.node_count()} nodes and {G.relationship_count()} relationships")
        except Exception as e:
            print(f"⚠️ Error creating graph projection: {e}")
            return
        
        try:
            print(f"🔍 Running Leiden algorithm...")
            
            # Run Leiden algorithm with GDS defaults (Neo4j LLM Graph Builder approach)
            # Note: Removed relationshipWeightProperty since relationships don't have weight property
            result = self.gds.leiden.write(
                G,
                writeProperty="communityId",  # Match Neo4j LLM Graph Builder naming
                includeIntermediateCommunities=True  # Enable hierarchy
            )
            
            total_communities = result['communityCount']
            levels_created = result.get('levels', max_levels)
            
            print(f"   ✅ Found {total_communities} communities across {levels_created} hierarchical levels")
            
            # Clean up projection
            try:
                self.gds.graph.drop(projection_name)
            except:
                pass
            
            if total_communities > 0:
                print(f"✅ Community detection completed. Processing hierarchical structure...")
                
                # Create community nodes with proper hierarchy
                self.create_hierarchical_community_nodes(max_levels)
                
                # Generate community summaries if enabled
                if self.community_summarization_enabled:
                    self.generate_community_summaries_hierarchical(max_levels)
            else:
                print("⚠️ No communities were created")
                
        except Exception as e:
            print(f"⚠️ Error in Leiden algorithm: {e}")
            # Clean up projection on error
            try:
                self.gds.graph.drop(projection_name)
            except:
                pass
    
    def create_hierarchical_community_nodes(self, max_levels: int):
        """Create hierarchical Community nodes based on official Neo4j GraphRAG approach"""
        print("🏗️ Creating hierarchical community nodes and relationships...")
        
        # Use the official Neo4j GraphRAG approach from their blog post
        # Reference: https://neo4j.com/blog/developer/global-graphrag-neo4j-langchain/
        
        # First, check if entities have the communityId property
        check_query = """
        MATCH (e:`__Entity__`) 
        WHERE e.communityId IS NOT NULL 
        RETURN count(e) as entities_with_communities, 
               collect(e.communityId)[0..3] as sample_community_ids
        """
        
        with self.driver.session() as session:
            result = session.run(check_query)
            record = result.single()
            if record['entities_with_communities'] == 0:
                print("   ⚠️ No entities have communityId property - Leiden algorithm may have failed")
                return
            print(f"   📊 Found {record['entities_with_communities']} entities with community assignments")
            print(f"   📋 Sample community IDs: {record['sample_community_ids']}")
        
        # Official Neo4j GraphRAG community materialization approach
        # Split into separate queries to avoid Cypher CALL subquery limitations
        
        # First, create level 0 communities and connect entities
        level_0_query = """
        MATCH (e:`__Entity__`)
        WHERE e.communityId IS NOT NULL AND size(e.communityId) > 0
        WITH e, e.communityId[0] as community_id
        MERGE (c:`__Community__` {id: '0-' + toString(community_id)})
        ON CREATE SET c.level = 0, c.community_id = community_id
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """
        
        # Then create higher level communities and hierarchy
        hierarchy_query = """
        MATCH (e:`__Entity__`)
        WHERE e.communityId IS NOT NULL AND size(e.communityId) > 1
        UNWIND range(1, size(e.communityId) - 1) AS idx
        WITH e, idx, e.communityId[idx] as current_id, e.communityId[idx-1] as parent_id
        MERGE (cur:`__Community__` {id: toString(idx) + '-' + toString(current_id)})
        ON CREATE SET cur.level = idx, cur.community_id = current_id
        MERGE (prev:`__Community__` {id: toString(idx-1) + '-' + toString(parent_id)})
        MERGE (prev)-[:PARENT_COMMUNITY]->(cur)
        """
        
        try:
            # Execute level 0 community creation
            result1 = self.driver.execute_query(level_0_query)
            print("   ✅ Created level 0 communities and entity connections")
            
            # Execute hierarchy creation
            result2 = self.driver.execute_query(hierarchy_query)
            print("   ✅ Created hierarchical community structure")
            
        except Exception as e:
            print(f"   ⚠️ Error creating communities: {e}")
            return
        
        # Add community ranking based on document mentions (Neo4j GraphRAG approach)
        self.add_community_ranking()
        
        print("✅ Hierarchical community nodes and relationships created")
    
    def add_community_ranking(self):
        """Add community ranking based on document mentions (Neo4j GraphRAG approach)"""
        print("📊 Adding community ranking...")
        
        try:
            # First, rank level 0 communities based on chunk connections (Neo4j LLM Graph Builder approach)
            # This will rank ALL communities that have entities with chunk connections
            level_0_ranking_query = """
            MATCH (c:__Community__ {level: 0})<-[:IN_COMMUNITY]-(e:__Entity__)<-[:HAS_ENTITY]-(chunk:Chunk)
            WITH c, count(distinct chunk) AS chunkCount
            SET c.community_rank = chunkCount,
                c.weight = chunkCount
            """
            
            result1 = self.driver.execute_query(level_0_ranking_query)
            print("   ✅ Level 0 community rankings assigned")
            
            # Then, rank higher level communities level by level to avoid race conditions
            # Process each level sequentially so inheritance works correctly
            for level in range(1, 4):  # Process levels 1, 2, 3
                level_ranking_query = f"""
                MATCH (parent:__Community__ {{level: {level}}})
                OPTIONAL MATCH (child:__Community__)-[:PARENT_COMMUNITY]->(parent)
                WITH parent, sum(coalesce(child.community_rank, 0)) as inherited_rank
                SET parent.community_rank = inherited_rank,
                    parent.weight = inherited_rank
                """
                
                result = self.driver.execute_query(level_ranking_query)
                print(f"   ✅ Level {level} community rankings inherited from children")
            
        except Exception as e:
            print(f"   ⚠️ Error adding community ranking: {e}")
    
    def create_community_hierarchy_proper(self, max_levels: int):
        """Create proper hierarchical relationships between community levels (Neo4j LLM Graph Builder style)"""
        print("🔗 Creating proper community hierarchy...")
        
        # The key insight: Leiden with includeIntermediateCommunities creates a hierarchical structure
        # where higher levels are aggregations of lower levels
        
        # Method 1: Use the actual Leiden hierarchical results
        hierarchy_query = """
        MATCH (e:__Entity__)
        WHERE e.community_id IS NOT NULL
        
        // Get all community levels this entity belongs to
        WITH e, e.community_id as community_array
        UNWIND range(0, $max_levels - 1) as level
        WITH e, level, 
             CASE 
                 WHEN level = 0 THEN toString(community_array[0])
                 WHEN level = 1 THEN toString(community_array[1])
                 WHEN level = 2 THEN toString(community_array[2])
                 ELSE toString(community_array[0])
             END as community_at_level
        WHERE community_at_level IS NOT NULL AND community_at_level <> ""
        
        // Find the community nodes for each level
        MATCH (c:__Community__ {level: level})
        WHERE c.community_id = community_at_level
        
        // Create entity to community relationships
        MERGE (e)-[:IN_COMMUNITY]->(c)
        
        // Skip complex hierarchy creation for now - Leiden provides natural hierarchy
        WITH c
        RETURN count(c) as communities_created
        """
        
        try:
            result = self.driver.execute_query(hierarchy_query, max_levels=max_levels)
            print("   ✅ Created hierarchical community structure based on Leiden results")
        except Exception as e:
            print(f"   ⚠️ Error creating community hierarchy: {e}")
        
        # Add community ranking and weights
        ranking_query = """
        MATCH (c:__Community__)
        SET c.community_rank = 
            CASE c.level
                WHEN 0 THEN c.member_count * 1.0      
                WHEN 1 THEN c.member_count * 2.0      
                WHEN 2 THEN c.member_count * 3.0      
                ELSE c.member_count * (c.level + 1.0) 
            END,
            c.weight = toFloat(c.member_count) / 10.0
        """
        
        try:
            self.driver.execute_query(ranking_query)
            print("   ✅ Community ranking and weights assigned")
        except Exception as e:
            print(f"   ⚠️ Error assigning community rankings: {e}")
        
        print("✅ Proper community hierarchy created successfully")
    
    def generate_community_summaries_hierarchical(self, max_levels: int) -> int:
        """Generate AI summaries for hierarchical communities (Neo4j LLM Graph Builder style)"""
        if not self.community_summarization_enabled:
            print("⚠️ Community summarization is disabled")
            return 0
        
        print("📝 Generating hierarchical community summaries...")
        
        total_summaries = 0
        
        # Process each level
        for level in range(max_levels):
            print(f"   Processing summaries for level {level}...")
            
            # Get communities for this level with proper entity aggregation
            if level == 0:
                # Level 0: Direct entity connections
                communities_query = f"""
                MATCH (c:__Community__ {{level: {level}}})
                OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, collect({{name: e.name, description: coalesce(e.description, e.ai_summary, e.name)}}) as entities
                WHERE size(entities) > 0
                RETURN c.id as community_id, 
                       c.level as level,
                       c.member_count as member_count,
                       entities
                ORDER BY coalesce(c.community_rank, 0) DESC
                """
            else:
                # Level 1+: Aggregate entities from child communities
                communities_query = f"""
                MATCH (c:__Community__ {{level: {level}}})
                OPTIONAL MATCH (child:__Community__)-[:PARENT_COMMUNITY]->(c)
                OPTIONAL MATCH (child)<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, collect(DISTINCT {{name: e.name, description: coalesce(e.description, e.ai_summary, e.name)}}) as entities
                WHERE size(entities) > 0
                RETURN c.id as community_id, 
                       c.level as level,
                       size(entities) as member_count,
                       entities
                ORDER BY coalesce(c.community_rank, 0) DESC
                """
            
            try:
                result = self.driver.execute_query(communities_query)
                communities = result.records
                
                if not communities:
                    print(f"     No communities found for level {level}")
                    continue
                
                print(f"     Found {len(communities)} communities to summarize at level {level}")
                
                # Generate summaries in batches with progress bar
                level_summaries = self._generate_community_summaries_batch(communities, level)
                total_summaries += level_summaries
                
            except Exception as e:
                print(f"     ⚠️ Error processing level {level}: {e}")
        
        print(f"✅ Generated {total_summaries} community summaries across {max_levels} levels")
        return total_summaries
    
    def _generate_community_summaries_batch(self, communities: list, level: int) -> int:
        """Generate summaries for a batch of communities"""
        summaries_created = 0
        
        # Create progress bar for this level
        progress_bar = tqdm(communities, desc=f"     Level {level} summaries", 
                           unit="community", leave=False, 
                           bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for community in progress_bar:
            try:
                community_id = community['community_id']
                entities = community['entities']
                member_count = community['member_count']
                
                # Create context from entities
                entity_context = "\n".join([
                    f"- {entity['name']}: {entity['description']}" 
                    for entity in entities[:20]  # Limit to prevent token overflow
                ])
                
                # Generate summary using LLM
                summary_prompt = f"""
                Analyze this community of {member_count} entities at hierarchical level {level} and provide:
                1. A descriptive title (2-4 words)
                2. A comprehensive summary (2-3 sentences)
                3. An importance rating (0-10) based on entity relationships and diversity
                
                Entities in this community:
                {entity_context}
                
                Respond in JSON format:
                {{
                    "title": "Community Title",
                    "summary": "Detailed summary of the community's main themes and relationships",
                    "rating": 7.5
                }}
                """
                
                try:
                    response = self.llm.invoke(summary_prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    # Enhanced JSON parsing for Ollama compatibility
                    import json
                    import re
                    
                    # Try to extract JSON from response if it's wrapped in text
                    content = content.strip()
                    
                    # Look for JSON object in the response
                    json_match = re.search(r'(\{.*?\})', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                    
                    # If still no valid JSON, try to extract from code blocks
                    if not content.startswith('{') and not content.startswith('['):
                        json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
                        if json_blocks:
                            content = json_blocks[0]
                    
                    summary_data = json.loads(content)
                    
                    # Update community with summary
                    update_query = """
                    MATCH (c:__Community__ {id: $community_id})
                    SET c.title = $title,
                        c.summary = $summary,
                        c.rating = $rating,
                        c.summarized_at = datetime()
                    """
                    
                    self.driver.execute_query(
                        update_query,
                        community_id=community_id,
                        title=summary_data.get('title', f'Community {community_id}'),
                        summary=summary_data.get('summary', 'No summary available'),
                        rating=float(summary_data.get('rating', 5.0))
                    )
                    
                    summaries_created += 1
                    # Update progress bar description with success count
                    progress_bar.set_postfix({"✅": summaries_created})
                    
                except json.JSONDecodeError:
                    progress_bar.write(f"     ⚠️ Could not parse JSON response for community {community_id}")
                except Exception as e:
                    progress_bar.write(f"     ⚠️ Error generating summary for community {community_id}: {e}")
                    
            except Exception as e:
                progress_bar.write(f"     ⚠️ Error processing community: {e}")
        
        # Close progress bar
        progress_bar.close()
        
        return summaries_created
    
    def create_community_hierarchy(self):
        """Create hierarchical relationships between community levels (Neo4j LLM Graph Builder style)"""
        print("🔗 Creating community hierarchy...")
        
        # Create parent-child relationships between community levels
        hierarchy_query = """
        MATCH (child:__Community__)
        MATCH (parent:__Community__)
        WHERE child.level = parent.level + 1
        MATCH (child)<-[:BELONGS_TO]-(member)
        MATCH (member)-[:BELONGS_TO]->(parent)
        WITH child, parent, count(member) as shared_members
        WHERE shared_members > 0
        MERGE (child)-[r:PART_OF]->(parent)
        SET r.shared_members = shared_members
        RETURN count(r) as hierarchical_relationships
        """
        
        result = self.driver.execute_query(hierarchy_query)
        if result.records:
            count = result.records[0]['hierarchical_relationships']
            print(f"   Created {count} hierarchical relationships between community levels")
        
        # Also create summary relationships for easier traversal
        summary_hierarchy_query = """
        MATCH (c:__Community__)
        WITH c.level as level, collect(c) as communities_at_level
        ORDER BY level
        UNWIND range(0, size(communities_at_level)-2) as i
        WITH communities_at_level[i] as current_level_communities, communities_at_level[i+1] as next_level_communities
        UNWIND current_level_communities as child
        UNWIND next_level_communities as parent
        WHERE child.level = parent.level - 1
        MERGE (child)-[:CHILD_OF]->(parent)
        """
        
        try:
            self.driver.execute_query(summary_hierarchy_query)
            print("   Created CHILD_OF relationships for community hierarchy")
        except Exception as e:
            print(f"   Warning: Could not create CHILD_OF relationships: {e}")
    
    def generate_community_summaries(self, max_levels: List[int] = [0, 1, 2], min_community_size: int = 2) -> int:
        """Generate AI summaries for detected communities"""
        if not self.community_summarization_enabled:
            print("⚠️ Community summarization is disabled")
            return 0
            
        print("📝 Generating community summaries...")
        
        # Get communities that need summaries
        communities_query = """
        MATCH (c:__Community__)
        WHERE c.ai_summary IS NULL
        MATCH (c)<-[:BELONGS_TO]-(member)
        WITH c, collect(member) as members
        WHERE size(members) >= $min_size
        RETURN c.id as community_id,
               c.level as level,
               elementId(c) as element_id,
               members
        ORDER BY c.level, size(members) DESC
        """
        
        communities = list(self.driver.execute_query(
            communities_query, 
            min_size=min_community_size
        ).records)
        
        if not communities:
            print("ℹ️ No communities need summaries")
            return 0
        
        print(f"📊 Processing {len(communities)} communities...")
        
        summaries_generated = 0
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all communities for processing
            future_to_community = {
                executor.submit(self._process_community, {
                    'community_id': record['community_id'],
                    'level': record['level'],
                    'element_id': record['element_id'],
                    'members': record['members']
                }): record['community_id'] 
                for record in communities
            }
            
            # Process completed summaries
            for future in tqdm(as_completed(future_to_community), total=len(communities), desc="Community summaries"):
                community_id = future_to_community[future]
                try:
                    result = future.result()
                    if result:
                        # Update database with summary
                        update_query = """
                        MATCH (c:__Community__ {id: $community_id})
                        SET c.ai_summary = $summary,
                            c.title = $title,
                            c.importance_rating = $rating,
                            c.summary_generated_at = datetime()
                        """
                        self.driver.execute_query(
                            update_query,
                            community_id=community_id,
                            summary=result['summary'],
                            title=result['title'],
                            rating=result['rating']
                        )
                        summaries_generated += 1
                except Exception as e:
                    print(f"⚠️ Error processing community {community_id}: {e}")
        
        print(f"✅ Generated {summaries_generated} community summaries")
        return summaries_generated
    
    def _process_community(self, community_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single community for summary generation"""
        try:
            community_string = self._prepare_community_string(community_data)
            
            prompt = ChatPromptTemplate.from_template("""
            You are an expert analyst tasked with understanding and summarizing communities of related entities.
            
            Community Level: {level}
            Community Data: {community_data}
            
            Analyze this community and provide:
            1. A descriptive title that captures the main theme
            2. A comprehensive summary of what this community represents
            3. An importance rating (0-10) based on the relationships and entities involved
            4. An explanation for the importance rating
            
            Focus on the relationships, patterns, and significance of this grouping.
            """)
            
            chain = prompt | self.llm.with_structured_output(CommunityReport)
            result = chain.invoke({
                "level": community_data['level'],
                "community_data": community_string
            })
            
            return {
                "summary": result.summary,
                "title": result.title,
                "rating": result.rating
            }
            
        except Exception as e:
            print(f"⚠️ Error generating summary for community {community_data['community_id']}: {e}")
            return None
    
    def _prepare_community_string(self, data: Dict[str, Any]) -> str:
        """Prepare a string representation of community data for LLM processing"""
        members = data['members']
        
        # Group members by type
        entities = []
        chunks = []
        
        for member in members:
            if 'Entity' in member.labels:
                name = member.get('name', 'Unknown')
                desc = member.get('description', member.get('ai_summary', ''))
                entities.append(f"{name}: {desc[:100]}..." if desc else name)
            elif 'Chunk' in member.labels:
                text = member.get('text', '')[:150] + "..." if len(member.get('text', '')) > 150 else member.get('text', '')
                chunks.append(text)
        
        parts = []
        if entities:
            parts.append(f"Entities ({len(entities)}): " + " | ".join(entities[:10]))  # Limit for token management
        if chunks:
            parts.append(f"Text Chunks ({len(chunks)}): " + " | ".join(chunks[:5]))  # Fewer chunks due to size
        
        return " || ".join(parts)
    
    # ==================== COST ESTIMATION ====================
    
    def estimate_processing_costs(self, stats: Dict[str, int]) -> Dict[str, Any]:
        """Estimate costs for advanced processing"""
        
        # Token cost estimates (rough)
        entity_summary_tokens_per_entity = 300  # input + output
        community_summary_tokens_per_community = 500
        cost_per_1k_tokens = 0.00015  # Approximate pricing (varies by model)
        
        # Entity summarization costs
        entity_cost = 0
        if stats.get('entities', 0) > 0:
            entity_tokens = stats['entities'] * entity_summary_tokens_per_entity
            entity_cost = (entity_tokens / 1000) * cost_per_1k_tokens
        
        # Community detection (estimate communities based on entities)
        estimated_communities = max(1, stats.get('entities', 0) // 5)  # Rough estimate
        community_tokens = estimated_communities * community_summary_tokens_per_community
        community_cost = (community_tokens / 1000) * cost_per_1k_tokens
        
        total_cost = entity_cost + community_cost
        
        return {
            "entity_summarization": {
                "entities": stats.get('entities', 0),
                "estimated_tokens": stats.get('entities', 0) * entity_summary_tokens_per_entity,
                "estimated_cost": entity_cost
            },
            "community_detection": {
                "estimated_communities": estimated_communities,
                "estimated_tokens": community_tokens,
                "estimated_cost": community_cost
            },
            "total_estimated_cost": total_cost,
            "currency": "USD"
        }
    
    def get_user_confirmation_for_costs(self, costs: Dict[str, Any]) -> bool:
        """Get user confirmation before proceeding with costly operations"""
        
        print("\n" + "="*60)
        print("💰 ADVANCED PROCESSING COST ESTIMATE")
        print("="*60)
        
        entity_info = costs["entity_summarization"]
        community_info = costs["community_detection"]
        
        print(f"📝 Entity Summarization:")
        print(f"   • {entity_info['entities']} entities to summarize")
        print(f"   • ~{entity_info['estimated_tokens']:,} tokens")
        print(f"   • ~${entity_info['estimated_cost']:.4f}")
        
        print(f"\n🏘️ Community Detection & Summarization:")
        print(f"   • ~{community_info['estimated_communities']} communities expected")
        print(f"   • ~{community_info['estimated_tokens']:,} tokens")
        print(f"   • ~${community_info['estimated_cost']:.4f}")
        
        print(f"\n💸 TOTAL ESTIMATED COST: ${costs['total_estimated_cost']:.4f}")
        print("="*60)
        
        if costs['total_estimated_cost'] < 0.10:  # Auto-approve small costs
            print("✅ Cost is minimal, proceeding automatically...")
            return True
        
        while True:
            response = input("\nProceed with advanced processing? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    # ==================== MAIN PROCESSING METHOD ====================
    
    def perform_advanced_processing(self, stats: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Perform all advanced processing with cost estimation and user confirmation
        """
        print("🚀 Starting Advanced Graph Processing...")
        
        # Get current stats if not provided
        if stats is None:
            stats = self.get_graph_statistics()
        
        # Estimate costs
        costs = self.estimate_processing_costs(stats)
        
        # Get user confirmation
        if not self.get_user_confirmation_for_costs(costs):
            print("❌ Advanced processing cancelled by user")
            return {"status": "cancelled", "reason": "user_declined"}
        
        results = {
            "status": "completed",
            "costs": costs,
            "element_summarization": {},
            "community_detection": {}
        }
        
        try:
            # Perform element summarization
            print("\n🔍 Starting element summarization...")
            self.perform_element_summarization(summarize_entities=True)
            results["element_summarization"]["status"] = "completed"
            
            # Perform community detection
            print("\n🏘️ Starting community detection...")
            self.perform_community_detection()
            results["community_detection"]["status"] = "completed"
            
            print("\n✅ Advanced processing completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error during advanced processing: {e}")
            results["status"] = "error"
            results["error"] = str(e)
        
        return results
    
    def get_graph_statistics(self) -> Dict[str, int]:
        """Get current graph statistics for cost estimation"""
        stats_query = """
        MATCH (e:Entity) WITH count(e) as entities
        MATCH (c:Chunk) WITH entities, count(c) as chunks
        MATCH (d:Document) WITH entities, chunks, count(d) as documents
        MATCH ()-[r]->() WITH entities, chunks, documents, count(r) as relationships
        RETURN entities, chunks, documents, relationships
        """
        
        result = self.driver.execute_query(stats_query)
        if result.records:
            record = result.records[0]
            return {
                "entities": record["entities"],
                "chunks": record["chunks"],
                "documents": record["documents"],
                "relationships": record["relationships"]
            }
        else:
            return {
                "entities": 0,
                "chunks": 0,
                "documents": 0,
                "relationships": 0
            }
