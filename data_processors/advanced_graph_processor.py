"""
Advanced Graph Processor for RFP Analysis with Element Summarization and Community Detection

This script enhances an existing graph created by graph_processor.py with:
1. Element summarization capabilities for enhanced entity descriptions
2. Community detection and summarization for hierarchical graph structure

PREREQUISITE: Run graph_processor.py first to create the base graph.
This processor only works on existing graphs and does not build from scratch.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dotenv import load_dotenv
import neo4j
from langchain_openai import ChatOpenAI
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

# Import the basic graph processor
try:
    from .graph_processor import CustomGraphProcessor
except ImportError:
    # Fallback for when running as a direct script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from data_processors.graph_processor import CustomGraphProcessor

# Load environment variables
load_dotenv()

# Configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
LLM = os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')  # Default to gpt-4o-mini if not set

@dataclass
class ElementSummary:
    """Data class for element summaries"""
    entity_id: str
    entity_type: str
    original_descriptions: List[str]
    summarized_description: str
    mention_count: int

class EntitySummary(BaseModel):
    """Individual entity summary"""
    id: str = Field(description="Entity ID/name")
    type: str = Field(description="Entity type")
    enhanced_description: str = Field(description="Enhanced description combining all contexts")

class ElementSummarization(BaseModel):
    """Pydantic model for element summarization responses"""
    summaries: List[EntitySummary] = Field(
        description="List of element summaries with id, type, and enhanced_description"
    )

class AdvancedGraphProcessor(CustomGraphProcessor):
    def __init__(self):
        super().__init__()
        
        # Element summarization components
        self.element_summarization_enabled = False  # Default to disabled due to cost
        self.element_batch_size = 10  # Process entities in batches to reduce LLM calls
        self.summarization_llm = ChatOpenAI(model=LLM, temperature=0.1)
        
        # Element summarization prompt
        element_system_prompt = """You are an expert information analyst. Your task is to create enhanced, comprehensive descriptions for entities and relationships based on multiple mentions across documents.

Rules for summarization:
1. Combine information from multiple descriptions of the same entity
2. Remove redundancy while preserving all unique information
3. Maintain factual accuracy - do not invent details
4. Create concise but comprehensive descriptions (2-3 sentences maximum)
5. Preserve important contextual information and relationships
6. Use professional, clear language
"""
        
        element_user_template = """
Please enhance the descriptions for the following entities. For each entity, you have:
- Entity ID/Name
- Entity Type  
- Multiple descriptions from different document contexts

Create a single, enhanced description that combines and refines all the information:

{entities_batch}

Return a JSON object with this structure:
{{
  "summaries": [
    {{
      "id": "entity_id",
      "type": "entity_type", 
      "enhanced_description": "comprehensive description combining all contexts"
    }}
  ]
}}
"""
        
        self.element_prompt = ChatPromptTemplate.from_messages([
            ("system", element_system_prompt),
            ("human", element_user_template),
        ])
        
        self.element_chain = self.element_prompt | self.summarization_llm.with_structured_output(ElementSummarization)
        
        # Community summarization components
        self.community_summarization_enabled = False  # Default to disabled
        self.community_llm = ChatOpenAI(model=LLM, temperature=0.1)
        
        # Community summarization prompt
        community_system_prompt = """You are an expert analyst specializing in understanding complex networks and relationships. 
Your task is to generate comprehensive summaries of graph communities based on the entities and relationships within them.

Rules for community summarization:
1. Identify the main themes and patterns in the community
2. Describe the key entities and their roles
3. Explain the relationships and connections between entities
4. Highlight any notable characteristics or insights
5. Keep summaries concise but informative (3-5 sentences)
6. Use clear, professional language
"""
        
        community_user_template = """Based on the provided nodes and relationships that belong to the same graph community,
generate a natural language summary of the provided information:

{community_info}

Summary:"""
        
        self.community_prompt = ChatPromptTemplate.from_messages([
            ("system", community_system_prompt),
            ("human", community_user_template),
        ])
        
        self.community_chain = self.community_prompt | self.community_llm | StrOutputParser()
        
    def collect_entity_descriptions(self) -> Dict[str, ElementSummary]:
        """Collect all entity descriptions from the graph for summarization"""
        with self.driver.session() as session:
            # Get all entities with their descriptions and mention counts
            result = session.run("""
                MATCH (e:__Entity__)
                OPTIONAL MATCH (c:Chunk)-[:HAS_ENTITY]->(e)
                WITH e, collect(DISTINCT c.text[0..200]) as chunk_contexts, count(c) as mention_count
                RETURN e.id as entity_id, 
                       e.entity_type as entity_type,
                       e.description as description,
                       chunk_contexts,
                       mention_count
                ORDER BY mention_count DESC
            """).data()
            
            entity_summaries = {}
            for record in result:
                entity_id = record['entity_id']
                if entity_id and record['description']:
                    # Collect all description contexts for this entity
                    descriptions = [record['description']]
                    # Add relevant chunk contexts (first 200 chars to keep token count manageable)
                    if record['chunk_contexts']:
                        descriptions.extend([ctx for ctx in record['chunk_contexts'][:3]])  # Limit to 3 contexts
                    
                    entity_summaries[entity_id] = ElementSummary(
                        entity_id=entity_id,
                        entity_type=record['entity_type'],
                        original_descriptions=descriptions,
                        summarized_description="",  # To be filled by LLM
                        mention_count=record['mention_count'] or 1
                    )
            
            print(f"Collected {len(entity_summaries)} entities for summarization")
            return entity_summaries

    def batch_summarize_entities(self, entity_summaries: Dict[str, ElementSummary], max_workers: int = 3) -> Dict[str, str]:
        """Summarize entities in batches to reduce LLM calls"""
        if not entity_summaries:
            return {}
            
        # Sort entities by mention count (most mentioned first)
        sorted_entities = sorted(
            entity_summaries.items(), 
            key=lambda x: x[1].mention_count, 
            reverse=True
        )
        
        # Create batches
        total_entities = len(sorted_entities)
        num_batches = ceil(total_entities / self.element_batch_size)
        batches = []
        
        for i in range(num_batches):
            start_idx = i * self.element_batch_size
            end_idx = min((i + 1) * self.element_batch_size, total_entities)
            batch_entities = sorted_entities[start_idx:end_idx]
            batches.append(batch_entities)
        
        print(f"Processing {total_entities} entities in {num_batches} batches of {self.element_batch_size}")
        
        # Process batches in parallel
        enhanced_descriptions = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(self._process_entity_batch, batch): batch_idx 
                for batch_idx, batch in enumerate(batches)
            }
            
            # Collect results
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Summarizing entity batches"):
                try:
                    batch_results = future.result()
                    enhanced_descriptions.update(batch_results)
                except Exception as e:
                    batch_idx = future_to_batch[future]
                    print(f"Error processing batch {batch_idx}: {e}")
        
        print(f"Successfully summarized {len(enhanced_descriptions)} entities")
        return enhanced_descriptions

    def _process_entity_batch(self, entity_batch: List[Tuple[str, ElementSummary]]) -> Dict[str, str]:
        """Process a single batch of entities"""
        # Prepare batch data for the LLM
        batch_data = []
        for entity_id, summary in entity_batch:
            entity_info = {
                "id": entity_id,
                "type": summary.entity_type,
                "descriptions": summary.original_descriptions,
                "mention_count": summary.mention_count
            }
            batch_data.append(entity_info)
        
        # Format for the prompt
        entities_text = ""
        for entity in batch_data:
            entities_text += f"\nEntity ID: {entity['id']}\n"
            entities_text += f"Type: {entity['type']}\n"
            entities_text += f"Mentions: {entity['mention_count']}\n"
            entities_text += f"Descriptions:\n"
            for i, desc in enumerate(entity['descriptions'][:5], 1):  # Limit to 5 descriptions
                entities_text += f"  {i}. {desc}\n"
            entities_text += "\n"
        
        try:
            # Call LLM for batch processing
            result = self.element_chain.invoke({"entities_batch": entities_text})
            
            # Extract enhanced descriptions
            enhanced_descriptions = {}
            if result.summaries:
                for summary in result.summaries:
                    entity_id = summary.id
                    enhanced_desc = summary.enhanced_description
                    if entity_id and enhanced_desc:
                        enhanced_descriptions[entity_id] = enhanced_desc
            
            return enhanced_descriptions
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return {}

    def update_entity_descriptions(self, enhanced_descriptions: Dict[str, str]):
        """Update entity descriptions in the graph with enhanced summaries"""
        if not enhanced_descriptions:
            return
            
        with self.driver.session() as session:
            updated_count = 0
            for entity_id, enhanced_desc in enhanced_descriptions.items():
                try:
                    result = session.run("""
                        MATCH (e:__Entity__ {id: $entity_id})
                        SET e.original_description = e.description,
                            e.description = $enhanced_description,
                            e.enhanced_summary = true,
                            e.enhanced_at = datetime()
                        RETURN e.id as updated_id
                    """, entity_id=entity_id, enhanced_description=enhanced_desc)
                    
                    if result.single():
                        updated_count += 1
                        
                except Exception as e:
                    print(f"Error updating entity {entity_id}: {e}")
            
            print(f"Updated {updated_count} entity descriptions in the graph")

    def collect_relationship_descriptions(self) -> List[Dict[str, Any]]:
        """Collect relationship descriptions for summarization"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e1:__Entity__)-[r:RELATES_TO]->(e2:__Entity__)
                RETURN e1.id as source_id, e1.entity_type as source_type,
                       e2.id as target_id, e2.entity_type as target_type,
                       r.co_occurrences as co_occurrences,
                       type(r) as relationship_type
                ORDER BY r.co_occurrences DESC
            """).data()
            
            print(f"Collected {len(result)} relationships for potential summarization")
            return result

    def perform_element_summarization(self, summarize_entities: bool = True, summarize_relationships: bool = False):
        """Perform element summarization on entities and relationships"""
        if not self.element_summarization_enabled:
            print("Element summarization is disabled. Enable it by setting element_summarization_enabled=True")
            return
        
        print("\n" + "="*60)
        print("STARTING ELEMENT SUMMARIZATION")
        print("="*60)
        
        if summarize_entities:
            print("Step 1: Collecting entity descriptions...")
            entity_summaries = self.collect_entity_descriptions()
            
            if entity_summaries:
                print("Step 2: Batch summarizing entities...")
                enhanced_descriptions = self.batch_summarize_entities(entity_summaries)
                
                print("Step 3: Updating entity descriptions in graph...")
                self.update_entity_descriptions(enhanced_descriptions)
            else:
                print("No entities found for summarization")
        
        if summarize_relationships:
            print("Step 4: Collecting relationship information...")
            relationships = self.collect_relationship_descriptions()
            # Note: Relationship summarization could be implemented similarly
            # but is more complex as relationships don't have direct descriptions
            print(f"Found {len(relationships)} relationships (relationship summarization not yet implemented)")
        
        print("Element summarization completed!")

    def enable_element_summarization(self, batch_size: int = 10):
        """Enable element summarization with specified batch size"""
        self.element_summarization_enabled = True
        self.element_batch_size = batch_size
        print(f"Element summarization enabled with batch size: {batch_size}")

    def disable_element_summarization(self):
        """Disable element summarization"""
        self.element_summarization_enabled = False
        print("Element summarization disabled")

    def perform_community_detection(self, max_levels: List[int] = [0, 1, 2], min_community_size: int = 2):
        """Perform community detection and summarization"""
        if not self.community_summarization_enabled:
            print("Community summarization is disabled. Enable it by setting community_summarization_enabled=True")
            return
        
        print("\n" + "="*60)
        print("STARTING COMMUNITY DETECTION AND SUMMARIZATION")
        print("="*60)
        
        try:
            # Step 1: Detect communities using Leiden algorithm
            print("Step 1: Detecting communities using Leiden algorithm...")
            self.detect_communities_leiden()
            
            # Step 2: Create community nodes and relationships
            print("Step 2: Creating community nodes and relationships...")
            self.create_community_nodes()
            
            # Step 3: Calculate community statistics and ranks
            print("Step 3: Calculating community statistics...")
            stats_df = self.calculate_community_statistics()
            print(f"Community statistics:\n{stats_df}")
            
            # Step 4: Generate community summaries
            print("Step 4: Generating community summaries...")
            summary_count = self.generate_community_summaries(max_levels=max_levels, min_community_size=min_community_size)
            
            print(f"✅ Community detection completed! Generated {summary_count} community summaries")
            
        except Exception as e:
            print(f"❌ Error during community detection: {e}")
            try:
                self.gds.graph.drop("communities")
            except:
                pass

    def detect_communities_leiden(self):
        """Detect communities using Leiden algorithm"""
        try:
            # Drop existing projection if it exists
            try:
                self.gds.graph.drop("communities")
            except:
                pass
            
            # Project graph for community detection
            G, result = self.gds.graph.project(
                "communities",  # Graph name
                "__Entity__",   # Node projection
                {
                    "_ALL_": {
                        "type": "*",
                        "orientation": "UNDIRECTED",
                        "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
                    }
                },
            )
            
            print(f"Graph projected with {G.node_count()} nodes and {G.relationship_count()} relationships")
            
            # Check weakly connected components
            wcc = self.gds.wcc.stats(G)
            print(f"Component count: {wcc['componentCount']}")
            print(f"Component distribution: {wcc['componentDistribution']}")
            
            # Run Leiden community detection
            self.gds.leiden.write(
                G,
                writeProperty="communities",
                includeIntermediateCommunities=True,
                relationshipWeightProperty="weight",
            )
            
            print("Leiden community detection completed")
            
        except Exception as e:
            print(f"Error in community detection: {e}")
            raise

    def create_community_nodes(self):
        """Create community nodes and relationships in the graph"""
        with self.driver.session() as session:
            # Create constraint for community nodes
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE")
            
            # Create community nodes and relationships
            result = session.run("""
                MATCH (e:`__Entity__`)
                WHERE e.communities IS NOT NULL
                UNWIND range(0, size(e.communities) - 1, 1) AS index
                CALL {
                  WITH e, index
                  WITH e, index
                  WHERE index = 0
                  MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
                  ON CREATE SET c.level = index
                  MERGE (e)-[:IN_COMMUNITY]->(c)
                  RETURN count(*) AS count_0
                }
                CALL {
                  WITH e, index
                  WITH e, index
                  WHERE index > 0
                  MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
                  ON CREATE SET current.level = index
                  MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
                  ON CREATE SET previous.level = index - 1
                  MERGE (previous)-[:IN_COMMUNITY]->(current)
                  RETURN count(*) AS count_1
                }
                RETURN count(*)
            """)
            
            # Calculate community ranks based on text chunks (Neo4j LLM Graph Builder approach)
            session.run("""
                MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:HAS_ENTITY]-(chunk:Chunk)
                WITH c, count(distinct chunk) AS rank
                SET c.community_rank = rank,
                    c.rank_explanation = "number of text chunks in which entities within the community appear",
                    c.title = CASE WHEN c.level = 0 THEN "Community " + split(c.id, "-")[1] 
                                   ELSE "Level " + split(c.id, "-")[0] + " Community " + split(c.id, "-")[1] END,
                    c.weight = toFloat(rank) / 100.0  // Normalized weight based on chunk frequency
            """)
            
            print("Community nodes and relationships created")
    
    def enhance_chunk_relationships(self):
        """
        Add LLM Graph Builder compatible chunk relationships:
        1. SIMILAR relationships between chunks (threshold=0.95, not used for retrieval)
        2. Additional chunk-community connections for graph completeness
        """
        print("Adding LLM Graph Builder compatible chunk relationships...")
        
        # Call the base class method for chunk similarity (matches LLM Graph Builder)
        self.create_chunk_similarity_relationships(similarity_threshold=0.95)
        
        # Add additional chunk-based community relationships for graph completeness
        with self.driver.session() as session:
            # Connect chunks to communities through their entities
            session.run("""
                MATCH (chunk:Chunk)-[:HAS_ENTITY]->(entity:__Entity__)-[:IN_COMMUNITY]->(community:__Community__)
                WITH chunk, community, count(distinct entity) as entity_count
                WHERE entity_count >= 2  // Only connect if chunk has multiple entities in the community
                MERGE (chunk)-[r:RELATES_TO_COMMUNITY]->(community)
                SET r.entity_count = entity_count,
                    r.strength = toFloat(entity_count) / 10.0  // Normalize strength
            """)
            
            print("Chunk-community relationships created")

    def calculate_community_statistics(self) -> pd.DataFrame:
        """Calculate and return community statistics"""
        with self.driver.session() as session:
            community_size_data = session.run("""
                MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(e:__Entity__)
                WITH c, count(distinct e) AS entities
                RETURN split(c.id, '-')[0] AS level, entities
            """).data()
            
            if not community_size_data:
                return pd.DataFrame()
            
            community_size_df = pd.DataFrame.from_records(community_size_data)
            percentiles_data = []
            
            for level in community_size_df["level"].unique():
                subset = community_size_df[community_size_df["level"] == level]["entities"]
                num_communities = len(subset)
                if num_communities > 0:
                    percentiles = np.percentile(subset, [25, 50, 75, 90, 99])
                    percentiles_data.append([
                        level,
                        num_communities,
                        percentiles[0],
                        percentiles[1], 
                        percentiles[2],
                        percentiles[3],
                        percentiles[4],
                        max(subset)
                    ])
            
            percentiles_df = pd.DataFrame(
                percentiles_data,
                columns=[
                    "Level",
                    "Number of communities", 
                    "25th Percentile",
                    "50th Percentile",
                    "75th Percentile", 
                    "90th Percentile",
                    "99th Percentile",
                    "Max"
                ],
            )
            
            return percentiles_df

    def generate_community_summaries(self, max_levels: List[int] = [0, 1, 2], min_community_size: int = 2) -> int:
        """Generate LLM summaries for communities"""
        with self.driver.session() as session:
            # Get community information for summarization
            community_data = session.run("""
                MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
                WHERE c.level IN $max_levels
                WITH c, collect(e) AS nodes
                WHERE size(nodes) >= $min_community_size
                CALL apoc.path.subgraphAll(nodes[0], {
                    whitelistNodes: nodes
                })
                YIELD relationships
                RETURN c.id AS communityId,
                       c.level AS level,
                       [n in nodes | {
                           id: n.id, 
                           description: n.description, 
                           type: [el in labels(n) WHERE el <> '__Entity__'][0]
                       }] AS nodes,
                       [r in relationships | {
                           start: startNode(r).id, 
                           type: type(r), 
                           end: endNode(r).id,
                           description: coalesce(r.description, '')
                       }] AS rels
                ORDER BY c.level, size(nodes) DESC
            """, max_levels=max_levels, min_community_size=min_community_size).data()
            
            if not community_data:
                print("No communities found for summarization")
                return 0
            
            print(f"Found {len(community_data)} communities to summarize")
            
            # Process communities with LLM
            summaries = []
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self._process_community, community): community 
                    for community in community_data
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing communities"):
                    try:
                        summary_result = future.result()
                        if summary_result:
                            summaries.append(summary_result)
                    except Exception as e:
                        print(f"Error processing community: {e}")
            
            # Store summaries in the graph
            if summaries:
                session.run("""
                    UNWIND $data AS row
                    MERGE (c:__Community__ {id: row.community})
                    SET c.summary = row.summary,
                        c.summary_generated_at = datetime()
                """, data=summaries)
                
                print(f"Stored {len(summaries)} community summaries")
            
            return len(summaries)

    def _process_community(self, community_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single community to generate summary"""
        try:
            community_info = self._prepare_community_string(community_data)
            summary = self.community_chain.invoke({'community_info': community_info})
            
            return {
                "community": community_data['communityId'],
                "summary": summary
            }
            
        except Exception as e:
            print(f"Error processing community {community_data.get('communityId', 'unknown')}: {e}")
            return None

    def _prepare_community_string(self, data: Dict[str, Any]) -> str:
        """Prepare community data as a string for LLM processing"""
        nodes_str = "Nodes are:\n"
        for node in data['nodes']:
            node_id = node['id']
            node_type = node['type']
            node_description = f", description: {node['description']}" if node.get('description') else ""
            nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

        rels_str = "Relationships are:\n"
        for rel in data['rels']:
            start = rel['start']
            end = rel['end']
            rel_type = rel['type']
            description = f", description: {rel['description']}" if rel.get('description') else ""
            rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

        return nodes_str + "\n" + rels_str

    def enable_community_summarization(self):
        """Enable community summarization"""
        self.community_summarization_enabled = True
        print("Community summarization enabled")

    def disable_community_summarization(self):
        """Disable community summarization"""
        self.community_summarization_enabled = False
        print("Community summarization disabled")

    def _check_existing_graph(self) -> Optional[Dict[str, int]]:
        """Check if graph exists and return node counts by type"""
        with self.driver.session() as session:
            # Check for basic graph structure
            result = session.run("""
                MATCH (n)
                WITH labels(n) as labels, count(n) as count
                RETURN labels, count
                ORDER BY count DESC
            """).data()
            
            if not result:
                return None
            
            # Process results into a readable format
            stats = {}
            for record in result:
                labels = record['labels']
                count = record['count']
                
                # Skip internal Neo4j labels
                if any(label.startswith('_') and label != '__Entity__' and label != '__Community__' for label in labels):
                    continue
                
                # Combine counts for nodes that have both specific type and __Entity__ label
                label_key = next((label for label in labels if label not in ['__Entity__', '__Community__']), None)
                if label_key:
                    stats[label_key] = stats.get(label_key, 0) + count
                elif '__Community__' in labels:
                    stats['Communities'] = count
            
            return stats if stats else None

    def _estimate_costs(self, stats: Dict[str, int], perform_element_summarization: bool, perform_community_detection: bool) -> Dict[str, Any]:
        """Estimate LLM API costs for graph enhancement operations"""
        costs = {}
        
        # Get entity count and description length in a single session
        with self.driver.session() as session:
            # Get entity count and average description length in one query
            result = session.run("""
                MATCH (e:__Entity__)
                WITH count(e) as entity_count,
                     avg(size(coalesce(e.description, ''))) as avg_length
                RETURN entity_count, avg_length
            """).single()
            
            entity_count = result['entity_count']
            avg_desc_length = result['avg_length'] or 0
        
        # Element summarization costs
        if perform_element_summarization:
            batch_count = ceil(entity_count / self.element_batch_size)
            avg_tokens_per_entity = 25  # Rough estimate for entity name + type
            avg_tokens_per_batch = self.element_batch_size * (avg_tokens_per_entity + avg_desc_length)
            
            costs['element_summarization'] = {
                'model': LLM,
                'entity_count': entity_count,
                'batch_count': batch_count,
                'estimated_tokens': avg_tokens_per_batch * batch_count,
                'estimated_cost': (avg_tokens_per_batch * batch_count * 0.01) / 1000  # $0.01 per 1K tokens
            }
        
        # Community detection costs
        if perform_community_detection:
            # Get community count estimate in a new session
            with self.driver.session() as session:
                try:
                    community_count = session.run("""
                        CALL gds.graph.project.estimate('entity-graph', ['__Entity__'], ['RELATES_TO'])
                        YIELD nodeCount
                        RETURN nodeCount
                    """).single()['nodeCount']
                except Exception as e:
                    # Fallback if GDS estimate fails
                    community_count = int(entity_count * 0.1)  # Estimate ~10% of entities form communities
            
            # Estimate costs for community summarization
            avg_tokens_per_community = 500  # Rough estimate for community context and generation
            total_community_tokens = community_count * avg_tokens_per_community
            
            costs['community_detection'] = {
                'model': LLM,
                'community_count': community_count,
                'estimated_tokens': total_community_tokens,
                'estimated_cost': (total_community_tokens * 0.01) / 1000  # $0.01 per 1K tokens
            }
        
        return costs

    def _get_user_confirmation(self, costs: Dict[str, Any], perform_element_summarization: bool, perform_community_detection: bool) -> bool:
        """Get user confirmation for LLM operations based on cost estimates"""
        total_cost = 0
        
        print("\n💰 Estimated costs:\n")
        
        if perform_element_summarization and "element_summarization" in costs:
            ec = costs["element_summarization"]
            print(f"Element Summarization:")
            print(f"  • {ec['entity_count']} entities in {ec['batch_count']} batches")
            print(f"  • Model: {ec['model']}")
            print(f"  • ~{ec['estimated_tokens']:,} tokens")
            print(f"  • ~${ec['estimated_cost']:.2f} USD")
            total_cost += ec['estimated_cost']
        
        if perform_community_detection and "community_detection" in costs:
            cc = costs["community_detection"]
            print(f"\nCommunity Detection:")
            print(f"  • {cc['community_count']} communities")
            print(f"  • Model: {cc['model']}")
            print(f"  • ~{cc['estimated_tokens']:,} tokens")
            print(f"  • ~${cc['estimated_cost']:.2f} USD")
            total_cost += cc['estimated_cost']
        
        print(f"\nTotal estimated cost: ${total_cost:.2f} USD\n")
        
        print("⚠️ " * 20)
        print("COST WARNING: Advanced graph enhancement uses LLM APIs")
        print("⚠️ " * 20 + "\n")
        
        while True:
            response = input("Do you want to proceed? [y/N]: ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['', 'n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no (or press Enter for no)")

    def enhance_existing_graph(self, perform_element_summarization: bool = False, perform_community_detection: bool = False) -> Dict[str, Any]:
        """Enhance an existing graph with element summarization and community detection.
        
        This method requires that graph_processor.py has already been run to create the base graph.
        """
        
        # Check if graph exists
        print("🔍 Checking existing graph...")
        stats = self._check_existing_graph()
        
        if not stats:
            raise ValueError(
                "❌ No graph data found! You must run graph_processor.py first to create the base graph.\n"
                "Run: python data_processors/graph_processor.py"
            )
        
        print("📊 Found existing graph with:")
        for label, count in stats.items():
            print(f"   {label}: {count} nodes")
        
        # Estimate costs and get user confirmation
        if perform_element_summarization or perform_community_detection:
            costs = self._estimate_costs(stats, perform_element_summarization, perform_community_detection)
            
            if not self._get_user_confirmation(costs, perform_element_summarization, perform_community_detection):
                print("\n❌ Enhancement cancelled by user")
                return {"status": "cancelled_by_user", "initial_stats": stats}
        
        # Perform element summarization if requested
        if perform_element_summarization:
            print("\n🔄 Starting element summarization...")
            self.element_summarization_enabled = True
            self.perform_element_summarization(summarize_entities=True, summarize_relationships=False)
        
        # Perform community detection if requested
        if perform_community_detection:
            print("\n🔄 Starting community detection...")
            self.community_summarization_enabled = True
            self.perform_community_detection(max_levels=[0, 1, 2], min_community_size=2)
        
        # Get final stats
        final_stats = self._check_existing_graph()
        
        return {
            "status": "enhanced_existing_graph",
            "initial_stats": stats,
            "final_stats": final_stats
        }


def main():
    """Main processing function - enhances existing graph created by graph_processor.py"""
    processor = AdvancedGraphProcessor()
    
    try:
        # Optional: Enable element summarization (WARNING: This increases LLM costs significantly)
        processor.enable_element_summarization(batch_size=10)
        
        # Optional: Enable community detection and summarization (WARNING: Additional LLM costs)
        processor.enable_community_summarization() 
        
        # Enhance existing graph with element summarization and community detection
        results = processor.enhance_existing_graph(
            perform_element_summarization=True,
            perform_community_detection=True
        )
        
        print("\n" + "="*60)
        print("ENHANCEMENT SUMMARY")
        print("="*60)
        
        if results.get('status') == 'enhanced_existing_graph':
            print("✅ Successfully enhanced existing graph")
            print("\nInitial graph contained:")
            for label, count in results.get('initial_stats', {}).items():
                print(f"   {label}: {count} nodes")
            
            if 'final_stats' in results:
                print("\nAfter creating __Entity__ wrappers:")
                for label, count in results.get('final_stats', {}).items():
                    print(f"   {label}: {count} nodes")
        elif results.get('status') == 'cancelled_by_user':
            print("⏹️ Enhancement cancelled by user")
            print("No changes were made to the graph")
            return
        else:
            print("❌ Enhancement failed or was not completed")
        
        # Print database stats
        with processor.driver.session() as session:
            stats = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """).data()
            
            print(f"\nFinal Database Statistics:")
            for stat in stats:
                print(f"  {stat['label']}: {stat['count']} nodes")
                
            # Show element summarization results
            enhanced_count = session.run("""
                MATCH (e:__Entity__ {enhanced_summary: true})
                RETURN count(e) as enhanced_count
            """).single()
            
            if enhanced_count and enhanced_count['enhanced_count'] > 0:
                print(f"\n📝 Element Summarization Results:")
                print(f"  ✅ Enhanced {enhanced_count['enhanced_count']} entity descriptions")
            
            # Show community detection results
            community_stats = session.run("""
                MATCH (c:__Community__)
                WITH count(c) as total_communities
                MATCH (c:__Community__)
                WHERE c.summary IS NOT NULL
                RETURN total_communities, count(c) as summarized_communities
            """).single()
            
            if community_stats and community_stats['total_communities'] > 0:
                print(f"\n🌐 Community Detection Results:")
                print(f"  ✅ Created {community_stats['total_communities']} communities")
                print(f"  ✅ Generated {community_stats['summarized_communities']} community summaries")
                
                # Show community level distribution
                level_distribution = session.run("""
                    MATCH (c:__Community__)
                    RETURN c.level as level, count(c) as count
                    ORDER BY c.level
                """).data()
                
                if level_distribution:
                    print(f"  Community hierarchy:")
                    for level_stat in level_distribution:
                        print(f"    Level {level_stat['level']}: {level_stat['count']} communities")
                
    finally:
        processor.close()


if __name__ == "__main__":
    main() 