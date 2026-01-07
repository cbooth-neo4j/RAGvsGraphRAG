"""
Fix Community Hierarchy Script

This script addresses the following issues in the community structure:
1. Generate missing summaries for levels 3 and 4
2. Embed community summaries and create a vector index for DRIFT-style search
3. Verify and report on the community hierarchy health

Based on the Neo4j GraphRAG approach from:
https://neo4j.com/blog/developer/global-graphrag-neo4j-langchain/
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_llm, get_embeddings
from utils.graph_rag_logger import setup_logging, get_logger

load_dotenv()
setup_logging()
logger = get_logger(__name__)

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
NEO4J_DATABASE = os.environ.get('CLIENT_NEO4J_DATABASE')

# Configuration
COMMUNITY_SUMMARY_MAX_WORKERS = int(os.getenv('COMMUNITY_SUMMARY_MAX_WORKERS', '10'))
EMBEDDING_BATCH_SIZE = 50


class CommunityHierarchyFixer:
    """Fixes and enhances community hierarchy for GraphRAG retrieval."""
    
    def __init__(self):
        import neo4j
        self.driver = neo4j.GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.max_workers = COMMUNITY_SUMMARY_MAX_WORKERS
    
    def close(self):
        self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_community_stats(self) -> Dict[str, Any]:
        """Get current community hierarchy statistics."""
        with self.driver.session(database=NEO4J_DATABASE) as session:
            # Get distribution by level
            level_stats = session.run("""
                MATCH (c:__Community__)
                RETURN c.level as level, 
                       count(*) as total,
                       count(c.summary) as with_summary,
                       avg(size(coalesce(c.summary, ''))) as avg_summary_length
                ORDER BY level
            """).data()
            
            # Get entity assignment info
            entity_stats = session.run("""
                MATCH (e:__Entity__)
                OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(c:__Community__)
                RETURN 
                  count(DISTINCT e) as total_entities,
                  count(DISTINCT c) as communities_with_entities
            """).single()
            
            # Get hierarchy connectivity
            hierarchy_stats = session.run("""
                MATCH (child:__Community__)-[:PARENT_COMMUNITY]->(parent:__Community__)
                RETURN child.level as child_level, 
                       parent.level as parent_level, 
                       count(*) as connections
                ORDER BY child_level
            """).data()
            
            return {
                'level_stats': level_stats,
                'entity_stats': dict(entity_stats),
                'hierarchy_stats': hierarchy_stats
            }
    
    def print_stats(self, stats: Dict[str, Any]):
        """Pretty print community statistics."""
        print("\n" + "="*60)
        print("COMMUNITY HIERARCHY STATUS")
        print("="*60)
        
        print("\n[*] Level Distribution:")
        print("-"*50)
        for level in stats['level_stats']:
            summary_pct = (level['with_summary'] / level['total'] * 100) if level['total'] > 0 else 0
            avg_len = level['avg_summary_length'] or 0
            status = "[OK]" if summary_pct == 100 else "[PARTIAL]" if summary_pct > 0 else "[MISSING]"
            print(f"  Level {level['level']}: {level['total']:,} communities | "
                  f"{status} {level['with_summary']:,}/{level['total']:,} summaries ({summary_pct:.1f}%) | "
                  f"avg length: {avg_len:.0f} chars")
        
        print(f"\n[*] Entity Coverage:")
        print("-"*50)
        print(f"  Total entities: {stats['entity_stats']['total_entities']:,}")
        print(f"  Communities with entity connections: {stats['entity_stats']['communities_with_entities']:,}")
        
        print(f"\n[*] Hierarchy Connectivity:")
        print("-"*50)
        for h in stats['hierarchy_stats']:
            print(f"  Level {h['child_level']} -> Level {h['parent_level']}: {h['connections']:,} connections")
        
        print("="*60 + "\n")
    
    async def generate_missing_summaries(self, levels: List[int] = [3, 4]) -> int:
        """Generate summaries for communities at specified levels that are missing summaries."""
        print(f"\n[*] Generating summaries for levels {levels}...")
        
        total_generated = 0
        
        for level in levels:
            print(f"\n   Processing level {level}...")
            
            # Get communities without summaries at this level
            # For higher levels, aggregate entities from child communities
            with self.driver.session(database=NEO4J_DATABASE) as session:
                communities = session.run("""
                    MATCH (c:__Community__ {level: $level})
                    WHERE c.summary IS NULL OR c.summary = ''
                    
                    // Get entities through child community hierarchy
                    OPTIONAL MATCH path = (child:__Community__)-[:PARENT_COMMUNITY*0..]->(c)
                    WHERE child.level = 0
                    OPTIONAL MATCH (child)<-[:IN_COMMUNITY]-(e:__Entity__)
                    
                    WITH c, collect(DISTINCT {
                        name: e.name, 
                        description: coalesce(e.ai_summary, e.description, e.name)
                    }) as entities
                    WHERE size(entities) > 0
                    
                    // Also get child community summaries for context
                    OPTIONAL MATCH (child_direct:__Community__)-[:PARENT_COMMUNITY]->(c)
                    WITH c, entities, collect(child_direct.summary) as child_summaries
                    
                    RETURN c.id as community_id, 
                           c.level as level,
                           size(entities) as member_count,
                           entities[0..30] as entities,
                           child_summaries[0..5] as child_summaries
                    ORDER BY size(entities) DESC
                """, level=level).data()
            
            if not communities:
                print(f"   No communities needing summaries at level {level}")
                continue
            
            print(f"   Found {len(communities)} communities to summarize at level {level}")
            
            # Process in parallel with semaphore
            semaphore = asyncio.Semaphore(self.max_workers)
            tasks = [
                self._generate_single_summary(c, semaphore)
                for c in communities
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            level_generated = sum(1 for r in results if r is True)
            total_generated += level_generated
            print(f"   [OK] Generated {level_generated}/{len(communities)} summaries for level {level}")
        
        return total_generated
    
    async def _generate_single_summary(
        self, 
        community: Dict[str, Any], 
        semaphore: asyncio.Semaphore
    ) -> bool:
        """Generate a summary for a single community."""
        async with semaphore:
            try:
                community_id = community['community_id']
                entities = community['entities']
                member_count = community['member_count']
                child_summaries = community.get('child_summaries', [])
                level = community['level']
                
                # Build context from entities
                entity_context = "\n".join([
                    f"- {e['name']}: {e['description']}"
                    for e in entities if e.get('name')
                ][:25])  # Limit to prevent token overflow
                
                # Add child summaries for higher level context
                child_context = ""
                if child_summaries:
                    child_context = "\n\nChild community summaries:\n" + "\n".join([
                        f"- {s}" for s in child_summaries if s
                    ][:5])
                
                # Generate summary using LLM
                summary_prompt = f"""
Analyze this community of {member_count} entities at hierarchical level {level} and provide:
1. A descriptive title (2-4 words) that captures the main theme
2. A comprehensive summary (2-3 sentences) of the community's main themes and relationships
3. An importance rating (0-10) based on entity relationships, diversity, and significance
4. A brief explanation of the importance rating

Entities in this community:
{entity_context}
{child_context}

Respond in JSON format:
{{
    "title": "Community Title",
    "summary": "Detailed summary of the community's main themes and relationships",
    "rating": 7.5,
    "rating_explanation": "Explanation of why this rating was assigned"
}}
"""
                
                try:
                    response = await self.llm.ainvoke(summary_prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    # Parse JSON from response
                    import re
                    content = content.strip()
                    
                    # Extract JSON from various formats
                    json_match = re.search(r'(\{.*?\})', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                    
                    # Try code blocks if needed
                    if not content.startswith('{'):
                        json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
                        if json_blocks:
                            content = json_blocks[0]
                    
                    summary_data = json.loads(content)
                    
                    # Update community in Neo4j
                    with self.driver.session(database=NEO4J_DATABASE) as session:
                        session.run("""
                            MATCH (c:__Community__ {id: $community_id})
                            SET c.title = $title,
                                c.summary = $summary,
                                c.rating = $rating,
                                c.rating_explanation = $rating_explanation,
                                c.summarized_at = datetime()
                        """,
                            community_id=community_id,
                            title=summary_data.get('title', f'Community {community_id}'),
                            summary=summary_data.get('summary', 'No summary available'),
                            rating=float(summary_data.get('rating', 5.0)),
                            rating_explanation=summary_data.get('rating_explanation', 'No explanation provided')
                        )
                    
                    return True
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error for community {community_id}: {e}")
                    return False
                except Exception as e:
                    logger.error(f"Error generating summary for community {community_id}: {e}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error processing community: {e}")
                return False
    
    def embed_community_summaries(self) -> int:
        """Embed all community summaries that don't have embeddings yet."""
        print("\n[*] Embedding community summaries...")
        
        # Get communities with summaries but no embeddings
        with self.driver.session(database=NEO4J_DATABASE) as session:
            communities = session.run("""
                MATCH (c:__Community__)
                WHERE c.summary IS NOT NULL 
                  AND c.summary <> ''
                  AND (c.embedding IS NULL OR size(c.embedding) = 0)
                RETURN c.id as id, c.summary as summary, c.title as title
            """).data()
        
        if not communities:
            print("   No communities need embedding")
            return 0
        
        print(f"   Embedding {len(communities)} community summaries...")
        
        embedded = 0
        
        # Process in batches
        for i in tqdm(range(0, len(communities), EMBEDDING_BATCH_SIZE), desc="   Embedding"):
            batch = communities[i:i + EMBEDDING_BATCH_SIZE]
            
            # Create text for embedding (title + summary)
            texts = [
                f"{c['title'] or ''}: {c['summary']}" 
                for c in batch
            ]
            
            try:
                # Generate embeddings
                embeddings = self.embeddings.embed_documents(texts)
                
                # Store embeddings in Neo4j
                with self.driver.session(database=NEO4J_DATABASE) as session:
                    for c, emb in zip(batch, embeddings):
                        session.run("""
                            MATCH (c:__Community__ {id: $id})
                            SET c.embedding = $embedding
                        """, id=c['id'], embedding=emb)
                
                embedded += len(batch)
                
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                continue
        
        print(f"   [OK] Embedded {embedded} community summaries")
        return embedded
    
    def create_community_vector_index(self) -> bool:
        """Create vector index on community summaries for DRIFT-style search."""
        print("\n[*] Creating community summary vector index...")
        
        # Get embedding dimension from a sample
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("""
                MATCH (c:__Community__)
                WHERE c.embedding IS NOT NULL AND size(c.embedding) > 0
                RETURN size(c.embedding) as dimension
                LIMIT 1
            """).single()
            
            if not result:
                print("   [WARNING] No embeddings found - embed summaries first")
                return False
            
            dimension = result['dimension']
            print(f"   Detected embedding dimension: {dimension}")
        
        # Create the vector index
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Drop existing index if it exists
                try:
                    session.run("DROP INDEX community_summary_embeddings IF EXISTS")
                except:
                    pass
                
                # Create new index
                session.run(f"""
                    CREATE VECTOR INDEX community_summary_embeddings IF NOT EXISTS
                    FOR (c:__Community__)
                    ON c.embedding
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {dimension},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """)
            
            print(f"   [OK] Created vector index 'community_summary_embeddings' with dimension {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            print(f"   [ERROR] Error creating index: {e}")
            return False
    
    def verify_indexes(self):
        """Verify all required indexes exist."""
        print("\n[*] Verifying indexes...")
        
        with self.driver.session(database=NEO4J_DATABASE) as session:
            indexes = session.run("""
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                RETURN name, type, labelsOrTypes, properties
            """).data()
        
        print("\n   Existing indexes:")
        for idx in indexes:
            idx_type = idx.get('type', 'unknown')
            labels = idx.get('labelsOrTypes', [])
            props = idx.get('properties', [])
            print(f"   - {idx['name']}: {idx_type} on {labels}.{props}")
        
        # Check for required indexes
        required = ['community_summary_embeddings', 'entity_embeddings', 'entity_fulltext_idx']
        existing_names = [idx['name'] for idx in indexes]
        
        print("\n   Required indexes status:")
        for req in required:
            status = "[OK]" if req in existing_names else "[MISSING]"
            print(f"   {status} {req}")


def run_all_fixes():
    """Run all community hierarchy fixes."""
    print("\n" + "="*60)
    print("COMMUNITY HIERARCHY FIX SCRIPT")
    print("="*60)
    
    with CommunityHierarchyFixer() as fixer:
        # 1. Get initial stats
        print("\n[*] Initial Status:")
        stats = fixer.get_community_stats()
        fixer.print_stats(stats)
        
        # 2. Generate missing summaries for levels 3-4
        missing_levels = [
            level['level'] for level in stats['level_stats']
            if level['with_summary'] == 0
        ]
        
        if missing_levels:
            print(f"\n[*] Fixing: Generating summaries for levels {missing_levels}")
            summaries_created = asyncio.run(fixer.generate_missing_summaries(missing_levels))
            print(f"\n   Total summaries created: {summaries_created}")
        else:
            print("\n[OK] All levels have summaries")
        
        # 3. Embed community summaries
        embedded = fixer.embed_community_summaries()
        
        # 4. Create vector index
        fixer.create_community_vector_index()
        
        # 5. Verify indexes
        fixer.verify_indexes()
        
        # 6. Final stats
        print("\n[*] Final Status:")
        final_stats = fixer.get_community_stats()
        fixer.print_stats(final_stats)
        
        print("\n[OK] Community hierarchy fix complete!")
        print("\nNext steps:")
        print("  1. Run the updated hybrid_cypher_retriever to test community traversal")
        print("  2. Test DRIFT retrieval with the new community_summary_embeddings index")


if __name__ == "__main__":
    run_all_fixes()

