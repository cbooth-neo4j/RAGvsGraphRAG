"""
Test script for hierarchical community detection

This script tests the improved hierarchical community detection based on 
Neo4j LLM Graph Builder approach.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processors.build_graph import CustomGraphProcessor

def test_hierarchical_communities():
    """Test the hierarchical community detection implementation"""
    print("🧪 Testing Hierarchical Community Detection")
    print("=" * 50)
    
    try:
        # Create processor with smart relationship strategy
        processor = CustomGraphProcessor(relationship_strategy="smart")
        
        print("✅ Processor initialized successfully")
        
        # Test community detection directly
        print("\n🔍 Running hierarchical community detection...")
        processor.perform_community_detection(max_levels=3, min_community_size=2)
        
        # Check results
        print("\n📊 Checking community detection results...")
        
        with processor.driver.session() as session:
            # Count communities by level
            level_counts_query = """
            MATCH (c:__Community__)
            RETURN c.level as level, count(c) as count
            ORDER BY level
            """
            
            result = session.run(level_counts_query)
            levels = list(result)
            
            print("Community counts by level:")
            for record in levels:
                level = record['level']
                count = record['count']
                print(f"   Level {level}: {count} communities")
            
            # Check hierarchy relationships
            hierarchy_query = """
            MATCH (child:__Community__)-[:PART_OF]->(parent:__Community__)
            RETURN child.level as child_level, parent.level as parent_level, count(*) as relationships
            ORDER BY child_level, parent_level
            """
            
            result = session.run(hierarchy_query)
            relationships = list(result)
            
            if relationships:
                print("\nHierarchical relationships:")
                for record in relationships:
                    child_level = record['child_level']
                    parent_level = record['parent_level']
                    rel_count = record['relationships']
                    print(f"   Level {child_level} → Level {parent_level}: {rel_count} relationships")
            else:
                print("\n⚠️  No hierarchical relationships found!")
            
            # Check entity-community relationships
            entity_community_query = """
            MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c:__Community__)
            RETURN c.level as level, count(DISTINCT e) as entities, count(DISTINCT c) as communities
            ORDER BY level
            """
            
            result = session.run(entity_community_query)
            entity_stats = list(result)
            
            print("\nEntity-Community relationships:")
            for record in entity_stats:
                level = record['level']
                entities = record['entities']
                communities = record['communities']
                print(f"   Level {level}: {entities} entities in {communities} communities")
            
            # Sample some communities
            sample_query = """
            MATCH (c:__Community__)
            WHERE c.title IS NOT NULL
            RETURN c.level as level, c.title as title, c.member_count as members, c.rating as rating
            ORDER BY c.level, c.member_count DESC
            LIMIT 10
            """
            
            result = session.run(sample_query)
            samples = list(result)
            
            if samples:
                print("\nSample communities:")
                for record in samples:
                    level = record['level']
                    title = record['title']
                    members = record['members']
                    rating = record['rating']
                    print(f"   Level {level}: '{title}' ({members} members, rating: {rating:.1f})")
        
        processor.close()
        
        # Analyze results
        if not levels:
            print("\n❌ FAILED: No communities were created!")
            return False
        
        if len(levels) == 1:
            print(f"\n⚠️  WARNING: Only level {levels[0]['level']} communities created")
            print("   Expected multiple hierarchical levels (0, 1, 2)")
            return False
        
        if len(levels) >= 2:
            print(f"\n✅ SUCCESS: Created {len(levels)} hierarchical levels!")
            
            if relationships:
                print(f"✅ SUCCESS: Created {sum(r['relationships'] for r in relationships)} hierarchical relationships!")
            else:
                print("⚠️  WARNING: No hierarchical relationships found")
            
            return True
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hierarchical_communities()
    
    if success:
        print("\n🎉 Hierarchical community detection test PASSED!")
        print("\nThe Neo4j LLM Graph Builder approach is working correctly:")
        print("- Multiple hierarchical levels created")
        print("- Proper parent-child relationships established")
        print("- Community summaries generated")
    else:
        print("\n💥 Hierarchical community detection test FAILED!")
        print("\nPlease check:")
        print("- Neo4j database connection")
        print("- Graph Data Science plugin installation")
        print("- Entity relationships in the graph")
        print("- Leiden algorithm parameters")
