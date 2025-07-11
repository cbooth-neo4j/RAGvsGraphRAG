"""
Test script for Community Detection feature

This script demonstrates how to use the community detection and summarization
feature to identify and summarize entity communities in the knowledge graph.

WARNING: This script will make LLM API calls which incur costs.
Make sure you understand the pricing before running.
"""

import os
from data_processors import AdvancedGraphProcessor
from dotenv import load_dotenv

def test_community_detection():
    """Test the community detection feature"""
    
    # Load environment variables
    load_dotenv()
    
    # Verify API keys are available
    if not os.environ.get('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    if not all([
        os.environ.get('NEO4J_URI'),
        os.environ.get('NEO4J_USERNAME'), 
        os.environ.get('NEO4J_PASSWORD')
    ]):
        print("❌ Neo4j credentials not found in environment variables")
        return
    
    print("🧪 Testing Community Detection Feature")
    print("="*50)
    
    # Initialize processor
    processor = AdvancedGraphProcessor()
    
    try:
        # Check if we have entities in the graph
        with processor.driver.session() as session:
            entity_count = session.run("MATCH (e:__Entity__) RETURN count(e) as count").single()['count']
            
            if entity_count == 0:
                print("❌ No entities found in the graph")
                print("   Run the main processor first to create entities")
                return
            
            print(f"📊 Found {entity_count} entities in the graph")
            
            # Check for relationships
            relationship_count = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count").single()['count']
            print(f"📊 Found {relationship_count} relationships in the graph")
            
            if relationship_count < 10:
                print("⚠️ Warning: Very few relationships found. Community detection works best with more connected entities.")
        
        # Enable community summarization
        print("🔧 Enabling community detection and summarization...")
        processor.enable_community_summarization()
        
        # Estimate costs (rough estimate)
        estimated_communities = max(10, entity_count // 20)  # Rough estimate
        print(f"💰 Estimated: ~{estimated_communities} communities to summarize")
        print(f"   At ~$0.002 per community = ~${estimated_communities * 0.002:.3f} USD")
        
        # Ask for confirmation
        response = input("\n❓ Continue with community detection? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Community detection cancelled")
            return
        
        # Perform community detection
        print("\n🚀 Starting community detection...")
        processor.perform_community_detection(
            max_levels=[0, 1, 2],  # Detect communities at levels 0, 1, and 2
            min_community_size=2   # Only summarize communities with 2+ entities
        )
        
        # Show results
        with processor.driver.session() as session:
            community_stats = session.run("""
                MATCH (c:__Community__)
                WITH count(c) as total_communities
                MATCH (c:__Community__)
                WHERE c.summary IS NOT NULL
                RETURN total_communities, count(c) as summarized_communities
            """).single()
            
            print(f"\n✅ Detected {community_stats['total_communities']} total communities")
            print(f"✅ Generated summaries for {community_stats['summarized_communities']} communities")
            
            # Show community level distribution
            level_distribution = session.run("""
                MATCH (c:__Community__)
                RETURN c.level as level, count(c) as count
                ORDER BY c.level
            """).data()
            
            print(f"\n📋 Community Level Distribution:")
            for level_stat in level_distribution:
                print(f"  Level {level_stat['level']}: {level_stat['count']} communities")
            
            # Show sample community summaries
            sample_communities = session.run("""
                MATCH (c:__Community__)
                WHERE c.summary IS NOT NULL
                RETURN c.id as id, c.level as level, c.summary as summary,
                       size([(c)<-[:IN_COMMUNITY*]-(e:__Entity__) | e]) as entity_count
                ORDER BY c.level, entity_count DESC
                LIMIT 3
            """).data()
            
            print(f"\n📋 Sample Community Summaries:")
            for community in sample_communities:
                print(f"  Community {community['id']} (Level {community['level']}, {community['entity_count']} entities):")
                print(f"    {community['summary'][:150]}...")
                print()
        
        print("✅ Community detection test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        
    finally:
        processor.close()

def show_community_stats():
    """Show detailed statistics about detected communities"""
    processor = AdvancedGraphProcessor()
    
    try:
        with processor.driver.session() as session:
            # Overall statistics
            overall_stats = session.run("""
                MATCH (c:__Community__)
                WITH count(c) as total_communities
                MATCH (c:__Community__)
                WHERE c.summary IS NOT NULL
                WITH total_communities, count(c) as summarized_communities
                MATCH (e:__Entity__)
                WITH total_communities, summarized_communities, count(e) as total_entities
                MATCH (c:__Community__)<-[:IN_COMMUNITY]-(e:__Entity__)
                RETURN total_communities, summarized_communities, total_entities,
                       count(distinct e) as entities_in_communities,
                       avg(size(c.summary)) as avg_summary_length
            """).single()
            
            if overall_stats and overall_stats['total_communities'] > 0:
                print("📊 Community Detection Statistics:")
                print(f"  Total communities: {overall_stats['total_communities']}")
                print(f"  Summarized communities: {overall_stats['summarized_communities']}")
                print(f"  Total entities: {overall_stats['total_entities']}")
                print(f"  Entities in communities: {overall_stats['entities_in_communities']}")
                if overall_stats['avg_summary_length']:
                    print(f"  Avg summary length: {overall_stats['avg_summary_length']:.0f} chars")
                
                # Level-wise statistics
                level_stats = session.run("""
                    MATCH (c:__Community__)
                    WITH c.level as level, count(c) as community_count,
                         avg(size([(c)<-[:IN_COMMUNITY*]-(e:__Entity__) | e])) as avg_entities
                    RETURN level, community_count, avg_entities
                    ORDER BY level
                """).data()
                
                print(f"\n📊 Level-wise Statistics:")
                for stat in level_stats:
                    print(f"  Level {stat['level']}: {stat['community_count']} communities, avg {stat['avg_entities']:.1f} entities")
                
                # Top communities by size
                top_communities = session.run("""
                    MATCH (c:__Community__)
                    WHERE c.summary IS NOT NULL
                    WITH c, size([(c)<-[:IN_COMMUNITY*]-(e:__Entity__) | e]) as entity_count
                    RETURN c.id as id, c.level as level, entity_count,
                           c.summary[0..100] + '...' as summary_preview
                    ORDER BY entity_count DESC
                    LIMIT 5
                """).data()
                
                print(f"\n📋 Top Communities by Size:")
                for community in top_communities:
                    print(f"  {community['id']} (L{community['level']}, {community['entity_count']} entities):")
                    print(f"    {community['summary_preview']}")
                    print()
                
            else:
                print("📊 No communities found in the graph")
                print("   Run community detection first")
            
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        
    finally:
        processor.close()

def explore_community(community_id: str = None):
    """Explore a specific community in detail"""
    processor = AdvancedGraphProcessor()
    
    try:
        with processor.driver.session() as session:
            if not community_id:
                # Show available communities
                communities = session.run("""
                    MATCH (c:__Community__)
                    WHERE c.summary IS NOT NULL
                    WITH c, size([(c)<-[:IN_COMMUNITY*]-(e:__Entity__) | e]) as entity_count
                    RETURN c.id as id, c.level as level, entity_count
                    ORDER BY entity_count DESC
                    LIMIT 10
                """).data()
                
                print("Available communities:")
                for i, community in enumerate(communities):
                    print(f"  {i+1}. {community['id']} (Level {community['level']}, {community['entity_count']} entities)")
                
                choice = input("\nEnter community number (1-10): ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(communities):
                        community_id = communities[idx]['id']
                    else:
                        print("Invalid choice")
                        return
                except ValueError:
                    print("Invalid input")
                    return
            
            # Get detailed community information
            community_detail = session.run("""
                MATCH (c:__Community__ {id: $community_id})
                OPTIONAL MATCH (c)<-[:IN_COMMUNITY*]-(e:__Entity__)
                WITH c, collect(e) as entities
                OPTIONAL MATCH (e1:__Entity__)-[r:RELATES_TO]->(e2:__Entity__)
                WHERE e1 in entities AND e2 in entities
                RETURN c.id as id, c.level as level, c.summary as summary,
                       [e in entities | {id: e.id, type: labels(e)[1], description: e.description}] as entities,
                       count(r) as internal_relationships
            """, community_id=community_id).single()
            
            if community_detail:
                print(f"\n🔍 Community Details: {community_detail['id']}")
                print(f"Level: {community_detail['level']}")
                print(f"Entities: {len(community_detail['entities'])}")
                print(f"Internal relationships: {community_detail['internal_relationships']}")
                print(f"\nSummary:")
                print(f"  {community_detail['summary']}")
                
                print(f"\nEntities in community:")
                for entity in community_detail['entities'][:10]:  # Show first 10
                    print(f"  - {entity['type']}: {entity['id']}")
                    if entity['description']:
                        print(f"    {entity['description'][:80]}...")
                
                if len(community_detail['entities']) > 10:
                    print(f"  ... and {len(community_detail['entities']) - 10} more entities")
            else:
                print(f"❌ Community {community_id} not found")
            
    except Exception as e:
        print(f"❌ Error exploring community: {e}")
        
    finally:
        processor.close()

if __name__ == "__main__":
    print("Community Detection Test Script")
    print("=" * 40)
    print("1. Test community detection")
    print("2. Show community statistics")
    print("3. Explore specific community")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        test_community_detection()
    elif choice == "2":
        show_community_stats()
    elif choice == "3":
        community_id = input("Enter community ID (or press Enter to choose from list): ").strip()
        if not community_id:
            community_id = None
        explore_community(community_id)
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice. Please select 1, 2, 3, or 4.") 