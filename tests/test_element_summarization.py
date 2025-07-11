"""
Test script for Element Summarization feature

This script demonstrates how to use the element summarization feature
to enhance entity descriptions in the knowledge graph.

WARNING: This script will make LLM API calls which incur costs.
Make sure you understand the pricing before running.
"""

import os
from data_processors import AdvancedGraphProcessor
from dotenv import load_dotenv

def test_element_summarization():
    """Test the element summarization feature"""
    
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
    
    print("🧪 Testing Element Summarization Feature")
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
            
            # Show sample entities before enhancement
            sample_entities = session.run("""
                MATCH (e:__Entity__)
                WHERE e.description IS NOT NULL
                RETURN e.id as id, e.entity_type as type, e.description as description
                LIMIT 5
            """).data()
            
            print("\n📋 Sample entities BEFORE summarization:")
            for entity in sample_entities:
                print(f"  {entity['type']}: {entity['id']}")
                print(f"    Description: {entity['description'][:100]}...")
                print()
        
        # Enable element summarization with small batch size for testing
        print("🔧 Enabling element summarization...")
        processor.enable_element_summarization(batch_size=5)  # Small batches for testing
        
        # Estimate costs
        estimated_batches = (entity_count // 5) + (1 if entity_count % 5 > 0 else 0)
        print(f"💰 Estimated cost: ~{estimated_batches} LLM calls")
        print(f"   At ~$0.0015 per batch = ~${estimated_batches * 0.0015:.3f} USD")
        
        # Ask for confirmation
        response = input("\n❓ Continue with element summarization? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Element summarization cancelled")
            return
        
        # Perform element summarization
        print("\n🚀 Starting element summarization...")
        processor.perform_element_summarization(
            summarize_entities=True,
            summarize_relationships=False
        )
        
        # Show results
        with processor.driver.session() as session:
            enhanced_count = session.run("""
                MATCH (e:__Entity__ {enhanced_summary: true})
                RETURN count(e) as count
            """).single()['count']
            
            print(f"\n✅ Enhanced {enhanced_count} entity descriptions")
            
            # Show sample enhanced entities
            enhanced_entities = session.run("""
                MATCH (e:__Entity__ {enhanced_summary: true})
                WHERE e.description IS NOT NULL AND e.original_description IS NOT NULL
                RETURN e.id as id, e.entity_type as type, 
                       e.original_description as original,
                       e.description as enhanced
                LIMIT 3
            """).data()
            
            print("\n📋 Sample entities AFTER summarization:")
            for entity in enhanced_entities:
                print(f"  {entity['type']}: {entity['id']}")
                print(f"    Original: {entity['original'][:80]}...")
                print(f"    Enhanced: {entity['enhanced'][:80]}...")
                print()
        
        print("✅ Element summarization test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        
    finally:
        processor.close()

def show_enhancement_stats():
    """Show statistics about enhanced entities"""
    processor = AdvancedGraphProcessor()
    
    try:
        with processor.driver.session() as session:
            stats = session.run("""
                MATCH (e:__Entity__)
                WITH count(e) as total_entities
                MATCH (enhanced:__Entity__ {enhanced_summary: true})
                WITH total_entities, count(enhanced) as enhanced_entities
                MATCH (e:__Entity__)
                WHERE e.enhanced_summary = true
                WITH total_entities, enhanced_entities, 
                     avg(size(e.description)) as avg_enhanced_length,
                     avg(size(e.original_description)) as avg_original_length
                RETURN total_entities, enhanced_entities,
                       avg_enhanced_length, avg_original_length
            """).single()
            
            if stats:
                print("📊 Element Summarization Statistics:")
                print(f"  Total entities: {stats['total_entities']}")
                print(f"  Enhanced entities: {stats['enhanced_entities']}")
                if stats['enhanced_entities'] > 0:
                    print(f"  Enhancement rate: {(stats['enhanced_entities']/stats['total_entities']*100):.1f}%")
                    print(f"  Avg original description length: {stats['avg_original_length']:.0f} chars")
                    print(f"  Avg enhanced description length: {stats['avg_enhanced_length']:.0f} chars")
                    improvement = ((stats['avg_enhanced_length'] - stats['avg_original_length']) / stats['avg_original_length'] * 100)
                    print(f"  Description length improvement: {improvement:+.1f}%")
                else:
                    print("  No enhanced entities found")
            
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        
    finally:
        processor.close()

if __name__ == "__main__":
    print("Element Summarization Test Script")
    print("=" * 40)
    print("1. Test element summarization")
    print("2. Show enhancement statistics")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        test_element_summarization()
    elif choice == "2":
        show_enhancement_stats()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice. Please select 1, 2, or 3.") 