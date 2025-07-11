"""
Test script for Enhanced Graph Processor with Entity Resolution

This script demonstrates how to use the enhanced processor that includes
entity resolution capabilities.
"""

import sys
import os
# Add parent directory to path to import data_processors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processors import AdvancedGraphProcessor
from dotenv import load_dotenv

def main():
    """Test the enhanced graph processor"""
    load_dotenv()
    
    # Check environment variables
    required_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set these in your .env file")
        return
    
    print("✅ All environment variables found")
    
    # Initialize processor
    processor = AdvancedGraphProcessor()
    
    try:
        # Test 1: Process documents with entity resolution
        print("\n" + "="*50)
        print("TEST 1: PROCESSING WITH ENTITY RESOLUTION")
        print("="*50)
        
        results = processor.process_directory("PDFs", perform_resolution=True)
        
        # Test 2: Process documents without entity resolution (for comparison)
        print("\n" + "="*50)
        print("TEST 2: PROCESSING WITHOUT ENTITY RESOLUTION")
        print("="*50)
        
        # Clear database and reprocess without resolution
        processor.clear_database()
        processor.setup_database_schema()
        
        results_no_resolution = processor.process_directory("PDFs", perform_resolution=False)
        
        # Show comparison
        print("\n" + "="*50)
        print("COMPARISON RESULTS")
        print("="*50)
        
        with processor.driver.session() as session:
            # Count entities before and after resolution
            entity_stats = session.run("""
                MATCH (e:__Entity__)
                RETURN e.entity_type as type, count(*) as count
                ORDER BY count DESC
            """).data()
            
            print("Final entity counts by type:")
            for stat in entity_stats:
                print(f"  {stat['type']}: {stat['count']}")
            
            # Show some sample merged entities (if any)
            sample_entities = session.run("""
                MATCH (e:__Entity__)
                WHERE e.description IS NOT NULL
                RETURN e.id, e.entity_type, e.description
                LIMIT 10
            """).data()
            
            print("\nSample entities:")
            for entity in sample_entities:
                print(f"  {entity['entity_type']}: {entity['id']} - {entity['description'][:50]}...")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        
    finally:
        processor.close()
        print("\n✅ Test completed")


if __name__ == "__main__":
    main() 