"""
Test script for smart RELATED_TO relationship extraction

This script demonstrates how the new LLM-guided relationship extraction works
to avoid the "everything connected to everything" problem.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_relationship_extraction():
    """Test the smart RELATED_TO relationship extraction"""
    print("🧪 Testing Smart RELATED_TO Relationship Extraction")
    print("=" * 60)
    
    try:
        from data_processors.build_graph.graph_operations import GraphOperationsMixin
        from config import get_llm
        import neo4j
        
        # Create a test instance
        class TestProcessor(GraphOperationsMixin):
            def __init__(self):
                self.llm = get_llm()
                self.driver = neo4j.GraphDatabase.driver(
                    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                    auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), 
                          os.getenv('NEO4J_PASSWORD', 'password'))
                )
        
        processor = TestProcessor()
        
        # Test case 1: Clear relationships should be extracted
        print("\n📝 Test Case 1: Clear Relationships")
        test_text_1 = """
        John Smith works for Microsoft Corporation as a Senior Engineer. 
        The company is headquartered in Seattle, Washington. 
        Microsoft develops software products and cloud services.
        """
        
        entity_ids_1 = [
            ("PERSON", "John Smith"),
            ("ORGANIZATION", "Microsoft Corporation"), 
            ("LOCATION", "Seattle"),
            ("LOCATION", "Washington"),
            ("ROLE", "Senior Engineer")
        ]
        
        relationships_1 = processor._extract_meaningful_relationships(entity_ids_1, test_text_1)
        
        print(f"   Entities: {len(entity_ids_1)}")
        print(f"   Relationships found: {len(relationships_1)}")
        for rel in relationships_1:
            print(f"   - {rel['entity1_id']} → {rel['entity2_id']}")
            print(f"     Evidence: {rel['evidence']}")
            print(f"     Confidence: {rel['confidence']:.2f}")
        
        # Test case 2: Weak relationships should be filtered out
        print("\n📝 Test Case 2: Weak Relationships (Should be filtered)")
        test_text_2 = """
        The report mentions various topics including artificial intelligence,
        machine learning, data science, and blockchain technology.
        These are important areas of research.
        """
        
        entity_ids_2 = [
            ("TECHNOLOGY", "artificial intelligence"),
            ("TECHNOLOGY", "machine learning"),
            ("TECHNOLOGY", "data science"),
            ("TECHNOLOGY", "blockchain technology"),
            ("CONCEPT", "research")
        ]
        
        relationships_2 = processor._extract_meaningful_relationships(entity_ids_2, test_text_2)
        
        print(f"   Entities: {len(entity_ids_2)}")
        print(f"   Relationships found: {len(relationships_2)}")
        for rel in relationships_2:
            print(f"   - {rel['entity1_id']} → {rel['entity2_id']}")
            print(f"     Evidence: {rel['evidence']}")
            print(f"     Confidence: {rel['confidence']:.2f}")
        
        # Test case 3: Proximity fallback for large entity sets
        print("\n📝 Test Case 3: Proximity Fallback")
        test_text_3 = """
        The quarterly financial report shows revenue growth. 
        Apple Inc reported strong iPhone sales in Q3 2024.
        The company's stock price increased significantly.
        """
        
        entity_ids_3 = [
            ("DOCUMENT", "quarterly financial report"),
            ("METRIC", "revenue growth"),
            ("ORGANIZATION", "Apple Inc"),
            ("PRODUCT", "iPhone"),
            ("METRIC", "sales"),
            ("TIME", "Q3 2024"),
            ("FINANCIAL", "stock price"),
            ("ORGANIZATION", "company"),
            ("TREND", "increased"),
        ]  # 9 entities - should trigger proximity fallback
        
        relationships_3 = processor._discover_proximity_relationships_simple(entity_ids_3, test_text_3)
        
        print(f"   Entities: {len(entity_ids_3)} (triggers proximity fallback)")
        print(f"   Relationships found: {len(relationships_3)}")
        for rel in relationships_3:
            print(f"   - {rel['entity1_id']} → {rel['entity2_id']}")
            print(f"     Evidence: {rel['evidence']}")
            print(f"     Confidence: {rel['confidence']:.2f}")
        
        processor.driver.close()
        
        # Analyze results
        print("\n📊 Analysis:")
        print(f"   Test 1 (Clear relationships): {len(relationships_1)} relationships")
        print(f"   Test 2 (Weak relationships): {len(relationships_2)} relationships") 
        print(f"   Test 3 (Proximity fallback): {len(relationships_3)} relationships")
        
        # Success criteria
        success = True
        if len(relationships_1) == 0:
            print("   ⚠️  Expected some relationships in Test 1")
            success = False
        
        if len(relationships_2) > len(entity_ids_2) // 2:
            print("   ⚠️  Too many relationships in Test 2 (should filter weak ones)")
            success = False
        
        if len(relationships_3) > 8:
            print("   ⚠️  Too many relationships in Test 3 (proximity should be limited)")
            success = False
        
        return success
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_relationship_strategies():
    """Demonstrate different relationship strategies"""
    print("\n🎯 Relationship Strategy Comparison")
    print("=" * 50)
    
    strategies = [
        ("smart", "LLM-guided + proximity fallback"),
        ("semantic", "LLM-guided only"),
        ("proximity", "Proximity-based only"),
        ("implicit", "No explicit relationships")
    ]
    
    for strategy, description in strategies:
        print(f"\n**{strategy.upper()}** Strategy:")
        print(f"   {description}")
        print(f"   Usage: processor = CustomGraphProcessor(relationship_strategy='{strategy}')")

if __name__ == "__main__":
    print("🚀 Smart RELATED_TO Relationship System")
    print("Based on Neo4j LLM Graph Builder approach")
    print("=" * 60)
    
    success = test_relationship_extraction()
    demonstrate_relationship_strategies()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Smart relationship extraction is working correctly!")
        print("\n🎯 Key Benefits:")
        print("- LLM analyzes text to find meaningful relationships")
        print("- Only creates relationships with clear evidence")
        print("- Confidence scoring filters weak connections")
        print("- Proximity fallback prevents relationship explosion")
        print("- Single RELATED_TO type with rich metadata")
    else:
        print("❌ Some tests failed - check the implementation")
        
    print("\n📚 How it avoids the 'everything connected' problem:")
    print("1. LLM only creates relationships with textual evidence")
    print("2. Confidence threshold filters weak relationships")  
    print("3. Entity limit prevents processing too many entities at once")
    print("4. Proximity fallback only connects adjacent entities")
    print("5. Relationship count limits prevent explosion")
