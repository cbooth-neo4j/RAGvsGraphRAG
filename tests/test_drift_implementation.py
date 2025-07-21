"""
Test script for DRIFT Implementation

This script tests the modular DRIFT system to ensure all components work correctly.
It verifies the context building, primer, state management, actions, and search orchestration.

WARNING: This script will make LLM API calls which incur costs.
Make sure you understand the pricing before running.
"""

import os
import sys
import warnings
from typing import Dict, List, Any
from dotenv import load_dotenv

# Add parent directory to path so we can import from the main project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import DRIFT modules
from retrievers.drift_modules.drift_context import DRIFTContextBuilder
from retrievers.drift_modules.drift_primer import DRIFTPrimer
from retrievers.drift_modules.drift_state import DRIFTQueryState
from retrievers.drift_modules.drift_action import DRIFTAction
from retrievers.drift_modules.drift_search import DRIFTSearch
from retrievers.drift_graphrag_retriever import DriftGraphRAGRetriever

def check_prerequisites():
    """Check if all required environment variables and dependencies are available"""
    print("🔍 Checking Prerequisites...")
    
    # Check environment variables
    required_vars = ['OPENAI_API_KEY', 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    # Check if Neo4j is accessible
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get('NEO4J_URI'),
            auth=(os.environ.get('NEO4J_USERNAME'), os.environ.get('NEO4J_PASSWORD'))
        )
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()['test']
            if test_value != 1:
                print("❌ Neo4j connection test failed")
                return False
                
        driver.close()
        print("✅ Neo4j connection successful")
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False
    
    print("✅ All prerequisites met")
    return True

def check_graph_data():
    """Check if the graph has the necessary data for DRIFT testing"""
    print("\n📊 Checking Graph Data...")
    
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get('NEO4J_URI'),
            auth=(os.environ.get('NEO4J_USERNAME'), os.environ.get('NEO4J_PASSWORD'))
        )
        
        with driver.session() as session:
            # Check entities
            entity_count = session.run("MATCH (e:__Entity__) RETURN count(e) as count").single()['count']
            
            # Check communities
            community_count = session.run("MATCH (c:__Community__) RETURN count(c) as count").single()['count']
            
            # Check chunks
            chunk_count = session.run("MATCH (c:__Chunk__) RETURN count(c) as count").single()['count']
            
            print(f"  📍 Entities: {entity_count}")
            print(f"  👥 Communities: {community_count}")
            print(f"  📄 Chunks: {chunk_count}")
            
            if entity_count == 0 or community_count == 0:
                print("❌ Insufficient graph data for testing")
                print("   Run the advanced graph processor first to create entities and communities")
                return False
            
            if chunk_count == 0:
                print("⚠️  No chunks found, but entities and communities are sufficient for DRIFT testing")
                
        driver.close()
        print("✅ Graph data looks good")
        return True
        
    except Exception as e:
        print(f"❌ Error checking graph data: {e}")
        return False

def test_drift_context_builder():
    """Test the DRIFT context builder"""
    print("\n🧪 Testing DRIFT Context Builder...")
    
    try:
        context_builder = DRIFTContextBuilder()
        
        # Test context building
        sample_query = "What are the main challenges in implementing AI systems?"
        context = context_builder.build_context(sample_query)
        
        print(f"  📝 Query: {sample_query}")
        print(f"  📚 Context built successfully")
        print(f"  🔤 Context length: {len(context)} characters")
        
        if len(context) > 0:
            print("✅ Context builder test passed")
            return True
        else:
            print("❌ Context builder returned empty context")
            return False
            
    except Exception as e:
        print(f"❌ Context builder test failed: {e}")
        return False

def test_drift_primer():
    """Test the DRIFT primer"""
    print("\n🧪 Testing DRIFT Primer...")
    
    try:
        primer = DRIFTPrimer()
        
        sample_query = "What are the key benefits of using knowledge graphs?"
        sample_context = "Knowledge graphs provide structured representation of information with entities and relationships..."
        
        # Test primer generation
        primer_result = primer.generate_primer(sample_query, sample_context)
        
        print(f"  📝 Original query: {sample_query}")
        print(f"  🎯 Primer generated successfully")
        print(f"  📋 Decomposed queries: {len(primer_result.decomposed_queries)}")
        print(f"  🎪 Follow-up questions: {len(primer_result.follow_up_questions)}")
        
        if len(primer_result.decomposed_queries) > 0:
            print("✅ Primer test passed")
            return True
        else:
            print("❌ Primer did not generate decomposed queries")
            return False
            
    except Exception as e:
        print(f"❌ Primer test failed: {e}")
        return False

def test_drift_state():
    """Test the DRIFT state management"""
    print("\n🧪 Testing DRIFT State Management...")
    
    try:
        state = DRIFTQueryState(
            original_query="What are the applications of machine learning?",
            context="Machine learning has various applications across industries...",
            max_actions=5
        )
        
        # Test state initialization
        print(f"  📝 Original query: {state.original_query}")
        print(f"  📊 Action graph nodes: {len(state.action_graph.nodes)}")
        print(f"  🎯 Max actions: {state.max_actions}")
        
        # Test adding an action
        action_id = state.add_action("test_action", "Testing action addition")
        print(f"  ➕ Added action: {action_id}")
        
        # Test action completion
        state.complete_action(action_id, {"test": "result"})
        print(f"  ✅ Action completed successfully")
        
        if action_id in state.action_graph.nodes:
            print("✅ State management test passed")
            return True
        else:
            print("❌ State management test failed")
            return False
            
    except Exception as e:
        print(f"❌ State management test failed: {e}")
        return False

def test_drift_action():
    """Test the DRIFT action execution"""
    print("\n🧪 Testing DRIFT Action Execution...")
    
    try:
        action = DRIFTAction()
        
        # Test action execution
        sample_query = "What are neural networks?"
        result = action.execute(sample_query, "local")
        
        print(f"  📝 Query: {sample_query}")
        print(f"  🎯 Action type: local")
        print(f"  📊 Result generated: {result.success}")
        
        if result.success and result.content:
            print(f"  📄 Content length: {len(result.content)} characters")
            print("✅ Action execution test passed")
            return True
        else:
            print("❌ Action execution failed")
            return False
            
    except Exception as e:
        print(f"❌ Action execution test failed: {e}")
        return False

def test_drift_search():
    """Test the complete DRIFT search system"""
    print("\n🧪 Testing DRIFT Search System...")
    
    try:
        search = DRIFTSearch()
        
        # Test search execution
        sample_query = "Explain the concept of deep learning and its applications"
        result = search.search(sample_query)
        
        print(f"  📝 Query: {sample_query}")
        print(f"  📊 Search completed successfully")
        print(f"  📄 Result length: {len(result)} characters")
        
        if len(result) > 0:
            print("✅ Search system test passed")
            return True
        else:
            print("❌ Search system returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Search system test failed: {e}")
        return False

def test_drift_retriever_modular():
    """Test the DRIFT retriever in modular mode"""
    print("\n🧪 Testing DRIFT Retriever (Modular Mode)...")
    
    try:
        from data_processors import AdvancedGraphProcessor
        processor = AdvancedGraphProcessor()
        retriever = DriftGraphRAGRetriever(processor)
        
        # Test retrieval
        sample_query = "What are the advantages of using graph databases?"
        import asyncio
        result = asyncio.run(retriever.search(sample_query, use_modular=True))
        
        print(f"  📝 Query: {sample_query}")
        print(f"  🔧 Mode: Modular")
        print(f"  📊 Result generated successfully")
        print(f"  📄 Result length: {len(result.response)} characters")
        
        if len(result.response) > 0:
            print("✅ Modular retriever test passed")
            return True
        else:
            print("❌ Modular retriever returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Modular retriever test failed: {e}")
        return False

def test_drift_retriever_legacy():
    """Test the DRIFT retriever in legacy mode"""
    print("\n🧪 Testing DRIFT Retriever (Legacy Mode)...")
    
    try:
        from data_processors import AdvancedGraphProcessor
        processor = AdvancedGraphProcessor()
        retriever = DriftGraphRAGRetriever(processor)
        
        # Test retrieval
        sample_query = "How do recommendation systems work?"
        import asyncio
        result = asyncio.run(retriever.search(sample_query, use_modular=False))
        
        print(f"  📝 Query: {sample_query}")
        print(f"  🔧 Mode: Legacy")
        print(f"  📊 Result generated successfully")
        print(f"  📄 Result length: {len(result.response)} characters")
        
        if len(result.response) > 0:
            print("✅ Legacy retriever test passed")
            return True
        else:
            print("❌ Legacy retriever returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Legacy retriever test failed: {e}")
        return False

def run_all_tests():
    """Run all DRIFT tests"""
    print("🚀 Starting DRIFT Implementation Tests")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return False
    
    # Check graph data
    if not check_graph_data():
        print("\n❌ Graph data not available. Please run the advanced graph processor first.")
        return False
    
    # Run tests
    tests = [
        test_drift_context_builder,
        test_drift_primer,
        test_drift_state,
        test_drift_action,
        test_drift_search,
        test_drift_retriever_modular,
        test_drift_retriever_legacy
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All DRIFT tests passed successfully!")
        return True
    else:
        print("❌ Some tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 