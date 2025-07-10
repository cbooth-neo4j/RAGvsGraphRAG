import os
from dotenv import load_dotenv
import neo4j

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

def check_labels():
    """Check what node labels exist in the database"""
    try:
        with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            print("="*60)
            print("NODE LABELS IN DATABASE")
            print("="*60)
            
            # Get all node labels
            result = driver.execute_query("""
                MATCH (n) 
                RETURN labels(n) as labels, count(*) as count 
                ORDER BY count DESC
            """)
            
            print("Node labels and counts:")
            for record in result.records:
                labels = record['labels']
                count = record['count']
                print(f"  {labels}: {count}")
            
            print("\n" + "="*60)
            print("ENTITY NODES ANALYSIS")
            print("="*60)
            
            # Check for nodes with 'name' property (likely entities)
            result = driver.execute_query("""
                MATCH (n) 
                WHERE n.name IS NOT NULL
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """)
            
            print("Nodes with 'name' property (entities):")
            for record in result.records:
                label = record['label']
                count = record['count']
                print(f"  {label}: {count}")
            
            print("\n" + "="*60)
            print("WARNING ANALYSIS")
            print("="*60)
            
            # Check if 'Entity' label exists
            result = driver.execute_query("MATCH (e:Entity) RETURN COUNT(e) as count")
            entity_count = result.records[0]['count']
            print(f"Nodes with 'Entity' label: {entity_count}")
            
            if entity_count == 0:
                print("❌ 'Entity' label does not exist in the database")
                print("✅ This explains the warning - the test query looks for 'Entity' label")
                print("✅ But entities are stored with specific labels like 'Organization', 'Location', etc.")
            else:
                print("✅ 'Entity' label exists")
                
    except Exception as e:
        print(f"Error checking labels: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_labels() 