import os
from dotenv import load_dotenv
import neo4j

load_dotenv()

driver = neo4j.GraphDatabase.driver(
    os.environ.get('NEO4J_URI'), 
    auth=(os.environ.get('NEO4J_USERNAME'), os.environ.get('NEO4J_PASSWORD'))
)

result = driver.execute_query('SHOW INDEXES')
print("Current indexes:")
for r in result.records:
    print(f"  {r['name']} - {r['type']} - {r['labelsOrTypes']} - {r['properties']}")

driver.close()
