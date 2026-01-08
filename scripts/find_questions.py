"""Find easy and hard questions based on current database state."""
import json
import os
from neo4j import GraphDatabase

# Load corpus
with open('data/hotpotqa/prepared_corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

questions = corpus['questions'][:100]

# Get ingested articles from Neo4j
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
neo4j_db = os.environ.get('CLIENT_NEO4J_DATABASE', 'neo4j')

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

with driver.session(database=neo4j_db) as session:
    result = session.run("""
        MATCH (d:Document)
        WHERE d.name STARTS WITH 'wiki_'
        WITH d.name as name
        WITH split(name, '_') as parts
        WITH parts[1..size(parts)-1] as title_parts
        WITH reduce(s = '', x IN title_parts | s + CASE WHEN s = '' THEN '' ELSE '_' END + x) as title
        RETURN collect(DISTINCT title) as titles
    """)
    record = result.single()
    ingested_titles = set(record['titles']) if record else set()

driver.close()

print(f"Ingested articles: {len(ingested_titles)}")
print()

# Analyze each question
question_analysis = []
for q in questions:
    q_id = q.get('_id', 'unknown')
    question_text = q['question']
    q_type = q.get('type', 'unknown')
    answer = q.get('answer', '')
    
    # Get required articles
    required = set()
    for sf in q.get('supporting_facts', []):
        if isinstance(sf, (list, tuple)) and len(sf) >= 1:
            required.add(sf[0])
    
    # Check coverage
    missing = required - ingested_titles
    coverage = len(required - missing) / len(required) if required else 0
    
    question_analysis.append({
        'id': q_id,
        'question': question_text,
        'answer': answer,
        'type': q_type,
        'required': required,
        'missing': missing,
        'coverage': coverage,
        'is_answerable': len(missing) == 0
    })

# Find answerable questions
answerable = [q for q in question_analysis if q['is_answerable']]
print(f"Total answerable questions: {len(answerable)}")

# Separate by type
bridge_answerable = [q for q in answerable if q['type'] == 'bridge']
comparison_answerable = [q for q in answerable if q['type'] == 'comparison']

print(f"  - Bridge questions: {len(bridge_answerable)}")
print(f"  - Comparison questions: {len(comparison_answerable)}")

# Show EASY question (simple bridge)
if bridge_answerable:
    print("\n" + "="*60)
    print("EASY QUESTION (Bridge - simpler multi-hop):")
    print("="*60)
    easy = bridge_answerable[0]
    print(f"Type: {easy['type']}")
    print(f"Question: {easy['question']}")
    print(f"Answer: {easy['answer']}")
    print(f"Required articles: {easy['required']}")

# Show HARD ANSWERABLE question (comparison type - requires comparing entities)
if comparison_answerable:
    print("\n" + "="*60)
    print("HARD ANSWERABLE QUESTION (Comparison - requires entity comparison):")
    print("="*60)
    hard = comparison_answerable[0]
    print(f"Type: {hard['type']}")
    print(f"Question: {hard['question']}")
    print(f"Answer: {hard['answer']}")
    print(f"Required articles: {hard['required']}")
else:
    # Fall back to bridge question with longer answer
    print("\n" + "="*60)
    print("HARDER ANSWERABLE QUESTION (longer answer / more complex):")
    print("="*60)
    sorted_by_ans_len = sorted(answerable, key=lambda x: len(str(x['answer'])), reverse=True)
    if sorted_by_ans_len:
        hard = sorted_by_ans_len[0]
        print(f"Type: {hard['type']}")
        print(f"Question: {hard['question']}")
        print(f"Answer: {hard['answer']}")
        print(f"Required articles: {hard['required']}")

# Show a few more comparison questions if available
if len(comparison_answerable) > 1:
    print("\n" + "="*60)
    print("OTHER COMPARISON QUESTIONS (also answerable):")
    print("="*60)
    for q in comparison_answerable[1:4]:
        print(f"Q: {q['question']}")
        print(f"A: {q['answer']}")
        print(f"Articles: {q['required']}")
        print()

# Update IngestionProgress
answerable_ids = [q['id'] for q in answerable]
print(f"\nUpdating Neo4j with {len(answerable_ids)} answerable questions...")

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
with driver.session(database=neo4j_db) as session:
    session.run("""
        MERGE (p:__IngestionProgress__ {id: 'current'})
        SET p.answerable_questions = $answerable_ids,
            p.answerable_count = $count,
            p.last_updated = datetime()
    """, answerable_ids=answerable_ids, count=len(answerable_ids))
    
    session.run("""
        MERGE (t:__TestableQuestions__ {id: 'current'})
        SET t.question_ids = $answerable_ids,
            t.testable_count = $count,
            t.ingested_articles = $ingested,
            t.computed_at = datetime()
    """, answerable_ids=answerable_ids, count=len(answerable_ids), ingested=len(ingested_titles))
driver.close()

print("Done!")

