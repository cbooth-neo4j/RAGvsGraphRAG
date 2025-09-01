import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings

load_dotenv()

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

FULLTEXT_INDEX = os.environ.get("NEO4J_FULLTEXT_INDEX", "chunk_text_fulltext")
VECTOR_INDEX = os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
out_path = os.path.join(os.getcwd(), "debug_out.txt")
out_lines = []

def run(q, **params):
    with driver.session(database=NEO4J_DATABASE) as s:
        return s.run(q, **params).data()

out_lines.append("== Node counts ==")
counts = run(
    """
    CALL {
      MATCH (c:Chunk) RETURN 'Chunk' AS label, count(c) AS count
      UNION ALL
      MATCH (d:Document) RETURN 'Document' AS label, count(d) AS count
      UNION ALL
      MATCH (e:__Entity__) RETURN '__Entity__' AS label, count(e) AS count
    }
    RETURN label, count
    """
)
for row in counts:
    out_lines.append(f"{row['label']}: {row['count']}")

out_lines.append("\n== Embedding dimensions (Chunk) ==")
dims = run(
    """
    MATCH (c:Chunk)
    WHERE c.embedding IS NOT NULL
    RETURN size(c.embedding) AS dim, count(*) AS n
    ORDER BY n DESC
    LIMIT 5
    """
)
if not dims:
    out_lines.append("No embeddings found on :Chunk nodes")
else:
    for row in dims:
        out_lines.append(f"dim={row['dim']} -> nodes={row['n']}")

out_lines.append("\n== Fulltext indexes ==")
fti = run(
    """
    SHOW FULLTEXT INDEXES YIELD name, entityType, labelsOrTypes, properties, state
    RETURN name, entityType, labelsOrTypes, properties, state
    ORDER BY name
    """
)
for row in fti:
    out_lines.append(str(row))

out_lines.append("\n== Vector indexes (SHOW INDEXES) ==")
vec = run(
    """
    SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state
    WHERE type = 'VECTOR'
    RETURN name, entityType, labelsOrTypes, properties, state
    ORDER BY name
    """
)
for row in vec:
    out_lines.append(str(row))

out_lines.append("\n== Fulltext smoke test (bank*) ==")
try:
    res = run(
        """
        CALL db.index.fulltext.queryNodes($idx, $q, {limit: 5})
        YIELD node, score
        RETURN labels(node) AS labels, coalesce(node.id, node.name, left(node.text, 60)) AS id, score
        """,
        idx=FULLTEXT_INDEX,
        q="bank*",
    )
    if not res:
        out_lines.append("No fulltext results for 'bank*'")
    else:
        for row in res:
            out_lines.append(str(row))
except Exception as e:
    out_lines.append(f"Fulltext query failed: {e}")

out_lines.append("\n== Fulltext smoke test (proposal) ==")
try:
    res2 = run(
        """
        CALL db.index.fulltext.queryNodes($idx, $q, {limit: 5})
        YIELD node, score
        RETURN labels(node) AS labels, coalesce(node.id, node.name, left(node.text, 60)) AS id, score
        """,
        idx=FULLTEXT_INDEX,
        q="proposal",
    )
    if not res2:
        out_lines.append("No fulltext results for 'proposal'")
    else:
        for row in res2:
            out_lines.append(str(row))
except Exception as e:
    out_lines.append(f"Fulltext query failed: {e}")

out_lines.append("\n== Vector index smoke test (query: 'banking services') ==")
try:
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vec = emb.embed_query("banking services")
    res3 = run(
        """
        CALL db.index.vector.queryNodes($idx, 5, $v)
        YIELD node, score
        RETURN labels(node) AS labels, coalesce(node.id, node.name, left(node.text, 80)) AS id, score
        """,
        idx=VECTOR_INDEX,
        v=vec,
    )
    if not res3:
        out_lines.append("No vector results for 'banking services'")
    else:
        for row in res3:
            out_lines.append(str(row))
except Exception as e:
    out_lines.append(f"Vector query failed: {e}")

driver.close()

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print(f"Wrote diagnostic output to {out_path}")


