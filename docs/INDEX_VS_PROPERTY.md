# Neo4j Index Names vs Property Names

## Understanding the Difference

### Property Name
**What it is**: The actual field name on your nodes where data is stored.

**Example**:
```cypher
// Node with 'embedding' property
(c:Chunk {
    id: "chunk_1",
    text: "Hello world",
    embedding: [0.1, 0.2, 0.3, ...]  // ← This is the PROPERTY
})
```

### Index Name
**What it is**: A label/identifier you give to an index structure for fast lookups.

**Example**:
```cypher
// Create index with NAME 'chunk_embeddings' on PROPERTY 'embedding'
CREATE VECTOR INDEX chunk_embeddings     // ← INDEX NAME
FOR (c:Chunk) 
ON c.embedding                           // ← PROPERTY NAME
```

## Why They're Different

The index name and property name serve different purposes:

1. **Property Name** (`embedding`)
   - Must match what you stored in the database
   - Used in Cypher queries to access data
   - Same across all node types (chunks, entities, documents all use `embedding`)

2. **Index Name** (`chunk_embeddings`, `entity_embeddings`, etc.)
   - Used to identify which index to use for searches
   - Can be any descriptive name
   - Helps distinguish between indexes on different node types
   - Referenced when performing vector searches

## Example: Complete Flow

### 1. Data Ingestion (storing the property)
```python
# Store chunk with 'embedding' property
session.run("""
    CREATE (c:Chunk {
        id: $chunk_id,
        text: $text,
        embedding: $embedding_vector  // ← Property: 'embedding'
    })
""", chunk_id="c1", text="Hello", embedding_vector=[0.1, 0.2, ...])
```

### 2. Index Creation (naming the index)
```cypher
-- Create index NAMED 'chunk_embeddings' on PROPERTY 'embedding'
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (c:Chunk) ON c.embedding
OPTIONS {indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
}}
```

### 3. Vector Search (using the index name)
```python
from neo4j_graphrag.retrievers import VectorRetriever

# Use the INDEX NAME to search
retriever = VectorRetriever(
    driver=driver,
    index_name="chunk_embeddings",  // ← Reference INDEX NAME
    embedder=embeddings
)

# The retriever will:
# 1. Use the 'chunk_embeddings' index
# 2. Which points to the 'embedding' property on Chunk nodes
# 3. To find similar vectors
results = retriever.search(query_text="hello world", top_k=5)
```

## Why Separate Names?

### Multiple Indexes on Same Property Name

You might have the same property name (`embedding`) on different node types:

```cypher
// All use property name 'embedding'
(c:Chunk {embedding: [...]})
(e:__Entity__ {embedding: [...]})
(d:Document {embedding: [...]})

// But separate indexes with descriptive names
CREATE VECTOR INDEX chunk_embeddings FOR (c:Chunk) ON c.embedding
CREATE VECTOR INDEX entity_embeddings FOR (e:__Entity__) ON e.embedding
CREATE VECTOR INDEX document_embeddings FOR (d:Document) ON d.embedding
```

This lets you:
- Search only chunks: use `chunk_embeddings` index
- Search only entities: use `entity_embeddings` index
- Search only documents: use `document_embeddings` index

### Benefits

1. **Clarity**: `chunk_embeddings` is more descriptive than `embedding`
2. **Multiple indexes**: Can have many indexes on properties with the same name
3. **Flexibility**: Can create composite indexes or indexes on different node types
4. **Performance**: Can optimize each index separately

## In Your RAGvsGraphRAG Project

### Current Setup

**Property name on nodes** (what's stored):
```python
# All nodes use 'embedding' as property name
CREATE (c:Chunk {embedding: $vec})
CREATE (e:__Entity__ {embedding: $vec})
CREATE (d:Document {embedding: $vec})
```

**Index names** (what retrievers reference):
```python
# retrievers/graph_rag_retriever.py
INDEX_NAME = "chunk_embeddings"  # ← Uses this INDEX NAME

# Which searches the 'embedding' PROPERTY on Chunk nodes
```

### How Retrievers Use It

```python
# In graph_rag_retriever.py
vector_retriever = VectorRetriever(
    driver=driver,
    index_name="chunk_embeddings",  # ← INDEX NAME
    embedder=self.embeddings
)

# Behind the scenes, Neo4j uses:
# 1. The 'chunk_embeddings' index
# 2. Which is configured to search Chunk.embedding property
# 3. Returns matching Chunk nodes
```

## Summary

| Aspect | Property Name | Index Name |
|--------|--------------|------------|
| **Purpose** | Store actual data | Fast lookup structure |
| **Defined** | When creating nodes | When creating indexes |
| **Used in** | Cypher MATCH, SET, etc. | Vector search queries |
| **Scope** | Per node | Per node type + property |
| **In your code** | `c.embedding` | `"chunk_embeddings"` |
| **Can be same?** | Yes, but not required | Yes, but not recommended |
| **Naming convention** | Usually singular (`embedding`) | Usually descriptive (`chunk_embeddings`) |

## Analogy

Think of it like a database table:

- **Property name** = Column name in the table
- **Index name** = The index you create on that column for fast searches

```sql
-- SQL Analogy
CREATE TABLE chunks (
    id TEXT,
    text TEXT,
    embedding FLOAT[]  -- ← This is the column (like property)
);

-- Create an index with a name
CREATE INDEX idx_chunk_embeddings  -- ← This is the index name
ON chunks(embedding);               -- ← On the column/property

-- Use the index by querying the column
SELECT * FROM chunks WHERE embedding <-> [0.1, 0.2, ...];
```

In Neo4j, it's the same concept:
- Property = where data lives
- Index = fast lookup structure with a name
- Searcher = uses index name to find data in properties

