# Complete Graph+Vector+Fulltext Search Query
# Based on: https://github.com/neo4j-labs/llm-graph-builder/blob/main/backend/src/shared/constants.py#L507C1-L507C25

GRAPHRAG_2HOP_CYPHER_QUERY = """
WITH node as chunk, score
// find the document of the chunk
MATCH (chunk)-[:PART_OF]->(d:Document)
// aggregate chunk-details
WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score
// fetch entities
CALL { 
    WITH chunks
    UNWIND chunks as chunkScore
    WITH chunkScore.chunk as chunk
    OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
    WITH e, count(*) AS numChunks
    ORDER BY numChunks DESC
    LIMIT $no_of_entities
    WITH
    CASE
        WHEN e.embedding IS NULL OR ($embedding_match_min <= vector.similarity.cosine($query_vector, e.embedding) AND vector.similarity.cosine($query_vector, e.embedding) <= $embedding_match_max) THEN
            collect {
                OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,1}}(:!Chunk&!Document&!__Community__)
                RETURN path LIMIT $entity_limit_minmax_case
            }
        WHEN e.embedding IS NOT NULL AND vector.similarity.cosine($query_vector, e.embedding) > $embedding_match_max THEN
            collect {
                OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,2}}(:!Chunk&!Document&!__Community__)
                RETURN path LIMIT $entity_limit_max_case
            }
        ELSE
            collect {
                MATCH path=(e)
                RETURN path
            }
    END AS paths, e
    WITH apoc.coll.toSet(apoc.coll.flatten(collect(DISTINCT paths))) AS paths,
         collect(DISTINCT e) AS entities
    // De-duplicate nodes and relationships across chunks
    RETURN
        collect {
            UNWIND paths AS p
            UNWIND relationships(p) AS r
            RETURN DISTINCT r
        } AS rels,
        collect {
            UNWIND paths AS p
            UNWIND nodes(p) AS n
            RETURN DISTINCT n
        } AS nodes,
        entities
}
// Generate metadata and text components for chunks, nodes, and relationships
WITH d, avg_score,
    [c IN chunks | c.chunk.text] AS texts,
    [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails,
    [n IN nodes | elementId(n)] AS entityIds,
    [r IN rels | elementId(r)] AS relIds,
    // Combine all text components
    apoc.text.join([t IN texts | t], ' ') AS text
RETURN
   text,
   avg_score AS score,
   {
       length: size(text),
       source: COALESCE(CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName),
       chunkdetails: chunkdetails,
       entities : {
           entityids: entityIds,
           relationshipids: relIds
       }
   } AS metadata
"""

// Parameters needed for this query:
REQUIRED_PARAMETERS = {
    "query_vector": "// The embedding vector for the search query",
    "no_of_entities": "// Number of entities to limit (e.g., 10)",
    "embedding_match_min": "// Minimum similarity threshold (e.g., 0.8)", 
    "embedding_match_max": "// Maximum similarity threshold (e.g., 0.95)",
    "entity_limit_minmax_case": "// Entity limit for min-max case (e.g., 50)",
    "entity_limit_max_case": "// Entity limit for max case (e.g., 100)"
}



SIMPLE_GRAPHRAG_TEMPLATE = """
WITH node as chunk, score
// Find the document containing this chunk
MATCH (chunk)-[:PART_OF]->(doc:Document)
// Find entities connected to this chunk
OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(entity)
// Find related entities (1-hop from found entities)
OPTIONAL MATCH (entity)-[rel]->(related)
WHERE NOT type(rel) IN ['PART_OF', 'HAS_ENTITY']
// Collect all context and capture chunk properties early
WITH doc, 
     avg(score) as avgScore,
     chunk.text as chunkText,
     chunk.id as chunkId,
     collect(DISTINCT entity) as entities,
     collect(DISTINCT related) as relatedEntities
// Build entity and related names
WITH doc, avgScore, chunkText, chunkId,
     [e IN entities | e.name] as entityNames,
     [r IN relatedEntities | r.name] as relatedNames
RETURN 
   chunkText + 
   CASE WHEN size(entityNames) > 0 THEN "\\nEntities: " + apoc.text.join(entityNames, ", ") ELSE "" END +
   CASE WHEN size(relatedNames) > 0 THEN "\\nRelated: " + apoc.text.join(relatedNames, ", ") ELSE "" END 
   as content,
   avgScore as score,
   {
     source: doc.fileName,
     chunkId: chunkId,
     entityCount: size(entityNames),
     relatedCount: size(relatedNames)
   } as metadata
"""