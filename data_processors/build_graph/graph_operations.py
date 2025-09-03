"""
Neo4j graph operations for graph building.
Handles database setup, node/relationship creation, and entity resolution.
Supports configurable LLM models.
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple, Optional
from neo4j import GraphDatabase
from pydantic import BaseModel


# Import centralized configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_llm

# Pydantic models for structured LLM outputs
class DuplicateEntities(BaseModel):
    duplicates: List[List[str]]

class Disambiguate(BaseModel):
    canonical_name: str
    reasoning: str


class GraphOperationsMixin:
    """
    Mixin for Neo4j graph operations with configurable models.
    Handles database setup, CRUD operations, and entity resolution.
    """
    
    def __init__(self):
        # Neo4j connection from environment variables
        neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
        
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Use configurable LLM for entity resolution
        self.llm = get_llm()
        
        super().__init__()
    
    def clear_database(self, drop_schema=True):
        """
        Comprehensive database cleanup.
        
        Args:
            drop_schema (bool): If True, also drop all indexes, constraints, and vector indexes
        """
        with self.driver.session() as session:
            print("🗑️  Clearing Neo4j database...")
            
            # 1. Delete all nodes and relationships
            print("  - Deleting all nodes and relationships...")
            session.run("MATCH (n) DETACH DELETE n")
            
            if drop_schema:
                # 2. Drop all vector indexes (this is crucial for dimension mismatches)
                print("  - Dropping vector indexes...")
                try:
                    vector_indexes = session.run("SHOW INDEXES YIELD name, type WHERE type = 'VECTOR'").data()
                    for index in vector_indexes:
                        index_name = index['name']
                        print(f"    - Dropping vector index: {index_name}")
                        session.run(f"DROP INDEX {index_name}")
                except Exception as e:
                    print(f"    - Vector index cleanup: {e}")
                
                # 3. Drop all fulltext indexes
                print("  - Dropping fulltext indexes...")
                try:
                    fulltext_indexes = session.run("SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT'").data()
                    for index in fulltext_indexes:
                        index_name = index['name']
                        print(f"    - Dropping fulltext index: {index_name}")
                        session.run(f"DROP INDEX {index_name}")
                except Exception as e:
                    print(f"    - Fulltext index cleanup: {e}")
                
                # 4. Drop all regular indexes
                print("  - Dropping regular indexes...")
                try:
                    regular_indexes = session.run("SHOW INDEXES YIELD name, type WHERE type IN ['BTREE', 'RANGE']").data()
                    for index in regular_indexes:
                        index_name = index['name']
                        print(f"    - Dropping regular index: {index_name}")
                        session.run(f"DROP INDEX {index_name}")
                except Exception as e:
                    print(f"    - Regular index cleanup: {e}")
                
                # 5. Drop all constraints
                print("  - Dropping constraints...")
                try:
                    constraints = session.run("SHOW CONSTRAINTS YIELD name").data()
                    for constraint in constraints:
                        constraint_name = constraint['name']
                        print(f"    - Dropping constraint: {constraint_name}")
                        session.run(f"DROP CONSTRAINT {constraint_name}")
                except Exception as e:
                    print(f"    - Constraint cleanup: {e}")
                
                # 6. Clear any GDS projections (for community detection)
                print("  - Clearing GDS projections...")
                try:
                    projections = session.run("CALL gds.graph.list() YIELD graphName").data()
                    for projection in projections:
                        graph_name = projection['graphName']
                        print(f"    - Dropping GDS projection: {graph_name}")
                        session.run(f"CALL gds.graph.drop('{graph_name}')")
                except Exception as e:
                    print(f"    - GDS cleanup: {e}")
            
            print("✅ Comprehensive database cleanup complete")
            
            # 7. Verify cleanup
            result = session.run("MATCH (n) RETURN count(n) as node_count").single()
            node_count = result['node_count']
            if node_count == 0:
                print("✅ Verification: Database is completely empty")
            else:
                print(f"⚠️  Warning: {node_count} nodes still remain")
    
    def setup_database_schema(self):
        """Set up Neo4j database schema with constraints and indexes."""
        # Get embedding dimensions from current model configuration
        from config import get_embeddings
        embeddings_model = get_embeddings()
        
        # Determine vector dimensions based on embedding model
        if hasattr(embeddings_model, 'model') and 'nomic' in embeddings_model.model.lower():
            vector_dimensions = 768  # Ollama nomic-embed-text
        elif hasattr(embeddings_model, 'model') and 'text-embedding-3' in embeddings_model.model:
            vector_dimensions = 1536  # OpenAI text-embedding-3-small/large
        elif hasattr(embeddings_model, 'model') and 'ada-002' in embeddings_model.model:
            vector_dimensions = 1536  # OpenAI text-embedding-ada-002
        else:
            # Test actual dimensions by generating a sample embedding
            try:
                test_embedding = embeddings_model.embed_query("test")
                vector_dimensions = len(test_embedding)
                print(f"🔍 Detected embedding dimensions: {vector_dimensions}")
            except Exception as e:
                print(f"⚠️  Could not detect embedding dimensions, defaulting to 768: {e}")
                vector_dimensions = 768
        
        print(f"📐 Setting up vector indexes with {vector_dimensions} dimensions")
        
        with self.driver.session() as session:
            # Create constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Constraint may already exist: {e}")
            
            # Create full-text search indexes
            indexes = [
                "CREATE FULLTEXT INDEX entity_text_index IF NOT EXISTS FOR (e:Entity) ON EACH [e.text]",
                "CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    print(f"Index may already exist: {e}")
            
            # Create vector indexes for embeddings with dynamic dimensions
            vector_indexes = [
                f"""
                CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
                FOR (d:Document) ON (d.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {vector_dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
                f"""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {vector_dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
                f"""
                CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {vector_dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            ]
            
            for index in vector_indexes:
                try:
                    session.run(index)
                except Exception as e:
                    print(f"Vector index may already exist: {e}")
        
        print("✅ Database schema setup complete")
    

    
    def create_document_node(self, session, doc_id: str, doc_name: str, 
                           source_info: str, text: str, embedding: List[float]) -> str:
        """Create a document node in Neo4j."""
        session.run("""
            CREATE (d:Document {
                id: $doc_id,
                name: $doc_name,
                text: $text,
                embedding: $embedding,
                created_at: datetime()
            })
        """, doc_id=doc_id, doc_name=doc_name,
            text=text[:1000], embedding=embedding)
        
        return doc_id
    
    def create_chunk_nodes(self, session, chunks: List[Dict[str, Any]], 
                          doc_id: str, embeddings: List[List[float]]) -> List[str]:
        """Create chunk nodes and link them to document."""
        chunk_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{chunk['index']}"
            
            session.run("""
                CREATE (c:Chunk {
                    id: $chunk_id,
                    text: $text,
                    index: $index,
                    type: $type,
                    embedding: $embedding,
                    created_at: datetime()
                })
            """, chunk_id=chunk_id, text=chunk['text'], index=chunk['index'],
                type=chunk.get('type', 'text'), embedding=embedding)
            
            # Link chunk to document
            session.run("""
                MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id})
                CREATE (c)-[:PART_OF]->(d)
            """, doc_id=doc_id, chunk_id=chunk_id)
            
            # Create FIRST_CHUNK and NEXT_CHUNK relationships for document sequencing
            if i == 0:
                # Mark first chunk
                session.run("""
                    MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id})
                    CREATE (d)-[:FIRST_CHUNK]->(c)
                """, doc_id=doc_id, chunk_id=chunk_id)
            
            if i > 0:
                # Link to previous chunk
                prev_chunk_id = f"{doc_id}_chunk_{chunks[i-1]['index']}"
                session.run("""
                    MATCH (prev:Chunk {id: $prev_chunk_id}), (curr:Chunk {id: $chunk_id})
                    CREATE (prev)-[:NEXT_CHUNK]->(curr)
                """, prev_chunk_id=prev_chunk_id, chunk_id=chunk_id)
            
            chunk_ids.append(chunk_id)
        
        return chunk_ids
    
    def create_entity_nodes(self, session, entities_by_type: Dict[str, List[Dict[str, Any]]], 
                           doc_id: str) -> List[Tuple[str, str]]:
        """Create entity nodes and return list of (entity_id, entity_type) tuples."""
        entity_ids = []
        
        for entity_type, entities in entities_by_type.items():
            for entity in entities:
                entity_text = entity['text']
                entity_id = f"entity_{entity_text.lower().replace(' ', '_')}_{entity_type.lower()}"
                
                # Create embedding for entity
                entity_embedding = self.create_embedding(
                    f"{entity_text} {entity.get('description', '')}"
                )
                
                # Create or update entity node
                session.run(f"""
                    MERGE (e:Entity {{id: $entity_id}})
                    ON CREATE SET 
                        e.text = $entity_text,
                        e.type = $entity_type,
                        e.description = $description,
                        e.embedding = $embedding,
                        e.created_at = datetime()
                    ON MATCH SET
                        e.description = CASE WHEN e.description = '' THEN $description ELSE e.description END
                """, entity_id=entity_id, entity_text=entity_text, 
                    entity_type=entity_type, description=entity.get('description', ''),
                    embedding=entity_embedding)
                
                # Add dynamic label to entity (using APOC if available, otherwise skip)
                try:
                    session.run(f"""
                        MATCH (e:Entity {{id: $entity_id}})
                        CALL apoc.create.addLabels([e], [$entity_type])
                        YIELD node
                        RETURN node
                    """, entity_id=entity_id, entity_type=entity_type)
                except Exception:
                    # If APOC is not available, just continue without dynamic labels
                    pass
                
                # Link entity to document (kept for backward compatibility)
                session.run("""
                    MATCH (d:Document {id: $doc_id}), (e:Entity {id: $entity_id})
                    MERGE (d)-[:MENTIONS]->(e)
                """, doc_id=doc_id, entity_id=entity_id)
                
                entity_ids.append((entity_id, entity_type))
        
        return entity_ids
    
    def create_entity_nodes_for_chunk(self, session, entities_by_type: Dict[str, List[Dict[str, Any]]], 
                                    chunk_id: str, doc_id: str) -> List[Tuple[str, str]]:
        """Create entity nodes and link them to a specific chunk (aligned with original graph_processor.py)."""
        entity_ids = []
        entity_counter = 0
        
        for entity_type, entities in entities_by_type.items():
            for entity in entities:
                entity_counter += 1
                entity_name = entity['text']
                entity_description = entity.get('description', '')
                
                # Use same ID format as original: "{entity_type}_{entity_name}"
                unique_id = f"{entity_type}_{entity_name}"
                
                # Create embedding for entity
                embedding_text = f"{entity_name}: {entity_description}" if entity_description else entity_name
                entity_embedding = self.create_embedding(embedding_text)
                
                # Create or update entity node with both __Entity__ and dynamic labels (like original)
                session.run(f"""
                    MERGE (e:__Entity__ {{id: $unique_id}})
                    ON CREATE SET e.name = $name, e.description = $description, e.embedding = $embedding,
                                e.entity_type = $entity_type, e.human_readable_id = $human_id,
                                e:{entity_type}
                    ON MATCH SET e.description = CASE WHEN e.description IS NULL THEN $description ELSE e.description END,
                               e.embedding = CASE WHEN e.embedding IS NULL THEN $embedding ELSE e.embedding END,
                               e.human_readable_id = CASE WHEN e.human_readable_id IS NULL THEN $human_id ELSE e.human_readable_id END
                    WITH e
                    MERGE (c:Chunk {{id: $chunk_id}})
                    MERGE (c)-[:HAS_ENTITY]->(e)
                """, unique_id=unique_id, name=entity_name, description=entity_description, 
                    embedding=entity_embedding, chunk_id=chunk_id, human_id=entity_counter, entity_type=entity_type)
                
                entity_ids.append((entity_type, entity_name))
        
        return entity_ids
    
    def create_entity_relationships_dynamic(self, session, entity_ids: List[Tuple[str, str]], chunk_text: str = None):
        """Create meaningful RELATED_TO relationships between entities (Neo4j LLM Graph Builder style)."""
        if len(entity_ids) < 2:
            return
        
        strategy = getattr(self, 'relationship_strategy', 'smart')
        
        if strategy == "implicit":
            # No explicit entity relationships - rely only on chunk connections
            return
        
        # Use LLM-guided relationship extraction (Neo4j LLM Graph Builder approach)
        if chunk_text and len(entity_ids) <= 8:  # Limit to prevent token overflow
            meaningful_relationships = self._extract_meaningful_relationships(entity_ids, chunk_text)
            for rel in meaningful_relationships:
                self._create_related_to_relationship(session, rel)
        else:
            # Fallback: proximity-based relationships for larger entity sets
            proximity_relationships = self._discover_proximity_relationships_simple(entity_ids, chunk_text)
            for rel in proximity_relationships:
                self._create_related_to_relationship(session, rel)
    
    def _extract_meaningful_relationships(self, entity_ids: List[Tuple[str, str]], chunk_text: str) -> List[Dict[str, Any]]:
        """Extract meaningful relationships using LLM analysis (Neo4j LLM Graph Builder style)."""
        if len(entity_ids) < 2 or not chunk_text:
            return []
        
        # Prepare entity list for LLM
        entity_list = []
        for i, (entity_type, entity_name) in enumerate(entity_ids):
            entity_list.append(f"{i}: {entity_name} ({entity_type})")
        
        entities_text = "\n".join(entity_list)
        
        # LLM prompt for relationship extraction (inspired by Neo4j LLM Graph Builder)
        relationship_prompt = f"""
        Analyze the following text and identify meaningful relationships between the entities listed below.
        Only create relationships that are explicitly mentioned or strongly implied in the text.
        
        TEXT:
        {chunk_text[:1000]}  # Limit text to prevent token overflow
        
        ENTITIES:
        {entities_text}
        
        RULES:
        1. Only create relationships that are clearly supported by the text
        2. Avoid creating relationships between every entity pair
        3. Focus on meaningful connections (not just co-occurrence)
        4. Each relationship should have a clear reason/evidence
        
        Respond with a JSON array of relationships:
        [
            {{
                "source": "entity_name",
                "target": "entity_name", 
                "relationship_type": "RELATED_TO",
                "evidence": "brief description of why they are related",
                "confidence": 0.8
            }}
        ]
        
        If no meaningful relationships exist, return an empty array: []
        """
        
        try:
            response = self.llm.invoke(relationship_prompt)
            # Handle different response types (LangChain vs native Ollama)
            if hasattr(response, 'content'):
                content = response.content  # LangChain response
            else:
                content = str(response)  # Native Ollama response
            
            # Clean and parse JSON response (handle Ollama formatting issues)
            import json
            import re
            
            # Try to extract JSON from response if it's wrapped in text
            content = content.strip()
            
            # Look for JSON array or object in the response
            json_match = re.search(r'(\[.*?\]|\{.*?\})', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            # If still no valid JSON, try to extract from code blocks
            if not content.startswith('[') and not content.startswith('{'):
                json_blocks = re.findall(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
                if json_blocks:
                    content = json_blocks[0]
            
            relationships = json.loads(content)
            
            # Ensure relationships is a list
            if not isinstance(relationships, list):
                print(f"   ⚠️ Expected list of relationships, got {type(relationships)}")
                return []
            
            # Validate and convert to internal format
            validated_relationships = []
            for rel in relationships:
                # Skip if rel is not a dictionary
                if not isinstance(rel, dict):
                    print(f"   ⚠️ Expected relationship dict, got {type(rel)}: {rel}")
                    continue
                    
                if (rel.get('source') and rel.get('target') and 
                    rel.get('confidence', 0) > 0.5):  # Only high-confidence relationships
                    
                    # Find entity IDs
                    source_id = self._find_entity_id(rel['source'], entity_ids)
                    target_id = self._find_entity_id(rel['target'], entity_ids)
                    
                    if source_id and target_id and source_id != target_id:
                        validated_relationships.append({
                            'entity1_id': source_id,
                            'entity2_id': target_id,
                            'relationship_type': 'RELATED_TO',
                            'evidence': rel.get('evidence', 'LLM-identified relationship'),
                            'confidence': float(rel.get('confidence', 0.7)),
                            'source': 'llm_extraction'
                        })
            
            return validated_relationships
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"   ⚠️ LLM relationship extraction failed: {e}")
            return []
    
    def _find_entity_id(self, entity_name: str, entity_ids: List[Tuple[str, str]]) -> str:
        """Find entity ID by name"""
        for entity_type, name in entity_ids:
            if name.lower() == entity_name.lower():
                return f"{entity_type}_{name}"
        return None
    
    def _discover_proximity_relationships_simple(self, entity_ids: List[Tuple[str, str]], chunk_text: str) -> List[Dict[str, Any]]:
        """Create simple proximity-based relationships as fallback."""
        if not chunk_text or len(entity_ids) < 2:
            return []
        
        relationships = []
        max_relationships = min(len(entity_ids) * 2, 10)  # Limit to prevent explosion
        
        # Only connect entities that are relatively close in the text
        entity_positions = {}
        for entity_type, entity_name in entity_ids:
            pos = chunk_text.lower().find(entity_name.lower())
            if pos != -1:
                entity_positions[f"{entity_type}_{entity_name}"] = pos
        
        # Sort by position and only connect adjacent entities
        sorted_entities = sorted(entity_positions.items(), key=lambda x: x[1])
        
        for i in range(min(len(sorted_entities) - 1, max_relationships)):
            entity1_id, pos1 = sorted_entities[i]
            entity2_id, pos2 = sorted_entities[i + 1]
            
            # Only connect if they're reasonably close (within 200 characters)
            if abs(pos2 - pos1) <= 200:
                relationships.append({
                    'entity1_id': entity1_id,
                    'entity2_id': entity2_id,
                    'relationship_type': 'RELATED_TO',
                    'evidence': f'Entities appear close together in text (distance: {abs(pos2 - pos1)} chars)',
                    'confidence': max(0.3, 1.0 - abs(pos2 - pos1) / 200.0),
                    'source': 'proximity_fallback'
                })
        
        return relationships
    
    def _create_related_to_relationship(self, session, relationship: Dict[str, Any]):
        """Create a RELATED_TO relationship with metadata."""
        session.run("""
            MATCH (e1:__Entity__ {id: $entity1_id})
            MATCH (e2:__Entity__ {id: $entity2_id})
            MERGE (e1)-[r:RELATED_TO]-(e2)
            ON CREATE SET r.evidence = $evidence,
                         r.confidence = $confidence,
                         r.source = $source,
                         r.created_at = datetime(),
                         r.count = 1
            ON MATCH SET r.count = r.count + 1,
                        r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
                        r.evidence = CASE WHEN $confidence > r.confidence THEN $evidence ELSE r.evidence END
        """,
        entity1_id=relationship['entity1_id'],
        entity2_id=relationship['entity2_id'],
        evidence=relationship['evidence'],
        confidence=relationship['confidence'],
        source=relationship['source'])
    
    def _identify_entity_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to identify relationships between entities."""
        if len(entities) < 2:
            return []
        
        # Prepare entity context
        entity_context = []
        for entity in entities:
            context = f"{entity['text']} ({entity['type']})"
            if entity['description']:
                context += f": {entity['description']}"
            entity_context.append(context)
        
        prompt = f"""
        Given these entities from a document:
        {chr(10).join(f"{i+1}. {ctx}" for i, ctx in enumerate(entity_context))}
        
        Identify the most likely relationships between these entities.
        Focus on meaningful, semantic relationships that would be useful in a knowledge graph.
        
        Return a JSON list of relationships:
        [
            {{
                "entity1_index": 0,
                "entity2_index": 1,
                "relationship_type": "WORKS_FOR",
                "confidence": 0.8
            }}
        ]
        
        Only include relationships with confidence >= 0.6.
        Limit to 5 most important relationships.
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # Clean response: remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]   # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
            
            relationships_data = json.loads(content)
            
            # Convert to our format
            relationships = []
            for rel in relationships_data:
                if (isinstance(rel, dict) and 
                    'entity1_index' in rel and 'entity2_index' in rel and
                    0 <= rel['entity1_index'] < len(entities) and
                    0 <= rel['entity2_index'] < len(entities) and
                    rel.get('confidence', 0) >= 0.6):
                    
                    relationships.append({
                        'entity1_id': entities[rel['entity1_index']]['id'],
                        'entity2_id': entities[rel['entity2_index']]['id'],
                        'relationship_type': rel.get('relationship_type', 'RELATED_TO'),
                        'confidence': rel.get('confidence', 0.7)
                    })
            
            return relationships
            
        except Exception as e:
            print(f"Warning: Relationship identification failed: {e}")
            return []
    
    def create_chunk_similarity_relationships(self, similarity_threshold: float = 0.95):
        """Create SIMILAR relationships between semantically similar chunks."""
        print(f"🔗 Creating chunk similarity relationships (threshold: {similarity_threshold})...")
        
        with self.driver.session() as session:
            # Get all chunk embeddings
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.embedding IS NOT NULL
                RETURN c.id as chunk_id, c.embedding as embedding
            """)
            
            chunks = [(record['chunk_id'], record['embedding']) for record in result]
            
            if len(chunks) < 2:
                print("Not enough chunks with embeddings for similarity analysis")
                return
            
            # Calculate similarities and create relationships
            relationships_created = 0
            for i in range(len(chunks)):
                for j in range(i + 1, len(chunks)):
                    chunk1_id, emb1 = chunks[i]
                    chunk2_id, emb2 = chunks[j]
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(emb1, emb2)
                    
                    if similarity >= similarity_threshold:
                        session.run("""
                            MATCH (c1:Chunk {id: $chunk1_id}), (c2:Chunk {id: $chunk2_id})
                            MERGE (c1)-[r:SIMILAR]->(c2)
                            SET r.similarity = $similarity
                        """, chunk1_id=chunk1_id, chunk2_id=chunk2_id, similarity=similarity)
                        
                        relationships_created += 1
            
            print(f"✅ Created {relationships_created} similarity relationships")
    
    # ==================== NEW RELATIONSHIP DISCOVERY METHODS ====================
    
    def _discover_semantic_relationships(self, entity_ids: List[Tuple[str, str]], chunk_text: str) -> List[Dict[str, Any]]:
        """Use LLM to discover meaningful semantic relationships between entities."""
        if len(entity_ids) < 2 or len(entity_ids) > 10:  # Avoid token overflow
            return []
        
        # Prepare entity context for LLM
        entity_context = []
        for entity_type, entity_name in entity_ids:
            entity_context.append(f"{entity_name} ({entity_type})")
        
        prompt = f"""
        Analyze the following text and identify ONLY the most important semantic relationships 
        between these entities. Focus on meaningful, explicit relationships that would be valuable 
        for knowledge retrieval.
        
        Entities: {', '.join(entity_context)}
        
        Text: {chunk_text[:2000]}  
        
        Return a JSON list of relationships with high confidence (>0.7):
        [
            {{
                "entity1": "entity_name1",
                "entity2": "entity_name2", 
                "relationship_type": "WORKS_FOR|LOCATED_IN|PART_OF|OWNS|etc",
                "confidence": 0.8,
                "evidence": "brief text evidence"
            }}
        ]
        
        Only include relationships explicitly supported by the text. Maximum 5 relationships.
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean JSON response
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            relationships = json.loads(content)
            
            # Validate and convert to our format
            valid_relationships = []
            for rel in relationships:
                if (isinstance(rel, dict) and 
                    'entity1' in rel and 'entity2' in rel and 
                    'relationship_type' in rel and
                    rel.get('confidence', 0) >= 0.7):
                    
                    # Find matching entity IDs
                    entity1_id = None
                    entity2_id = None
                    for entity_type, entity_name in entity_ids:
                        if entity_name.lower() == rel['entity1'].lower():
                            entity1_id = f"{entity_type}_{entity_name}"
                        if entity_name.lower() == rel['entity2'].lower():
                            entity2_id = f"{entity_type}_{entity_name}"
                    
                    if entity1_id and entity2_id and entity1_id != entity2_id:
                        valid_relationships.append({
                            'entity1_id': entity1_id,
                            'entity2_id': entity2_id,
                            'relationship_type': rel['relationship_type'],
                            'confidence': rel['confidence'],
                            'evidence': rel.get('evidence', ''),
                            'source': 'semantic_discovery'
                        })
            
            return valid_relationships
            
        except Exception as e:
            print(f"Warning: Semantic relationship discovery failed: {e}")
            return []
    
    def _discover_proximity_relationships(self, entity_ids: List[Tuple[str, str]], chunk_text: str, max_distance: int = 100) -> List[Dict[str, Any]]:
        """Discover relationships based on proximity in text."""
        if len(entity_ids) < 2:
            return []
        
        # Find positions of entities in text
        entity_positions = {}
        text_lower = chunk_text.lower()
        
        for entity_type, entity_name in entity_ids:
            entity_id = f"{entity_type}_{entity_name}"
            pos = text_lower.find(entity_name.lower())
            if pos != -1:
                entity_positions[entity_id] = pos
        
        # Create proximity relationships
        proximity_relationships = []
        processed_pairs = set()
        
        for i, (type1, name1) in enumerate(entity_ids):
            entity1_id = f"{type1}_{name1}"
            if entity1_id not in entity_positions:
                continue
                
            for j, (type2, name2) in enumerate(entity_ids[i+1:], i+1):
                entity2_id = f"{type2}_{name2}"
                if entity2_id not in entity_positions:
                    continue
                
                # Avoid duplicate pairs
                pair_key = tuple(sorted([entity1_id, entity2_id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                # Calculate distance
                pos1 = entity_positions[entity1_id]
                pos2 = entity_positions[entity2_id]
                distance = abs(pos1 - pos2)
                
                if distance <= max_distance:
                    # Calculate proximity score (closer = higher score)
                    proximity_score = max(0.1, 1.0 - (distance / max_distance))
                    
                    proximity_relationships.append({
                        'entity1_id': entity1_id,
                        'entity2_id': entity2_id,
                        'relationship_type': 'CO_OCCURS',
                        'distance': distance,
                        'proximity_score': proximity_score,
                        'source': 'proximity_based'
                    })
        
        return proximity_relationships
    
    def _create_semantic_relationship(self, session, relationship: Dict[str, Any]):
        """Create a semantic relationship in the graph."""
        session.run(f"""
            MATCH (e1:__Entity__ {{id: $entity1_id}})
            MATCH (e2:__Entity__ {{id: $entity2_id}})
            MERGE (e1)-[r:{relationship['relationship_type']}]->(e2)
            SET r.confidence = $confidence,
                r.evidence = $evidence,
                r.source = $source,
                r.created_at = datetime()
        """, 
        entity1_id=relationship['entity1_id'],
        entity2_id=relationship['entity2_id'], 
        confidence=relationship['confidence'],
        evidence=relationship['evidence'],
        source=relationship['source'])
    
    def _create_proximity_relationship(self, session, relationship: Dict[str, Any]):
        """Create a proximity-based relationship in the graph."""
        session.run("""
            MATCH (e1:__Entity__ {id: $entity1_id})
            MATCH (e2:__Entity__ {id: $entity2_id})
            MERGE (e1)-[r:CO_OCCURS]-(e2)
            ON CREATE SET r.distance = $distance,
                         r.proximity_score = $proximity_score,
                         r.source = $source,
                         r.co_occurrences = 1,
                         r.created_at = datetime()
            ON MATCH SET r.co_occurrences = r.co_occurrences + 1,
                        r.distance = CASE WHEN $distance < r.distance THEN $distance ELSE r.distance END,
                        r.proximity_score = CASE WHEN $proximity_score > r.proximity_score THEN $proximity_score ELSE r.proximity_score END
        """,
        entity1_id=relationship['entity1_id'],
        entity2_id=relationship['entity2_id'],
        distance=relationship['distance'],
        proximity_score=relationship['proximity_score'],
        source=relationship['source'])
    
    def _update_cooccurrence_stats(self, session, entity_ids: List[Tuple[str, str]]):
        """Update co-occurrence statistics without creating explicit relationships."""
        if len(entity_ids) < 2:
            return
        
        # Create or update a co-occurrence summary for this chunk's entities
        entity_list = [f"{entity_type}_{entity_name}" for entity_type, entity_name in entity_ids]
        
        # Store co-occurrence data as properties on entities rather than relationships
        for entity_type, entity_name in entity_ids:
            entity_id = f"{entity_type}_{entity_name}"
            cooccurring_entities = [eid for eid in entity_list if eid != entity_id]
            
            session.run("""
                MATCH (e:__Entity__ {id: $entity_id})
                SET e.cooccurring_entities = CASE 
                    WHEN e.cooccurring_entities IS NULL THEN $cooccurring_entities
                    ELSE e.cooccurring_entities + [x IN $cooccurring_entities WHERE NOT x IN e.cooccurring_entities]
                END,
                e.cooccurrence_count = CASE
                    WHEN e.cooccurrence_count IS NULL THEN size($cooccurring_entities)
                    ELSE e.cooccurrence_count + size($cooccurring_entities)
                END
            """, entity_id=entity_id, cooccurring_entities=cooccurring_entities)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def entity_resolution(self, entities: List[str]) -> Optional[List[str]]:
        """Resolve duplicate entities using LLM analysis."""
        if len(entities) < 2:
            return None
        
        prompt = f"""
        Analyze these entity names and group any that refer to the same real-world entity:
        {json.dumps(entities)}
        
        Consider:
        - Exact matches (case-insensitive)
        - Abbreviations and full forms
        - Synonyms and alternative names
        - Typos and variations
        
        Return groups of duplicates as a JSON object:
        {{"duplicates": [["entity1", "entity2"], ["entity3", "entity4", "entity5"]]}}
        
        Only include groups with 2+ entities. If no duplicates found, return {{"duplicates": []}}.
        """
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)
            return result.get('duplicates', [])
        except Exception as e:
            print(f"Warning: Entity resolution failed: {e}")
            return None
    
    def perform_entity_resolution(self, similarity_threshold: float = 0.95, 
                                word_edit_distance: int = 3, max_workers: int = 4):
        """Perform comprehensive entity resolution across the graph."""
        print("🔍 Starting entity resolution...")
        
        with self.driver.session() as session:
            # Get all entities
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.id as id, e.text as text, e.type as type
            """)
            
            entities_by_type = {}
            for record in result:
                entity_type = record['type']
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append({
                    'id': record['id'],
                    'text': record['text']
                })
            
            total_merged = 0
            
            # Process each entity type separately
            for entity_type, entities in entities_by_type.items():
                if len(entities) < 2:
                    continue
                
                print(f"🔍 Resolving {entity_type} entities ({len(entities)} entities)...")
                
                # Use LLM for duplicate detection
                entity_texts = [e['text'] for e in entities]
                duplicates = self.entity_resolution(entity_texts)
                
                if duplicates:
                    for duplicate_group in duplicates:
                        if len(duplicate_group) >= 2:
                            # Find entity IDs for this group
                            entity_ids = []
                            for text in duplicate_group:
                                for entity in entities:
                                    if entity['text'].lower() == text.lower():
                                        entity_ids.append(entity['id'])
                                        break
                            
                            if len(entity_ids) >= 2:
                                merged_count = self._merge_entities(session, entity_ids)
                                total_merged += merged_count
            
            print(f"✅ Entity resolution complete. Merged {total_merged} entities.")
    
    def _merge_entities(self, session, entity_ids: List[str]) -> int:
        """Merge duplicate entities into the first one."""
        if len(entity_ids) < 2:
            return 0
        
        primary_id = entity_ids[0]
        duplicate_ids = entity_ids[1:]
        
        # Transfer all relationships to primary entity
        for dup_id in duplicate_ids:
            # Transfer incoming relationships
            session.run("""
                MATCH (n)-[r]->(e:Entity {id: $dup_id})
                MATCH (primary:Entity {id: $primary_id})
                CREATE (n)-[r2:RELATED_TO]->(primary)
                SET r2 = properties(r)
                DELETE r
            """, dup_id=dup_id, primary_id=primary_id)
            
            # Transfer outgoing relationships
            session.run("""
                MATCH (e:Entity {id: $dup_id})-[r]->(n)
                MATCH (primary:Entity {id: $primary_id})
                CREATE (primary)-[r2:RELATED_TO]->(n)
                SET r2 = properties(r)
                DELETE r
            """, dup_id=dup_id, primary_id=primary_id)
            
            # Delete duplicate entity
            session.run("""
                MATCH (e:Entity {id: $dup_id})
                DELETE e
            """, dup_id=dup_id)
        
        return len(duplicate_ids)
    
    def close(self):
        """Close the Neo4j driver."""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
