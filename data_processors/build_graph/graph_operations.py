"""
Neo4j graph operations for graph building.
Handles database setup, node/relationship creation, and entity resolution.
Supports configurable LLM models.
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple, Optional, Type

import neo4j
from langchain_core.utils.json_schema import dereference_refs
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from utils.graph_rag_logger import setup_logging, get_logger
from dotenv import load_dotenv

from utils.llms import get_vertex_llm, generate_dereferenced_schema  # get_vertex_llm kept for backwards compatibility

load_dotenv()

setup_logging()
logger = get_logger(__name__)


# Import centralized configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_llm

# Pydantic models for structured LLM outputs
class DuplicateEntities(BaseModel):
    duplicates: List[List[str]]
    #duplicates: List[List[str]] = Field(default_factory=list, description="List of commonly grouped entities")

class Disambiguate(BaseModel):
    canonical_name: str
    reasoning: str


class GraphOperationsMixin:
    """
    Mixin for Neo4j graph operations with configurable models.
    Handles database setup, CRUD operations, and entity resolution.
    """
    
    def _normalize_entity_id(self, entity_type: str, entity_name: str) -> str:
        """Generate normalized entity ID matching create_entity_nodes_for_chunk format."""
        return f"{entity_type}_{entity_name.lower().strip().replace(' ', '_')}"
    
    def __init__(self):
        # Neo4j connection from environment variables
        neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
        neo4j_db = os.environ.get('CLIENT_NEO4J_DATABASE', 'neo4j')  # Changed default from neo4j_db to neo4j
        self.neo4j_db = neo4j_db

        logger.info(f'GraphDatabase driver setup. Database is: {neo4j_db}')
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password), database=neo4j_db)
        
        # Use configurable LLM for entity resolution
        self.llm = get_llm()
        
        super().__init__()
    
    def clear_database(self, drop_schema=True):
        """
        Comprehensive database cleanup.
        
        Args:
            drop_schema (bool): If True, also drop all indexes, constraints, and vector indexes
        """
        with self.driver.session(database=self.neo4j_db) as session:
            print("[CLEANUP] Clearing Neo4j database...")
            
            # 1. Delete all nodes and relationships
            print("  - Deleting all nodes and relationships...")
            session.run("MATCH (n) DETACH DELETE n")
            
            if drop_schema:
                # 2. Drop all vector indexes (this is crucial for dimension mismatches)
                print("  - Dropping vector indexes...")
                logger.info(f'GraphDatabase driver setup. Dropping vector indices')

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
            
            print("[OK] Comprehensive database cleanup complete")
            
            # 7. Verify cleanup
            result = session.run("MATCH (n) RETURN count(n) as node_count").single()
            node_count = result['node_count']
            if node_count == 0:
                print("[OK] Verification: Database is completely empty")
            else:
                print(f"[WARNING] {node_count} nodes still remain")
    
    def setup_database_schema(self):
        """Set up Neo4j database schema with constraints and indexes."""
        # Get embedding dimensions from current model configuration
        logger.debug("In setup_database_schema from GraphOperationsMixin")
        from config import get_embeddings
        embeddings_model = get_embeddings()
        logger.debug(f"Embeddings model: {embeddings_model}")
        #logger.debug(f"Embeddings model name: {embeddings_model.model}")
        if hasattr(embeddings_model, 'model_name'):
            logger.debug(f"Embeddings model model_name: {embeddings_model.model_name}")

        # Determine vector dimensions based on embedding model
        if hasattr(embeddings_model, 'model') and 'nomic' in embeddings_model.model.lower():
            vector_dimensions = 768  # Ollama nomic-embed-text
        elif hasattr(embeddings_model, 'model') and 'text-embedding-3' in embeddings_model.model:
            vector_dimensions = 1536  # OpenAI text-embedding-3-small/large
        elif hasattr(embeddings_model, 'model') and 'ada-002' in embeddings_model.model:
            vector_dimensions = 1536  # OpenAI text-embedding-ada-002
        elif hasattr(embeddings_model, 'model_name') and 'text-embedding-005' in embeddings_model.model_name:
            logger.debug(f'In setup_database_schema: Return 768 dimension...')
            vector_dimensions = 768  # VectorAI text-embedding-005
        else:
            # Test actual dimensions by generating a sample embedding
            try:
                test_embedding = embeddings_model.embed_query("test")
                vector_dimensions = len(test_embedding)
                print(f"[DETECT] Detected embedding dimensions: {vector_dimensions}")
            except Exception as e:
                print(f"[WARNING] Could not detect embedding dimensions, defaulting to 768: {e}")
                logger.warning(f"Could not detect embedding dimensions, defaulting to 768: {e}")
                vector_dimensions = 768
        
        print(f"[SCHEMA] Setting up vector indexes with {vector_dimensions} dimensions")
        
        with self.driver.session(database=self.neo4j_db) as session:
            # Create constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    import traceback
                    logger.error(f"Unable to create constraint. {traceback.print_exc()}. Failed with exception {e}")
                    print(f"Constraint may already exist: {e}")
            
            # Create full-text search indexes
            indexes = [
                "CREATE FULLTEXT INDEX entity_text_index IF NOT EXISTS FOR (e:Entity) ON EACH [e.text]",
                "CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
                ## Added these from graph processor as retrievers are looking for it
                "CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.name, e.description]",
                "CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    import traceback
                    logger.error(f"Unable to create  full-text search indexes. {traceback.print_exc()}. Failed with exception {e}")
                    print(f"Index may already exist: {e}")
            
            # Drop existing vector indexes first to handle dimension changes
            drop_indexes = [
                "DROP INDEX document_embeddings IF EXISTS",
                "DROP INDEX chunk_embeddings IF EXISTS",
                "DROP INDEX entity_embeddings IF EXISTS",
                "DROP INDEX entity_embeddings_old IF EXISTS"
            ]
            
            for drop_query in drop_indexes:
                try:
                    session.run(drop_query)
                except Exception:
                    pass  # Index may not exist, that's fine
            
            # Create vector indexes for embeddings with dynamic dimensions
            vector_indexes = [
                f"""
                CREATE VECTOR INDEX document_embeddings 
                FOR (d:Document) ON (d.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {vector_dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
                f"""
                CREATE VECTOR INDEX chunk_embeddings 
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {vector_dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
                f"""
                CREATE VECTOR INDEX entity_embeddings_old
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {vector_dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
                f"""
                CREATE VECTOR INDEX entity_embeddings
                FOR (e:__Entity__) ON (e.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {vector_dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            ]

            for index in vector_indexes:
                try:
                    logger.debug(f"Creating index: {index}.")
                    session.run(index)
                except neo4j.exceptions.ClientError as e:
                    # Silently ignore "already exists" errors - this is expected on reruns
                    # Handle both EquivalentSchemaRuleAlreadyExists and IndexAlreadyExists
                    if "AlreadyExists" in str(e) or "already exists" in str(e).lower():
                        logger.debug(f"Vector index already exists (skipping): {e}")
                    else:
                        # Log actual errors
                        logger.error(f"Unable to create vector index: {e}")
                        raise
                except Exception as e:
                    # Unexpected errors
                    logger.error(f"Unexpected error creating index: {e}")
                    raise
        
        print("[OK] Database schema setup complete")
        logger.info("Database schema setup complete")
    

    
    def create_document_node(self, session, doc_id: str, doc_name: str, 
                           source_info: str, text: str, embedding: List[float]) -> str:
        """Create a document node in Neo4j."""
        logger.info(f"Creating document node for doc_name: {doc_name}")
        session.run("""
            CREATE (d:Document {
                id: $doc_id,
                name: $doc_name,
                text: $text,
                embedding: $embedding,
                created_at: datetime()
            })
        """, doc_id=doc_id, doc_name=doc_name,  text=text, embedding=embedding) #text=text[:1000], embedding=embedding)
        
        return doc_id
    
    def create_chunk_nodes(self, session, chunks: List[Dict[str, Any]], 
                          doc_id: str, embeddings: List[List[float]]) -> List[str]:
        """Create chunk nodes and link them to document.
        
        Optimized with UNWIND batch operations - creates all chunks in 2-3 queries
        instead of 3-4 queries per chunk, reducing database round-trips by 50-80%.
        """
        if not chunks:
            return []
        
        # Prepare batch data
        chunk_data = []
        chunk_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{chunk['index']}"
            chunk_ids.append(chunk_id)
            chunk_data.append({
                'chunk_id': chunk_id,
                'text': chunk['text'],
                'index': chunk['index'],
                'type': chunk.get('type', 'text'),
                'embedding': embedding,
                'is_first': i == 0,
                'prev_chunk_id': f"{doc_id}_chunk_{chunks[i-1]['index']}" if i > 0 else None
            })
        
        # Batch 1: Create all chunk nodes in ONE query using UNWIND
        session.run("""
            UNWIND $chunks AS chunk
            CREATE (c:Chunk {
                id: chunk.chunk_id,
                text: chunk.text,
                index: chunk.index,
                type: chunk.type,
                embedding: chunk.embedding,
                created_at: datetime()
            })
        """, chunks=chunk_data)
        
        # Batch 2: Link all chunks to document (PART_OF) in ONE query
        session.run("""
            MATCH (d:Document {id: $doc_id})
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {id: chunk_id})
            CREATE (c)-[:PART_OF]->(d)
        """, doc_id=doc_id, chunk_ids=chunk_ids)
        
        # Batch 3: Create FIRST_CHUNK relationship (single query)
        if chunk_ids:
            session.run("""
                MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $first_chunk_id})
                CREATE (d)-[:FIRST_CHUNK]->(c)
            """, doc_id=doc_id, first_chunk_id=chunk_ids[0])
        
        # Batch 4: Create all NEXT_CHUNK relationships in ONE query
        next_chunk_pairs = [
            {'prev_id': chunk_data[i]['chunk_id'], 'curr_id': chunk_data[i+1]['chunk_id']}
            for i in range(len(chunk_data) - 1)
        ]
        
        if next_chunk_pairs:
            session.run("""
                UNWIND $pairs AS pair
                MATCH (prev:Chunk {id: pair.prev_id}), (curr:Chunk {id: pair.curr_id})
                CREATE (prev)-[:NEXT_CHUNK]->(curr)
            """, pairs=next_chunk_pairs)
        
        logger.debug(f"Created {len(chunk_ids)} chunks for document {doc_id} (batch mode)")
        return chunk_ids
    
    def create_entity_nodes(self, session, entities_by_type: Dict[str, List[Dict[str, Any]]], 
                           doc_id: str) -> List[Tuple[str, str]]:
        """Create entity nodes and return list of (entity_id, entity_type) tuples.
        
        Optimized with batch embedding generation - generates all embeddings in a single
        API call instead of one call per entity.
        """
        logger.debug(f'In create_entity_nodes (batch optimized)')
        
        if not entities_by_type:
            return []
        
        # Phase 1: Collect all entities and prepare embedding texts
        all_entities = []
        embedding_texts = []
        
        for entity_type, entities in entities_by_type.items():
            for entity in entities:
                entity_text = entity['text']
                entity_id = f"entity_{entity_text.lower().replace(' ', '_')}_{entity_type.lower()}"
                description = entity.get('description', '')
                
                embedding_texts.append(f"{entity_text} {description}")
                all_entities.append({
                    'entity_id': entity_id,
                    'entity_text': entity_text,
                    'entity_type': entity_type,
                    'description': description
                })
        
        if not all_entities:
            return []
        
        # Phase 2: Batch generate all embeddings in ONE API call
        logger.debug(f'Batch generating {len(embedding_texts)} embeddings for entities')
        entity_embeddings = self.create_embeddings_batch(embedding_texts)
        
        # Phase 3: Write entities to Neo4j with pre-computed embeddings
        entity_ids = []
        
        for entity_data, embedding in zip(all_entities, entity_embeddings):
            entity_id = entity_data['entity_id']
            entity_text = entity_data['entity_text']
            entity_type = entity_data['entity_type']
            description = entity_data['description']
            
            # Create or update entity node (Entity label)
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
                entity_type=entity_type, description=description,
                embedding=embedding)
            
            # Create or update entity node (__Entity__ label)
            session.run(f"""
                MERGE (e:__Entity__ {{id: $entity_id}})
                ON CREATE SET 
                    e.text = $entity_text,
                    e.entity_type = $entity_type,
                    e.description = $description,
                    e.embedding = $embedding,
                    e.created_at = datetime()
                ON MATCH SET
                    e.description = CASE WHEN e.description = '' THEN $description ELSE e.description END
            """, entity_id=entity_id, entity_text=entity_text,
                entity_type=entity_type, description=description,
                embedding=embedding)
            
            # Add dynamic label to entity (using APOC if available)
            try:
                session.run(f"""
                    MATCH (e:Entity {{id: $entity_id}})
                    CALL apoc.create.addLabels([e], [$entity_type])
                    YIELD node
                    RETURN node
                """, entity_id=entity_id, entity_type=entity_type)

                session.run(f"""
                    MATCH (e:__Entity__ {{id: $entity_id}})
                    CALL apoc.create.addLabels([e], [$entity_type])
                    YIELD node
                    RETURN node
                """, entity_id=entity_id, entity_type=entity_type)
            except Exception:
                # If APOC is not available, continue without dynamic labels
                pass
            
            # Link entity to document
            session.run("""
                MATCH (d:Document {id: $doc_id}), (e:__Entity__ {id: $entity_id})
                MERGE (d)-[:MENTIONS]->(e)
            """, doc_id=doc_id, entity_id=entity_id)
            
            entity_ids.append((entity_id, entity_type))
        
        logger.debug(f'Created {len(entity_ids)} entities for document {doc_id}')
        return entity_ids
    
    def create_entity_nodes_for_chunk(self, session, entities_by_type: Dict[str, List[Dict[str, Any]]], 
                                    chunk_id: str, doc_id: str) -> List[Tuple[str, str]]:
        """Create entity nodes and link them to a specific chunk.
        
        Optimized with batch embedding generation - generates all embeddings in a single
        API call instead of one call per entity, reducing network overhead by 20-40x.
        """
        logger.debug(f"create_entity_nodes_for_chunk: Processing entities for chunk {chunk_id}")
        
        if not entities_by_type:
            return []
        
        # Phase 1: Collect all entities and prepare embedding texts
        all_entities = []  # List of (entity_type, entity_name, entity_description, unique_id, human_id)
        embedding_texts = []
        entity_counter = 0
        
        for entity_type, entities in entities_by_type.items():
            for entity in entities:
                entity_counter += 1
                entity_name = entity['text']
                entity_description = entity.get('description', '')
                unique_id = f"{entity_type}_{entity_name.lower().strip().replace(' ', '_')}"
                
                # Prepare embedding text
                embedding_text = f"{entity_name}: {entity_description}" if entity_description else entity_name
                embedding_texts.append(embedding_text.lower())
                
                all_entities.append({
                    'entity_type': entity_type,
                    'name': entity_name,
                    'description': entity_description,
                    'unique_id': unique_id,
                    'human_id': entity_counter
                })
        
        if not all_entities:
            return []
        
        # Phase 2: Batch generate all embeddings in ONE API call
        logger.debug(f"Batch generating {len(embedding_texts)} embeddings...")
        entity_embeddings = self.create_embeddings_batch(embedding_texts)
        
        # Phase 3: Write all entities to Neo4j with pre-computed embeddings
        entity_ids = []
        for entity_data, embedding in zip(all_entities, entity_embeddings):
            try:
                session.run(f"""
                    MERGE (e:__Entity__ {{id: $unique_id}})
                    ON CREATE SET e.name = $name, e.description = $description, e.embedding = $embedding,
                                e.entity_type = $entity_type, e.human_readable_id = $human_id,
                                e:{entity_data['entity_type']}
                    ON MATCH SET e.description = CASE WHEN e.description IS NULL THEN $description ELSE e.description END,
                               e.embedding = CASE WHEN e.embedding IS NULL THEN $embedding ELSE e.embedding END,
                               e.human_readable_id = CASE WHEN e.human_readable_id IS NULL THEN $human_id ELSE e.human_readable_id END
                    WITH e
                    MERGE (c:Chunk {{id: $chunk_id}})
                    MERGE (c)-[:HAS_ENTITY]->(e)
                """, 
                unique_id=entity_data['unique_id'], 
                name=entity_data['name'], 
                description=entity_data['description'],
                embedding=embedding, 
                chunk_id=chunk_id, 
                human_id=entity_data['human_id'], 
                entity_type=entity_data['entity_type'])

                entity_ids.append((entity_data['entity_type'], entity_data['name']))
            except Exception as e:
                logger.error(f"Failed to create/update entity {entity_data['name']} - {e}")
        
        logger.debug(f"Created {len(entity_ids)} entities for chunk {chunk_id}")
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
        {chunk_text} #[:1000]  # Limit text to prevent token overflow
        
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
                print(f"   [WARNING] Expected list of relationships, got {type(relationships)}")
                return []
            
            # Validate and convert to internal format
            validated_relationships = []
            for rel in relationships:
                # Skip if rel is not a dictionary
                if not isinstance(rel, dict):
                    print(f"   [WARNING] Expected relationship dict, got {type(rel)}: {rel}")
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
            print(f"   [WARNING] LLM relationship extraction failed: {e}")
            return []
    
    def _find_entity_id(self, entity_name: str, entity_ids: List[Tuple[str, str]]) -> str:
        """Find entity ID by name"""
        for entity_type, name in entity_ids:
            if name.lower() == entity_name.lower():
                return self._normalize_entity_id(entity_type, name)
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
                entity_positions[self._normalize_entity_id(entity_type, entity_name)] = pos
        
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
        print(f"[*] Creating chunk similarity relationships (threshold: {similarity_threshold})...")
        
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
            
            print(f"[OK] Created {relationships_created} similarity relationships")
    
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
                            entity1_id = self._normalize_entity_id(entity_type, entity_name)
                        if entity_name.lower() == rel['entity2'].lower():
                            entity2_id = self._normalize_entity_id(entity_type, entity_name)
                    
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
            entity_id = self._normalize_entity_id(entity_type, entity_name)
            pos = text_lower.find(entity_name.lower())
            if pos != -1:
                entity_positions[entity_id] = pos
        
        # Create proximity relationships
        proximity_relationships = []
        processed_pairs = set()
        
        for i, (type1, name1) in enumerate(entity_ids):
            entity1_id = self._normalize_entity_id(type1, name1)
            if entity1_id not in entity_positions:
                continue
                
            for j, (type2, name2) in enumerate(entity_ids[i+1:], i+1):
                entity2_id = self._normalize_entity_id(type2, name2)
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
        entity_list = [self._normalize_entity_id(entity_type, entity_name) for entity_type, entity_name in entity_ids]
        
        # Store co-occurrence data as properties on entities rather than relationships
        for entity_type, entity_name in entity_ids:
            entity_id = self._normalize_entity_id(entity_type, entity_name)
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
        logger.debug(f'In entity_resolution: {entities}')
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
            logger.debug(f"ER Response: {response}")
            
            # Handle different response types (LangChain vs native)
            if hasattr(response, 'content'):
                content = response.content.strip()
            else:
                content = str(response).strip()
            
            # Clean JSON response - remove markdown code blocks if present
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            return result.get('duplicates', [])
        except Exception as e:
            import traceback
            logger.error(f"Entity resolution failed. Exception: {e}. Traceback\n: {traceback.print_exc()}")
            print(f"Warning: Entity resolution failed: {e}")
            return None
    
    def perform_entity_resolution(self, similarity_threshold: float = 0.95, 
                                word_edit_distance: int = 3, max_workers: int = 4):
        """Perform comprehensive entity resolution across the graph.
        
        Two-phase approach:
        1. Exact name match (case-insensitive) - merge entities with same name but different types
        2. LLM-based similarity detection - find entities that refer to the same thing
        """
        print("[RESOLVE] Starting entity resolution...")
        logger.info(f"Starting entity resolution....")
        
        total_merged = 0
        
        with self.driver.session() as session:
            # ================================================================
            # PHASE 1: Merge exact name duplicates (different types)
            # ================================================================
            print("[RESOLVE] Phase 1: Merging exact name duplicates across types...")
            
            # Find entities with same normalized name but different types
            result = session.run("""
                MATCH (e:__Entity__)
                WITH toLower(trim(e.name)) as normalized_name, collect(e) as entities
                WHERE size(entities) > 1
                RETURN normalized_name, 
                       [x IN entities | {id: x.id, name: x.name, type: x.entity_type, description: x.description}] as duplicates
            """)
            
            exact_duplicates = list(result)
            print(f"[RESOLVE] Found {len(exact_duplicates)} entities with exact name duplicates")
            
            for record in exact_duplicates:
                duplicates = record['duplicates']
                if len(duplicates) >= 2:
                    # Merge preserving all type labels
                    merged_count = self._merge_entities_preserve_all_types(session, duplicates)
                    total_merged += merged_count
            
            # ================================================================
            # PHASE 2: GDS-based similarity resolution (KNN + WCC)
            # ================================================================
            print("[RESOLVE] Phase 2: Embedding-based similarity detection...")
            
            phase2_merged, flagged_for_review = self._resolve_entities_with_gds(
                session, 
                high_confidence_threshold=0.95,
                medium_confidence_threshold=0.85
            )
            total_merged += phase2_merged
            
            print(f"[RESOLVE] Entity resolution complete. Merged {total_merged} entities.")
            logger.info(f"Entity resolution complete. Merged {total_merged} entities.")
            
            # ================================================================
            # PHASE 3: User review of flagged entities
            # ================================================================
            if flagged_for_review:
                self._prompt_user_review(session, flagged_for_review)
    
    def _merge_entities_preserve_all_types(self, session, duplicates: List[dict]) -> int:
        """Merge duplicate entities, preserving ALL type labels and combining descriptions.
        
        Domain-agnostic approach: instead of picking one "best" type, we add all
        type labels to the merged node. This preserves all contextual uses of the entity.
        
        Example: "Michigan" with types [STATE, ACADEMIC_INSTITUTION, GEOGRAPHIC_LOCATION]
        becomes a single node with all three labels.
        """
        if len(duplicates) < 2:
            return 0
        
        # Use first entity as primary (arbitrary choice since we keep all types)
        primary = duplicates[0]
        primary_id = primary['id']
        
        # Collect all unique types and descriptions
        all_types = list(set(d.get('type', 'ENTITY') for d in duplicates if d.get('type')))
        descriptions = [d.get('description', '') for d in duplicates if d.get('description')]
        combined_description = ' | '.join(filter(None, descriptions))
        
        logger.info(f"Merging duplicates of '{primary.get('name')}': combining types {all_types}, merging {len(duplicates)-1} others")
        
        # First, add all type labels to the primary entity
        for entity_type in all_types:
            if entity_type:
                try:
                    # Add the type as a label (Neo4j labels are added with SET)
                    session.run(f"""
                        MATCH (e:__Entity__ {{id: $primary_id}})
                        SET e:`{entity_type}`
                    """, primary_id=primary_id)
                except Exception as e:
                    logger.error(f'Failed to add label {entity_type} to {primary_id}: {e}')
        
        # Store all types in a property as well (for easy querying)
        try:
            session.run("""
                MATCH (e:__Entity__ {id: $primary_id})
                SET e.entity_types = $types
            """, primary_id=primary_id, types=all_types)
        except Exception as e:
            logger.error(f'Failed to set entity_types property for {primary_id}: {e}')
        
        merged_count = 0
        for dup in duplicates[1:]:
            dup_id = dup['id']
            try:
                # Transfer incoming relationships (except from Chunk - use MERGE to avoid duplicates)
                session.run("""
                    MATCH (n)-[r]->(e:__Entity__ {id: $dup_id})
                    WHERE NOT n:Chunk
                    MATCH (primary:__Entity__ {id: $primary_id})
                    MERGE (n)-[r2:RELATED_TO]->(primary)
                    SET r2 = properties(r)
                    DELETE r
                """, dup_id=dup_id, primary_id=primary_id)
                
                # Transfer HAS_ENTITY relationships from Chunks
                session.run("""
                    MATCH (c:Chunk)-[r:HAS_ENTITY]->(e:__Entity__ {id: $dup_id})
                    MATCH (primary:__Entity__ {id: $primary_id})
                    MERGE (c)-[:HAS_ENTITY]->(primary)
                    DELETE r
                """, dup_id=dup_id, primary_id=primary_id)
                
                # Transfer outgoing relationships
                session.run("""
                    MATCH (e:__Entity__ {id: $dup_id})-[r]->(n)
                    MATCH (primary:__Entity__ {id: $primary_id})
                    MERGE (primary)-[r2:RELATED_TO]->(n)
                    SET r2 = properties(r)
                    DELETE r
                """, dup_id=dup_id, primary_id=primary_id)
                
                # Delete duplicate entity
                session.run("""
                    MATCH (e:__Entity__ {id: $dup_id})
                    DETACH DELETE e
                """, dup_id=dup_id)
                
                merged_count += 1
                
            except Exception as e:
                logger.error(f'Failed to merge entity {dup_id} into {primary_id}: {e}')
        
        # Update primary entity with combined description if we merged anything
        if merged_count > 0 and combined_description:
            try:
                session.run("""
                    MATCH (e:__Entity__ {id: $primary_id})
                    SET e.description = $description
                """, primary_id=primary_id, description=combined_description[:1000])  # Limit length
            except Exception as e:
                logger.error(f'Failed to update description for {primary_id}: {e}')
        
        return merged_count
    
    def _merge_entities(self, session, entity_ids: List[str]) -> int:
        """Merge duplicate entities into the first one."""
        logger.debug(f'Merging entities: {entity_ids}')
        merged_entities = [] #sd43372

        if len(entity_ids) < 2:
            return 0
        
        primary_id = entity_ids[0]
        logger.info(f'Primary Id: {primary_id}')
        duplicate_ids = entity_ids[1:]
        logger.info(f'Duplicate Id: {duplicate_ids}')
        
        # Transfer all relationships to primary entity
        for dup_id in duplicate_ids:
            # Transfer incoming relationships
            logger.debug(f"Duplicate Id: {dup_id}")
            try: #Added try/catch so that process continues even if merging fails for some reason sd43372
                session.run("""
                    MATCH (n)-[r]->(e:__Entity__ {id: $dup_id})
                    MATCH (primary:__Entity__ {id: $primary_id})
                    CREATE (n)-[r2:RELATED_TO]->(primary)
                    SET r2 = properties(r)
                    DELETE r
                """, dup_id=dup_id, primary_id=primary_id)

                # Transfer outgoing relationships
                session.run("""
                    MATCH (e:__Entity__ {id: $dup_id})-[r]->(n)
                    MATCH (primary:__Entity__ {id: $primary_id})
                    CREATE (primary)-[r2:RELATED_TO]->(n)
                    SET r2 = properties(r)
                    DELETE r
                """, dup_id=dup_id, primary_id=primary_id)

                # Delete duplicate entity
                logger.debug(f'Trying to delete entity with id : {dup_id}')
                session.run("""
                    MATCH (e:__Entity__ {id: $dup_id})
                    DELETE e
                """, dup_id=dup_id)
                merged_entities.append(dup_id)
            except neo4j.exceptions.ConstraintError as nec:
                logger.error(f'Failed to delete node with duplicate id: {dup_id}. Primary id: {primary_id}')
            except Exception as e:
                logger.error(f'Failed to delete node with duplicate id: {dup_id}. Primary id: {primary_id}')
        return len(merged_entities)
    
    def _resolve_entities_with_gds(self, session, high_confidence_threshold: float = 0.95,
                                   medium_confidence_threshold: float = 0.85) -> tuple:
        """
        Use Neo4j GDS for embedding-based entity resolution.
        
        Approach:
        1. Create KNN graph based on entity embeddings
        2. Use WCC to find connected components (clusters)
        3. Auto-merge high confidence clusters
        4. LLM verify medium confidence clusters
        5. Flag low confidence / large clusters for user review
        
        Returns:
            (merged_count, flagged_for_review)
        """
        merged_count = 0
        flagged_for_review = []
        
        try:
            # Check if GDS is available
            gds_check = session.run("RETURN gds.version() as version").single()
            if not gds_check:
                print("[RESOLVE] GDS not available, skipping embedding-based resolution")
                return 0, []
            print(f"[RESOLVE] Using Neo4j GDS {gds_check['version']}")
        except Exception as e:
            print(f"[RESOLVE] GDS not available ({e}), skipping embedding-based resolution")
            return 0, []
        
        try:
            # Step 1: Create in-memory graph projection with entity embeddings
            print("[RESOLVE] Creating graph projection for similarity detection...")
            
            # Drop existing projection if exists
            try:
                session.run("CALL gds.graph.drop('entity-similarity', false)")
            except:
                pass
            
            # Create projection with entities and their embeddings
            session.run("""
                CALL gds.graph.project(
                    'entity-similarity',
                    {
                        __Entity__: {
                            properties: ['embedding']
                        }
                    },
                    '*'
                )
            """)
            
            # Step 2: Run KNN to find similar entity pairs
            print(f"[RESOLVE] Finding similar entities (threshold: {medium_confidence_threshold})...")
            
            knn_result = session.run("""
                CALL gds.knn.stream('entity-similarity', {
                    nodeProperties: ['embedding'],
                    topK: 3,
                    similarityCutoff: $threshold,
                    concurrency: 4
                })
                YIELD node1, node2, similarity
                WITH gds.util.asNode(node1) as e1, gds.util.asNode(node2) as e2, similarity
                RETURN e1.id as id1, e1.name as name1, e1.entity_type as type1,
                       e2.id as id2, e2.name as name2, e2.entity_type as type2,
                       similarity
                ORDER BY similarity DESC
            """, threshold=medium_confidence_threshold)
            
            # Collect candidate pairs
            candidates = list(knn_result)
            print(f"[RESOLVE] Found {len(candidates)} candidate pairs")
            
            if not candidates:
                session.run("CALL gds.graph.drop('entity-similarity', false)")
                return 0, []
            
            # Step 3: Group candidates using Union-Find (simpler than WCC for this case)
            # Build clusters from pairs
            clusters = self._build_clusters_from_pairs(candidates, high_confidence_threshold)
            
            print(f"[RESOLVE] Grouped into {len(clusters)} clusters")
            
            # Step 4: Process each cluster based on confidence
            for cluster in clusters:
                cluster_size = len(cluster['entities'])
                avg_similarity = cluster['avg_similarity']
                
                if cluster_size > 5:
                    # Large cluster - flag for review (might be over-merged)
                    flagged_for_review.append({
                        'reason': 'large_cluster',
                        'entities': cluster['entities'],
                        'avg_similarity': avg_similarity
                    })
                elif avg_similarity >= high_confidence_threshold:
                    # High confidence - auto-merge
                    entity_data = [{'id': e['id'], 'name': e['name'], 'type': e['type'], 'description': ''} 
                                   for e in cluster['entities']]
                    merged = self._merge_entities_preserve_all_types(session, entity_data)
                    merged_count += merged
                    logger.info(f"Auto-merged high confidence cluster: {[e['name'] for e in cluster['entities']]}")
                elif avg_similarity >= medium_confidence_threshold:
                    # Medium confidence - LLM verification
                    entity_names = [e['name'] for e in cluster['entities']]
                    if self._llm_verify_duplicates(entity_names):
                        entity_data = [{'id': e['id'], 'name': e['name'], 'type': e['type'], 'description': ''} 
                                       for e in cluster['entities']]
                        merged = self._merge_entities_preserve_all_types(session, entity_data)
                        merged_count += merged
                        logger.info(f"LLM-verified and merged: {entity_names}")
                    else:
                        # LLM said no - flag for review
                        flagged_for_review.append({
                            'reason': 'llm_uncertain',
                            'entities': cluster['entities'],
                            'avg_similarity': avg_similarity
                        })
            
            # Cleanup
            session.run("CALL gds.graph.drop('entity-similarity', false)")
            
        except Exception as e:
            logger.error(f"GDS-based resolution failed: {e}")
            import traceback
            traceback.print_exc()
            # Try to cleanup
            try:
                session.run("CALL gds.graph.drop('entity-similarity', false)")
            except:
                pass
        
        return merged_count, flagged_for_review
    
    def _build_clusters_from_pairs(self, candidates: list, high_threshold: float) -> list:
        """Build clusters from candidate pairs using Union-Find algorithm."""
        # Union-Find data structure
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Build entity info map and union similar entities
        entity_info = {}
        pair_similarities = {}
        
        for c in candidates:
            id1, id2 = c['id1'], c['id2']
            entity_info[id1] = {'id': id1, 'name': c['name1'], 'type': c['type1']}
            entity_info[id2] = {'id': id2, 'name': c['name2'], 'type': c['type2']}
            union(id1, id2)
            pair_similarities[(id1, id2)] = c['similarity']
        
        # Group by cluster
        cluster_map = {}
        for entity_id in entity_info:
            root = find(entity_id)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].append(entity_info[entity_id])
        
        # Calculate average similarity per cluster
        clusters = []
        for root, entities in cluster_map.items():
            if len(entities) < 2:
                continue
            
            # Calculate average similarity of pairs in this cluster
            similarities = []
            entity_ids = [e['id'] for e in entities]
            for i, id1 in enumerate(entity_ids):
                for id2 in entity_ids[i+1:]:
                    key = (id1, id2) if (id1, id2) in pair_similarities else (id2, id1)
                    if key in pair_similarities:
                        similarities.append(pair_similarities[key])
            
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.85
            
            clusters.append({
                'entities': entities,
                'avg_similarity': avg_sim
            })
        
        return clusters
    
    def _llm_verify_duplicates(self, entity_names: list) -> bool:
        """Use LLM to verify if a list of entity names refer to the same entity."""
        if len(entity_names) < 2:
            return False
        
        prompt = f"""Do these names refer to the SAME real-world entity? 
Names: {entity_names}

Consider:
- Abbreviations vs full names (e.g., "NY Times" = "The New York Times")
- Nicknames vs formal names (e.g., "JFK" = "John F. Kennedy")
- Spelling variations

Reply with ONLY "yes" or "no"."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip().lower() if hasattr(response, 'content') else str(response).strip().lower()
            return content.startswith('yes')
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return False  # Conservative: don't merge if unsure
    
    def _prompt_user_review(self, session, flagged_for_review: list):
        """Present flagged entities to user for review and merge decisions."""
        if not flagged_for_review:
            return
        
        print("\n" + "="*70)
        print("ENTITY RESOLUTION - REVIEW REQUIRED")
        print("="*70)
        print(f"\n{len(flagged_for_review)} entity groups need your review:\n")
        
        for i, item in enumerate(flagged_for_review, 1):
            entities = item['entities']
            reason = item['reason']
            similarity = item.get('avg_similarity', 0)
            
            reason_text = {
                'large_cluster': 'Large cluster (>5 entities) - may be over-merged',
                'llm_uncertain': 'LLM uncertain if these are the same entity'
            }.get(reason, reason)
            
            print(f"[{i}] {reason_text}")
            print(f"    Similarity: {similarity:.2%}")
            print(f"    Entities:")
            for e in entities:
                print(f"      - {e['name']} ({e.get('type', 'unknown')})")
            print()
        
        print("-"*70)
        print("Options:")
        print("  Enter numbers to MERGE (e.g., '1 3 5' or '1,3,5')")
        print("  Enter 'all' to merge ALL")
        print("  Enter 'none' or press Enter to skip ALL")
        print("-"*70)
        
        try:
            user_input = input("\nYour choice: ").strip().lower()
        except EOFError:
            print("[SKIP] No terminal input available, skipping review")
            return
        
        if not user_input or user_input == 'none':
            print("[SKIP] No entities merged from review")
            return
        
        if user_input == 'all':
            merge_indices = list(range(len(flagged_for_review)))
        else:
            # Parse numbers
            try:
                merge_indices = [int(x.strip()) - 1 for x in user_input.replace(',', ' ').split() 
                                if x.strip().isdigit()]
            except:
                print("[ERROR] Invalid input, skipping review")
                return
        
        # Merge selected groups
        merged_count = 0
        for idx in merge_indices:
            if 0 <= idx < len(flagged_for_review):
                entities = flagged_for_review[idx]['entities']
                entity_data = [{'id': e['id'], 'name': e['name'], 'type': e.get('type', 'ENTITY'), 'description': ''} 
                               for e in entities]
                merged = self._merge_entities_preserve_all_types(session, entity_data)
                merged_count += merged
                print(f"[MERGED] Group {idx+1}: {[e['name'] for e in entities]}")
        
        print(f"\n[REVIEW COMPLETE] Merged {merged_count} additional entities from review")
    
    def close(self):
        """Close the Neo4j driver."""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
