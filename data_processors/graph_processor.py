"""
Dynamic Graph Processor

This script processes RFP documents by:
1. Corpus-wide entity discovery using LLM analysis with caching
2. PDF text extraction with table extraction (Camelot/Tabula)
3. Intelligent text chunking with RecursiveCharacterTextSplitter
4. Creating embeddings with OpenAI (text-embedding-3-small)
5. Dynamic entity extraction using LLM with discovered entity types
6. Building a unified graph schema in Neo4j with vector indexes
7. Advanced entity resolution using Graph Data Science algorithms
8. Creating semantic relationships between entities and chunks
"""

import os
import json
import hashlib
import re
from pypdf import PdfReader
try:
    import camelot  # type: ignore
    _HAS_CAMELOT = True
except Exception:
    _HAS_CAMELOT = False
try:
    import tabula  # type: ignore
    _HAS_TABULA = True
except Exception:
    _HAS_TABULA = False
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dotenv import load_dotenv
import neo4j
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from graphdatascience import GraphDataScience
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from retry import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Import centralized configuration
from config import get_model_config, get_embeddings, get_llm, ModelProvider

# Configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
LLM = os.environ.get('OPENAI_MODEL_NAME', os.environ.get('LLM_FALLBACK_MODEL', 'llama3.1:8b'))  # Use configurable fallback

class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="Entities that represent the same object or real-world entity and should be merged"
    )

class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )

class CustomGraphProcessor:
    def __init__(self, model_config=None):
        self.config = model_config or get_model_config()
        self.driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Initialize models based on configuration
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        
        # Validate embedding dimensions for Neo4j vector indexes
        expected_dims = self.config.embedding_dimensions
        if expected_dims != 1536 and self.config.embedding_provider == ModelProvider.OPENAI:
            import warnings
            warnings.warn(f"Neo4j vector indexes expect 1536 dimensions, but {self.config.embedding_model.value} has {expected_dims} dimensions. You may need to recreate indexes.")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Always use dynamic entity discovery with corpus-wide approach
        self.corpus_discovery = True
        self.discovered_labels: Optional[List[str]] = None
        self.schema_cache_file = "data_processors/.schema_cache.json"
        
        # Entity resolution components
        self.gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.extraction_llm = get_llm().with_structured_output(Disambiguate)
        
        # Entity resolution prompt
        system_prompt = """You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

Here are the rules for identifying duplicates:
1. Entities with minor typographical differences should be considered duplicates.
2. Entities with different formats but the same content should be considered duplicates.
3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
4. If it refers to different numbers, dates, or products, do not merge results
"""
        user_template = """
Here is the list of entities to process:
{entities}

Please identify duplicates, merge them, and provide the merged list.
"""
        
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template),
        ])
        
        self.extraction_chain = self.extraction_prompt | self.extraction_llm
        
    # ---------------------- Corpus-wide discovery utilities ----------------------
    def _compute_corpus_hash(self, pdf_files: List[Path]) -> str:
        """Compute hash of corpus for schema cache key."""
        file_info = []
        for pdf_path in sorted(pdf_files):
            try:
                stat = pdf_path.stat()
                file_info.append(f"{pdf_path.name}:{stat.st_size}:{stat.st_mtime}")
            except Exception:
                file_info.append(f"{pdf_path.name}:unknown")
        corpus_str = "|".join(file_info)
        return hashlib.md5(corpus_str.encode()).hexdigest()[:12]

    def _load_schema_cache(self, corpus_hash: str) -> Optional[List[str]]:
        """Load cached schema labels if available."""
        try:
            if os.path.exists(self.schema_cache_file):
                with open(self.schema_cache_file, 'r') as f:
                    cache = json.load(f)
                    if cache.get('corpus_hash') == corpus_hash:
                        return cache.get('labels', [])
        except Exception:
            pass
        return None

    def _save_schema_cache(self, corpus_hash: str, labels: List[str]):
        """Save approved labels to cache."""
        try:
            cache = {
                'corpus_hash': corpus_hash,
                'labels': labels,
                'created_at': str(os.path.getctime(self.schema_cache_file) if os.path.exists(self.schema_cache_file) else 'new')
            }
            with open(self.schema_cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save schema cache: {e}")

    def _extract_entity_rich_patterns(self, text: str) -> str:
        """Extract entity-rich patterns: ALL CAPS, quoted text, bullet points."""
        patterns = []
        # ALL CAPS words (likely entities)
        caps_matches = re.findall(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', text)
        patterns.extend(caps_matches[:10])  # Limit to avoid noise
        
        # Quoted text (often entity names)
        quoted_matches = re.findall(r'"([^"]{2,50})"', text)
        patterns.extend(quoted_matches[:5])
        
        # Bullet points and list items (often entity lists)
        bullet_matches = re.findall(r'(?:^|\n)\s*[•\-\*]\s*([^\n]{5,100})', text, re.MULTILINE)
        patterns.extend(bullet_matches[:8])
        
        return " | ".join(patterns)

    def _sample_corpus_text(self, pdf_files: List[Path]) -> str:
        """Hybrid sampling: first/last 500 chars + patterns + stratified sampling."""
        samples = []
        total_chars = 0
        max_chars = 8000  # Token budget
        
        for pdf_path in pdf_files[:20]:  # Limit to first 20 files for performance
            try:
                # Extract basic text
                text = self.extract_text_from_pdf(str(pdf_path))
                if not text.strip():
                    continue
                
                # Document title/filename
                doc_title = f"Document: {pdf_path.stem}"
                samples.append(doc_title)
                total_chars += len(doc_title)
                
                # First 500 + last 500 chars (intro/conclusion entity density)
                first_part = text[:500].strip()
                last_part = text[-500:].strip() if len(text) > 1000 else ""
                
                if first_part:
                    samples.append(f"Beginning: {first_part}")
                    total_chars += len(first_part) + 12
                
                if last_part and last_part != first_part:
                    samples.append(f"End: {last_part}")
                    total_chars += len(last_part) + 5
                
                # Entity-rich patterns
                patterns = self._extract_entity_rich_patterns(text)
                if patterns:
                    samples.append(f"Patterns: {patterns}")
                    total_chars += len(patterns) + 11
                
                # Stop if we hit budget
                if total_chars >= max_chars:
                    break
                    
            except Exception as e:
                print(f"Warning: Could not sample from {pdf_path}: {e}")
                continue
        
        return "\n\n".join(samples)[:max_chars]

    def discover_corpus_labels(self, pdf_files: List[Path]) -> List[str]:
        """Discover labels corpus-wide using hybrid sampling strategy."""
        corpus_hash = self._compute_corpus_hash(pdf_files)
        
        # Check cache first
        cached_labels = self._load_schema_cache(corpus_hash)
        if cached_labels:
            print(f"📋 Using cached labels from previous run: {cached_labels}")
            return cached_labels
        
        # Sample corpus text
        print("🔍 Sampling corpus text for entity discovery...")
        corpus_sample = self._sample_corpus_text(pdf_files)
        
        if not corpus_sample.strip():
            print("⚠️ No text could be sampled from corpus")
            return []
        
        # Discover labels
        print("🧠 Analyzing corpus with LLM...")
        proposed_labels = self.discover_labels_for_text(corpus_sample)
        
        # CLI approval
        approved_labels = self._approve_labels_cli(proposed_labels)
        
        # Cache results
        if approved_labels:
            self._save_schema_cache(corpus_hash, approved_labels)
        
        return approved_labels

    def discover_labels_for_text(self, text: str, max_labels: int = 12) -> List[str]:
        """Discover entity labels from text using LLM."""
        # Truncate text to fit in prompt
        text_sample = text[:12000]
        
        prompt = f"""
        Analyze the following text and propose up to {max_labels} entity types (labels) that would be most useful for knowledge graph construction.

        Focus on:
        - Domain-specific entities relevant to this content
        - Entities that appear frequently and have relationships
        - Concrete, actionable entity types (not abstract concepts)
        
        Examples of good entity types: Contract, Vendor, Deliverable, Timeline, Budget, Compliance, Product, Service, Location, Person, Organization, Requirement, Technology, Process

        Text to analyze:
        {text_sample}
        
        Return ONLY a comma-separated list of entity type names (no descriptions, no extra text):
        """
        
        response = self.llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Parse the response
        labels = [label.strip() for label in response_text.split(',') if label.strip()]
        
        # Clean and validate labels
        clean_labels = []
        for label in labels[:max_labels]:
            # Sanitize label name - replace spaces with underscores, remove special chars
            clean_label = re.sub(r'[^a-zA-Z0-9_]', '', label.replace(' ', '_').replace('-', '_'))
            if clean_label and len(clean_label) > 1:
                # Ensure it starts with a letter (Neo4j requirement)
                if not clean_label[0].isalpha():
                    clean_label = 'Entity_' + clean_label
                clean_labels.append(clean_label.title())
        
        return clean_labels

    def _approve_labels_cli(self, proposed_labels: List[str]) -> List[str]:
        """CLI interface for user to approve/modify discovered labels."""
        if not proposed_labels:
            print("⚠️ No labels were discovered")
            return []
        
        print(f"\n📋 Proposed entity types: {', '.join(proposed_labels)}")
        
        while True:
            choice = input("\n✅ Approve these entities? (y/n/edit): ").lower().strip()
            
            if choice in ['y', 'yes']:
                return proposed_labels
            elif choice in ['n', 'no']:
                print("❌ Labels rejected. Using fallback to CONSTRAINED mode.")
                return []
            elif choice in ['e', 'edit']:
                print("📝 Enter your preferred entity types (comma-separated):")
                user_input = input("> ").strip()
                if user_input:
                    user_labels = [label.strip().title() for label in user_input.split(',') if label.strip()]
                    if user_labels:
                        return user_labels
                print("⚠️ No valid labels entered, trying again...")
            else:
                print("⚠️ Please enter 'y', 'n', or 'edit'")

    def extract_entities_dynamic(self, text: str, allowed_labels: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities dynamically using discovered or free labels."""
        if allowed_labels:
            labels_constraint = f"ONLY extract entities of these types: {', '.join(allowed_labels)}"
        else:
            labels_constraint = "Extract any relevant entities you find"
        
        prompt = f"""
        Extract entities from the following text. {labels_constraint}
        
        Return ONLY a valid JSON object with this structure:
        {{
            "entities": [
                {{"type": "EntityType", "name": "entity name", "description": "brief description"}}
            ]
        }}
        
        Guidelines:
        - Keep entity names concise and specific
        - Provide brief, relevant descriptions
        - Focus on entities that have clear relationships to other entities
        
        Text to analyze:
        {text[:3000]}
        
        Return only the JSON object, no other text.
        """
        
        response = self.llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Clean response
        if response_text.startswith('```json'):
            response_text = response_text[7:-3]
        elif response_text.startswith('```'):
            response_text = response_text[3:-3]
        
        try:
            result = json.loads(response_text)
            entities = result.get('entities', [])
            
            # Group by type for compatibility with existing code
            grouped = {}
            for entity in entities:
                entity_type = entity.get('type', 'Unknown')
                # Sanitize entity type same as labels
                entity_type = re.sub(r'[^a-zA-Z0-9_]', '', entity_type.replace(' ', '_').replace('-', '_'))
                if entity_type and not entity_type[0].isalpha():
                    entity_type = 'Entity_' + entity_type
                entity_type = entity_type.title()
                
                if entity_type not in grouped:
                    grouped[entity_type] = []
                
                grouped[entity_type].append({
                    'text': entity.get('name', ''),
                    'description': entity.get('description', ''),
                    'label': entity_type.upper(),
                    'start': 0,
                    'end': len(entity.get('name', ''))
                })
            
            print(f"✅ Extracted {sum(len(v) for v in grouped.values())} dynamic entities")
            return grouped
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in dynamic extraction: {e}")
            return {}

    def create_entity_relationships_dynamic(self, session, entity_ids: List[Tuple[str, str]]):
        """Create generic RELATES_TO relationships for dynamic entities."""
        if len(entity_ids) < 2:
            return
        
        # Create relationships between all entities in the chunk (co-occurrence)
        for i, (label1, name1) in enumerate(entity_ids):
            for j, (label2, name2) in enumerate(entity_ids[i+1:], i+1):
                if label1 == label2 and name1 == name2:
                    continue
                
                # Create bidirectional relationships with co-occurrence count
                unique_id1 = f"{label1}_{name1}"
                unique_id2 = f"{label2}_{name2}"
                session.run("""
                    MATCH (e1:__Entity__ {id: $id1}), (e2:__Entity__ {id: $id2})
                    MERGE (e1)-[r:RELATES_TO]->(e2)
                    ON CREATE SET r.co_occurrences = 1
                    ON MATCH SET r.co_occurrences = r.co_occurrences + 1
                """, id1=unique_id1, id2=unique_id2)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        if not text.strip():
            raise ValueError(f"No text content extracted from PDF: {pdf_path}")
            
        return text

    def extract_tables(self, pdf_path: str, source_file: str, start_index: int) -> List[Dict[str, Any]]:
        """Extract tables using Camelot, falling back to Tabula, else return empty list.

        Tables are converted to CSV text and emitted as atomic 'table' chunks.
        """
        chunks: List[Dict[str, Any]] = []
        table_count = 0
        # Try Camelot first
        if _HAS_CAMELOT:
            try:
                # Prefer lattice for ruled tables, fallback to stream
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
                if tables.n == 0:
                    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
                for idx in range(tables.n):
                    try:
                        df = tables[idx].df
                        csv_text = df.to_csv(index=False)
                        chunks.append({
                            'text': csv_text[:20000],
                            'index': start_index + table_count,
                            'source': source_file,
                            'type': 'table'
                        })
                        table_count += 1
                    except Exception:
                        continue
                return chunks
            except Exception:
                pass
        # Fallback to Tabula
        if _HAS_TABULA:
            try:
                dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
                for df in dfs or []:
                    try:
                        csv_text = df.to_csv(index=False)
                        chunks.append({
                            'text': csv_text[:20000],
                            'index': start_index + table_count,
                            'source': source_file,
                            'type': 'table'
                        })
                        table_count += 1
                    except Exception:
                        continue
            except Exception:
                pass
        # If neither available or no tables detected, return empty -> default to text chunking only
        return chunks
    
    def chunk_text(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Split text into chunks using RecursiveCharacterTextSplitter"""
        if not text.strip():
            raise ValueError("Cannot split empty text")
            
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                'text': chunk_text,
                'index': i,
                'source': source_file,
                'type': 'content'
            })
                
        return chunks
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using OpenAI via langchain"""
        try:
            # Use langchain's OpenAIEmbeddings
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return []
    

    
    def setup_database_schema(self):
        """Create the Neo4j schema with proper indexes"""
        # Clear existing data
        self.clear_database()
        
        with self.driver.session() as session:
            # Create constraints for unique entities (dynamic approach)
            constraints = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"Created constraint: {constraint.split()[2]}")
                except Exception as e:
                    print(f"Constraint may already exist: {e}")
            
            # Create full-text indexes first (required for Neo4j GraphRAG compatibility)
            fulltext_indexes = [
                "CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.name, e.description]",
                "CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]"
            ]
            
            for index in fulltext_indexes:
                try:
                    session.run(index)
                    index_name = index.split()[3]  # Extract index name for logging
                    print(f"✅ Created/verified full-text index: {index_name}")
                except Exception as e:
                    print(f"⚠️ Full-text index creation issue: {e}")
                    # Continue processing - index might already exist
            
            # Create unified vector index for all entities
            entity_vector_index = """
                CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (e:__Entity__) ON (e.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """
            
            try:
                session.run(entity_vector_index)
                print("Created unified entity vector index")
            except Exception as e:
                print(f"Entity vector index may already exist: {e}")
    
    def clear_database(self):
        """Clear all data from the Neo4j database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Database cleared - all nodes and relationships deleted")
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF document"""
        doc_name = Path(pdf_path).stem
        print(f"Processing {doc_name}...")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters")
        
        return self._process_document_text(text, doc_name, pdf_path)
    
    def process_text_document(self, text: str, doc_name: str, source_info: str = None) -> Dict[str, Any]:
        """Process a document from text content (for RAGBench, etc.)"""
        print(f"Processing {doc_name}...")
        print(f"Text length: {len(text)} characters")
        
        return self._process_document_text(text, doc_name, source_info or doc_name)
    
    def _process_document_text(self, text: str, doc_name: str, source_info: str) -> Dict[str, Any]:
        """Internal method to process document text (shared by PDF and text processing)"""
        
        # Per-document discovery if needed (fallback if corpus-wide failed)
        if not self.discovered_labels:
            print("\n🔎 Discovering entity labels from document text...")
            proposed_labels = self.discover_labels_for_text(text)
            self.discovered_labels = self._approve_labels_cli(proposed_labels)
            print(f"\n✅ Using labels: {self.discovered_labels}")
        
        # Create document embedding
        doc_embedding = self.create_embedding(text[:8000])  # Limit for embedding
        
        # Chunk text
        chunks = self.chunk_text(text, doc_name)
        # Attempt table extraction and append as atomic chunks (only for PDFs)
        table_chunks = []
        if source_info.endswith('.pdf') and os.path.exists(source_info):
            table_chunks = self.extract_tables(source_info, doc_name, start_index=len(chunks))
        if table_chunks:
            chunks.extend(table_chunks)
        print(f"Created {len(chunks)} chunks (including {len(table_chunks)} table chunks)")
        
        # Process with Neo4j
        with self.driver.session() as session:
            # Create document node
            doc_id = f"doc_{doc_name}"
            session.run("""
                CREATE (d:Document {
                    id: $doc_id,
                    name: $doc_name,
                    path: $source_info,
                    text: $text,
                    embedding: $embedding,
                    chunk_count: $chunk_count,
                    created_at: datetime()
                })
            """, doc_id=doc_id, doc_name=doc_name, source_info=source_info, 
                text=text[:1000], embedding=doc_embedding, chunk_count=len(chunks))
            
            chunk_ids = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{chunk['index']}"
                chunk_ids.append(chunk_id)
                
                # Create embedding
                chunk_embedding = self.create_embedding(chunk['text'])
                
                # Create chunk node
                session.run("""
                    CREATE (c:Chunk {
                        id: $chunk_id,
                        text: $text,
                        index: $index,
                        embedding: $embedding,
                        source: $source,
                        type: $type
                    })
                """, chunk_id=chunk_id, text=chunk['text'], index=chunk['index'],
                    embedding=chunk_embedding, source=chunk['source'], 
                    type=chunk['type'])
                
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
                
                # Extract and process entities for this chunk (always dynamic)
                dynamic_entities = self.extract_entities_dynamic(
                    chunk['text'],
                    allowed_labels=self.discovered_labels,
                                )
                # Process dynamic entities
                chunk_entity_ids = []
                entity_counter = 0
                
                for entity_type, entity_list in dynamic_entities.items():
                    for entity in entity_list:
                        entity_counter += 1
                        entity_name = entity['text']
                        entity_description = entity.get('description', '')
                        
                        # Create embedding
                        embedding_text = f"{entity_name}: {entity_description}" if entity_description else entity_name
                        entity_embedding = self.create_embedding(embedding_text)
                        
                        # Create dynamic entity with both specific label and __Entity__
                        # Use a unique identifier combining name and type to avoid conflicts
                        unique_id = f"{entity_type}_{entity_name}"
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
                        
                        chunk_entity_ids.append((entity_type, entity_name))
                
                # Create dynamic relationships
                self.create_entity_relationships_dynamic(session, chunk_entity_ids)
        
        return {
            'document_id': doc_id,
            'chunks_created': len(chunks),
            'status': 'success'
        }
    
    def process_directory(self, pdf_dir: str, perform_resolution: bool = True) -> Dict[str, Any]:
        """Process all PDFs in a directory and optionally perform entity resolution"""
        results = {}
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Corpus-wide discovery (always enabled)
        if not self.discovered_labels:
            print("\n🌐 Running corpus-wide entity discovery...")
            self.discovered_labels = self.discover_corpus_labels(pdf_files)
            if self.discovered_labels:
                print(f"✅ Corpus-wide labels approved: {self.discovered_labels}")
            else:
                print("⚠️ No labels discovered, falling back to per-document discovery")
        
        # Setup database schema first
        self.setup_database_schema()
        
        # Process each file
        for pdf_path in pdf_files:
            try:
                result = self.process_document(str(pdf_path))
                results[pdf_path.name] = result
                print(f"✅ Successfully processed {pdf_path.name}")
            except Exception as e:
                print(f"❌ Error processing {pdf_path.name}: {e}")
                results[pdf_path.name] = {'status': 'error', 'error': str(e)}
        
        # Perform entity resolution if requested
        if perform_resolution:
            try:
                self.perform_entity_resolution()
                print("✅ Entity resolution completed")
            except Exception as e:
                print(f"❌ Error during entity resolution: {e}")
        
        return results
    
    def create_chunk_similarity_relationships(self, similarity_threshold: float = 0.95):
        """
        Create SIMILAR relationships between chunks based on embedding similarity.
        Matches LLM Graph Builder implementation (threshold=0.95, not used for retrieval).
        """
        print(f"Creating chunk similarity relationships with threshold {similarity_threshold}")
        
        try:
            # Create graph projection for chunks
            with self.driver.session() as session:
                # Drop existing projection if it exists
                try:
                    self.gds.graph.drop("chunk_similarity_graph")
                except:
                    pass
                
                # Create new projection
                G_chunks = self.gds.graph.project(
                    "chunk_similarity_graph",
                    "Chunk",
                    "*",
                    nodeProperties=["embedding"]
                )
                
                # Create k-nearest neighbor relationships between chunks
                self.gds.knn.mutate(
                    G_chunks,
                    nodeProperties=['embedding'],
                    mutateRelationshipType='SIMILAR',
                    mutateProperty='similarity_score',
                    similarityCutoff=similarity_threshold,
                    topK=5  # Each chunk connects to top 5 similar chunks
                )
                
                # Write back to Neo4j
                self.gds.graph.writeRelationship(
                    G_chunks,
                    relationshipType="SIMILAR",
                    relationshipProperty="similarity_score"
                )
                
                # Clean up projection
                self.gds.graph.drop(G_chunks)
                
                print(f"Chunk similarity relationships created successfully")
                
        except Exception as e:
            print(f"Error creating chunk similarity relationships: {e}")
            # Continue without chunk similarities - not critical for basic functionality
    
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    @retry(tries=3, delay=2)
    def entity_resolution(self, entities: List[str]) -> Optional[List[str]]:
        """Use LLM to determine which entities should be merged"""
        try:
            result = self.extraction_chain.invoke({"entities": entities})
            if result.merge_entities:
                return [el.entities for el in result.merge_entities]
            return None
        except Exception as e:
            print(f"Error in entity resolution: {e}")
            return None

    def perform_entity_resolution(self, similarity_threshold: float = 0.95, word_edit_distance: int = 3, max_workers: int = 4):
        """Perform entity resolution using k-nearest neighbors and LLM evaluation"""
        print("\n" + "="*60)
        print("STARTING ENTITY RESOLUTION")
        print("="*60)
        
        with self.driver.session() as session:
            # Check if we have any entities
            entity_count = session.run("MATCH (e:__Entity__) RETURN count(e) as count").single()['count']
            if entity_count == 0:
                print("No entities found for resolution")
                return
                
            print(f"Found {entity_count} entities to process")
        
        try:
            # Step 1: Create vector embeddings for existing entities (already done during creation)
            print("Step 1: Using existing embeddings for entities")
            
            # Step 2: Project graph for GDS
            print("Step 2: Creating graph projection")
            try:
                # Drop existing projection if it exists
                try:
                    self.gds.graph.drop("entities")
                except:
                    pass
                
                G, result = self.gds.graph.project(
                    "entities",                   # Graph name
                    "__Entity__",                 # Node projection
                    "*",                          # Relationship projection
                    nodeProperties=["embedding"]  # Configuration parameters
                )
                print(f"Graph projected with {G.node_count()} nodes")
                
            except Exception as e:
                print(f"Error creating graph projection: {e}")
                return
            
            # Step 3: Create k-nearest neighbor relationships
            print("Step 3: Creating k-nearest neighbor relationships")
            try:
                self.gds.knn.mutate(
                    G,
                    nodeProperties=['embedding'],
                    mutateRelationshipType='SIMILAR',
                    mutateProperty='score',
                    similarityCutoff=similarity_threshold
                )
                print(f"KNN relationships created with similarity threshold {similarity_threshold}")
            except Exception as e:
                print(f"Error creating KNN relationships: {e}")
                return
            
            # Step 4: Find weakly connected components
            print("Step 4: Finding weakly connected components")
            try:
                self.gds.wcc.write(
                    G,
                    writeProperty="wcc",
                    relationshipTypes=["SIMILAR"]
                )
                print("Weakly connected components identified")
            except Exception as e:
                print(f"Error finding components: {e}")
                return
            
            # Step 5: Find potential duplicate candidates with word distance filtering
            print("Step 5: Finding potential duplicate candidates")
            with self.driver.session() as session:
                potential_duplicate_candidates = session.run("""
                    MATCH (e:`__Entity__`)
                    WHERE size(e.id) > 4 // longer than 4 characters
                    WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
                    WHERE count > 1
                    UNWIND nodes AS node
                    // Add text distance
                    WITH distinct
                      [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] AS intermediate_results
                    WHERE size(intermediate_results) > 1
                    WITH collect(intermediate_results) AS results
                    // combine groups together if they share elements
                    UNWIND range(0, size(results)-1, 1) as index
                    WITH results, index, results[index] as result
                    WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
                            CASE WHEN index <> index2 AND
                                size(apoc.coll.intersection(acc, results[index2])) > 0
                                THEN apoc.coll.union(acc, results[index2])
                                ELSE acc
                            END
                    )) as combinedResult
                    WITH distinct(combinedResult) as combinedResult
                    // extra filtering
                    WITH collect(combinedResult) as allCombinedResults
                    UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
                    WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
                    WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
                        WHERE x <> combinedResultIndex
                        AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
                    )
                    RETURN combinedResult
                """, distance=word_edit_distance).data()
                
                print(f"Found {len(potential_duplicate_candidates)} groups of potential duplicates")
                
                # Step 6: Use LLM to evaluate and merge entities
                print("Step 6: Using LLM to evaluate entity merges")
                merged_entities = []
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submitting all tasks and creating a list of future objects
                    futures = [
                        executor.submit(self.entity_resolution, el['combinedResult'])
                        for el in potential_duplicate_candidates
                    ]

                    for future in tqdm(
                        as_completed(futures), total=len(futures), desc="Processing entity groups"
                    ):
                        to_merge = future.result()
                        if to_merge:
                            merged_entities.extend(to_merge)
                
                print(f"LLM identified {len(merged_entities)} groups for merging")
                
                # Step 7: Perform the actual merges
                if merged_entities:
                    print("Step 7: Performing entity merges")
                    merge_result = session.run("""
                        UNWIND $data AS candidates
                        CALL {
                          WITH candidates
                          MATCH (e:__Entity__) WHERE e.id IN candidates
                          RETURN collect(e) AS nodes
                        }
                        CALL apoc.refactor.mergeNodes(nodes, {properties: {
                            `.*`: 'discard'
                        }})
                        YIELD node
                        RETURN count(*) as merged_count
                    """, data=merged_entities).single()
                    
                    print(f"Successfully merged {merge_result['merged_count']} entity groups")
                else:
                    print("No entities needed merging")
            
            # Clean up graph projection
            G.drop()
            print("Graph projection cleaned up")
            
        except Exception as e:
            print(f"Error during entity resolution: {e}")
            try:
                self.gds.graph.drop("entities")
            except:
                pass

    def create_entity_relationships(self, session, entity_ids: List[Tuple[str, str]]):
        """Create RELATES_TO relationships between entities that appear in the same chunk"""
        if len(entity_ids) < 2:
            return
            
        # Define meaningful relationship patterns
        # Only create relationships between these entity type pairs to reduce clutter
        meaningful_pairs = {
            ('Organization', 'Requirement'),
            ('Organization', 'Location'), 
            ('Organization', 'Person'),
            ('Organization', 'Financial'),
            ('Requirement', 'Date'),
            ('Requirement', 'Financial'),
            ('Organization', 'Date'),
            ('Location', 'Date'),
            ('Person', 'Location'),
            ('Person', 'Date'),
            ('Financial', 'Date')
        }
        
        # Create selective relationships between entities in the chunk
        for i, (label1, name1) in enumerate(entity_ids):
            for j, (label2, name2) in enumerate(entity_ids[i+1:], i+1):
                # Skip same entity
                if label1 == label2 and name1 == name2:
                    continue
                
                # Check if this is a meaningful relationship pair
                pair = tuple(sorted([label1, label2]))
                if pair in meaningful_pairs:
                    # Create unidirectional relationship with co-occurrence count
                    session.run(f"""
                        MATCH (e1:{label1} {{name: $name1}}), (e2:{label2} {{name: $name2}})
                        MERGE (e1)-[r:RELATES_TO]->(e2)
                        ON CREATE SET r.co_occurrences = 1
                        ON MATCH SET r.co_occurrences = r.co_occurrences + 1
                    """, name1=name1, name2=name2)


def main():
    """Main processing function"""
    processor = CustomGraphProcessor()
    
    try:
        # Process PDFs directory
        results = processor.process_directory("PDFs")
        
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        for filename, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                chunks = result.get('chunks_created', 0)
                print(f"✅ {filename}: {chunks} chunks created")
            else:
                error = result.get('error', 'Unknown error')
                print(f"❌ {filename}: {error}")
        
        # Print database stats
        with processor.driver.session() as session:
            stats = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """).data()
            
            print(f"\nDatabase Statistics:")
            for stat in stats:
                print(f"  {stat['label']}: {stat['count']} nodes")
                
    finally:
        processor.close()


if __name__ == "__main__":
    main() 