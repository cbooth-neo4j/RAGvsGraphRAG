"""
Custom Graph Processor for RFP Analysis

This script processes RFP documents by:
1. Manually chunking text
2. Creating embeddings with OpenAI
3. Extracting entities with spaCy
4. Building a proper graph schema in Neo4j
"""

import os
import json
from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
import neo4j
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables
load_dotenv()

# Configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

class CustomGraphProcessor:
    def __init__(self):
        self.driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
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
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using GPT-4o-mini for intelligent entity extraction from RFP documents"""
        
        # Initialize entities structure
        entities = {
            'organizations': [],
            'locations': [],
            'dates': [],
            'persons': [],
            'financial': [],
            'requirements': []
        }
        
        # Create prompt for GPT-4o-mini
        prompt = f"""
        Extract entities from the following RFP document text. Return ONLY a valid JSON object with the following structure:
        
        {{
            "organizations": [{{"text": "org name", "description": "brief description of the organization", "label": "ORG"}}],
            "locations": [{{"text": "location name", "description": "brief description of the location", "label": "GPE"}}],
            "dates": [{{"text": "date/time", "description": "context of the date", "label": "DATE"}}],
            "persons": [{{"text": "person name", "description": "role or title of the person", "label": "PERSON"}}],
            "financial": [{{"text": "financial amount/term", "description": "context of the financial term", "label": "MONEY"}}],
            "requirements": [{{"text": "specific requirement", "description": "brief description of the requirement", "label": "REQUIREMENT"}}]
        }}
        
        Guidelines:
        - Organizations: Company names, institutions, government agencies with their role/purpose
        - Locations: Cities, states, countries, addresses with context
        - Dates: Specific dates, timeframes, deadlines with their significance
        - Persons: Individual names, titles, contact persons with their role
        - Financial: Dollar amounts, percentages, financial terms with context
        - Requirements: Specific service requirements, capabilities needed, scope items (keep concise, max 50 chars each) with brief description
        
        Text to analyze:
        {text[:3000]}  # Limit to first 3000 chars to avoid token limits
        
        Return only the JSON object, no other text.
        """
        
        # Get response from GPT-4o-mini
        response = self.llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Try to extract JSON from response
        if response_text.startswith('```json'):
            response_text = response_text[7:-3]  # Remove ```json and ```
        elif response_text.startswith('```'):
            response_text = response_text[3:-3]  # Remove ``` and ```
        
        # Clean up any extra text before/after JSON
        response_text = response_text.strip()
        
        # Parse JSON response
        try:
            extracted_entities = json.loads(response_text)
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {json_error}")
            print(f"Response text: {response_text[:200]}...")
            raise
        
        # Validate and clean the extracted entities
        for entity_type, entity_list in extracted_entities.items():
            if entity_type in entities and isinstance(entity_list, list):
                for entity in entity_list:
                    if isinstance(entity, dict) and 'text' in entity:
                        # Clean and validate entity text
                        entity_text = entity['text'].strip()
                        entity_description = entity.get('description', '').strip()
                        
                        if entity_text and len(entity_text) <= 200:  # Reasonable length limit
                            entity_info = {
                                'text': entity_text,
                                'description': entity_description,
                                'label': entity.get('label', entity_type.upper()),
                                'start': 0,  # We don't have exact positions from GPT
                                'end': len(entity_text)
                            }
                            entities[entity_type].append(entity_info)
        
        print(f"✅ Extracted {sum(len(entities[et]) for et in entities)} entities using GPT-4o-mini")
        
        return entities
    
    def setup_database_schema(self):
        """Create the Neo4j schema with proper indexes"""
        # Clear existing data
        self.clear_database()
        
        with self.driver.session() as session:
            # Create constraints for unique entities
            constraints = [
                "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
                "CREATE CONSTRAINT loc_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE", 
                "CREATE CONSTRAINT req_name IF NOT EXISTS FOR (r:Requirement) REQUIRE r.name IS UNIQUE",
                "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT financial_name IF NOT EXISTS FOR (f:Financial) REQUIRE f.name IS UNIQUE",
                "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"Created constraint: {constraint.split()[2]}")
                except Exception as e:
                    print(f"Constraint may already exist: {e}")
            
            # Create vector indexes for embeddings
            vector_indexes = [
                """
                CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
                """
                CREATE VECTOR INDEX document_embedding IF NOT EXISTS  
                FOR (d:Document) ON (d.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
                """
                CREATE VECTOR INDEX organization_embedding IF NOT EXISTS
                FOR (o:Organization) ON (o.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
                """
                CREATE VECTOR INDEX location_embedding IF NOT EXISTS
                FOR (l:Location) ON (l.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
                """
                CREATE VECTOR INDEX date_embedding IF NOT EXISTS
                FOR (dt:Date) ON (dt.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
                """
                CREATE VECTOR INDEX requirement_embedding IF NOT EXISTS
                FOR (r:Requirement) ON (r.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
                """
                CREATE VECTOR INDEX person_embedding IF NOT EXISTS
                FOR (p:Person) ON (p.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
                """
                CREATE VECTOR INDEX financial_embedding IF NOT EXISTS
                FOR (f:Financial) ON (f.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
            ]
            
            for index in vector_indexes:
                try:
                    session.run(index)
                    print("Created vector index")
                except Exception as e:
                    print(f"Vector index may already exist: {e}")
    
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
        
        # Create document embedding
        doc_embedding = self.create_embedding(text[:8000])  # Limit for embedding
        
        # Chunk text
        chunks = self.chunk_text(text, doc_name)
        print(f"Created {len(chunks)} chunks")
        
        # Process with Neo4j
        with self.driver.session() as session:
            # Create document node
            doc_id = f"doc_{doc_name}"
            session.run("""
                CREATE (d:Document {
                    id: $doc_id,
                    name: $doc_name,
                    path: $pdf_path,
                    text: $text,
                    embedding: $embedding,
                    chunk_count: $chunk_count,
                    created_at: datetime()
                })
            """, doc_id=doc_id, doc_name=doc_name, pdf_path=pdf_path, 
                text=text[:1000], embedding=doc_embedding, chunk_count=len(chunks))
            
            chunk_ids = []
            
            # Process each chunk
            for chunk in chunks:
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
                
                # Extract and process entities for this chunk
                entities = self.extract_entities(chunk['text'])
                
                # Store entity IDs for creating RELATES_TO relationships
                chunk_entity_ids = []
                
                # Create entity nodes with embeddings and HAS_ENTITY relationships
                for org in entities['organizations']:
                    org_name = org['text']
                    org_description = org.get('description', '')
                    # Create embedding from both name and description
                    embedding_text = f"{org_name}: {org_description}" if org_description else org_name
                    org_embedding = self.create_embedding(embedding_text)
                    session.run("""
                        MERGE (o:Organization {name: $name})
                        ON CREATE SET o.description = $description, o.embedding = $embedding
                        ON MATCH SET o.description = CASE WHEN o.description IS NULL THEN $description ELSE o.description END,
                                   o.embedding = CASE WHEN o.embedding IS NULL THEN $embedding ELSE o.embedding END
                        WITH o
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:HAS_ENTITY]->(o)
                    """, name=org_name, description=org_description, embedding=org_embedding, chunk_id=chunk_id)
                    chunk_entity_ids.append(('Organization', org_name))
                
                for loc in entities['locations']:
                    loc_name = loc['text']
                    loc_description = loc.get('description', '')
                    # Create embedding from both name and description
                    embedding_text = f"{loc_name}: {loc_description}" if loc_description else loc_name
                    loc_embedding = self.create_embedding(embedding_text)
                    session.run("""
                        MERGE (l:Location {name: $name})
                        ON CREATE SET l.description = $description, l.embedding = $embedding
                        ON MATCH SET l.description = CASE WHEN l.description IS NULL THEN $description ELSE l.description END,
                                   l.embedding = CASE WHEN l.embedding IS NULL THEN $embedding ELSE l.embedding END
                        WITH l
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:HAS_ENTITY]->(l)
                    """, name=loc_name, description=loc_description, embedding=loc_embedding, chunk_id=chunk_id)
                    chunk_entity_ids.append(('Location', loc_name))
                
                for date in entities['dates']:
                    date_name = date['text']
                    date_description = date.get('description', '')
                    # Create embedding from both name and description
                    embedding_text = f"{date_name}: {date_description}" if date_description else date_name
                    date_embedding = self.create_embedding(embedding_text)
                    session.run("""
                        MERGE (dt:Date {name: $name})
                        ON CREATE SET dt.description = $description, dt.embedding = $embedding
                        ON MATCH SET dt.description = CASE WHEN dt.description IS NULL THEN $description ELSE dt.description END,
                                   dt.embedding = CASE WHEN dt.embedding IS NULL THEN $embedding ELSE dt.embedding END
                        WITH dt
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:HAS_ENTITY]->(dt)
                    """, name=date_name, description=date_description, embedding=date_embedding, chunk_id=chunk_id)
                    chunk_entity_ids.append(('Date', date_name))
                
                for req in entities['requirements']:
                    req_name = req['text']  # GPT-4o-mini already provides concise requirements
                    req_description = req.get('description', '')
                    # Create embedding from both name and description
                    embedding_text = f"{req_name}: {req_description}" if req_description else req_name
                    req_embedding = self.create_embedding(embedding_text)
                    session.run("""
                        MERGE (r:Requirement {name: $name})
                        ON CREATE SET r.description = $description, r.embedding = $embedding
                        ON MATCH SET r.description = CASE WHEN r.description IS NULL THEN $description ELSE r.description END,
                                   r.embedding = CASE WHEN r.embedding IS NULL THEN $embedding ELSE r.embedding END
                        WITH r
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:HAS_ENTITY]->(r)
                    """, name=req_name, description=req_description, embedding=req_embedding, chunk_id=chunk_id)
                    chunk_entity_ids.append(('Requirement', req_name))
                
                for person in entities['persons']:
                    person_name = person['text']
                    person_description = person.get('description', '')
                    # Create embedding from both name and description
                    embedding_text = f"{person_name}: {person_description}" if person_description else person_name
                    person_embedding = self.create_embedding(embedding_text)
                    session.run("""
                        MERGE (p:Person {name: $name})
                        ON CREATE SET p.description = $description, p.embedding = $embedding
                        ON MATCH SET p.description = CASE WHEN p.description IS NULL THEN $description ELSE p.description END,
                                   p.embedding = CASE WHEN p.embedding IS NULL THEN $embedding ELSE p.embedding END
                        WITH p
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:HAS_ENTITY]->(p)
                    """, name=person_name, description=person_description, embedding=person_embedding, chunk_id=chunk_id)
                    chunk_entity_ids.append(('Person', person_name))
                
                for financial in entities['financial']:
                    financial_name = financial['text']
                    financial_description = financial.get('description', '')
                    # Create embedding from both name and description
                    embedding_text = f"{financial_name}: {financial_description}" if financial_description else financial_name
                    financial_embedding = self.create_embedding(embedding_text)
                    session.run("""
                        MERGE (f:Financial {name: $name})
                        ON CREATE SET f.description = $description, f.embedding = $embedding
                        ON MATCH SET f.description = CASE WHEN f.description IS NULL THEN $description ELSE f.description END,
                                   f.embedding = CASE WHEN f.embedding IS NULL THEN $embedding ELSE f.embedding END
                        WITH f
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:HAS_ENTITY]->(f)
                    """, name=financial_name, description=financial_description, embedding=financial_embedding, chunk_id=chunk_id)
                    chunk_entity_ids.append(('Financial', financial_name))
                
                # Create RELATES_TO relationships between entities in the same chunk
                self.create_entity_relationships(session, chunk_entity_ids)
        
        return {
            'document_id': doc_id,
            'chunks_created': len(chunks),
            'status': 'success'
        }
    
    def process_directory(self, pdf_dir: str) -> Dict[str, Any]:
        """Process all PDFs in a directory"""
        results = {}
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files")
        
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
        
        return results
    
    def close(self):
        """Close database connection"""
        self.driver.close()

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