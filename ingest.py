#!/usr/bin/env python3
"""
Unified data ingestion pipeline for RAGvsGraphRAG.

Single entry point for building knowledge graphs from any supported data source.
Benchmark evaluation is separate - this script ONLY builds the graph.

Usage:
    python ingest.py --source hotpotqa --quantity 100 --lean --new     # Fresh start, clears DB
    python ingest.py --source hotpotqa --quantity 100 --lean --resume  # Resume from where it stopped
    python ingest.py --source pdf --quantity 10 --full --new
    
Required arguments:
    --source      Data source: pdf | hotpotqa
    --quantity    Number of questions (hotpotqa) or documents (pdf) to process
    --lean        Build minimal graph (no summaries, no communities)
    --full        Build complete graph (with summaries + communities)
    --new         Clear database and start fresh
    --resume      Resume from previous progress (skips already-processed articles)
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from utils.graph_rag_logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def ingest_pdfs(pdf_dir: str, quantity: int, lean_mode: bool) -> dict:
    """
    Ingest PDF documents into the knowledge graph.
    
    Args:
        pdf_dir: Directory containing PDF files
        quantity: Maximum number of PDFs to process
        lean_mode: If True, skip summaries and communities
    """
    from data_processors.build_graph import CustomGraphProcessor
    
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    # Find all PDFs
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    # Apply quantity limit
    total_available = len(pdf_files)
    if quantity >= total_available:
        print(f"[INFO] Requested {quantity}, found {total_available}. Processing all {total_available}.")
        quantity = total_available
    else:
        print(f"[INFO] Processing {quantity} of {total_available} available PDFs")
    
    # Select PDFs (first N)
    selected_pdfs = pdf_files[:quantity]
    
    print(f"\n{'='*60}")
    print(f"INGESTING PDFs")
    print(f"{'='*60}")
    print(f"   Source: {pdf_dir}")
    print(f"   Documents: {quantity}")
    print(f"   Mode: {'LEAN (no summaries/communities)' if lean_mode else 'FULL (with summaries/communities)'}")
    print(f"{'='*60}\n")
    
    # Initialize processor
    processor = CustomGraphProcessor()
    
    try:
        # Process the directory with selected files
        result = processor.process_directory(
            pdf_dir=str(pdf_dir),
            perform_resolution=True,
            prompt_for_advanced=not lean_mode,
            auto_advanced=not lean_mode,
            mode='fresh',
            lean_mode=lean_mode
        )
        return result
    finally:
        processor.close()


def get_existing_documents(driver, db_name: str) -> set:
    """Query Neo4j for existing document source_info values."""
    with driver.session(database=db_name) as session:
        result = session.run("MATCH (d:Document) RETURN d.source_info AS source_info")
        return {r["source_info"] for r in result if r["source_info"]}


def get_ingestion_progress(driver, db_name: str) -> dict:
    """Get current ingestion progress from Neo4j."""
    with driver.session(database=db_name) as session:
        result = session.run("""
            MATCH (p:__IngestionProgress__ {id: 'current'})
            RETURN p.processed_titles AS processed_titles,
                   p.total_expected AS total_expected,
                   p.started_at AS started_at,
                   p.last_updated AS last_updated,
                   p.questions_count AS questions_count,
                   p.source AS source,
                   p.lean_mode AS lean_mode,
                   p.answerable_questions AS answerable_questions,
                   p.answerable_count AS answerable_count
        """)
        record = result.single()
        if record:
            return {
                'processed_titles': record['processed_titles'] or [],
                'total_expected': record['total_expected'],
                'started_at': record['started_at'],
                'last_updated': record['last_updated'],
                'questions_count': record['questions_count'],
                'source': record['source'],
                'lean_mode': record['lean_mode'],
                'answerable_questions': record['answerable_questions'] or [],
                'answerable_count': record['answerable_count'] or 0
            }
        return None


def update_ingestion_progress(driver, db_name: str, processed_titles: list, total_expected: int, 
                               questions_count: int, source: str = "hotpotqa", lean_mode: bool = True,
                               answerable_questions: list = None):
    """Update ingestion progress in Neo4j."""
    import time
    with driver.session(database=db_name) as session:
        session.run("""
            MERGE (p:__IngestionProgress__ {id: 'current'})
            SET p.processed_titles = $processed_titles,
                p.total_expected = $total_expected,
                p.questions_count = $questions_count,
                p.last_updated = $last_updated,
                p.source = $source,
                p.lean_mode = $lean_mode,
                p.answerable_questions = $answerable_questions,
                p.answerable_count = $answerable_count
            ON CREATE SET p.started_at = $last_updated
        """, 
            processed_titles=processed_titles,
            total_expected=total_expected,
            questions_count=questions_count,
            last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
            source=source,
            lean_mode=lean_mode,
            answerable_questions=answerable_questions or [],
            answerable_count=len(answerable_questions) if answerable_questions else 0
        )


def clear_ingestion_progress(driver, db_name: str):
    """Clear ingestion progress node."""
    with driver.session(database=db_name) as session:
        session.run("MATCH (p:__IngestionProgress__) DELETE p")


def get_testable_questions(corpus_path: str = "data/hotpotqa/prepared_corpus.json", 
                           quantity: int = None) -> dict:
    """
    Determine which questions can be tested based on currently ingested articles.
    
    A question is "testable" only if ALL its supporting articles are in the database.
    
    Args:
        corpus_path: Path to the prepared corpus JSON
        quantity: Limit to first N questions (default: all)
    
    Returns:
        dict with:
            - testable_questions: list of question objects that have full article coverage
            - testable_count: number of testable questions
            - total_questions: total questions checked
            - missing_coverage: dict mapping question_id to missing article titles
            - ingested_articles: set of article titles in database
    """
    import json
    from neo4j import GraphDatabase
    
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    neo4j_db = os.environ.get('CLIENT_NEO4J_DATABASE', 'neo4j')
    
    # Load corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    questions = corpus["questions"]
    if quantity:
        questions = questions[:quantity]
    
    # Get ingested articles from Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    db_name = neo4j_db
    
    try:
        with driver.session(database=db_name) as session:
            # Try to get from progress node first
            result = session.run("""
                MATCH (p:__IngestionProgress__ {id: 'current'})
                RETURN p.processed_titles AS titles
            """)
            record = result.single()
            
            if record and record['titles']:
                ingested_titles = set(record['titles'])
            else:
                # Fall back to scanning Document nodes
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
    finally:
        driver.close()
    
    # Check each question for complete coverage
    testable_questions = []
    missing_coverage = {}
    
    for q in questions:
        # Get all articles referenced by this question
        referenced = set()
        for ctx in q.get("context", []):
            if isinstance(ctx, (list, tuple)) and len(ctx) >= 1:
                referenced.add(ctx[0])
        for sf in q.get("supporting_facts", []):
            if isinstance(sf, (list, tuple)) and len(sf) >= 1:
                referenced.add(sf[0])
        
        # Check if all referenced articles are ingested
        missing = referenced - ingested_titles
        q_id = q.get('_id', f"q_{questions.index(q)}")
        
        if not missing:
            testable_questions.append(q)
        else:
            missing_coverage[q_id] = list(missing)
    
    return {
        'testable_questions': testable_questions,
        'testable_count': len(testable_questions),
        'total_questions': len(questions),
        'coverage_percent': round(len(testable_questions) / len(questions) * 100, 1) if questions else 0,
        'missing_coverage': missing_coverage,
        'ingested_articles': ingested_titles,
        'ingested_count': len(ingested_titles)
    }


def ingest_hotpotqa(quantity: int, lean_mode: bool, cache_dir: str = "data/hotpotqa", resume: bool = False) -> dict:
    """
    Ingest HotpotQA Wikipedia articles into the knowledge graph.
    
    Args:
        quantity: Number of questions (and their supporting articles) to process
        lean_mode: If True, skip summaries and communities
        cache_dir: Directory for caching downloaded data
        resume: If True, resume from previous progress instead of starting fresh
    """
    from data_processors.build_graph import CustomGraphProcessor
    from benchmark.hotpotqa.data_loader import prepare_corpus, load_cached_corpus, save_corpus
    from neo4j import GraphDatabase
    
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    neo4j_db = os.environ.get('CLIENT_NEO4J_DATABASE', 'neo4j')
    
    print(f"\n{'='*60}")
    print(f"INGESTING HOTPOTQA WIKIPEDIA ARTICLES")
    print(f"{'='*60}")
    print(f"   Questions: {quantity}")
    print(f"   Mode: {'LEAN (no summaries/communities)' if lean_mode else 'FULL (with summaries/communities)'}")
    print(f"   Resume: {resume}")
    print(f"   Cache: {cache_dir}")
    print(f"{'='*60}\n")
    
    # Load or prepare corpus
    print("[PHASE 1] Preparing corpus...")
    corpus = load_cached_corpus(cache_dir)
    
    if corpus and len(corpus.get("questions", [])) >= quantity:
        print(f"[CACHE] Using cached corpus ({len(corpus['questions'])} questions available)")
    else:
        print(f"[DOWNLOAD] Preparing fresh corpus...")
        corpus = prepare_corpus(
            questions_split="dev",
            cache_dir=cache_dir,
            question_limit=quantity
        )
        save_corpus(corpus, cache_dir)
    
    questions = corpus["questions"][:quantity]
    articles = corpus["articles"]
    
    # Filter articles to match selected questions
    referenced_titles = set()
    for q in questions:
        for ctx in q.get("context", []):
            if isinstance(ctx, (list, tuple)) and len(ctx) >= 1:
                referenced_titles.add(ctx[0])
        for sf in q.get("supporting_facts", []):
            if isinstance(sf, (list, tuple)) and len(sf) >= 1:
                referenced_titles.add(sf[0])
    
    articles = [a for a in articles if a.get("title") in referenced_titles]
    
    print(f"[INFO] Processing {len(articles)} articles for {len(questions)} questions")
    
    # Build question -> required articles mapping for tracking answerability
    question_requirements = {}
    for q in questions:
        q_id = q.get('_id', f"q_{questions.index(q)}")
        required_titles = set()
        for sf in q.get('supporting_facts', []):
            if isinstance(sf, (list, tuple)) and len(sf) >= 1:
                required_titles.add(sf[0])
        question_requirements[q_id] = required_titles
    
    print(f"[INFO] Tracking answerability for {len(question_requirements)} questions")
    
    # Prepare texts and sources for ingestion
    texts = [a["text"] for a in articles]
    sources = [a["title"] for a in articles]
    
    # Initialize processor
    processor = CustomGraphProcessor()
    
    # Get Neo4j driver for progress tracking
    progress_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    db_name = neo4j_db
    
    try:
        # Handle resume vs fresh start
        already_processed = set()
        
        if resume:
            # Check for existing progress
            progress = get_ingestion_progress(progress_driver, db_name)
            if progress:
                already_processed = set(progress['processed_titles'])
                print(f"[RESUME] Found existing progress: {len(already_processed)}/{progress['total_expected']} articles processed")
                print(f"[RESUME] Started at: {progress['started_at']}, Last updated: {progress['last_updated']}")
                
                # Validate that the quantity matches
                if progress['questions_count'] != quantity:
                    print(f"[WARNING] Previous run used {progress['questions_count']} questions, current run uses {quantity}")
                    user_input = input("Continue anyway? (y/n): ").strip().lower()
                    if user_input != 'y':
                        print("[ABORT] User cancelled resume")
                        return {'status': 'cancelled'}
            else:
                # No progress found, check for existing documents
                existing_docs = get_existing_documents(progress_driver, db_name)
                if existing_docs:
                    # Extract source titles from source_info (format: "wikipedia:Title")
                    for doc_source in existing_docs:
                        if doc_source and doc_source.startswith("wikipedia:"):
                            already_processed.add(doc_source.replace("wikipedia:", ""))
                    print(f"[RESUME] Found {len(already_processed)} existing documents in database")
                else:
                    print("[RESUME] No existing progress or documents found, starting fresh")
        else:
            # Fresh start - clear database
            print("[FRESH] Clearing database for fresh ingestion...")
            processor.clear_database()
            clear_ingestion_progress(progress_driver, db_name)
        
        # Discover labels from corpus sample (only if not already set)
        if not processor.discovered_labels:
            sample_text = "\n\n".join(text[:2000] for text in texts[:20])
            proposed_labels = processor.discover_labels_for_text(sample_text)
            processor.discovered_labels = processor._approve_labels_cli(proposed_labels)
        
        # Process each article
        results = []
        total_chunks = 0
        total_entities = 0
        processed_titles = list(already_processed)  # Start with already processed
        processed_titles_set = set(already_processed)  # For fast lookup
        skipped_count = 0
        
        # Track answerable questions
        answerable_questions = []
        last_answerable_count = 0
        
        # Check which questions are already answerable from resumed progress
        for q_id, required in question_requirements.items():
            if required and required.issubset(processed_titles_set):
                answerable_questions.append(q_id)
        if answerable_questions:
            print(f"[RESUME] {len(answerable_questions)} questions already answerable from previous progress")
            last_answerable_count = len(answerable_questions)
        
        for i, (text, source) in enumerate(zip(texts, sources)):
            # Skip if already processed
            if source in already_processed:
                skipped_count += 1
                if skipped_count <= 5 or skipped_count % 50 == 0:
                    print(f"[SKIP] Already processed: {source} ({skipped_count} skipped so far)")
                continue
                
            try:
                doc_name = f"wiki_{source}_{i}"
                result = processor.process_text_document(
                    text=text,
                    doc_name=doc_name,
                    source_info=f"wikipedia:{source}"
                )
                results.append(result)
                total_chunks += result.get('chunks_created', 0)
                total_entities += result.get('entities_created', 0)
                processed_titles.append(source)
                processed_titles_set.add(source)
                
                # Check if any new questions became answerable
                for q_id, required in question_requirements.items():
                    if q_id not in answerable_questions:
                        if required and required.issubset(processed_titles_set):
                            answerable_questions.append(q_id)
                
                # Report when new questions become answerable
                if len(answerable_questions) > last_answerable_count:
                    new_count = len(answerable_questions) - last_answerable_count
                    print(f"[ANSWERABLE] +{new_count} questions now answerable! Total: {len(answerable_questions)}/{len(question_requirements)}")
                    last_answerable_count = len(answerable_questions)
                    
                    # Update progress immediately when questions become answerable
                    update_ingestion_progress(
                        progress_driver, db_name,
                        processed_titles, len(texts), quantity,
                        source="hotpotqa", lean_mode=lean_mode,
                        answerable_questions=answerable_questions
                    )
                
                # Also update progress every 50 articles for resume safety
                elif len(processed_titles) % 50 == 0:
                    update_ingestion_progress(
                        progress_driver, db_name,
                        processed_titles, len(texts), quantity,
                        source="hotpotqa", lean_mode=lean_mode,
                        answerable_questions=answerable_questions
                    )
                
                if (i + 1) % 20 == 0:
                    print(f"[PROGRESS] Processed {i + 1}/{len(texts)} articles | Answerable: {len(answerable_questions)}/{len(question_requirements)} questions")
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {source}: {e}")
                results.append({'status': 'failed', 'error': str(e)})
        
        if skipped_count > 0:
            print(f"[RESUME] Skipped {skipped_count} already-processed articles")
        
        # Final progress save before entity resolution
        update_ingestion_progress(
            progress_driver, db_name,
            processed_titles, len(texts), quantity,
            source="hotpotqa", lean_mode=lean_mode,
            answerable_questions=answerable_questions
        )
        print(f"[CHECKPOINT] Saved progress: {len(processed_titles)}/{len(texts)} articles")
        print(f"[CHECKPOINT] Answerable questions: {len(answerable_questions)}/{len(question_requirements)}")
        
        # Also update TestableQuestions node for benchmark use
        save_testable_questions_to_neo4j(
            answerable_questions,
            len(answerable_questions),
            len(question_requirements),
            len(processed_titles)
        )
        
        # Entity resolution
        print("\n[RESOLVE] Performing entity resolution...")
        processor.create_chunk_similarity_relationships()
        processor.perform_entity_resolution()
        
        # Advanced processing (if not lean mode)
        if not lean_mode:
            print("\n[ADVANCED] Running advanced processing...")
            graph_stats = processor.get_graph_statistics()
            processor.perform_advanced_processing(graph_stats)
        else:
            print("\n[LEAN] Skipping advanced processing (summaries/communities)")
        
        # Save ingestion manifest to Neo4j (source of truth for benchmark pairing)
        import json
        import time
        manifest = {
            'source': 'hotpotqa',
            'ingested_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'questions_count': len(questions),
            'articles_count': len(texts),
            'article_titles': sources,
            'question_ids': [q.get('_id', f'q_{i}') for i, q in enumerate(questions)],
            'lean_mode': lean_mode,
            'total_chunks': total_chunks,
            'total_entities': total_entities
        }
        
        # Store manifest in Neo4j as a node
        from neo4j import GraphDatabase
        manifest_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        try:
            with manifest_driver.session(database=db_name) as session:
                session.run("""
                    MERGE (m:__IngestionManifest__ {id: 'current'})
                    SET m.source = $source,
                        m.ingested_at = $ingested_at,
                        m.questions_count = $questions_count,
                        m.articles_count = $articles_count,
                        m.article_titles = $article_titles,
                        m.question_ids = $question_ids,
                        m.lean_mode = $lean_mode,
                        m.total_chunks = $total_chunks,
                        m.total_entities = $total_entities
                """, **manifest)
            print(f"[MANIFEST] Saved ingestion manifest to Neo4j (:__IngestionManifest__)")
        finally:
            manifest_driver.close()
        
        summary = {
            'source': 'hotpotqa',
            'questions_requested': quantity,
            'articles_processed': len(texts),
            'total_chunks': total_chunks,
            'total_entities': total_entities,
            'lean_mode': lean_mode,
            'status': 'completed'
        }
        
        print(f"\n{'='*60}")
        print("INGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"   Articles: {len(texts)}")
        print(f"   Chunks: {total_chunks:,}")
        print(f"   Entities: {total_entities:,}")
        print(f"   Mode: {'LEAN' if lean_mode else 'FULL'}")
        print(f"   Manifest: Stored in Neo4j (:__IngestionManifest__)")
        print(f"{'='*60}\n")
        
        # Clear progress tracking on successful completion
        clear_ingestion_progress(progress_driver, db_name)
        print(f"[CLEANUP] Cleared progress tracking (ingestion complete)")
        
        return summary
        
    finally:
        processor.close()
        progress_driver.close()


def load_cached_corpus(cache_dir: str):
    """Load cached corpus if available."""
    import json
    corpus_file = Path(cache_dir) / "prepared_corpus.json"
    if corpus_file.exists():
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_testable_questions_to_neo4j(testable_ids: list, testable_count: int, 
                                      total_questions: int, ingested_count: int):
    """Save testable question IDs to Neo4j for database-specific tracking."""
    import time
    from neo4j import GraphDatabase
    
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    neo4j_db = os.environ.get('CLIENT_NEO4J_DATABASE', 'neo4j')
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session(database=neo4j_db) as session:
            session.run("""
                MERGE (t:__TestableQuestions__ {id: 'current'})
                SET t.question_ids = $question_ids,
                    t.testable_count = $testable_count,
                    t.total_questions = $total_questions,
                    t.ingested_articles = $ingested_articles,
                    t.computed_at = $computed_at
            """,
                question_ids=testable_ids,
                testable_count=testable_count,
                total_questions=total_questions,
                ingested_articles=ingested_count,
                computed_at=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        print(f"[NEO4J] Saved testable questions to :__TestableQuestions__ node")
    finally:
        driver.close()


def get_testable_questions_from_neo4j() -> dict:
    """Get testable question IDs from Neo4j."""
    from neo4j import GraphDatabase
    
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    neo4j_db = os.environ.get('CLIENT_NEO4J_DATABASE', 'neo4j')
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session(database=neo4j_db) as session:
            result = session.run("""
                MATCH (t:__TestableQuestions__ {id: 'current'})
                RETURN t.question_ids as question_ids,
                       t.testable_count as testable_count,
                       t.total_questions as total_questions,
                       t.ingested_articles as ingested_articles,
                       t.computed_at as computed_at
            """)
            record = result.single()
            if record:
                return {
                    'question_ids': record['question_ids'] or [],
                    'testable_count': record['testable_count'],
                    'total_questions': record['total_questions'],
                    'ingested_articles': record['ingested_articles'],
                    'computed_at': record['computed_at']
                }
        return None
    finally:
        driver.close()


def check_testable_cmd(quantity: int, cache_dir: str):
    """CLI handler for --check-testable command."""
    corpus_path = f"{cache_dir}/prepared_corpus.json"
    
    print(f"\n{'='*60}")
    print("CHECKING TESTABLE QUESTIONS")
    print(f"{'='*60}\n")
    
    try:
        result = get_testable_questions(corpus_path, quantity)
    except FileNotFoundError:
        print(f"[ERROR] Corpus not found at {corpus_path}")
        print("Run ingestion first to download the corpus.")
        return 1
    
    print(f"Ingested Articles: {result['ingested_count']}")
    print(f"Questions Checked: {result['total_questions']}")
    print(f"Testable Questions: {result['testable_count']} ({result['coverage_percent']}%)")
    
    if result['missing_coverage']:
        print(f"\nQuestions with incomplete coverage: {len(result['missing_coverage'])}")
        # Show first 5 examples
        examples = list(result['missing_coverage'].items())[:5]
        for q_id, missing in examples:
            print(f"  - {q_id}: missing {len(missing)} article(s)")
            for title in missing[:2]:
                # Handle Unicode characters that may not display on Windows console
                safe_title = title.encode('ascii', 'replace').decode('ascii')
                print(f"      - {safe_title}")
            if len(missing) > 2:
                print(f"      - ... and {len(missing)-2} more")
    
    print(f"\n{'='*60}")
    print(f"You can run benchmarks on {result['testable_count']} questions")
    print(f"{'='*60}\n")
    
    # Get testable question IDs
    testable_ids = [q.get('_id', f"q_{i}") for i, q in enumerate(result['testable_questions'])]
    
    # Save to Neo4j (primary storage - database-specific)
    save_testable_questions_to_neo4j(
        testable_ids, 
        result['testable_count'],
        result['total_questions'],
        result['ingested_count']
    )
    
    # Also save to local file (backup/convenience)
    import json
    output_file = f"{cache_dir}/testable_questions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'testable_question_ids': testable_ids,
            'testable_count': result['testable_count'],
            'total_questions': result['total_questions'],
            'ingested_articles': result['ingested_count']
        }, f, indent=2)
    print(f"[FILE] Also saved to: {output_file}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified data ingestion for RAGvsGraphRAG knowledge graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --source hotpotqa --quantity 100 --lean --new      # Fresh start
  python ingest.py --source hotpotqa --quantity 100 --lean --resume   # Resume interrupted run
  python ingest.py --source pdf --quantity 10 --full --new
  python ingest.py --check-testable --quantity 100                    # Check which questions can be tested

After ingestion, run benchmarks:
  python -m benchmark.hotpotqa.benchmark_pipeline mini
        """
    )
    
    # Check-testable mode (standalone command)
    parser.add_argument(
        '--check-testable',
        action='store_true',
        help='Check which questions can be tested based on currently ingested articles. Use alone with --quantity.'
    )
    
    # Required arguments (for ingestion)
    parser.add_argument(
        '--source',
        choices=['pdf', 'hotpotqa'],
        help='Data source to ingest (required for ingestion, not needed for --check-testable)'
    )
    
    parser.add_argument(
        '--quantity',
        type=int,
        required=True,
        help='Number of questions (hotpotqa) or documents (pdf) to process'
    )
    
    # Build mode (mutually exclusive, required for ingestion)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--lean',
        action='store_true',
        help='Build minimal graph: Document->Chunk->Entity with RELATES_TO. No summaries, no communities.'
    )
    mode_group.add_argument(
        '--full',
        action='store_true',
        help='Build complete graph with AI summaries and community detection.'
    )
    
    # Start mode (mutually exclusive, required for ingestion)
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        '--new',
        action='store_true',
        help='Clear database and start fresh ingestion.'
    )
    start_group.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous progress. Skips already-processed articles. Use after interrupted runs.'
    )
    
    # Optional arguments
    parser.add_argument(
        '--pdf-dir',
        default='PDFs',
        help='Directory containing PDF files (default: ./PDFs)'
    )
    
    parser.add_argument(
        '--cache-dir',
        default='data/hotpotqa',
        help='Cache directory for HotpotQA data (default: data/hotpotqa)'
    )
    
    args = parser.parse_args()
    
    # Handle --check-testable mode (standalone command)
    if args.check_testable:
        return check_testable_cmd(args.quantity, args.cache_dir)
    
    # For ingestion mode, validate required arguments
    if not args.source:
        parser.error("--source is required for ingestion (use --check-testable for standalone check)")
    if not (args.lean or args.full):
        parser.error("--lean or --full is required for ingestion")
    if not (args.new or args.resume):
        parser.error("--new or --resume is required for ingestion")
    
    # Determine modes
    lean_mode = args.lean  # True if --lean, False if --full
    resume_mode = args.resume  # True if --resume, False if --new
    
    print(f"\n{'='*60}")
    print("RAGvsGraphRAG INGESTION PIPELINE")
    print(f"{'='*60}")
    print(f"   Source: {args.source}")
    print(f"   Quantity: {args.quantity}")
    print(f"   Build Mode: {'LEAN' if lean_mode else 'FULL'}")
    print(f"   Start Mode: {'RESUME (continuing previous run)' if resume_mode else 'NEW (fresh start)'}")
    print(f"{'='*60}\n")
    
    try:
        if args.source == 'pdf':
            if resume_mode:
                print("[WARNING] --resume not yet implemented for PDF source. Using --new behavior.")
            result = ingest_pdfs(
                pdf_dir=args.pdf_dir,
                quantity=args.quantity,
                lean_mode=lean_mode
            )
        elif args.source == 'hotpotqa':
            result = ingest_hotpotqa(
                quantity=args.quantity,
                lean_mode=lean_mode,
                cache_dir=args.cache_dir,
                resume=resume_mode
            )
        
        print("[SUCCESS] Ingestion completed successfully")
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

