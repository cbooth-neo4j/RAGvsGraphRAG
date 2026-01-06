"""
HotpotQA Data Loader

Downloads and processes the HotpotQA fullwiki dataset and corresponding
Wikipedia articles for RAG benchmarking.
"""

import json
import os
import time
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import wikipediaapi
    WIKIPEDIA_API_AVAILABLE = True
except ImportError:
    WIKIPEDIA_API_AVAILABLE = False
    print("WARNING: wikipedia-api not installed. Run: pip install wikipedia-api")

from .configs import (
    DEFAULT_CACHE_DIR,
    WIKIPEDIA_API_SETTINGS,
    HOTPOTQA_URLS
)


def load_hotpotqa_fullwiki(
    split: str = "dev",
    cache_dir: str = DEFAULT_CACHE_DIR
) -> List[Dict[str, Any]]:
    """
    Download and load HotpotQA fullwiki dataset from official source.
    
    Args:
        split: Dataset split - "dev" (default) or "test"
        cache_dir: Directory to cache downloaded files
        
    Returns:
        List of question dictionaries with keys:
        - id: Question ID
        - question: The question text
        - answer: Ground truth answer
        - type: Question type (bridge/comparison)
        - level: Difficulty level (easy/medium/hard)
        - supporting_facts: List of (title, sentence_idx) tuples
        - context: List of (title, sentences) tuples (for dev set)
    """
    # Determine URL based on split
    if split == "dev":
        url = HOTPOTQA_URLS["dev_fullwiki"]
    elif split == "test":
        url = HOTPOTQA_URLS["test_fullwiki"]
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'dev' or 'test'.")
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Cache file path
    cache_file = cache_path / f"hotpot_{split}_fullwiki.json"
    
    # Download if not cached
    if not cache_file.exists():
        print(f"[DOWNLOAD] Downloading HotpotQA {split} fullwiki dataset...")
        print(f"   Source: {url}")
        
        try:
            urllib.request.urlretrieve(url, cache_file)
            print(f"   [OK] Saved to {cache_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to download HotpotQA dataset: {e}")
    else:
        print(f"[CACHE] Using cached HotpotQA {split} dataset: {cache_file}")
    
    # Load and parse
    print(f"[LOAD] Loading questions from {cache_file}...")
    with open(cache_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Normalize to consistent format
    questions = []
    for item in data:
        question = {
            "id": item.get("_id", item.get("id")),
            "question": item["question"],
            "answer": item.get("answer", ""),  # Empty for test set
            "type": item.get("type", "unknown"),
            "level": item.get("level", "unknown"),
            "supporting_facts": item.get("supporting_facts", []),
            "context": item.get("context", [])
        }
        questions.append(question)
    
    print(f"   [OK] Loaded {len(questions)} questions")
    
    # Print distribution
    types = {}
    levels = {}
    for q in questions:
        types[q["type"]] = types.get(q["type"], 0) + 1
        levels[q["level"]] = levels.get(q["level"], 0) + 1
    
    print(f"   Types: {types}")
    print(f"   Levels: {levels}")
    
    return questions


def extract_referenced_titles(questions: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract Wikipedia article titles referenced in HotpotQA questions.
    
    For HotpotQA, the relevant titles come from:
    1. The 'context' field (for dev set with distractor/gold paragraphs)
    2. The 'supporting_facts' field (title, sentence_idx pairs)
    
    Args:
        questions: List of question dictionaries from load_hotpotqa_fullwiki
        
    Returns:
        Set of unique Wikipedia article titles
    """
    titles = set()
    
    for q in questions:
        # Extract from context (list of [title, sentences] pairs)
        for ctx in q.get("context", []):
            if isinstance(ctx, (list, tuple)) and len(ctx) >= 1:
                title = ctx[0]
                if title:
                    titles.add(title)
        
        # Extract from supporting facts (list of [title, sent_idx] pairs)
        for sf in q.get("supporting_facts", []):
            if isinstance(sf, (list, tuple)) and len(sf) >= 1:
                title = sf[0]
                if title:
                    titles.add(title)
    
    print(f"[EXTRACT] Found {len(titles)} unique article titles from {len(questions)} questions")
    
    return titles


def download_wikipedia_articles(
    titles: Set[str],
    cache_dir: str = DEFAULT_CACHE_DIR,
    language: str = "en",
    max_workers: int = 4
) -> List[Dict[str, str]]:
    """
    Download Wikipedia articles via API with local caching.
    
    Args:
        titles: Set of Wikipedia article titles to download
        cache_dir: Directory to cache downloaded articles
        language: Wikipedia language code (default: "en")
        max_workers: Number of parallel download threads
        
    Returns:
        List of article dictionaries with keys:
        - title: Article title
        - text: Full article text
        - url: Wikipedia URL
        - summary: Article summary
    """
    if not WIKIPEDIA_API_AVAILABLE:
        raise ImportError(
            "wikipedia-api package required. Install with: pip install wikipedia-api"
        )
    
    # Create cache directory
    cache_path = Path(cache_dir) / "articles"
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize Wikipedia API
    wiki = wikipediaapi.Wikipedia(
        user_agent=WIKIPEDIA_API_SETTINGS["user_agent"],
        language=language
    )
    
    articles = []
    cached_count = 0
    downloaded_count = 0
    failed_titles = []
    
    def get_cache_filename(title: str) -> Path:
        """Generate a safe cache filename for a title."""
        # Create hash of title for safe filename
        title_hash = hashlib.md5(title.encode()).hexdigest()[:12]
        safe_title = "".join(c if c.isalnum() else "_" for c in title[:50])
        return cache_path / f"{safe_title}_{title_hash}.json"
    
    def download_single_article(title: str) -> Optional[Dict[str, str]]:
        """Download a single article with caching."""
        cache_file = get_cache_filename(title)
        
        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass  # Re-download if cache is corrupted
        
        # Download from Wikipedia
        try:
            time.sleep(WIKIPEDIA_API_SETTINGS["rate_limit_delay"])
            
            page = wiki.page(title)
            
            if not page.exists():
                return None
            
            article = {
                "title": page.title,
                "text": page.text,
                "url": page.fullurl,
                "summary": page.summary[:500] if page.summary else ""
            }
            
            # Cache the article
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(article, f, ensure_ascii=False, indent=2)
            
            return article
            
        except Exception as e:
            print(f"   [WARN] Failed to download '{title}': {e}")
            return None
    
    print(f"[DOWNLOAD] Downloading {len(titles)} Wikipedia articles...")
    print(f"   Cache directory: {cache_path}")
    
    # Download articles with progress tracking
    titles_list = list(titles)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_title = {
            executor.submit(download_single_article, title): title 
            for title in titles_list
        }
        
        for i, future in enumerate(as_completed(future_to_title)):
            title = future_to_title[future]
            
            try:
                article = future.result()
                
                if article:
                    articles.append(article)
                    # Check if it was cached
                    if get_cache_filename(title).exists():
                        cached_count += 1
                    else:
                        downloaded_count += 1
                else:
                    failed_titles.append(title)
                    
            except Exception as e:
                print(f"   [ERROR] Exception for '{title}': {e}")
                failed_titles.append(title)
            
            # Progress update every 100 articles
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i + 1}/{len(titles_list)} "
                      f"({len(articles)} successful, {len(failed_titles)} failed)")
    
    print(f"\n[DONE] Article download complete:")
    print(f"   Total successful: {len(articles)}")
    print(f"   From cache: {cached_count}")
    print(f"   Downloaded: {downloaded_count}")
    print(f"   Failed: {len(failed_titles)}")
    
    if failed_titles and len(failed_titles) <= 20:
        print(f"   Failed titles: {failed_titles}")
    
    return articles


def prepare_corpus(
    questions_split: str = "dev",
    cache_dir: str = DEFAULT_CACHE_DIR,
    question_limit: Optional[int] = None,
    include_context_articles: bool = True
) -> Dict[str, Any]:
    """
    Main entry point - downloads questions and matching Wikipedia articles.
    
    This function:
    1. Downloads the HotpotQA fullwiki questions
    2. Extracts referenced article titles from questions
    3. Downloads the Wikipedia articles
    4. Returns a complete corpus ready for ingestion
    
    Args:
        questions_split: Dataset split ("dev" or "test")
        cache_dir: Directory for caching downloaded data
        question_limit: Optional limit on number of questions (for testing)
        include_context_articles: If True, also includes articles from context field
        
    Returns:
        Dictionary with:
        - questions: List of question dictionaries
        - articles: List of article dictionaries  
        - stats: Processing statistics
    """
    print("\n" + "="*60)
    print("PREPARING HOTPOTQA CORPUS")
    print("="*60)
    
    # Step 1: Load questions
    questions = load_hotpotqa_fullwiki(split=questions_split, cache_dir=cache_dir)
    
    # Apply limit if specified
    if question_limit and question_limit < len(questions):
        print(f"\n[LIMIT] Limiting to {question_limit} questions (from {len(questions)})")
        questions = questions[:question_limit]
    
    # Step 2: Extract article titles
    titles = extract_referenced_titles(questions)
    
    # Step 3: Download Wikipedia articles
    articles = download_wikipedia_articles(titles, cache_dir=cache_dir)
    
    # Calculate coverage stats
    article_titles = {a["title"] for a in articles}
    questions_with_articles = 0
    
    for q in questions:
        q_titles = set()
        for sf in q.get("supporting_facts", []):
            if isinstance(sf, (list, tuple)) and len(sf) >= 1:
                q_titles.add(sf[0])
        
        if q_titles and q_titles.issubset(article_titles):
            questions_with_articles += 1
    
    stats = {
        "total_questions": len(questions),
        "unique_titles_referenced": len(titles),
        "articles_downloaded": len(articles),
        "questions_with_full_coverage": questions_with_articles,
        "coverage_percentage": round(questions_with_articles / len(questions) * 100, 2),
        "total_text_chars": sum(len(a["text"]) for a in articles),
        "avg_article_length": round(sum(len(a["text"]) for a in articles) / len(articles)) if articles else 0
    }
    
    print("\n" + "="*60)
    print("CORPUS PREPARATION COMPLETE")
    print("="*60)
    print(f"   Questions: {stats['total_questions']}")
    print(f"   Articles: {stats['articles_downloaded']}")
    print(f"   Coverage: {stats['coverage_percentage']}%")
    print(f"   Total text: {stats['total_text_chars']:,} characters")
    print(f"   Avg article: {stats['avg_article_length']:,} characters")
    print("="*60 + "\n")
    
    return {
        "questions": questions,
        "articles": articles,
        "stats": stats
    }


def load_cached_corpus(cache_dir: str = DEFAULT_CACHE_DIR) -> Optional[Dict[str, Any]]:
    """
    Load a previously prepared corpus from cache.
    
    Args:
        cache_dir: Cache directory
        
    Returns:
        Corpus dictionary if found, None otherwise
    """
    corpus_file = Path(cache_dir) / "prepared_corpus.json"
    
    if corpus_file.exists():
        print(f"[CACHE] Loading cached corpus from {corpus_file}...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None


def save_corpus(corpus: Dict[str, Any], cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """
    Save prepared corpus to cache for faster future loads.
    
    Args:
        corpus: Corpus dictionary from prepare_corpus
        cache_dir: Cache directory
        
    Returns:
        Path to saved file
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    corpus_file = cache_path / "prepared_corpus.json"
    
    print(f"[SAVE] Saving corpus to {corpus_file}...")
    with open(corpus_file, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False)
    
    return str(corpus_file)


if __name__ == "__main__":
    # Test the data loader
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HotpotQA data loader")
    parser.add_argument("--limit", type=int, default=50, help="Question limit")
    parser.add_argument("--split", default="dev", help="Dataset split")
    args = parser.parse_args()
    
    corpus = prepare_corpus(
        questions_split=args.split,
        question_limit=args.limit
    )
    
    print(f"\nSample question: {corpus['questions'][0]}")
    print(f"\nSample article title: {corpus['articles'][0]['title']}")

