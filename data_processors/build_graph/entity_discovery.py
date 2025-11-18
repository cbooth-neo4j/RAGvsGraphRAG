"""
Entity discovery and schema management for graph building.
Includes enhanced sampling and discovery techniques.
"""

import json
import re
import hashlib
import random
import pandas as pd
import os
import sys
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
# Optional dependencies for enhanced features
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Import centralized configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_llm
from utils.graph_rag_logger import setup_logging, get_logger
from dotenv import load_dotenv

load_dotenv()

setup_logging()
logger = get_logger(__name__)

class EntityDiscoveryMixin:
    """
    Mixin for enhanced entity discovery capabilities.
    Combines current approach with research-based improvements.
    """
    
    def __init__(self):
        # Schema caching
        self.schema_cache_file = "schema_cache.json"
        self.discovered_labels = []
        
        # Enhanced sampling parameters
        self.max_sample_size = 12000
        self.min_documents = 5
        self.max_documents = 50
        
        # Domain-specific patterns
        self.domain_patterns = {
            'financial': ['FINANCIAL_INSTRUMENT', 'CURRENCY', 'MARKET', 'INVESTMENT'],
            'medical': ['DISEASE', 'TREATMENT', 'MEDICATION', 'SYMPTOM', 'ANATOMY'],
            'legal': ['LAW', 'CASE', 'COURT', 'CONTRACT', 'REGULATION'],
            'technical': ['SYSTEM', 'PROCESS', 'METHOD', 'ALGORITHM', 'PROTOCOL'],
            'academic': ['THEORY', 'RESEARCH', 'PUBLICATION', 'INSTITUTION', 'FIELD']
        }
        
        # Initialize LLM for entity discovery
        self.llm = get_llm()
        
        super().__init__()
    
    def _compute_corpus_hash(self, pdf_files: List[Path]) -> str:
        """Compute hash of corpus for caching."""
        file_info = []
        for pdf_path in pdf_files:
            try:
                stat = pdf_path.stat()
                file_info.append(f"{pdf_path.name}:{stat.st_size}:{stat.st_mtime}")
            except:
                file_info.append(f"{pdf_path.name}:unknown")
        
        corpus_signature = "|".join(sorted(file_info))
        return hashlib.md5(corpus_signature.encode()).hexdigest()[:12]
    
    def _load_schema_cache(self, corpus_hash: str) -> Optional[List[str]]:
        """Load cached schema labels if available."""
        try:
            with open(self.schema_cache_file, 'r') as f:
                cache = json.load(f)
                return cache.get(corpus_hash)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def _save_schema_cache(self, corpus_hash: str, labels: List[str]):
        """Save schema labels to cache."""
        try:
            cache = {}
            try:
                with open(self.schema_cache_file, 'r') as f:
                    cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            
            cache[corpus_hash] = labels
            cache['_metadata'] = {
                'last_updated': str(pd.Timestamp.now()),
                'total_schemas': len(cache) - 1
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
    
    def _sample_corpus_text_enhanced(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced corpus sampling using multiple strategies.
        
        Args:
            documents: List of {'text': str, 'source': str, 'metadata': dict}
            
        Returns:
            {
                'sample_text': str,
                'sampling_stats': dict,
                'document_coverage': list,
                'diversity_score': float
            }
        """
        if not documents:
            return {'sample_text': '', 'sampling_stats': {}, 'document_coverage': [], 'diversity_score': 0.0}
        
        print(f"🔬 Enhanced corpus sampling from {len(documents)} documents...")
        
        # Step 1: Document characterization
        doc_features = self._characterize_documents(documents)
        
        # Step 2: Stratified selection of documents
        selected_docs = self._stratified_document_selection(documents, doc_features)
        
        # Step 3: Multi-strategy text sampling
        sample_components = self._multi_strategy_sampling(selected_docs)
        
        # Step 4: Combine and optimize sample
        final_sample = self._combine_and_optimize_sample(sample_components)
        
        # Step 5: Calculate metrics
        stats = self._calculate_sampling_metrics(documents, selected_docs, final_sample)
        
        return {
            'sample_text': final_sample['text'],
            'sampling_stats': stats,
            'document_coverage': [doc['source'] for doc in selected_docs],
            'diversity_score': final_sample['diversity_score']
        }
    
    def _characterize_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Characterize documents by length, complexity, and domain indicators."""
        features = []
        
        for doc in documents:
            text = doc['text']
            
            # Basic statistics
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text))
            
            # Complexity indicators
            avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
            caps_ratio = len(re.findall(r'[A-Z]', text)) / char_count if char_count > 0 else 0
            
            # Entity density indicators
            potential_entities = len(re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', text))
            number_density = len(re.findall(r'\b\d+(?:\.\d+)?\b', text)) / word_count if word_count > 0 else 0
            
            features.append({
                'doc_index': len(features),
                'char_count': char_count,
                'word_count': word_count,
                'complexity_score': avg_word_length * caps_ratio + potential_entities / word_count if word_count > 0 else 0
            })
        
        return features
    
    def _stratified_document_selection(self, 
                                     documents: List[Dict[str, Any]], 
                                     features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select documents using stratified sampling based on characteristics."""
        
        # Determine number of documents to select
        n_docs = min(max(self.min_documents, len(documents) // 4), 
                    min(self.max_documents, len(documents)))
        
        if n_docs >= len(documents):
            return documents
        
        # Create strata based on document characteristics
        complexity_scores = [f['complexity_score'] for f in features]
        char_counts = [f['char_count'] for f in features]
        
        # Divide into strata (low/medium/high complexity, short/medium/long length)
        complexity_terciles = np.percentile(complexity_scores, [33, 67])
        length_terciles = np.percentile(char_counts, [33, 67])
        
        strata = {}
        for i, (doc, feat) in enumerate(zip(documents, features)):
            complexity_level = 0 if feat['complexity_score'] <= complexity_terciles[0] else \
                             1 if feat['complexity_score'] <= complexity_terciles[1] else 2
            length_level = 0 if feat['char_count'] <= length_terciles[0] else \
                          1 if feat['char_count'] <= length_terciles[1] else 2
            
            stratum = (complexity_level, length_level)
            if stratum not in strata:
                strata[stratum] = []
            strata[stratum].append((i, doc, feat))
        
        # Sample from each stratum
        selected_docs = []
        docs_per_stratum = max(1, n_docs // len(strata))
        
        for stratum_docs in strata.values():
            n_from_stratum = min(docs_per_stratum, len(stratum_docs))
            # Prefer high entity density within stratum
            stratum_docs.sort(key=lambda x: x[2]['complexity_score'], reverse=True)
            selected_docs.extend([doc for _, doc, _ in stratum_docs[:n_from_stratum]])
        
        print(f"📊 Selected {len(selected_docs)} documents from {len(strata)} strata")
        return selected_docs[:n_docs]
    
    def _multi_strategy_sampling(self, documents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Apply multiple sampling strategies to selected documents."""
        
        components = {
            'diversity_samples': [],
            'pattern_samples': [],
            'random_samples': []
        }
        
        # Calculate budget allocation (40% diversity, 30% patterns, 30% random)
        diversity_budget = int(self.max_sample_size * 0.4)
        pattern_budget = int(self.max_sample_size * 0.3)
        random_budget = int(self.max_sample_size * 0.3)
        
        # 1. Diversity-based sampling using TF-IDF clustering
        if diversity_budget > 0:
            components['diversity_samples'] = self._diversity_based_sampling(
                documents, diversity_budget)
        
        # 2. Pattern-based sampling (enhanced version of current approach)
        if pattern_budget > 0:
            components['pattern_samples'] = self._enhanced_pattern_sampling(
                documents, pattern_budget)
        
        # 3. Random sampling for coverage
        if random_budget > 0:
            components['random_samples'] = self._strategic_random_sampling(
                documents, random_budget)
        
        return components
    
    def _diversity_based_sampling(self, documents: List[Dict[str, Any]], budget: int) -> List[str]:
        """Use TF-IDF clustering to sample diverse text segments (if sklearn available)."""
        
        # Extract text segments (sentences or paragraphs)
        segments = []
        segment_sources = []
        
        for doc in documents:
            text = doc['text']
            # Split into sentences or paragraphs
            sentences = re.split(r'[.!?]+', text)
            paragraphs = text.split('\n\n')
            
            # Use paragraphs if available, otherwise sentences
            text_units = [p.strip() for p in paragraphs if len(p.strip()) > 50] or \
                        [s.strip() for s in sentences if len(s.strip()) > 20]
            
            for unit in text_units[:10]:  # Limit per document
                if len(unit) > 20:  # Minimum meaningful length
                    segments.append(unit)
                    segment_sources.append(doc['source'])
        
        if not segments:
            return []
        
        # Use TF-IDF clustering if sklearn is available
        if _HAS_SKLEARN:
            try:
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2
                )
                tfidf_matrix = vectorizer.fit_transform(segments)
                
                # K-means clustering for diversity
                n_clusters = min(10, len(segments) // 3)
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(tfidf_matrix)
                    
                    # Sample from each cluster
                    selected_segments = []
                    current_length = 0
                    
                    for cluster_id in range(n_clusters):
                        cluster_indices = np.where(clusters == cluster_id)[0]
                        if len(cluster_indices) == 0:
                            continue
                        
                        # Find most representative segment in cluster
                        cluster_center = kmeans.cluster_centers_[cluster_id]
                        similarities = cosine_similarity(
                            tfidf_matrix[cluster_indices], 
                            cluster_center.reshape(1, -1)
                        ).flatten()
                        
                        best_idx = cluster_indices[np.argmax(similarities)]
                        segment = segments[best_idx]
                        
                        if current_length + len(segment) <= budget:
                            selected_segments.append(f"[{segment_sources[best_idx]}] {segment}")
                            current_length += len(segment)
                    
                    return selected_segments
                
            except Exception as e:
                print(f"⚠️  TF-IDF clustering failed: {e}, falling back to random sampling")
        else:
            print("⚠️  sklearn not available, using random sampling for diversity")
        
        # Fallback: random sampling
        selected = random.sample(segments, min(5, len(segments)))
        return [f"[Random] {seg[:200]}..." for seg in selected]
    
    def _enhanced_pattern_sampling(self, documents: List[Dict[str, Any]], budget: int) -> List[str]:
        """Enhanced pattern-based sampling with semantic awareness."""
        
        samples = []
        current_length = 0
        
        for doc in documents:
            if current_length >= budget:
                break
                
            text = doc['text']
            
            # Extract patterns using current method
            patterns = self._extract_entity_rich_patterns(text)
            if patterns:
                sample = f"[Patterns - {doc['source']}] {patterns}"
                if current_length + len(sample) <= budget:
                    samples.append(sample)
                    current_length += len(sample)
        
        return samples
    
    def _strategic_random_sampling(self, documents: List[Dict[str, Any]], budget: int) -> List[str]:
        """Strategic random sampling focusing on mid-document content."""
        
        samples = []
        current_length = 0
        
        for doc in documents:
            if current_length >= budget:
                break
                
            text = doc['text']
            if len(text) < 200:
                continue
            
            # Sample from middle portions (often contain main content)
            quarter_len = len(text) // 4
            middle_start = quarter_len
            middle_end = 3 * quarter_len
            
            middle_text = text[middle_start:middle_end]
            
            # Extract random sentences from middle
            sentences = re.split(r'[.!?]+', middle_text)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            
            if valid_sentences:
                sample_size = min(2, len(valid_sentences))
                selected = random.sample(valid_sentences, sample_size)
                
                for sentence in selected:
                    if current_length + len(sentence) <= budget:
                        samples.append(f"[Random - {doc['source']}] {sentence}")
                        current_length += len(sentence)
        
        return samples
    
    def _combine_and_optimize_sample(self, components: Dict[str, List[str]]) -> Dict[str, Any]:
        """Combine samples from different strategies and optimize."""
        
        all_samples = []
        for strategy, samples in components.items():
            all_samples.extend(samples)
        
        # Remove duplicates and overly similar content
        unique_samples = []
        seen_hashes = set()
        
        for sample in all_samples:
            # Create hash of normalized content
            normalized = re.sub(r'\s+', ' ', sample.lower().strip())
            sample_hash = hashlib.md5(normalized.encode()).hexdigest()[:8]
            
            if sample_hash not in seen_hashes:
                seen_hashes.add(sample_hash)
                unique_samples.append(sample)
        
        # Combine samples within budget
        final_text = ""
        used_samples = []
        
        for sample in unique_samples:
            if len(final_text) + len(sample) + 2 <= self.max_sample_size:  # +2 for separators
                final_text += sample + "\n\n"
                used_samples.append(sample)
        
        # Calculate diversity score (simple heuristic)
        diversity_score = len(set(sample.split()[0] for sample in used_samples if sample.split())) / len(used_samples) if used_samples else 0.0
        
        return {
            'text': final_text.strip(),
            'used_samples': used_samples,
            'diversity_score': diversity_score
        }
    
    def _calculate_sampling_metrics(self, 
                                  all_documents: List[Dict[str, Any]], 
                                  selected_documents: List[Dict[str, Any]], 
                                  final_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics about the sampling process."""
        
        total_chars = sum(len(doc['text']) for doc in all_documents)
        selected_chars = sum(len(doc['text']) for doc in selected_documents)
        sample_chars = len(final_sample['text'])
        
        return {
            'total_documents': len(all_documents),
            'selected_documents': len(selected_documents),
            'document_coverage_ratio': len(selected_documents) / len(all_documents),
            'total_corpus_chars': total_chars,
            'selected_corpus_chars': selected_chars,
            'final_sample_chars': sample_chars,
            'compression_ratio': sample_chars / total_chars if total_chars > 0 else 0,
            'diversity_score': final_sample['diversity_score'],
            'samples_used': len(final_sample['used_samples'])
        }
    
    # Original methods (backward compatibility)
    def _sample_corpus_text(self, pdf_files: List[Path]) -> str:
        """Original hybrid sampling method - kept for backward compatibility."""
        samples = []
        total_chars = 0
        max_chars = 350000 #8000  # Token budget - close to 500,000 tokens
        
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
                first_part = text[:15000].strip() #text[:500].strip()
                last_part = text[-15000:].strip() if len(text) > 1000 else "" #text[-500:].strip() if len(text) > 1000 else ""
                
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
        
        sample_corpus_text = "\n\n".join(samples)[:max_chars]
        logger.debug(f'Sample Corpus Text for entity label extraction is : {len(sample_corpus_text)}')
        return sample_corpus_text
    
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
        logger.debug("🔍 Sampling corpus text for entity discovery...")
        corpus_sample = self._sample_corpus_text(pdf_files)
        logger.debug(f"Corpus Sample length: {len(corpus_sample)}")

        if not corpus_sample.strip():
            print("⚠️ No text could be sampled from corpus")
            return []
        
        # Discover labels
        print("🧠 Analyzing corpus with LLM...")
        proposed_labels = self.discover_labels_for_text(corpus_sample, 30) ##added 30
        
        # CLI approval
        approved_labels = self._approve_labels_cli(proposed_labels)
        
        # Cache results
        if approved_labels:
            self._save_schema_cache(corpus_hash, approved_labels)
        
        return approved_labels
    
    def discover_labels_for_text(self, text: str, max_labels: int = 30) -> List[str]:
        """Discover entity labels from text using LLM."""
        # Truncate text to fit in prompt
        text_sample = text  #text[:12000]
        
        prompt = f"""
        Analyze the following text and propose up to {max_labels} entity types (labels) that would be most useful for knowledge graph construction.

        Focus on:
        - Domain-specific entities relevant to this content
        - Entities that appear frequently and have relationships
        - Entities that would be valuable for search and analysis
        - Balance between specificity and reusability

        Text sample:
        {text_sample}

        Return only a JSON list of entity type names (uppercase, noun-based):
        ["ENTITY_TYPE_1", "ENTITY_TYPE_2", ...]
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # Debug: Check response content
            if not response.content or response.content.strip() == "":
                print(f"Warning: LLM returned empty response")
                return ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "DOCUMENT"]
            
            # Enhanced JSON parsing for Ollama compatibility
            content = response.content.strip()
            
            # Try to extract JSON from response if it's wrapped in text
            # Look for JSON array in the response
            json_match = re.search(r'(\[.*?\])', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            # If still no valid JSON, try to extract from code blocks
            elif not content.startswith('['):
                json_blocks = re.findall(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL | re.IGNORECASE)
                if json_blocks:
                    content = json_blocks[0]
                else:
                    # Clean response: remove markdown code blocks if present
                    if content.startswith('```json'):
                        content = content[7:]  # Remove ```json
                    if content.startswith('```'):
                        content = content[3:]   # Remove ```
                    if content.endswith('```'):
                        content = content[:-3]  # Remove trailing ```
                    content = content.strip()
            
            labels = json.loads(content)
            
            # Validate and clean labels
            cleaned_labels = []
            for label in labels:
                if isinstance(label, str) and len(label) > 1 and label.isupper():
                    cleaned_labels.append(label)
            
            return cleaned_labels[:max_labels]
            
        except json.JSONDecodeError as e:
            print(f"Warning: LLM entity discovery JSON parsing failed: {e}")
            print(f"Raw response: '{response.content if 'response' in locals() else 'No response'}'")
            # Fallback to basic entity types
            return ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "DOCUMENT"]
        except Exception as e:
            print(f"Warning: LLM entity discovery failed: {e}")
            # Fallback to basic entity types
            return ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "DOCUMENT"]
    
    def _approve_labels_cli(self, proposed_labels: List[str]) -> List[str]:
        """Interactive CLI for approving discovered entity labels."""
        if not proposed_labels:
            return []
        
        print(f"\n🎯 Discovered Entity Labels ({len(proposed_labels)} proposed):")
        for i, label in enumerate(proposed_labels, 1):
            print(f"  {i:2d}. {label}")
        
        print(f"\nOptions:")
        print(f"  [a] Accept all")
        print(f"  [1,2,5] Select specific numbers (comma-separated)")
        print(f"  [n] None (use basic labels)")
        
        while True:
            choice = 'a' # input("Your choice: ").strip().lower()
            
            if choice == 'a':
                print(f"✅ Approved all {len(proposed_labels)} labels")
                return proposed_labels
            elif choice == 'n':
                basic_labels = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"]
                print(f"✅ Using basic labels: {basic_labels}")
                return basic_labels
            elif ',' in choice:
                try:
                    indices = [int(x.strip()) - 1 for x in choice.split(',')]
                    selected = [proposed_labels[i] for i in indices if 0 <= i < len(proposed_labels)]
                    if selected:
                        print(f"✅ Approved {len(selected)} labels: {selected}")
                        return selected
                    else:
                        print("❌ Invalid selection")
                except (ValueError, IndexError):
                    print("❌ Invalid format. Use comma-separated numbers (e.g., 1,3,5)")
            else:
                print("❌ Invalid choice. Use 'a', 'n', or comma-separated numbers")
    
    def extract_entities_dynamic(self, text: str, allowed_labels: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using dynamically discovered labels."""
        if not allowed_labels:
            allowed_labels = self.discovered_labels or ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"]
        
        # Truncate text if too long
        text_sample = text[:8000]
        
        prompt = f"""
        Extract entities from the following text. Only identify entities that match these allowed types:
        {', '.join(allowed_labels)}

        Text:
        {text_sample}

        Return a JSON object where keys are entity types and values are lists of entity objects:
        {{
            "ENTITY_TYPE": [
                {{
                    "text": "entity name",
                    "description": "brief description"
                }}
            ]
        }}

        Only include entity types that have at least one entity found in the text.
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # Enhanced JSON parsing for Ollama compatibility
            content = response.content.strip()
            
            # Try multiple approaches to extract valid JSON
            json_content = None
            
            # Approach 1: Look for complete JSON object
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
            
            # Approach 2: Extract from code blocks
            if not json_content:
                json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
                if json_blocks:
                    json_content = json_blocks[0]
            
            # Approach 3: Clean markdown and try direct parsing
            if not json_content:
                cleaned = content
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                elif cleaned.startswith('```'):
                    cleaned = cleaned[3:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                if cleaned.startswith('{') and cleaned.endswith('}'):
                    json_content = cleaned
            
            # Approach 4: Try to fix common JSON issues
            if not json_content and '{' in content and '}' in content:
                # Extract everything between first { and last }
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_content = content[start:end+1]
                    # Try to fix common issues like trailing commas
                    json_content = re.sub(r',\s*}', '}', json_content)
                    json_content = re.sub(r',\s*]', ']', json_content)
            
            if not json_content:
                raise json.JSONDecodeError("No valid JSON found in response", content, 0)

            logger.debug(f"JSON content: {json_content}")
            entities = json.loads(json_content)
            
            # Validate structure
            validated_entities = {}
            for entity_type, entity_list in entities.items():
                if entity_type in allowed_labels and isinstance(entity_list, list):
                    valid_entities = []
                    for entity in entity_list:
                        if isinstance(entity, dict) and 'text' in entity:
                            valid_entities.append({
                                'text': entity['text'],
                                'description': entity.get('description', ''),
                                'type': entity_type
                            })
                    if valid_entities:
                        validated_entities[entity_type] = valid_entities
            
            return validated_entities
            
        except json.JSONDecodeError as e:
            print(f"Warning: Entity extraction JSON parsing failed: {e}")
            print(f"Raw response content: '{content[:200]}...'")
            return {}
        except Exception as e:
            print(f"Warning: Entity extraction failed: {e}")
            return {}
    
    def discover_labels_for_text_enhanced(self, 
                                        documents: List[Dict[str, Any]], 
                                        domain_hint: Optional[str] = None,
                                        max_labels: int = 12) -> List[str]:
        """
        Enhanced entity discovery using improved sampling.
        
        Args:
            documents: List of {'text': str, 'source': str, 'metadata': dict}
            domain_hint: Optional domain hint (e.g., 'financial', 'medical')
            max_labels: Maximum number of entity types to discover
        
        Returns:
            List of discovered entity type names
        """
        print("🚀 Starting enhanced entity discovery...")
        logger.debug(f"discover_labels_for_text_enhanced: Starting enhanced entity discovery")
        
        # Enhanced sampling
        sample_result = self._sample_corpus_text_enhanced(documents)
        sample_text = sample_result['sample_text']
        
        if not sample_text.strip():
            print("⚠️ No text could be sampled from corpus")
            return []
        
        print(f"📊 Sampling metrics:")
        stats = sample_result['sampling_stats']
        print(f"   Documents: {stats.get('selected_documents', 0)}/{stats.get('total_documents', 0)}")
        print(f"   Sample size: {stats.get('final_sample_chars', 0):,} chars")
        print(f"   Diversity score: {sample_result.get('diversity_score', 0):.2f}")
        
        # Domain analysis
        domain_context = ""
        if domain_hint and domain_hint.lower() in self.domain_patterns:
            domain_entities = self.domain_patterns[domain_hint.lower()]
            domain_context = f"\nDomain: {domain_hint}\nDomain-specific entities to consider: {', '.join(domain_entities)}"
        
        # Enhanced LLM prompt
        prompt = f"""
        You are an expert in knowledge graph construction and ontology design.
        
        {domain_context}
        
        Analyze the following diverse text sample and identify the most important entity types 
        that would be valuable for building a comprehensive knowledge graph.
        
        TEXT SAMPLE:
        {sample_text}
        
        Instructions:
        1. Focus on entity types that appear multiple times or are central to the domain
        2. Consider both explicit entities (named things) and implicit concepts
        3. Think about entities that would have meaningful relationships
        4. Balance specificity with reusability - avoid overly narrow types
        5. Consider hierarchical relationships between entity types
        
        Return only a JSON list of entity type names (uppercase, noun-based):
        ["ENTITY_TYPE_1", "ENTITY_TYPE_2", ...]
        
        Limit to {max_labels} most important entity types.
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # Enhanced JSON parsing for Ollama compatibility
            content = response.content.strip()
            
            # Try to extract JSON from response if it's wrapped in text
            # Look for JSON array in the response
            json_match = re.search(r'(\[.*?\])', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            # If still no valid JSON, try to extract from code blocks
            elif not content.startswith('['):
                json_blocks = re.findall(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL | re.IGNORECASE)
                if json_blocks:
                    content = json_blocks[0]
            
            labels = json.loads(content)
            
            # Validate and clean labels
            cleaned_labels = []
            for label in labels:
                if isinstance(label, str) and len(label) > 1 and label.isupper():
                    cleaned_labels.append(label)
            
            discovered_labels = cleaned_labels[:max_labels]
            print(f"🎯 Discovered {len(discovered_labels)} entity types: {', '.join(discovered_labels)}")
            
            return discovered_labels
            
        except Exception as e:
            print(f"Warning: Enhanced entity discovery failed: {e}")
            # Fallback to original method
            return self.discover_labels_for_text(sample_text, max_labels)
