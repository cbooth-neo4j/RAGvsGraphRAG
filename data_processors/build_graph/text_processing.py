"""
Text processing utilities for graph building.
Handles PDF extraction, text chunking, and embeddings.
Supports configurable embedding models.
"""

import os
import sys
from typing import List, Dict, Any
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import centralized configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_embeddings

# Optional PDF processing dependency
try:
    from PyPDF2 import PdfReader
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

# Optional table extraction dependencies
try:
    import camelot
    _HAS_CAMELOT = True
except ImportError:
    _HAS_CAMELOT = False

try:
    import tabula
    _HAS_TABULA = True
except ImportError:
    _HAS_TABULA = False


class TextProcessingMixin:
    """
    Mixin for text processing capabilities.
    Handles PDF extraction, chunking, and embedding creation with configurable models.
    """
    
    def __init__(self):
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings with configurable model
        self.embeddings = get_embeddings()
        
        super().__init__()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        if not _HAS_PDF:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
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
                return chunks
            except Exception:
                pass
        
        # No table extraction available
        return chunks
    
    def chunk_text(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        chunks = self.text_splitter.split_text(text)
        
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_objects.append({
                'text': chunk,
                'index': i,
                'source': source_file,
                'type': 'text'
            })
        
        return chunk_objects
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI."""
        try:
            # Truncate text if too long (OpenAI has token limits)
            max_chars = 8000  # Conservative limit
            if len(text) > max_chars:
                text = text[:max_chars]
            
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"Warning: Failed to create embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536  # text-embedding-3-small dimension
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts efficiently."""
        try:
            # Truncate texts if too long
            max_chars = 8000
            truncated_texts = []
            for text in texts:
                if len(text) > max_chars:
                    truncated_texts.append(text[:max_chars])
                else:
                    truncated_texts.append(text)
            
            embeddings = self.embeddings.embed_documents(truncated_texts)
            return embeddings
        except Exception as e:
            print(f"Warning: Failed to create batch embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 1536 for _ in texts]
    
    def prepare_documents_for_sampling(self, pdf_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Convert PDF files to format expected by enhanced sampling.
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            List of document dictionaries for sampling
        """
        documents = []
        for pdf_path in pdf_files:
            try:
                text = self.extract_text_from_pdf(str(pdf_path))
                if text.strip():
                    documents.append({
                        'text': text,
                        'source': pdf_path.stem,
                        'metadata': {'file_path': str(pdf_path), 'type': 'pdf'}
                    })
            except Exception as e:
                print(f"Warning: Could not process {pdf_path}: {e}")
        return documents
    
    def prepare_ragbench_documents_for_sampling(self, 
                                              texts: List[str], 
                                              sources: List[str]) -> List[Dict[str, Any]]:
        """
        Convert RAGBench documents to format expected by enhanced sampling.
        
        Args:
            texts: List of document texts
            sources: List of source identifiers
            
        Returns:
            List of document dictionaries for sampling
        """
        return [
            {
                'text': text,
                'source': source,
                'metadata': {'dataset': 'ragbench', 'index': i}
            }
            for i, (text, source) in enumerate(zip(texts, sources))
        ]
