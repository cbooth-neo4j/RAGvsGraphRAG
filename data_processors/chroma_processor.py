import os
from typing import List, Dict
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import re
import chromadb
from chromadb.config import Settings
import shutil

# Disable Chroma telemetry
chromadb.Client(Settings(anonymized_telemetry=False))

load_dotenv(override=True)

# Configure Chroma
PERSIST_DIRECTORY = "chroma_db"

def get_pdf_files(directory: str) -> List[str]:
    """Get all PDF files from a directory"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    pdf_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {directory}")
        
    return pdf_files

def reset_chroma():
    """Reset the Chroma database by removing the persistence directory"""
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

def read_pdf(file_path: str) -> str:
    """Read a PDF file and return its text content"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
        
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    if not text.strip():
        raise ValueError(f"No text content extracted from PDF: {file_path}")
        
    return text

def split_text(text: str, source_file: str, maintain_sections: bool = True) -> List[Dict[str, str]]:
    """Split text into chunks while maintaining section structure if possible"""
    if not text.strip():
        raise ValueError("Cannot split empty text")
        
    # First try to identify sections
    section_pattern = r'(?i)^(?:Section|Chapter|Part)\s+\d+[.:]\s*(.*?)(?=(?:Section|Chapter|Part)\s+\d+[.:]|\Z)'
    sections = list(re.finditer(section_pattern, text, re.MULTILINE | re.DOTALL))
    
    chunks = []
    if maintain_sections and sections:
        for section in sections:
            section_text = section.group(0)
            section_title = section.group(1).strip()
            
            # Split section content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            section_chunks = text_splitter.split_text(section_text)
            
            # Add section metadata to chunks
            for chunk in section_chunks:
                chunks.append({
                    'content': chunk,
                    'section': section_title,
                    'type': 'section',
                    'source': source_file
                })
    else:
        # Fall back to simple chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        simple_chunks = text_splitter.split_text(text)
        chunks = [{'content': chunk, 'type': 'content', 'source': source_file} for chunk in simple_chunks]
    
    if not chunks:
        raise ValueError("Text splitting produced no chunks")
        
    return chunks

def create_vector_store(all_chunks: List[Dict[str, str]], collection_name: str) -> Chroma:
    """Create a vector store from text chunks with metadata"""
    if not all_chunks:
        raise ValueError("Cannot create vector store from empty chunks")
        
    embeddings = OpenAIEmbeddings()
    
    # Extract texts and metadata
    texts = [chunk['content'] for chunk in all_chunks]
    metadatas = [{k: v for k, v in chunk.items() if k != 'content'} for chunk in all_chunks]
    
    # Print debug info
    print(f"Creating vector store with {len(texts)} chunks")
    print(f"First chunk preview: {texts[0][:100]}...")
    
    # Create a new Chroma instance with persistence
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=PERSIST_DIRECTORY,
        metadatas=metadatas
    )
    return vector_store

def process_pdfs(pdf_directory: str, collection_name: str, maintain_sections: bool = True) -> Chroma:
    """Process all PDFs in a directory and return a combined vector store"""
    # Reset Chroma before processing
    reset_chroma()
    
    # Get all PDF files
    pdf_files = get_pdf_files(pdf_directory)
    print(f"Found {len(pdf_files)} PDF files to process")
    
    all_chunks = []
    
    # Process each PDF
    for pdf_path in pdf_files:
        print(f"\nProcessing PDF: {pdf_path}")
        
        # Read PDF
        text = read_pdf(pdf_path)
        print(f"Extracted {len(text)} characters of text")
        
        # Split into chunks while maintaining structure
        file_chunks = split_text(text, os.path.basename(pdf_path), maintain_sections)
        print(f"Created {len(file_chunks)} text chunks")
        
        all_chunks.extend(file_chunks)
    
    print(f"\nTotal chunks across all PDFs: {len(all_chunks)}")
    
    # Create combined vector store
    vector_store = create_vector_store(all_chunks, collection_name)
    
    return vector_store

if __name__ == "__main__":
    # Example usage
    pdf_directory = "PDFs"
    collection_name = "rfp_docs"
    vector_store = process_pdfs(pdf_directory, collection_name) 