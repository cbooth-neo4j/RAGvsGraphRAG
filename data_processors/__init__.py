"""
Data Processors Module - Centralized Interface for Document and Graph Processing

This module provides easy access to all data processing implementations:
- PDF Processor: Extract and chunk text from PDF documents
- Graph Processor: Basic Neo4j graph construction from processed documents
- Advanced Graph Processor: Enhanced graph processing with entity resolution, element summarization, and community detection

Usage:
    from data_processors import AdvancedGraphProcessor, process_pdfs, create_basic_graph
"""

# Import main classes and functions from each processor
try:
    from .chroma_processor import process_pdfs
    CHROMA_PROCESSOR_AVAILABLE = True
    
    # Create a wrapper class for compatibility
    class PDFProcessor:
        def __init__(self):
            pass
        
        def process_directory(self, pdf_directory: str, collection_name: str = "rfp_docs", maintain_sections: bool = True):
            return process_pdfs(pdf_directory, collection_name, maintain_sections)
    
    # Create wrapper function for compatibility
    def process_pdf_directory(pdf_directory: str, output_directory: str = "processed_docs"):
        return process_pdfs(pdf_directory, "rfp_docs")
    
    print("PDF processor imported successfully")
except ImportError as e:
    print(f"WARNING: PDF processor not available: {e}")
    CHROMA_PROCESSOR_AVAILABLE = False
    PDFProcessor = None
    process_pdf_directory = None
    process_pdfs = None

try:
    # Import the new refactored processor with advanced features
    from .build_graph import CustomGraphProcessor
    # Keep the old GraphProcessor for backwards compatibility
    try:
        from .graph_processor import GraphProcessor as LegacyGraphProcessor
        GraphProcessor = LegacyGraphProcessor  # Alias for compatibility
    except ImportError:
        GraphProcessor = CustomGraphProcessor  # Use new one if old doesn't exist
    
    BASIC_GRAPH_PROCESSOR_AVAILABLE = True
    
    # Create wrapper function for compatibility
    def create_graph_from_chunks(chunks_directory: str = "processed_docs"):
        processor = CustomGraphProcessor()
        try:
            return processor.process_directory("PDFs")  # Assumes PDFs directory
        finally:
            processor.close()
    
    print("Enhanced graph processor imported successfully (build_graph with advanced features)")
except ImportError as e:
    print(f"WARNING: Enhanced graph processor not available: {e}")
    try:
        # Fallback to old processor
        from .graph_processor import CustomGraphProcessor
        from .graph_processor import GraphProcessor
        BASIC_GRAPH_PROCESSOR_AVAILABLE = True
        print("Fallback: Legacy graph processor imported successfully")
    except ImportError as e2:
        print(f"WARNING: No graph processors available: {e2}")
        BASIC_GRAPH_PROCESSOR_AVAILABLE = False
        GraphProcessor = None
        CustomGraphProcessor = None
        create_graph_from_chunks = None

# Advanced processing is now integrated into build_graph/CustomGraphProcessor
# AdvancedGraphProcessor is an alias for CustomGraphProcessor for backward compatibility
ADVANCED_GRAPH_PROCESSOR_AVAILABLE = BASIC_GRAPH_PROCESSOR_AVAILABLE
AdvancedGraphProcessor = CustomGraphProcessor  # Alias for backward compatibility with DRIFT, etc.

# Central registry of all available processors
AVAILABLE_PROCESSORS = {
    'chroma_processor': {
        'class': PDFProcessor if CHROMA_PROCESSOR_AVAILABLE else None,
        'function': process_pdf_directory if CHROMA_PROCESSOR_AVAILABLE else None,
        'name': 'Chroma Processor',
        'description': 'Extract and chunk text from PDF documents for ChromaDB',
        'available': CHROMA_PROCESSOR_AVAILABLE
    },
    'graph_processor': {
        'class': CustomGraphProcessor if BASIC_GRAPH_PROCESSOR_AVAILABLE else None,
        'function': create_graph_from_chunks if BASIC_GRAPH_PROCESSOR_AVAILABLE else None,
        'name': 'Graph Processor',
        'description': 'Complete Neo4j graph processing with entity discovery, summarization, and community detection (always includes advanced features)',
        'available': BASIC_GRAPH_PROCESSOR_AVAILABLE
    }
}


def get_available_processors():
    """Get list of available data processors"""
    return {k: v for k, v in AVAILABLE_PROCESSORS.items() if v['available']}


def create_chroma_processor():
    """Create a Chroma processor instance"""
    if not CHROMA_PROCESSOR_AVAILABLE:
        raise ImportError("Chroma processor not available. Check dependencies.")
    return PDFProcessor()


def create_basic_graph_processor():
    """Create a basic graph processor instance"""
    if not BASIC_GRAPH_PROCESSOR_AVAILABLE:
        raise ImportError("Basic graph processor not available. Check dependencies.")
    return GraphProcessor()


# create_advanced_graph_processor is deprecated - use create_basic_graph_processor (includes advanced features)


def process_pdfs(pdf_directory: str, output_directory: str = "processed_docs"):
    """
    Process all PDFs in a directory using the Chroma processor
    
    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Directory to save processed chunks
    
    Returns:
        Results from PDF processing
    """
    if not CHROMA_PROCESSOR_AVAILABLE:
        raise ImportError("Chroma processor not available. Check dependencies.")
    
    # Import the function directly to avoid circular imports
    from .chroma_processor import process_pdfs as _process_pdfs
    return _process_pdfs(pdf_directory, "rfp_docs")


def create_basic_graph(chunks_directory: str = "processed_docs"):
    """
    Create a basic graph from processed document chunks
    
    Args:
        chunks_directory: Directory containing processed document chunks
    
    Returns:
        Results from basic graph creation
    """
    if not BASIC_GRAPH_PROCESSOR_AVAILABLE:
        raise ImportError("Basic graph processor not available. Check dependencies.")
    
    return create_graph_from_chunks(chunks_directory)


def create_advanced_graph(
    pdf_directory: str = "PDFs", 
    perform_resolution: bool = True,
    perform_element_summarization: bool = False,
    perform_community_detection: bool = False
):
    """
    Create an advanced graph with entity resolution and optional advanced features
    
    Args:
        pdf_directory: Directory containing PDF files
        perform_resolution: Whether to perform entity resolution
        perform_element_summarization: Whether to perform element summarization
        perform_community_detection: Whether to perform community detection
    
    Returns:
        Results from advanced graph processing
    """
    if not ADVANCED_GRAPH_PROCESSOR_AVAILABLE:
        raise ImportError("Advanced graph processor not available. Check dependencies.")
    
    processor = AdvancedGraphProcessor()
    
    try:
        # Enable optional features if requested
        if perform_element_summarization:
            processor.enable_element_summarization()
        if perform_community_detection:
            processor.enable_community_summarization()
        
        # Process the directory
        results = processor.process_directory(
            pdf_directory,
            perform_resolution=perform_resolution,
            perform_element_summarization=perform_element_summarization,
            perform_community_detection=perform_community_detection
        )
        
        return results
        
    finally:
        processor.close()


def get_processor_class(processor_type: str):
    """Get processor class by type name"""
    if processor_type in AVAILABLE_PROCESSORS:
        processor_info = AVAILABLE_PROCESSORS[processor_type]
        if processor_info['available']:
            return processor_info['class']
        else:
            raise ImportError(f"Processor '{processor_type}' is not available. Check dependencies.")
    else:
        raise ValueError(f"Unknown processor type: {processor_type}. Available: {list(AVAILABLE_PROCESSORS.keys())}")


def test_all_processors():
    """
    Test all available data processors
    
    Returns:
        Dictionary of test results
    """
    results = {}
    available = get_available_processors()
    
    print(f"Testing {len(available)} available data processors")
    print("=" * 60)
    
    for processor_type, info in available.items():
        print(f"\nTesting {info['name']}...")
        try:
            if processor_type == 'chroma_processor':
                processor = create_chroma_processor()
                results[processor_type] = {'status': 'available', 'class': type(processor).__name__}
                print(f"OK {info['name']}: Available")
            elif processor_type == 'basic_graph_processor':
                processor = create_basic_graph_processor()
                results[processor_type] = {'status': 'available', 'class': type(processor).__name__}
                print(f"OK {info['name']}: Available")
            # advanced_graph_processor is deprecated - integrated into basic_graph_processor
        except Exception as e:
            print(f"FAILED {info['name']}: Failed - {e}")
            results[processor_type] = {'status': 'error', 'error': str(e)}
    
    print("\n" + "=" * 60)
    successful = len([r for r in results.values() if r.get('status') == 'available'])
    total = len(results)
    print(f"Testing completed. {successful}/{total} processors available.")
    
    return results


# Export all main functions and classes
__all__ = [
    # Main processor classes
    'PDFProcessor',
    'GraphProcessor', 
    'AdvancedGraphProcessor',
    
    # Factory functions
    'create_chroma_processor',
    'create_basic_graph_processor',
    # 'create_advanced_graph_processor',  # Deprecated - integrated into basic
    
    # High-level processing functions
    'process_pdfs',
    'create_basic_graph',
    'create_advanced_graph',
    
    # Utility functions
    'get_available_processors',
    'get_processor_class',
    'test_all_processors',
    
    # Registry
    'AVAILABLE_PROCESSORS'
]

# Conditional exports based on availability
if CHROMA_PROCESSOR_AVAILABLE:
    __all__.extend(['process_pdf_directory', 'process_pdfs'])

if BASIC_GRAPH_PROCESSOR_AVAILABLE:
    __all__.extend(['create_graph_from_chunks', 'CustomGraphProcessor'])

print(f"Data processors module loaded. Available processors: {list(get_available_processors().keys())}") 