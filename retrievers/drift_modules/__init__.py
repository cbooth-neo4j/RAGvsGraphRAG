"""
DRIFT GraphRAG Modular Implementation

This module provides a modular implementation of Microsoft's DRIFT (Dynamic Iterative Refinement) 
algorithm, integrated with the advanced GraphRAG system.

Key Features:
- Intelligent routing between local and global search using existing QueryClassifier
- Neo4j integration maintained through existing GraphRAGLocalRetriever/GraphRAGGlobalRetriever
- Benchmark compatibility preserved
- Modular, production-ready design inspired by Microsoft's implementation
- Builds on top of existing retrievers rather than replacing them

Components:
- DRIFTContextBuilder: Context building and management using existing community detection
- DRIFTPrimer: Query decomposition using community reports
- DRIFTQueryState: NetworkX-based action graph management
- DRIFTAction: Individual search actions with adaptive routing
- DRIFTSearch: Main search orchestration

Usage:
    from retrievers.drift_modules import DRIFTSearch, DRIFTConfig
    from data_processors import AdvancedGraphProcessor
    
    # Initialize with existing graph processor
    processor = AdvancedGraphProcessor()
    drift_search = DRIFTSearch(processor)
    
    # Perform DRIFT search
    result = await drift_search.search("What are the main requirements?")
"""

from .drift_context import DRIFTContextBuilder
from .drift_primer import DRIFTPrimer  
from .drift_state import DRIFTQueryState
from .drift_action import DRIFTAction
from .drift_search import DRIFTSearch, DRIFTConfig, query_drift_search, create_drift_search
from .drift_query_classifier import DRIFTQueryClassifier, create_drift_query_classifier

__all__ = [
    "DRIFTContextBuilder",
    "DRIFTPrimer",
    "DRIFTQueryState", 
    "DRIFTAction",
    "DRIFTSearch",
    "DRIFTConfig",
    "query_drift_search",
    "create_drift_search",
    "DRIFTQueryClassifier",
    "create_drift_query_classifier"
]

# Version info
__version__ = "1.0.0"
__author__ = "Advanced GraphRAG Team"
__description__ = "Modular DRIFT implementation for GraphRAG" 