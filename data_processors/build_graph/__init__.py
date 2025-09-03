"""
Graph building components - refactored from monolithic graph_processor.py

This module breaks down graph processing into logical components:
- entity_discovery: Enhanced entity discovery and schema management
- text_processing: Text extraction, chunking, and embedding
- graph_operations: Neo4j operations and relationship creation
- main_processor: Main orchestrator class
"""

from .main_processor import CustomGraphProcessor
from .entity_discovery import EntityDiscoveryMixin
from .text_processing import TextProcessingMixin
from .graph_operations import GraphOperationsMixin

__all__ = [
    'CustomGraphProcessor',
    'EntityDiscoveryMixin', 
    'TextProcessingMixin',
    'GraphOperationsMixin'
]
