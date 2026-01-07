"""
Neo4j Agent Tools - Direct Neo4j driver wrappers for Deep Agents

This module provides Python function wrappers around Neo4j operations,
mirroring the functionality of the Neo4j MCP server tools:
- get_schema: Introspect labels, relationship types, property keys
- read_cypher: Execute read-only Cypher queries
- list_gds_procedures: List available Graph Data Science procedures

These tools are designed for use with Deep Agents (create_deep_agent).
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import neo4j

from utils.graph_rag_logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# Neo4j configuration from environment
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
NEO4J_DB = os.environ.get('CLIENT_NEO4J_DATABASE', 'neo4j')


class Neo4jAgentTools:
    """
    Neo4j tools for Deep Agent graph exploration.
    
    Provides the same capabilities as the Neo4j MCP server but as
    Python functions callable by an LLM agent.
    """
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = None
    ):
        """
        Initialize Neo4j tools with connection parameters.
        
        Args:
            uri: Neo4j connection URI (defaults to NEO4J_URI env var)
            username: Database username (defaults to NEO4J_USERNAME env var)
            password: Database password (defaults to NEO4J_PASSWORD env var)
            database: Database name (defaults to CLIENT_NEO4J_DATABASE env var)
        """
        self.uri = uri or NEO4J_URI
        self.username = username or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.database = database or NEO4J_DB
        
        if not all([self.uri, self.username, self.password]):
            raise ValueError("Neo4j connection parameters required: uri, username, password")
        
        logger.info(f"Neo4j Agent Tools initialized for database: {self.database}")
    
    def _get_driver(self) -> neo4j.Driver:
        """Create a new Neo4j driver connection."""
        return neo4j.GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )
    
    def get_schema(self) -> str:
        """
        Get the Neo4j database schema including node labels, relationship types,
        and property keys.
        
        This mirrors the Neo4j MCP 'get-schema' tool functionality.
        
        Returns:
            String representation of the database schema
        """
        logger.debug("Fetching Neo4j schema")
        
        try:
            with self._get_driver() as driver:
                with driver.session(database=self.database) as session:
                    # Get node labels and their properties
                    labels_result = session.run("""
                        CALL db.labels() YIELD label
                        RETURN collect(label) as labels
                    """)
                    labels = labels_result.single()["labels"]
                    
                    # Get relationship types
                    rel_result = session.run("""
                        CALL db.relationshipTypes() YIELD relationshipType
                        RETURN collect(relationshipType) as types
                    """)
                    rel_types = rel_result.single()["types"]
                    
                    # Get property keys
                    props_result = session.run("""
                        CALL db.propertyKeys() YIELD propertyKey
                        RETURN collect(propertyKey) as keys
                    """)
                    prop_keys = props_result.single()["keys"]
                    
                    # Get sample node counts per label
                    label_counts = {}
                    for label in labels:
                        count_result = session.run(f"MATCH (n:`{label}`) RETURN count(n) as count")
                        label_counts[label] = count_result.single()["count"]
                    
                    # Get sample properties per label (from first few nodes)
                    label_properties = {}
                    for label in labels:
                        try:
                            sample = session.run(f"""
                                MATCH (n:`{label}`) 
                                RETURN keys(n) as props 
                                LIMIT 1
                            """)
                            record = sample.single()
                            if record:
                                label_properties[label] = record["props"]
                        except:
                            label_properties[label] = []
                    
                    # Get relationship patterns
                    rel_patterns = []
                    try:
                        pattern_result = session.run("""
                            CALL db.schema.visualization() 
                            YIELD nodes, relationships
                            RETURN nodes, relationships
                        """)
                        for record in pattern_result:
                            rel_patterns.append(str(record))
                    except:
                        # Fallback if schema visualization not available
                        pass
                    
                    # Format schema output
                    schema_parts = []
                    schema_parts.append("=== NEO4J DATABASE SCHEMA ===\n")
                    
                    schema_parts.append("NODE LABELS (with counts):")
                    for label in sorted(labels):
                        props = label_properties.get(label, [])
                        props_str = ", ".join(props[:10])  # First 10 properties
                        if len(props) > 10:
                            props_str += f"... (+{len(props)-10} more)"
                        schema_parts.append(f"  - {label}: {label_counts.get(label, 0)} nodes")
                        if props_str:
                            schema_parts.append(f"    Properties: {props_str}")
                    
                    schema_parts.append("\nRELATIONSHIP TYPES:")
                    for rel_type in sorted(rel_types):
                        schema_parts.append(f"  - {rel_type}")
                    
                    schema_parts.append("\nKEY PROPERTY KEYS:")
                    # Show most relevant properties (excluding internal ones)
                    relevant_props = [p for p in prop_keys if not p.startswith('_')][:30]
                    schema_parts.append(f"  {', '.join(sorted(relevant_props))}")
                    
                    schema = "\n".join(schema_parts)
                    logger.debug(f"Schema extracted: {len(schema)} characters")
                    return schema
                    
        except Exception as e:
            error_msg = f"Error fetching schema: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def read_cypher(self, query: str, params: Dict[str, Any] = None) -> str:
        """
        Execute a read-only Cypher query against the Neo4j database.
        
        This mirrors the Neo4j MCP 'read-cypher' tool functionality.
        Write operations (CREATE, MERGE, DELETE, SET) will be rejected.
        
        Args:
            query: The Cypher query to execute
            params: Optional parameters for the query
            
        Returns:
            String representation of query results
        """
        logger.debug(f"Executing Cypher query: {query[:100]}...")
        
        # Check for write operations
        query_upper = query.upper().strip()
        write_keywords = ['CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE', 'DROP', 'DETACH']
        
        for keyword in write_keywords:
            # Check if keyword appears as a standalone word (not in comments)
            if f' {keyword} ' in f' {query_upper} ' or query_upper.startswith(keyword):
                error_msg = f"Write operations not allowed in read_cypher. Found '{keyword}' clause."
                logger.warning(error_msg)
                return f"ERROR: {error_msg}"
        
        try:
            with self._get_driver() as driver:
                with driver.session(database=self.database) as session:
                    result = session.run(query, params or {})
                    records = list(result)
                    
                    if not records:
                        return "Query returned no results."
                    
                    # Format results
                    output_parts = []
                    output_parts.append(f"Query returned {len(records)} result(s):\n")
                    
                    for i, record in enumerate(records[:50], 1):  # Limit to 50 results
                        record_dict = dict(record)
                        # Format each record nicely
                        formatted = {}
                        for key, value in record_dict.items():
                            if hasattr(value, '__dict__'):
                                # Neo4j Node or Relationship
                                if hasattr(value, 'labels'):
                                    # Node
                                    formatted[key] = {
                                        'labels': list(value.labels),
                                        'properties': dict(value)
                                    }
                                elif hasattr(value, 'type'):
                                    # Relationship
                                    formatted[key] = {
                                        'type': value.type,
                                        'properties': dict(value)
                                    }
                                else:
                                    formatted[key] = str(value)
                            else:
                                formatted[key] = value
                        output_parts.append(f"[{i}] {formatted}")
                    
                    if len(records) > 50:
                        output_parts.append(f"\n... and {len(records) - 50} more results (truncated)")
                    
                    output = "\n".join(output_parts)
                    logger.debug(f"Query returned {len(records)} results")
                    return output
                    
        except neo4j.exceptions.ClientError as e:
            error_msg = f"Cypher error: {e.message}"
            logger.warning(error_msg)
            return f"ERROR: {error_msg}"
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
    
    def list_gds_procedures(self) -> str:
        """
        List available Graph Data Science (GDS) procedures.
        
        This mirrors the Neo4j MCP 'list-gds-procedures' tool functionality.
        
        Returns:
            String listing available GDS procedures
        """
        logger.debug("Listing GDS procedures")
        
        try:
            with self._get_driver() as driver:
                with driver.session(database=self.database) as session:
                    # Check if GDS is installed
                    try:
                        result = session.run("""
                            CALL gds.list() YIELD name, description
                            RETURN name, description
                            ORDER BY name
                        """)
                        procedures = list(result)
                    except:
                        return "GDS (Graph Data Science) library is not installed on this database."
                    
                    if not procedures:
                        return "No GDS procedures found."
                    
                    # Format output
                    output_parts = []
                    output_parts.append(f"=== GDS PROCEDURES ({len(procedures)} available) ===\n")
                    
                    # Group by category
                    categories = {}
                    for proc in procedures:
                        name = proc["name"]
                        desc = proc["description"]
                        category = name.split('.')[1] if '.' in name and len(name.split('.')) > 1 else "other"
                        if category not in categories:
                            categories[category] = []
                        categories[category].append((name, desc))
                    
                    for category in sorted(categories.keys()):
                        output_parts.append(f"\n{category.upper()}:")
                        for name, desc in categories[category][:10]:  # Limit per category
                            output_parts.append(f"  - {name}")
                            if desc:
                                output_parts.append(f"    {desc[:100]}...")
                        if len(categories[category]) > 10:
                            output_parts.append(f"  ... and {len(categories[category]) - 10} more")
                    
                    return "\n".join(output_parts)
                    
        except Exception as e:
            error_msg = f"Error listing GDS procedures: {str(e)}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"


# Create standalone tool functions for Deep Agent
# These functions will be passed to create_deep_agent(tools=[...])

_tools_instance = None

def _get_tools() -> Neo4jAgentTools:
    """Get or create the singleton tools instance."""
    global _tools_instance
    if _tools_instance is None:
        _tools_instance = Neo4jAgentTools()
    return _tools_instance


def neo4j_get_schema() -> str:
    """
    Get the Neo4j database schema.
    
    Returns information about:
    - Node labels and their counts
    - Properties on each node type
    - Relationship types
    - Property keys
    
    Use this FIRST to understand what's in the database before writing queries.
    
    Returns:
        String describing the database schema
    """
    return _get_tools().get_schema()


def neo4j_read_cypher(query: str) -> str:
    """
    Execute a read-only Cypher query against the Neo4j graph database.
    
    Use this to:
    - Find specific entities by name or property
    - Traverse relationships between entities
    - Aggregate and analyze graph data
    - Search for patterns in the knowledge graph
    
    IMPORTANT TIPS:
    - Always check schema first with neo4j_get_schema()
    - Use CONTAINS for fuzzy text matching: WHERE n.name CONTAINS 'term'
    - For entities, check the 'ai_summary' property - it often has the answer
    - RELATED_TO relationships have 'evidence' property with relationship details
    - LIMIT your results (e.g., LIMIT 10) to avoid overwhelming output
    
    Args:
        query: A valid Cypher query (read-only operations only)
        
    Returns:
        Query results as formatted string, or error message
    """
    return _get_tools().read_cypher(query)


def neo4j_list_gds() -> str:
    """
    List available Graph Data Science (GDS) procedures.
    
    GDS provides graph algorithms for:
    - Centrality: PageRank, Betweenness, Degree
    - Community Detection: Louvain, Label Propagation
    - Similarity: Node Similarity, KNN
    - Path Finding: Shortest Path, All Shortest Paths
    
    Use this to discover what advanced analytics are available.
    
    Returns:
        List of available GDS procedures, or message if GDS not installed
    """
    return _get_tools().list_gds_procedures()


# Tool registry for Deep Agent
AGENT_TOOLS = [
    neo4j_get_schema,
    neo4j_read_cypher,
    neo4j_list_gds
]

# Minimal tools (without GDS) for simpler use cases
AGENT_TOOLS_MINIMAL = [
    neo4j_get_schema,
    neo4j_read_cypher
]

