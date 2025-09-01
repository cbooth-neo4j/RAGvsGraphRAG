"""
Constrained LLM Graph Builder (Neo4j GraphRAG SimpleKGPipeline)

This module defines a constrained Knowledge Graph schema mirroring the entity
types used in `graph_processor.py` and provides helpers to construct a
`SimpleKGPipeline` from the Neo4j GraphRAG Python library.

Entity types mirrored from `graph_processor.py`:
- Organization
- Location
- Date
- Person
- Financial
- Requirement

Notes:
- This file only sets up the constrained schema and pipeline builder. In a
  follow-up step, the LLM will analyze documents and propose entities to feed
  into the graph builder.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables for potential convenience in CLI usage
load_dotenv()


def get_node_types() -> List[Dict[str, Any]]:
    """Return constrained node type definitions with basic properties.

    Mirrors entities used in `graph_processor.py` and requires a `name` property
    for each node type, with an optional `description`.
    """
    base_properties = [
        {"name": "name", "type": "STRING", "required": True},
        {"name": "description", "type": "STRING"},
    ]

    return [
        {"label": "Organization", "properties": base_properties},
        {"label": "Location", "properties": base_properties},
        {"label": "Date", "properties": base_properties},
        {"label": "Person", "properties": base_properties},
        {"label": "Financial", "properties": base_properties},
        {"label": "Requirement", "properties": base_properties},
    ]


def get_relationship_types() -> List[Dict[str, Any]]:
    """Return constrained relationship type definitions.

    We keep a single generic relationship label to mirror `RELATES_TO` used in
    `graph_processor.py`. Properties are intentionally unspecified for now.
    """
    return [
        {"label": "RELATES_TO"},
    ]


def get_meaningful_entity_pairs() -> List[Tuple[str, str]]:
    """Pairs of entity labels considered meaningful co-occurrences.

    Derived from `create_entity_relationships` in `graph_processor.py`.
    """
    return [
        ("Organization", "Requirement"),
        ("Organization", "Location"),
        ("Organization", "Person"),
        ("Organization", "Financial"),
        ("Requirement", "Date"),
        ("Requirement", "Financial"),
        ("Organization", "Date"),
        ("Location", "Date"),
        ("Person", "Location"),
        ("Person", "Date"),
        ("Financial", "Date"),
    ]


def get_patterns(bidirectional: bool = True) -> List[Tuple[str, str, str]]:
    """Generate schema patterns based on meaningful entity pairs.

    If `bidirectional` is True, include both directions for each pair.
    """
    pairs = get_meaningful_entity_pairs()
    patterns: List[Tuple[str, str, str]] = []

    for a, b in pairs:
        patterns.append((a, "RELATES_TO", b))
        if bidirectional and a != b:
            patterns.append((b, "RELATES_TO", a))

    return patterns


def get_constrained_schema() -> Dict[str, Any]:
    """Return a constrained schema dict for `SimpleKGPipeline`.

    - Disallows additional node and relationship types
    - Disallows additional patterns by default
    - Requires `name` property (defined at node-type level)
    """
    return {
        "node_types": get_node_types(),
        "relationship_types": get_relationship_types(),
        "patterns": get_patterns(bidirectional=True),
        "additional_node_types": False,
        "additional_relationship_types": False,
        "additional_patterns": False,
    }


def build_pipeline(
    llm: Any,
    driver: Any,
    embedder: Optional[Any] = None,
    neo4j_database: Optional[str] = None,
    perform_entity_resolution: bool = True,
    lexical_graph_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Construct a `SimpleKGPipeline` with the constrained schema.

    Parameters:
        llm: Instance implementing the library's LLMInterface used for extraction
        driver: Neo4j driver instance
        embedder: Optional embedder instance for chunk embeddings
        neo4j_database: Target database name (if not default)
        perform_entity_resolution: Whether to run entity resolution step after run
        lexical_graph_config: Optional lexical graph builder configuration

    Returns:
        An instance of `SimpleKGPipeline`
    """
    # Import locally to avoid import errors at module import time if the package
    # isn't installed in certain environments.
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

    schema = get_constrained_schema()

    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=schema,
        neo4j_database=neo4j_database,
        perform_entity_resolution=perform_entity_resolution,
        lexical_graph_config=lexical_graph_config,
    )


def _build_driver_from_env() -> Optional[Any]:
    """Best-effort helper to create a Neo4j driver from environment variables.

    Env vars: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    Returns None if variables are missing.
    """
    try:
        import neo4j  # type: ignore
    except Exception:
        return None

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and password):
        return None
    try:
        return neo4j.GraphDatabase.driver(uri, auth=(user, password))
    except Exception:
        return None


if __name__ == "__main__":
    # Minimal CLI helper to verify schema setup; does not run the pipeline yet.
    print("Constrained LLM Graph Builder schema (for SimpleKGPipeline):")
    schema = get_constrained_schema()
    print({
        "node_types": [nt["label"] if isinstance(nt, dict) else nt for nt in schema["node_types"]],
        "relationship_types": [rt["label"] if isinstance(rt, dict) else rt for rt in schema["relationship_types"]],
        "patterns": schema["patterns"][:5],  # preview
        "additional_node_types": schema["additional_node_types"],
        "additional_relationship_types": schema["additional_relationship_types"],
        "additional_patterns": schema["additional_patterns"],
    })

    driver = _build_driver_from_env()
    if driver is None:
        print("\nTip: Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD to initialize a driver.")
    else:
        print("\nNeo4j driver created from environment (not running pipeline in this script).")


