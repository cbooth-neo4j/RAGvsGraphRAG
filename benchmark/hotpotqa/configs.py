"""
HotpotQA Benchmark Configuration Presets

Provides configurable presets for different benchmark scenarios,
from quick smoke tests to full evaluation runs.
"""

from typing import Dict, List, Optional, Any

BENCHMARK_PRESETS: Dict[str, Dict[str, Any]] = {
    "micro": {
        "description": "Minimal ETL test (1 question, ~10 articles)",
        "question_limit": 1,
        "retrievers": ["chroma"],
        "estimated_articles": 10,  # 2 gold + ~8 distractors per question
        "estimated_cost": "<$0.10",
        "estimated_time_minutes": 1
    },
    "mini_smoke": {
        "description": "Ultra-quick validation (25 questions, ~50 articles)",
        "question_limit": 25,
        "retrievers": ["chroma", "graphrag"],
        "estimated_articles": 50,
        "estimated_cost": "$2",
        "estimated_time_minutes": 8
    },
    "smoke": {
        "description": "Quick validation run (50 questions, ~500 articles)",
        "question_limit": 50,
        "retrievers": ["chroma", "graphrag"],
        "estimated_articles": 500,  # ~10 articles per question with some deduplication
        "estimated_cost": "$5",
        "estimated_time_minutes": 15
    },
    "dev": {
        "description": "Development benchmark (500 questions, ~4000 articles)",
        "question_limit": 500,
        "retrievers": ["chroma", "graphrag", "hybrid-cypher", "advanced-graphrag"],
        "estimated_articles": 4000,  # Higher deduplication at scale
        "estimated_cost": "$20",
        "estimated_time_minutes": 60
    },
    "full": {
        "description": "Full dev set evaluation",
        "question_limit": None,  # All ~7400 questions
        "retrievers": "all",
        "estimated_articles": 10000,
        "estimated_cost": "$100+",
        "estimated_time_minutes": 300
    },
    "mini": {
        "description": "Minimal test (10 questions, ~109 articles)",
        "question_limit": 10,
        "retrievers": ["chroma"],
        "estimated_articles": 109,  # Actual count from prepared corpus
        "estimated_cost": "$1",
        "estimated_time_minutes": 5
    }
}

# All available retrievers in the system (use hyphens for CLI consistency)
ALL_RETRIEVERS = [
    "chroma",
    "graphrag",
    "advanced-graphrag",
    "drift-graphrag",
    "hybrid-cypher",
    "neo4j-vector",
    "text2cypher",
    "agentic-text2cypher"
]

# Default cache directory for downloaded data
DEFAULT_CACHE_DIR = "data/hotpotqa"

# Wikipedia API settings
WIKIPEDIA_API_SETTINGS = {
    "user_agent": "RAGvsGraphRAG-Benchmark/1.0 (research project)",
    "rate_limit_delay": 0.1,  # seconds between requests
    "max_retries": 3,
    "timeout": 30
}

# HotpotQA dataset URLs
HOTPOTQA_URLS = {
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "dev_distractor": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "dev_fullwiki": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
    "test_fullwiki": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json"
}


def get_preset_config(preset: str) -> Dict[str, Any]:
    """
    Get configuration for a specific preset.
    
    Args:
        preset: Name of the preset (smoke, dev, full, mini)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If preset is not found
    """
    if preset not in BENCHMARK_PRESETS:
        available = ", ".join(BENCHMARK_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    config = BENCHMARK_PRESETS[preset].copy()
    
    # Expand "all" retrievers
    if config["retrievers"] == "all":
        config["retrievers"] = ALL_RETRIEVERS.copy()
    
    return config


def get_retrievers_for_preset(preset: str) -> List[str]:
    """Get the list of retrievers for a preset."""
    config = get_preset_config(preset)
    return config["retrievers"]


def estimate_cost(preset: str) -> str:
    """Get estimated cost for a preset."""
    config = get_preset_config(preset)
    return config["estimated_cost"]


def print_preset_info(preset: str) -> None:
    """Print detailed information about a preset."""
    config = get_preset_config(preset)
    
    print(f"\n{'='*60}")
    print(f"Preset: {preset.upper()}")
    print(f"{'='*60}")
    print(f"Description: {config['description']}")
    print(f"Questions: {config['question_limit'] or 'All (~7400)'}")
    print(f"Estimated articles: {config['estimated_articles']}")
    print(f"Retrievers: {', '.join(config['retrievers'])}")
    print(f"Estimated cost: {config['estimated_cost']}")
    print(f"Estimated time: {config['estimated_time_minutes']} minutes")
    print(f"{'='*60}\n")


def list_presets() -> None:
    """List all available presets with brief descriptions."""
    print("\n Available Benchmark Presets:")
    print("-" * 50)
    for name, config in BENCHMARK_PRESETS.items():
        print(f"  {name:10} - {config['description']}")
        print(f"             Questions: {config['question_limit'] or 'All'}, "
              f"Cost: {config['estimated_cost']}")
    print("-" * 50)

