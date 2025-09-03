"""
Configuration presets for RAGBench ingestion
"""

# Dataset size information from HuggingFace
DATASET_SIZES = {
    "covidqa": {"total": 1770, "test": 246},
    "cuad": {"total": 2550, "test": 354}, 
    "delucionqa": {"total": 1830, "test": 254},
    "emanual": {"total": 1320, "test": 183},
    "expertqa": {"total": 2030, "test": 282},
    "finqa": {"total": 16600, "test": 2306},
    "hagrid": {"total": 4530, "test": 629},
    "hotpotqa": {"total": 2700, "test": 375},
    "msmarco": {"total": 2690, "test": 374},
    "pubmedqa": {"total": 24500, "test": 3402},
    "tatqa": {"total": 33100, "test": 4597},
    "techqa": {"total": 1810, "test": 251}
}

# Ingestion presets for different scales
INGESTION_PRESETS = {
    "nano": {
        "description": "Ultra-tiny test with 10 records from finqa for initial validation",
        "datasets": ["finqa"],
        "split": "test",
        "max_records": 10,
        "sampling": "first",
        "processor_type": "basic",
        "estimated_docs": 40,   # 10 records × 4 docs
        "estimated_storage_gb": 0.1,
        "estimated_ram_gb": 2,
        "estimated_cost_usd": 2
    },
    
    "micro": {
        "description": "Tiny test with 50 records from covidqa",
        "datasets": ["covidqa"],
        "split": "test",
        "max_records": 50,
        "sampling": "first",
        "processor_type": "basic",
        "estimated_docs": 200,  # 50 records × 4 docs
        "estimated_storage_gb": 1,
        "estimated_ram_gb": 4,
        "estimated_cost_usd": 10
    },
    
    "small": {
        "description": "Small test with 500 records from medical domains",
        "datasets": ["covidqa", "expertqa"],
        "split": "test", 
        "max_records": 500,
        "sampling": "stratified",  # Balanced across datasets
        "processor_type": "basic",
        "estimated_docs": 2000,
        "estimated_storage_gb": 8,
        "estimated_ram_gb": 16,
        "estimated_cost_usd": 100
    },
    
    "medium": {
        "description": "Medium test with 2K records from diverse domains",
        "datasets": ["finqa", "hotpotqa", "msmarco"],
        "split": "test",
        "max_records": 2000,
        "sampling": "weighted",  # Weight by dataset size
        "processor_type": "basic",
        "estimated_docs": 8000,
        "estimated_storage_gb": 32,
        "estimated_ram_gb": 64,
        "estimated_cost_usd": 400
    },
    
    "large": {
        "description": "Large test with 10K records, enhanced processing",
        "datasets": ["finqa", "tatqa", "pubmedqa", "hotpotqa"],
        "split": "test",
        "max_records": 10000,
        "sampling": "weighted",
        "processor_type": "enhanced",  # Element summarization
        "estimated_docs": 40000,
        "estimated_storage_gb": 160,
        "estimated_ram_gb": 256,
        "estimated_cost_usd": 2000
    },
    
    "full_test": {
        "description": "Full test split from all datasets",
        "datasets": "all",  # All 12 datasets
        "split": "test", 
        "max_records": None,  # No limit
        "sampling": "all",
        "processor_type": "enhanced",
        "estimated_docs": 60000,  # ~15K test records × 4 docs
        "estimated_storage_gb": 240,
        "estimated_ram_gb": 512,
        "estimated_cost_usd": 3000
    }
}

# Domain categorization for analysis
DOMAIN_CATEGORIES = {
    "medical": ["covidqa", "pubmedqa"],
    "financial": ["finqa", "tatqa"],
    "legal": ["cuad"],
    "technical": ["techqa", "emanual"],
    "qa": ["expertqa", "hotpotqa", "msmarco"],
    "multimodal": ["hagrid"],
    "other": ["delucionqa"]
}

# Sampling strategies
SAMPLING_STRATEGIES = {
    "first": "Take first N records (fastest, but potentially biased)",
    "random": "Random sampling across the dataset",
    "stratified": "Balanced sampling across multiple datasets", 
    "weighted": "Sample proportionally to dataset sizes",
    "all": "Use all available records"
}
