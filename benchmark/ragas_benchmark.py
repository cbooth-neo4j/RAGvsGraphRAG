"""
RAGAS Benchmark: RAG vs GraphRAG Evaluation

This script evaluates both ChromaDB and GraphRAG approaches using RAGAS framework
following the exact pattern from the RAGAS documentation.
"""

import pandas as pd
import json
import time
import math
import sys
import os
import argparse
import asyncio
import threading
import queue
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# Custom progress bar to replace RAGAS tqdm
class CustomProgressBar:
    def __init__(self, total, desc="Evaluating", unit="it"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.unit = unit
        self.start_time = time.time()
        
    def update(self, n=1):
        self.current += n
        elapsed = time.time() - self.start_time
        if self.total > 0:
            progress = (self.current / self.total) * 100
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            # Format similar to tqdm but with actual progress
            bar_length = 50
            filled = int(bar_length * self.current / self.total)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r{self.desc}: {progress:3.0f}%|{bar}| {self.current}/{self.total} [{elapsed:02.0f}s<{eta:02.0f}s, {rate:.2f}{self.unit}/s]", end='', flush=True)
    
    def close(self):
        print()  # New line when done
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        self.close()

def patch_ragas_progress():
    """Monkey patch RAGAS to use our custom progress bar"""
    try:
        # Import tqdm and replace it
        import tqdm
        import ragas
        
        # Store original tqdm
        original_tqdm = tqdm.tqdm
        
        class TqdmWrapper:
            def __init__(self, *args, **kwargs):
                total = kwargs.get('total', 0)
                desc = kwargs.get('desc', 'Evaluating')
                self.pbar = CustomProgressBar(total, desc)
                
            def update(self, n=1):
                self.pbar.update(n)
                
            def close(self):
                self.pbar.close()
                
            def __enter__(self):
                return self
                
            def __exit__(self, *args):
                self.close()
        
        # Replace tqdm in various places RAGAS might use it
        tqdm.tqdm = TqdmWrapper
        
        # Also try to patch it in ragas modules
        try:
            import ragas.evaluation
            if hasattr(ragas.evaluation, 'tqdm'):
                ragas.evaluation.tqdm = TqdmWrapper
        except:
            pass
            
        try:
            import ragas.metrics
            if hasattr(ragas.metrics, 'tqdm'):
                ragas.metrics.tqdm = TqdmWrapper
        except:
            pass
            
        return original_tqdm
        
    except Exception as e:
        print(f"⚠️ Could not patch RAGAS progress bar: {e}")
        return None

# Import visualization module
try:
    from .visualizations import create_visualizations
except ImportError:
    # Fallback for when script is run directly
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from visualizations import create_visualizations

# Add parent directory to path so we can import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# RAGAS imports
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# Import our RAG systems from the new retrievers module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from retrievers import (
        query_chroma_rag,
        query_graphrag, 
        query_advanced_graphrag,
        query_drift_graphrag,
        query_text2cypher_rag,
        query_neo4j_vector_rag,
        query_hybrid_cypher_rag,
        get_available_retrievers,
        AVAILABLE_RETRIEVERS
    )
    
    # Check availability of each retriever
    available_retrievers = get_available_retrievers()
    
    CHROMA_AVAILABLE = 'chroma' in available_retrievers
    GRAPHRAG_AVAILABLE = 'graphrag' in available_retrievers
    ADVANCED_GRAPHRAG_AVAILABLE = 'advanced_graphrag' in available_retrievers
    DRIFT_GRAPHRAG_AVAILABLE = 'drift_graphrag' in available_retrievers
    TEXT2CYPHER_AVAILABLE = 'text2cypher' in available_retrievers
    NEO4J_VECTOR_AVAILABLE = 'neo4j_vector' in available_retrievers
    HYBRID_CYPHER_AVAILABLE = 'hybrid_cypher' in available_retrievers
    
    print("✅ Retrievers module imported successfully")
    print(f"📋 Available retrievers: {list(available_retrievers.keys())}")
    
except ImportError as e:
    print(f"❌ Error importing retrievers module: {e}")
    print("   Please ensure the retrievers module is properly set up.")
    
    # Set all retrievers as unavailable if import fails
    CHROMA_AVAILABLE = False
    GRAPHRAG_AVAILABLE = False
    ADVANCED_GRAPHRAG_AVAILABLE = False
    DRIFT_GRAPHRAG_AVAILABLE = False
    TEXT2CYPHER_AVAILABLE = False
    NEO4J_VECTOR_AVAILABLE = False

# Import centralized model configuration
try:
    from config.model_factory import get_llm, get_embeddings
    from config.model_config import get_model_config
    
    print("✅ Model configuration imported successfully")
    model_config = get_model_config()
    print(f"🔧 LLM Provider: {model_config.llm_provider.value}")
    print(f"🔧 LLM Model: {model_config.llm_model}")
    print(f"🔧 Embedding Provider: {model_config.embedding_provider.value}")
    print(f"🔧 Embedding Model: {model_config.embedding_model}")
    
    # Initialize LLM and embeddings for RAGAS using centralized configuration
    SEED = model_config.seed or 42
    llm = get_llm(
        seed=SEED,
        max_retries=0
    )
    embeddings = get_embeddings()
    
    print("✅ RAGAS models initialized with centralized configuration")
    
except ImportError as e:
    raise ImportError(f"Could not import centralized model configuration: {e}. Please ensure your .env file is properly configured.")

def load_benchmark_data(csv_path: str = "benchmark/benchmark.csv") -> List[Dict[str, str]]:
    """Load benchmark questions and ground truth answers from CSV"""
    df = pd.read_csv(csv_path, delimiter=';')
    
    benchmark_data = []
    for _, row in df.iterrows():
        benchmark_data.append({
            'question': row['question'],
            'ground_truth': row['ground_truth']
        })
    
    print(f"✅ Loaded {len(benchmark_data)} benchmark questions from CSV")
    return benchmark_data

def load_benchmark_data_jsonl(jsonl_path: str) -> List[Dict[str, str]]:
    """Load benchmark questions and ground truth answers from JSONL"""
    benchmark_data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                benchmark_data.append({
                    'question': record['question'],
                    'ground_truth': record['ground_truth'],
                    # Preserve additional metadata
                    'record_id': record.get('record_id'),
                    'source_dataset': record.get('source_dataset'),
                    'domain': record.get('domain')
                })
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping invalid JSON on line {line_num}: {e}")
    
    print(f"✅ Loaded {len(benchmark_data)} benchmark questions from JSONL")
    return benchmark_data

def collect_evaluation_data_simple(benchmark_data: List[Dict[str, str]], approach: str = "chroma") -> List[Dict[str, Any]]:
    """
    Collect evaluation data following RAGAS documentation pattern
    """
    print(f"\n🔄 Collecting evaluation data for {approach.upper()} approach...")
    
    dataset = []
    
    for i, item in enumerate(benchmark_data, 1):
        query = item['question']
        reference = item['ground_truth']
        
        print(f"  Processing question {i}/{len(benchmark_data)}: {query[:60]}...")
        print(f"  🔍 DEBUG: Reference value: '{reference}' (length: {len(reference)})")
        
        try:
            # Query the appropriate RAG system using new retriever functions
            if approach == "chroma" and CHROMA_AVAILABLE:
                result = query_chroma_rag(query, k=1)
            elif approach == "graphrag" and GRAPHRAG_AVAILABLE:
                try:
                    result = query_graphrag(query, k=5)
                    # Ensure result has the expected structure
                    if not result or not result.get('final_answer'):
                        result = {
                            'final_answer': 'GraphRAG retrieval failed to generate a response.',
                            'retrieval_details': [],
                            'method': 'graphrag_fallback'
                        }
                except Exception as e:
                    print(f"   ⚠️ GraphRAG query failed: {e}")
                    result = {
                        'final_answer': f'GraphRAG error: {str(e)}',
                        'retrieval_details': [],
                        'method': 'graphrag_error'
                    }
            elif approach == "text2cypher" and TEXT2CYPHER_AVAILABLE:
                result = query_text2cypher_rag(query)
            elif approach == "advanced_graphrag" and ADVANCED_GRAPHRAG_AVAILABLE:
                try:
                    # Advanced GraphRAG uses async, so we need to handle it properly
                    import asyncio
                    loop = None
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Import and run the async function
                    from retrievers.advanced_graphrag_retriever import query_advanced_graphrag_global
                    result = loop.run_until_complete(query_advanced_graphrag_global(query, k=5))
                    
                    # Ensure result has the expected structure
                    if not result or not result.get('final_answer'):
                        result = {
                            'final_answer': 'Advanced GraphRAG retrieval failed to generate a response.',
                            'retrieval_details': [],
                            'method': 'advanced_graphrag_fallback'
                        }
                except Exception as e:
                    print(f"   ⚠️ Advanced GraphRAG query failed: {e}")
                    result = {
                        'final_answer': f'Advanced GraphRAG error: {str(e)}',
                        'retrieval_details': [],
                        'method': 'advanced_graphrag_error'
                    }
            elif approach == "drift_graphrag" and DRIFT_GRAPHRAG_AVAILABLE:
                try:
                    # DRIFT GraphRAG also uses async
                    import asyncio
                    loop = None
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Run the async function
                    result = loop.run_until_complete(query_drift_graphrag(query, use_modular=True))
                    
                    # Ensure result has the expected structure
                    if not result or not result.get('final_answer'):
                        result = {
                            'final_answer': 'DRIFT GraphRAG retrieval failed to generate a response.',
                            'retrieval_details': [],
                            'method': 'drift_graphrag_fallback'
                        }
                except Exception as e:
                    print(f"   ⚠️ DRIFT GraphRAG query failed: {e}")
                    result = {
                        'final_answer': f'DRIFT GraphRAG error: {str(e)}',
                        'retrieval_details': [],
                        'method': 'drift_graphrag_error'
                    }
            elif approach == "neo4j_vector" and NEO4J_VECTOR_AVAILABLE:
                result = query_neo4j_vector_rag(query, k=5)
            elif approach == "hybrid_cypher" and HYBRID_CYPHER_AVAILABLE:
                # Use the main hybrid cypher function with reduced k for benchmark stability
                result = query_hybrid_cypher_rag(query, k=5)
            else:
                # Handle unavailable retrievers or unknown approaches
                if approach == "chroma" and not CHROMA_AVAILABLE:
                    raise ValueError(f"ChromaDB retriever not available")
                elif approach == "graphrag" and not GRAPHRAG_AVAILABLE:
                    raise ValueError(f"GraphRAG retriever not available")
                elif approach == "text2cypher" and not TEXT2CYPHER_AVAILABLE:
                    raise ValueError(f"Text2Cypher retriever not available")
                elif approach == "advanced_graphrag" and not ADVANCED_GRAPHRAG_AVAILABLE:
                    raise ValueError(f"Advanced GraphRAG retriever not available")
                elif approach == "drift_graphrag" and not DRIFT_GRAPHRAG_AVAILABLE:
                    raise ValueError(f"DRIFT GraphRAG retriever not available")
                elif approach == "neo4j_vector" and not NEO4J_VECTOR_AVAILABLE:
                    raise ValueError(f"Neo4j Vector retriever not available")
                elif approach == "hybrid_cypher" and not HYBRID_CYPHER_AVAILABLE:
                    raise ValueError(f"Hybrid Cypher retriever not available")
                else:
                    raise ValueError(f"Unknown approach: {approach}")
            
            # Extract retrieved contexts as simple list of strings (RAGAS format)
            retrieved_contexts = []
            for detail in result.get('retrieval_details', []):
                content = detail.get('content', '')
                if not content:
                    # Build a compact string from hybrid details if no 'content'
                    neighbor_summaries = detail.get('neighbor_summaries')
                    anchor = detail.get('anchor')
                    if neighbor_summaries:
                        content = f"Anchor: {str(anchor)[:200]} | Neighbors: " + ", ".join(map(str, neighbor_summaries[:20]))
                if content:
                    # Don't truncate contexts too aggressively - RAGAS needs sufficient context
                    # for meaningful comparison with ground truth
                    if len(content) > 2000:  # Increased from 1000 to 2000
                        content = content[:2000] + "..."
                    retrieved_contexts.append(content)
            
            # Ensure we have at least some context - but make it more meaningful
            if not retrieved_contexts:
                retrieved_contexts = ["No relevant context was retrieved for this query."]
            
            # Get the response
            response = result.get('final_answer', '')
            if not response:
                response = "No response generated."
            
            # Create RAGAS-compatible data structure with EXACT field names required by RAGAS
            # Based on debug testing, RAGAS requires exactly: user_input, retrieved_contexts, response, reference
            print(f"  🔍 DEBUG: About to add to dataset - reference: '{reference}' (length: {len(reference)})")
            dataset.append({
                "user_input": query,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": reference
            })
            
            # Small delay to avoid overwhelming the systems
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    ❌ Error processing question {i}: {e}")
            # Add a failure record to maintain dataset consistency with exact RAGAS format
            dataset.append({
                "user_input": query,
                "retrieved_contexts": ["Error: Could not retrieve context"],
                "response": f"Error: {str(e)}",
                "reference": reference
            })
    
    print(f"✅ Collected {len(dataset)} evaluation records for {approach.upper()}")
    return dataset

def debug_dataset_format(dataset: List[Dict[str, Any]], approach_name: str):
    """
    Debug function to examine dataset format and content quality
    """
    print(f"\n🔍 Debugging dataset format for {approach_name}...")
    
    if not dataset:
        print("   ⚠️  Dataset is empty!")
        return
    
    sample = dataset[0]
    print(f"   📋 Sample keys: {list(sample.keys())}")
    
    # Check contexts quality
    contexts = sample.get('retrieved_contexts', [])
    print(f"   📄 Number of contexts: {len(contexts)}")
    if contexts:
        print(f"   📄 First context length: {len(contexts[0])} chars")
        print(f"   📄 First context preview: {contexts[0][:200]}...")
    
    # Check ground truth quality - check both possible field names
    ground_truth = sample.get('ground_truth', '')
    reference = sample.get('reference', '')
    print(f"   🎯 Ground truth (ground_truth field) length: {len(ground_truth)} chars")
    print(f"   🎯 Ground truth (reference field) length: {len(reference)} chars")
    print(f"   🎯 Ground truth preview: {ground_truth[:200]}...")
    print(f"   🎯 Reference preview: {reference[:200]}...")
    
    # Check response quality
    response = sample.get('response', '')
    print(f"   💬 Response length: {len(response)} chars")
    print(f"   💬 Response preview: {response[:200]}...")

def evaluate_with_ragas_simple(dataset: List[Dict[str, Any]], approach_name: str) -> Dict[str, float]:
    """
    Evaluate the dataset using RAGAS following the documentation pattern
    """
    print(f"\n📊 Evaluating {approach_name} using RAGAS metrics...")
    print(f"   📋 Dataset size: {len(dataset)} items")
    
    # Debug the dataset format
    debug_dataset_format(dataset, approach_name)
    
    # Patch RAGAS progress bar for incremental updates
    original_tqdm = patch_ragas_progress()
    
    # Force sequential processing to prevent parallel job timeouts
    os.environ['RAGAS_MAX_WORKERS'] = '1'
    os.environ['RAGAS_DISABLE_PARALLEL'] = 'true'
    os.environ['RAGAS_EXECUTOR_WORKERS'] = '1'
    os.environ['RAGAS_ASYNC_MAX_WORKERS'] = '1'
    
    try:
        # Debug: Print first dataset item to check format
        if dataset:
            print(f"   📋 Dataset sample format: {list(dataset[0].keys())}")
        
        # Create RAGAS evaluation dataset
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        
        # Monkey patch RAGAS timeout settings
        try:
            import ragas
            # Set global timeout for RAGAS operations - increase default timeout
            timeout_value = int(os.getenv('RAGAS_METRIC_TIMEOUT', '300'))  # Increased from 120 to 300
            if hasattr(ragas, '_timeout'):
                ragas._timeout = timeout_value
            
            # Set additional timeout configurations
            os.environ['RAGAS_TIMEOUT'] = str(timeout_value)
            os.environ['RAGAS_LLM_TIMEOUT'] = str(timeout_value)
        except Exception as e:
            print(f"   ⚠️  Could not set RAGAS timeout: {e}")
        
        # Prepare evaluator LLM with timeout configuration
        evaluator_llm = LangchainLLMWrapper(llm)
        
        # LLM timeout configuration is handled by the model factory
        # If there are actual timeout issues, they will surface as runtime errors
        
        # Use basic metrics with enhanced configuration for smaller models
        # Configure metrics with explicit parameters to avoid issues
        try:
            context_recall_metric = LLMContextRecall()
            faithfulness_metric = Faithfulness()
            factual_correctness_metric = FactualCorrectness(mode="f1")
            
            metrics = [context_recall_metric, faithfulness_metric, factual_correctness_metric]
        except Exception as e:
            print(f"   ⚠️  Error configuring metrics: {e}")
            # Fallback to simpler configuration
            metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]
        
        # Configure metrics for smaller models
        for metric in metrics:
            if hasattr(metric, 'llm'):
                # Set timeout and retry configurations
                if hasattr(metric.llm, 'request_timeout'):
                    metric.llm.request_timeout = 300
                if hasattr(metric, '_max_retries'):
                    metric._max_retries = 5
        
        # Enhanced error handling for smaller models
        def safe_evaluate_metrics(dataset, metrics_list, llm_wrapper):
            """Safely evaluate metrics with fallback handling for parser errors"""
            results = {}
            
            for i, metric in enumerate(metrics_list):
                metric_name = metric.__class__.__name__
                print(f"   📊 Evaluating {metric_name} ({i+1}/{len(metrics_list)})...")
                
                try:
                    # Single metric evaluation to isolate failures
                    single_result = evaluate(
                        dataset=dataset,
                        metrics=[metric],
                        llm=llm_wrapper,
                        raise_exceptions=False
                    )
                    
                    # Extract the score
                    if hasattr(single_result, 'to_pandas'):
                        df = single_result.to_pandas()
                        if not df.empty:
                            for col in df.columns:
                                if col not in ['user_input', 'retrieved_contexts', 'response', 'reference']:
                                    score = df[col].mean() if not df[col].isna().all() else 0.0
                                    results[col] = score
                                    print(f"     ✅ {metric_name}: {col} = {score:.3f}")
                    
                except Exception as e:
                    print(f"     ❌ {metric_name} failed: {e}")
                    print(f"     🔍 Error details: {type(e).__name__}: {str(e)}")
                    
                    # Try to provide a basic heuristic score for context recall
                    if 'context_recall' in metric_name.lower():
                        try:
                            # Simple heuristic: check if any retrieved context contains keywords from ground truth
                            heuristic_score = calculate_heuristic_context_recall(dataset)
                            results['context_recall'] = heuristic_score
                            print(f"     🔄 Using heuristic context recall: {heuristic_score:.3f}")
                        except:
                            results['context_recall'] = 0.0
                    elif 'faithfulness' in metric_name.lower():
                        results['faithfulness'] = 0.0
                    elif 'factual' in metric_name.lower():
                        results['factual_correctness'] = 0.0
            
            return results
        
        def calculate_heuristic_context_recall(dataset):
            """Calculate a simple heuristic context recall score"""
            if not dataset:
                return 0.0
                
            total_score = 0.0
            for item in dataset:
                contexts = item.get('retrieved_contexts', [])
                ground_truth = item.get('ground_truth', '')
                
                if not contexts or not ground_truth:
                    continue
                    
                # Simple keyword overlap approach
                gt_words = set(ground_truth.lower().split())
                context_text = ' '.join(contexts).lower()
                context_words = set(context_text.split())
                
                # Calculate overlap ratio
                if gt_words:
                    overlap = len(gt_words.intersection(context_words))
                    score = overlap / len(gt_words)
                    total_score += min(score, 1.0)  # Cap at 1.0
            
            return total_score / len(dataset) if dataset else 0.0
        
        # Run evaluation with retry logic
        max_retries = int(os.getenv('RAGAS_MAX_RETRIES', '3'))  # Increased from 2 to 3
        for attempt in range(max_retries + 1):
            try:
                print(f"   🔄 Evaluation attempt {attempt + 1}/{max_retries + 1}")
                
                # Force sequential processing to avoid parallel job timeouts
                import asyncio
                try:
                    # Close any existing event loop to prevent conflicts
                    try:
                        loop = asyncio.get_running_loop()
                        if loop and not loop.is_closed():
                            loop.close()
                    except RuntimeError:
                        pass  # No running loop
                    
                    # Set event loop policy for Windows compatibility
                    if os.name == 'nt' and hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
                        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                    
                    # Create new event loop
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    
                except Exception as e:
                    print(f"   ⚠️  Async setup warning: {e}")
                    pass
                
                # Set environment variables to force single-threaded execution
                original_workers = os.environ.get('RAGAS_MAX_WORKERS')
                original_executor = os.environ.get('RAGAS_EXECUTOR_WORKERS')
                original_async = os.environ.get('RAGAS_ASYNC_MAX_WORKERS')
                
                os.environ['RAGAS_MAX_WORKERS'] = '1'
                os.environ['RAGAS_EXECUTOR_WORKERS'] = '1'
                os.environ['RAGAS_ASYNC_MAX_WORKERS'] = '1'
                os.environ['RAGAS_DISABLE_PARALLEL'] = 'true'
                
                # Increase various timeout settings - but make them more reasonable
                os.environ['RAGAS_REQUEST_TIMEOUT'] = '180'  # 3 minutes per request
                os.environ['RAGAS_ASYNC_TIMEOUT'] = '180'
                os.environ['RAGAS_EVALUATION_TIMEOUT'] = '600'  # 10 minutes for entire evaluation
                
                # Additional environment variables that might help
                os.environ['HTTPX_TIMEOUT'] = '180'  # For HTTP client timeouts
                os.environ['OPENAI_REQUEST_TIMEOUT'] = '180'  # For OpenAI API calls
                os.environ['OLLAMA_REQUEST_TIMEOUT'] = '180'  # For Ollama API calls
                
                # Add batch processing settings to reduce timeouts
                os.environ['RAGAS_BATCH_SIZE'] = '1'  # Process one item at a time
                os.environ['RAGAS_CONCURRENT_LIMIT'] = '1'  # No concurrent processing
                
                try:
                    # Force sequential evaluation by evaluating one metric at a time
                    print(f"   🔄 Evaluating metrics sequentially to avoid timeouts...")
                    
                    individual_results = {}
                    total_metrics = len(metrics)
                    
                    print(f"   🔄 Starting sequential evaluation of {total_metrics} metrics...")
                    
                    for i, metric in enumerate(metrics):
                        metric_name = type(metric).__name__
                        print(f"   📊 Evaluating {metric_name} ({i+1}/{total_metrics})...")
                        
                        try:
                            single_result = evaluate(
                                dataset=evaluation_dataset,
                                metrics=[metric],
                                llm=evaluator_llm,
                                raise_exceptions=False
                            )
                            
                            # Extract the result for this metric
                            if hasattr(single_result, 'to_pandas'):
                                df = single_result.to_pandas()
                                if not df.empty:
                                    for col in df.columns:
                                        if col.lower() not in ['user_input', 'retrieved_contexts', 'response', 'reference']:
                                            score = df[col].mean()
                                            individual_results[col] = score
                                            print(f"     ✅ {metric_name}: {col} = {score:.3f}")
                            
                        except Exception as metric_error:
                            print(f"     ❌ {metric_name} failed: {str(metric_error)[:100]}...")
                            # Set default value for failed metric
                            individual_results[metric_name.lower()] = 0.0
                    
                    print(f"\n   ✅ Evaluation completed for {approach_name}")
                    
                    # Create a mock result object with individual results
                    class MockResult:
                        def __init__(self, scores):
                            self.scores = scores
                        
                        def to_pandas(self):
                            import pandas as pd
                            return pd.DataFrame([self.scores])
                    
                    result = MockResult(individual_results)
                finally:
                    # Restore original tqdm
                    if original_tqdm:
                        import tqdm
                        tqdm.tqdm = original_tqdm
                    
                    # Restore original settings
                    if original_workers is not None:
                        os.environ['RAGAS_MAX_WORKERS'] = original_workers
                    else:
                        os.environ.pop('RAGAS_MAX_WORKERS', None)
                    if original_executor is not None:
                        os.environ['RAGAS_EXECUTOR_WORKERS'] = original_executor
                    else:
                        os.environ.pop('RAGAS_EXECUTOR_WORKERS', None)
                    if original_async is not None:
                        os.environ['RAGAS_ASYNC_MAX_WORKERS'] = original_async
                    else:
                        os.environ.pop('RAGAS_ASYNC_MAX_WORKERS', None)
                    os.environ.pop('RAGAS_DISABLE_PARALLEL', None)
                break  # Success, exit retry loop
            except Exception as e:
                if "TimeoutError" in str(e) or "timeout" in str(e).lower():
                    if attempt < max_retries:
                        print(f"   ⏱️  Timeout on attempt {attempt + 1}, retrying...")
                        continue
                    else:
                        print(f"   ❌ All retry attempts failed due to timeout")
                        print(f"   🔄 Attempting evaluation with reduced metrics...")
                        
                        # Try with just one metric as fallback
                        try:
                            print(f"   🔄 Attempting reduced metric evaluation (Faithfulness only)...")
                            reduced_metrics = [Faithfulness()]  # Most reliable metric
                            
                            result = evaluate(
                                dataset=evaluation_dataset,
                                metrics=reduced_metrics,
                                llm=evaluator_llm,
                                raise_exceptions=False
                            )
                            print(f"   ✅ Reduced metric evaluation succeeded")
                            break
                        except Exception as fallback_error:
                            print(f"   ❌ Reduced metric evaluation also failed: {fallback_error}")
                            raise e
                else:
                    # Non-timeout error, don't retry
                    print(f"   ❌ Evaluation failed: {e}")
                    print(f"   💡 Suggestions:")
                    print(f"      - Check if Ollama is running and accessible")
                    print(f"      - Increase timeout: export OLLAMA_REQUEST_TIMEOUT=300")
                    print(f"      - Reduce dataset size for testing")
                    print(f"      - Try with OpenAI models instead")
                    raise e
        
        print(f"✅ {approach_name} evaluation completed")
        
        # Convert result to dictionary format with correct metric names
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            print(f"   📊 Raw DataFrame columns: {df.columns.tolist()}")
            
            # Only calculate mean for numeric metric columns
            numeric_cols = []
            for col in df.columns:
                if any(metric in col.lower() for metric in ['context_recall', 'faithfulness', 'factual_correctness', 'precision', 'relevance']):
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        numeric_cols.append(col)
                    except:
                        continue
            
            if numeric_cols:
                # Calculate success rate for each metric
                total_samples = len(df)
                for col in numeric_cols:
                    successful_samples = df[col].count()  # Non-NaN count
                    success_rate = successful_samples / total_samples
                    print(f"   📊 {col}: {successful_samples}/{total_samples} samples ({success_rate:.1%} success)")
                
                scores = df[numeric_cols].mean().to_dict()
                print(f"   📊 Numeric scores: {scores}")
            else:
                # Fallback: look for direct score attributes
                scores = {}
                if hasattr(result, 'binary_score'):
                    scores.update(result.binary_score)
                print(f"   📊 Fallback scores: {scores}")
        else:
            scores = result
        
        # Map RAGAS metric names to our expected names
        mapped_scores = {}
        
        # Ensure we have some scores
        if not scores:
            print(f"   ⚠️  No scores found, using default values")
            return {
                'context_recall': 0.0,
                'faithfulness': 0.0,
                'factual_correctness': 0.0
            }
        
        for key, value in scores.items():
            try:
                # Convert value to float
                numeric_value = float(value) if (value is not None and not (isinstance(value, float) and math.isnan(value))) else 0.0
                
                # Map to our expected metric names
                if 'context_recall' in key.lower():
                    mapped_scores['context_recall'] = numeric_value
                elif 'faithfulness' in key.lower():
                    mapped_scores['faithfulness'] = numeric_value
                elif 'factual_correctness' in key.lower():
                    mapped_scores['factual_correctness'] = numeric_value
                else:
                    # Keep original key as fallback
                    mapped_scores[key.lower()] = numeric_value
                    
            except (ValueError, TypeError) as e:
                print(f"   ⚠️  Could not convert {key}={value} to numeric: {e}")
                continue
        
        print(f"   ✅ Final mapped scores: {mapped_scores}")
        return mapped_scores
        
    except Exception as e:
        print(f"❌ Error during {approach_name} evaluation: {e}")
        import traceback
        print(f"   🔍 Full error traceback:")
        traceback.print_exc()
        
        # Return default scores if evaluation fails
        return {
            'context_recall': 0.0,
            'faithfulness': 0.0,
            'factual_correctness': 0.0
        }

def create_comparison_table_simple(chroma_results: Dict, graphrag_results: Dict) -> pd.DataFrame:
    """Create a simple comparison table"""
    
    # Extract scores
    def extract_scores(results):
        if isinstance(results, dict):
            return results
        return {}
    
    chroma_scores = extract_scores(chroma_results)
    graphrag_scores = extract_scores(graphrag_results)
    
    # Create comparison dataframe
    metrics = []
    chroma_values = []
    graphrag_values = []
    improvements = []
    
    # Map metric names to display names
    metric_display_names = {
        'context_recall': 'Context Recall',
        'faithfulness': 'Faithfulness', 
        'factual_correctness': 'Factual Correctness'
    }
    
    for metric_key, display_name in metric_display_names.items():
        chroma_val = chroma_scores.get(metric_key, 0.0)
        graphrag_val = graphrag_scores.get(metric_key, 0.0)
        
        metrics.append(display_name)
        chroma_values.append(round(chroma_val, 4))
        graphrag_values.append(round(graphrag_val, 4))
        
        # Calculate improvement percentage
        if chroma_val > 0:
            improvement = ((graphrag_val - chroma_val) / chroma_val) * 100
            improvements.append(f"{improvement:+.2f}%")
        else:
            improvements.append("N/A")
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Metric': metrics,
        'ChromaDB RAG': chroma_values,
        'GraphRAG': graphrag_values,
        'Improvement': improvements
    })
    
    return comparison_df

def create_multi_approach_comparison_table(results_dict: Dict[str, Dict], approach_names: Dict[str, str]) -> pd.DataFrame:
    """Create a comparison table for multiple approaches"""
    
    # Extract scores for all approaches
    approach_scores = {}
    for approach_key, results in results_dict.items():
        if isinstance(results, dict):
            approach_scores[approach_key] = results
        else:
            approach_scores[approach_key] = {}
    
    # Create comparison dataframe
    metrics = []
    approach_columns = {}
    
    # Initialize columns for each approach
    for approach_key in approach_scores.keys():
        approach_columns[approach_names.get(approach_key, approach_key)] = []
    
    # Map metric names to display names
    metric_display_names = {
        'context_recall': 'Context Recall',
        'faithfulness': 'Faithfulness', 
        'factual_correctness': 'Factual Correctness'
    }
    
    for metric_key, display_name in metric_display_names.items():
        metrics.append(display_name)
        
        for approach_key, scores in approach_scores.items():
            approach_name = approach_names.get(approach_key, approach_key)
            score = scores.get(metric_key, 0.0)
            approach_columns[approach_name].append(round(score, 4))
    
    # Create comparison table
    comparison_data = {'Metric': metrics}
    comparison_data.update(approach_columns)
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def create_three_way_comparison_table(chroma_results: Dict, graphrag_results: Dict, text2cypher_results: Dict) -> pd.DataFrame:
    """Create a three-way comparison table for all approaches"""
    
    # Extract scores
    def extract_scores(results):
        if isinstance(results, dict):
            return results
        return {}
    
    chroma_scores = extract_scores(chroma_results)
    graphrag_scores = extract_scores(graphrag_results)
    text2cypher_scores = extract_scores(text2cypher_results)
    
    # Create comparison dataframe
    metrics = []
    chroma_values = []
    graphrag_values = []
    text2cypher_values = []
    
    # Map metric names to display names
    metric_display_names = {
        'context_recall': 'Context Recall',
        'faithfulness': 'Faithfulness', 
        'factual_correctness': 'Factual Correctness'
    }
    
    for metric_key, display_name in metric_display_names.items():
        chroma_val = chroma_scores.get(metric_key, 0.0)
        graphrag_val = graphrag_scores.get(metric_key, 0.0)
        text2cypher_val = text2cypher_scores.get(metric_key, 0.0)
        
        metrics.append(display_name)
        chroma_values.append(round(chroma_val, 4))
        graphrag_values.append(round(graphrag_val, 4))
        text2cypher_values.append(round(text2cypher_val, 4))
    
    # Create three-way comparison table
    comparison_df = pd.DataFrame({
        'Metric': metrics,
        'ChromaDB RAG': chroma_values,
        'GraphRAG': graphrag_values,
        'Text2Cypher': text2cypher_values
    })
    
    return comparison_df



def save_results_simple(chroma_dataset: List, graphrag_dataset: List, text2cypher_dataset: List,
                       chroma_results: Dict, graphrag_results: Dict, text2cypher_results: Dict,
                       comparison_table: pd.DataFrame, output_dir: str = "benchmark_outputs"):
    """Save results to files in organized folder structure"""
    
    # Create timestamped subdirectory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    
    # Create timestamped output directory if it doesn't exist
    os.makedirs(timestamped_output_dir, exist_ok=True)
    
    # Save datasets
    chroma_df = pd.DataFrame(chroma_dataset)
    graphrag_df = pd.DataFrame(graphrag_dataset)
    text2cypher_df = pd.DataFrame(text2cypher_dataset)
    
    chroma_df.to_csv(f'{timestamped_output_dir}/simple_benchmark_chroma.csv', index=False)
    graphrag_df.to_csv(f'{timestamped_output_dir}/simple_benchmark_graphrag.csv', index=False)
    text2cypher_df.to_csv(f'{timestamped_output_dir}/simple_benchmark_text2cypher.csv', index=False)
    
    # Save comparison table
    comparison_table.to_csv(f'{timestamped_output_dir}/simple_benchmark_three_way_comparison.csv', index=False)
    
    # Save results
    chroma_avg = comparison_table['ChromaDB RAG'].mean()
    graphrag_avg = comparison_table['GraphRAG'].mean()
    text2cypher_avg = comparison_table['Text2Cypher'].mean()
    
    # Determine best overall approach by highest average
    scores = {'ChromaDB RAG': chroma_avg, 'GraphRAG': graphrag_avg, 'Text2Cypher': text2cypher_avg}
    best_overall = max(scores, key=scores.get)
    
    with open(f'{timestamped_output_dir}/simple_benchmark_results.json', 'w') as f:
        json.dump({
            'chroma_results': chroma_results,
            'graphrag_results': graphrag_results,
            'text2cypher_results': text2cypher_results,
            'comparison_summary': {
                'chroma_avg': chroma_avg,
                'graphrag_avg': graphrag_avg,
                'text2cypher_avg': text2cypher_avg,
                'best_overall': best_overall
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to timestamped folder: '{timestamped_output_dir}/':")
    print("  - simple_benchmark_chroma.csv")
    print("  - simple_benchmark_graphrag.csv")
    print("  - simple_benchmark_text2cypher.csv") 
    print("  - simple_benchmark_three_way_comparison.csv")
    print("  - simple_benchmark_results.json")
    return timestamped_output_dir

def save_results_selective(datasets: Dict, results: Dict, comparison_table: pd.DataFrame, 
                          approaches: List[str], output_dir: str = "benchmark_outputs"):
    """Save results for selected approaches only"""
    
    # Create timestamped subdirectory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    
    # Create timestamped output directory if it doesn't exist
    os.makedirs(timestamped_output_dir, exist_ok=True)
    
    # Save datasets for selected approaches
    for approach in approaches:
        if approach in datasets and datasets[approach]:
            df = pd.DataFrame(datasets[approach])
            df.to_csv(f'{timestamped_output_dir}/simple_benchmark_{approach}.csv', index=False)
            print(f"  - simple_benchmark_{approach}.csv")
    
    # Save comparison table
    comparison_table.to_csv(f'{timestamped_output_dir}/simple_benchmark_comparison.csv', index=False)
    print(f"  - simple_benchmark_comparison.csv")
    
    # Calculate averages for selected approaches
    averages = {}
    for col in comparison_table.columns:
        if col != 'Metric' and col != 'Improvement':  # Skip non-numeric columns
            try:
                # Check if column contains numeric data
                if comparison_table[col].dtype in ['float64', 'int64'] or comparison_table[col].apply(lambda x: isinstance(x, (int, float))).all():
                    averages[col] = comparison_table[col].mean()
            except (TypeError, ValueError):
                continue
    
    # Determine best overall approach
    if averages:
        best_overall = max(averages, key=averages.get)
    else:
        best_overall = "None"
    
    # Save results JSON
    results_data = {
        'selected_approaches': approaches,
        'comparison_summary': {
            'averages': averages,
            'best_overall': best_overall
        }
    }
    
    # Add individual results
    for approach in approaches:
        if approach in results:
            results_data[f'{approach}_results'] = results[approach]
    
    with open(f'{timestamped_output_dir}/simple_benchmark_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"  - simple_benchmark_results.json")
    
    # Create detailed human-readable reports
    create_detailed_reports(datasets, results, approaches, timestamped_output_dir)
    
    print(f"\n💾 Results saved to timestamped folder: '{timestamped_output_dir}/'")
    return timestamped_output_dir


def create_detailed_reports(datasets: Dict, results: Dict, approaches: List[str], output_dir: str):
    """Create detailed human-readable reports with individual responses and scores"""
    
    try:
        # Import the results formatter
        import sys
        from pathlib import Path
        
        # Add ragbench module to path if not already there
        ragbench_path = Path(__file__).parent / "ragbench"
        if str(ragbench_path) not in sys.path:
            sys.path.append(str(ragbench_path))
        
        try:
            from ragbench.results_formatter import RAGBenchResultsFormatter
        except ImportError:
            from .ragbench.results_formatter import RAGBenchResultsFormatter
        
        print(f"\n📄 Creating detailed human-readable reports...")
        
        # Initialize formatter
        formatter = RAGBenchResultsFormatter()
        
        # Process each approach's dataset
        approach_names = {
            'chroma': 'ChromaDB RAG',
            'graphrag': 'GraphRAG', 
            'text2cypher': 'Text2Cypher',
            'advanced_graphrag': 'Advanced GraphRAG',
            'drift_graphrag': 'DRIFT GraphRAG',
            'neo4j_vector': 'Neo4j Vector RAG',
            'hybrid_cypher': 'Hybrid Cypher RAG'
        }
        
        # We need to correlate questions across approaches to group them properly
        question_groups = {}
        
        for approach in approaches:
            if approach not in datasets or not datasets[approach]:
                continue
                
            dataset = datasets[approach]
            approach_results = results.get(approach, {})
            retriever_name = approach_names.get(approach, approach)
            
            for i, record in enumerate(dataset):
                question = record.get("user_input", record.get("question", ""))
                ground_truth = record.get("reference", record.get("ground_truth", ""))
                response = record.get("response", "")
                contexts = record.get("retrieved_contexts", [])
                
                # Use question as key to group responses from different retrievers
                if question not in question_groups:
                    question_groups[question] = {
                        "ground_truth": ground_truth,
                        "responses": []
                    }
                
                # For individual scores, we use the average scores (limitation of current RAGAS integration)
                # In future, we could modify evaluate_with_ragas_simple to return per-question scores
                individual_scores = {
                    metric: score for metric, score in approach_results.items()
                    if isinstance(score, (int, float))
                }
                
                question_groups[question]["responses"].append({
                    "retriever_name": retriever_name,
                    "response": response,
                    "contexts": contexts,
                    "scores": individual_scores
                })
        
        # Add all results to formatter
        for question, question_data in question_groups.items():
            for response_data in question_data["responses"]:
                formatter.add_evaluation_result(
                    question=question,
                    ground_truth=question_data["ground_truth"],
                    retriever_name=response_data["retriever_name"],
                    retriever_response=response_data["response"],
                    retrieved_contexts=response_data["contexts"],
                    ragas_scores=response_data["scores"],
                    metadata={"evaluation_type": "ragas_benchmark"}
                )
        
        # Generate reports
        csv_path = formatter.create_comparison_csv(
            output_path=f"{output_dir}/detailed_comparison.csv"
        )
        
        print(f"  - detailed_comparison.csv (detailed data)")
        
    except ImportError as e:
        print(f"⚠️  Could not create detailed reports: {e}")
        print("   Detailed reporting requires the ragbench.results_formatter module")
    except Exception as e:
        print(f"⚠️  Error creating detailed reports: {e}")
        import traceback
        traceback.print_exc()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RAGAS Benchmark: Compare RAG approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ragas_benchmark.py --all                                # Test all available approaches  
      python ragas_benchmark.py --chroma --graphrag           # Test ChromaDB vs GraphRAG
    python ragas_benchmark.py --graphrag --advanced-graphrag # Test GraphRAG vs Advanced GraphRAG
  python ragas_benchmark.py --advanced-graphrag --drift-graphrag # Test Advanced vs DRIFT GraphRAG
  python ragas_benchmark.py --chroma --advanced-graphrag --drift-graphrag # Test 3-way comparison
  python ragas_benchmark.py --drift-graphrag                     # Test DRIFT GraphRAG only
  python ragas_benchmark.py --chroma                             # Test ChromaDB only
        """
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Test all three approaches (ChromaDB, GraphRAG, Text2Cypher)'
    )
    parser.add_argument(
        '--chroma', 
        action='store_true',
        help='Include ChromaDB RAG in testing'
    )
    parser.add_argument(
        '--graphrag', 
        action='store_true',
        help='Include GraphRAG in testing'
    )
    parser.add_argument(
        '--text2cypher', 
        action='store_true',
        help='Include Text2Cypher in testing'
    )
    parser.add_argument(
        '--advanced-graphrag', 
        action='store_true',
        help='Include Advanced GraphRAG (intelligent global/local/hybrid) in testing'
    )
    parser.add_argument(
        '--drift-graphrag', 
        action='store_true',
        help='Include DRIFT GraphRAG (iterative refinement) in testing'
    )
    parser.add_argument(
        '--neo4j-vector', 
        action='store_true',
        help='Include Neo4j Vector RAG (pure vector similarity) in testing'
    )
    parser.add_argument(
        '--hybrid-cypher', 
        action='store_true',
        help='Include Hybrid Cypher RAG (hybrid + generic neighborhood) in testing'
    )
    parser.add_argument(
        '--output-dir',
        default='benchmark_outputs',
        help='Output directory for results (default: benchmark_outputs)'
    )
    parser.add_argument(
        '--csv',
        help='Path to CSV benchmark file (default: benchmark/benchmark.csv)'
    )
    parser.add_argument(
        '--jsonl',
        help='Path to JSONL benchmark file (overrides --csv if provided)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit the number of evaluation samples (useful for testing)'
    )
    
    args = parser.parse_args()
    
    # Determine which approaches to test
    approaches = []
    if args.all:
        base_approaches = ['chroma', 'graphrag', 'text2cypher']
        if ADVANCED_GRAPHRAG_AVAILABLE:
            base_approaches.append('advanced_graphrag')
        if DRIFT_GRAPHRAG_AVAILABLE:
            base_approaches.append('drift_graphrag')
        if NEO4J_VECTOR_AVAILABLE:
            base_approaches.append('neo4j_vector')
        if HYBRID_CYPHER_AVAILABLE:
            base_approaches.append('hybrid_cypher')
        approaches = base_approaches
    else:
        if args.chroma:
            approaches.append('chroma')
        if args.graphrag:
            approaches.append('graphrag')
        if args.text2cypher:
            approaches.append('text2cypher')
        if getattr(args, 'advanced_graphrag', False) and ADVANCED_GRAPHRAG_AVAILABLE:
            approaches.append('advanced_graphrag')
        if getattr(args, 'drift_graphrag', False) and DRIFT_GRAPHRAG_AVAILABLE:
            approaches.append('drift_graphrag')
        if getattr(args, 'neo4j_vector', False) and NEO4J_VECTOR_AVAILABLE:
            approaches.append('neo4j_vector')
        if getattr(args, 'hybrid_cypher', False) and HYBRID_CYPHER_AVAILABLE:
            approaches.append('hybrid_cypher')
    
    # If no approaches specified, default to all
    if not approaches:
        print("⚠️  No approaches specified. Defaulting to all available approaches.")
        default_approaches = ['chroma', 'graphrag', 'text2cypher']
        if ADVANCED_GRAPHRAG_AVAILABLE:
            default_approaches.append('advanced_graphrag')
        if DRIFT_GRAPHRAG_AVAILABLE:
            default_approaches.append('drift_graphrag')
        if NEO4J_VECTOR_AVAILABLE:
            default_approaches.append('neo4j_vector')
        if HYBRID_CYPHER_AVAILABLE:
            default_approaches.append('hybrid_cypher')
        approaches = default_approaches
    
    return approaches, args.output_dir, args.csv, args.jsonl, args.limit

def main_selective(approaches: List[str], output_dir: str = "benchmark_outputs", 
                  csv_path: str = None, jsonl_path: str = None, limit: int = None):
    """Main benchmarking function with selective approach testing"""
    
    approach_names = {
        'chroma': 'ChromaDB RAG',
        'graphrag': 'GraphRAG', 
        'text2cypher': 'Text2Cypher',
        'advanced_graphrag': 'Advanced GraphRAG',
        'drift_graphrag': 'DRIFT GraphRAG',
        'neo4j_vector': 'Neo4j Vector RAG',
        'hybrid_cypher': 'Hybrid Cypher RAG'
    }
    
    selected_names = [approach_names[approach] for approach in approaches]
    
    print(f"🚀 Starting Selective RAGAS Benchmark: {' vs '.join(selected_names)}")
    print("=" * 80)
    
    # Load benchmark data (prefer JSONL over CSV)
    if jsonl_path:
        benchmark_data = load_benchmark_data_jsonl(jsonl_path)
    elif csv_path:
        benchmark_data = load_benchmark_data(csv_path)
    else:
        benchmark_data = load_benchmark_data()  # Default CSV
    
    # Apply limit if specified
    if limit and limit > 0:
        original_count = len(benchmark_data)
        benchmark_data = benchmark_data[:limit]
        print(f"🔢 Limited evaluation data from {original_count} to {len(benchmark_data)} samples")
    
    # Collect evaluation data for selected approaches
    print(f"\n📋 Phase 1: Data Collection")
    datasets = {}
    for approach in approaches:
        datasets[approach] = collect_evaluation_data_simple(benchmark_data, approach=approach)
    
    # Evaluate selected approaches with RAGAS
    print(f"\n📊 Phase 2: RAGAS Evaluation")
    results = {}
    for approach in approaches:
        results[approach] = evaluate_with_ragas_simple(datasets[approach], approach_names[approach])
    
    # Create comparison table for selected approaches
    print(f"\n📈 Phase 3: Results Analysis")
    
    if len(approaches) == 1:
        # Single approach - create simple results display
        approach = approaches[0]
        result = results[approach]
        print(f"\n📊 Results for {approach_names[approach]}:")
        print("-" * 50)
        for metric, score in result.items():
            print(f"{metric.replace('_', ' ').title()}: {score:.4f}")
        
        # Create simple comparison table
        comparison_table = pd.DataFrame({
            'Metric': [metric.replace('_', ' ').title() for metric in result.keys()],
            approach_names[approach]: list(result.values())
        })
        
    elif len(approaches) == 2:
        # Two approaches - create comparison table
        comparison_table = create_comparison_table_simple(
            results[approaches[0]], 
            results[approaches[1]]
        )
        # Rename columns to match selected approaches
        comparison_table.columns = ['Metric', approach_names[approaches[0]], approach_names[approaches[1]], 'Improvement']
        
    else:
        # Multiple approaches - create multi-approach comparison
        comparison_table = create_multi_approach_comparison_table(results, approach_names)
    
    # Display results
    print(f"\n" + "=" * 90)
    print(f"🏆 BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print(comparison_table.to_string(index=False))
    
    # Calculate overall performance for selected approaches
    print(f"\n📊 OVERALL PERFORMANCE SUMMARY:")
    print("-" * 50)
    
    averages = {}
    for col in comparison_table.columns:
        if col != 'Metric' and col != 'Improvement':  # Skip non-numeric columns
            try:
                # Check if column contains numeric data
                if comparison_table[col].dtype in ['float64', 'int64'] or comparison_table[col].apply(lambda x: isinstance(x, (int, float))).all():
                    averages[col] = comparison_table[col].mean()
                    print(f"{col} Average Score: {averages[col]:.4f}")
            except (TypeError, ValueError):
                print(f"⚠️  Skipping non-numeric column: {col}")
                continue
    
    # Determine overall winner
    if averages:
        winner = max(averages, key=averages.get)
        winner_score = averages[winner]
        print(f"\n🏆 Overall Winner: {winner} (Score: {winner_score:.4f})")
        
        # Show improvements if multiple approaches
        if len(approaches) > 1:
            print(f"\n📈 Performance Comparisons:")
            baseline = list(averages.values())[0]
            baseline_name = list(averages.keys())[0]
            
            for name, score in averages.items():
                if name != baseline_name:
                    if score > baseline:
                        improvement = ((score - baseline) / baseline) * 100
                        print(f"📈 {name} vs {baseline_name}: +{improvement:.2f}%")
                    else:
                        decline = ((baseline - score) / baseline) * 100
                        print(f"📉 {name} vs {baseline_name}: -{decline:.2f}%")
    
    # Save detailed results
    print(f"\n💾 Phase 4: Saving Results")
    timestamped_dir = save_results_selective(datasets, results, comparison_table, approaches, output_dir)
    
    # Generate visualizations
    print(f"\n📊 Phase 5: Generating Visualizations")
    create_visualizations(comparison_table, output_dir=timestamped_dir)
    
    print(f"\n✅ BENCHMARK COMPLETE!")
    print("=" * 80)
    
    return {
        'approaches': approaches,
        'results': results,
        'comparison_table': comparison_table,
        'datasets': datasets
    }



if __name__ == "__main__":
    # Parse command line arguments
    approaches, output_dir, csv_path, jsonl_path, limit = parse_arguments()
    
    # Run selective benchmark
    results = main_selective(approaches, output_dir, csv_path, jsonl_path, limit) 