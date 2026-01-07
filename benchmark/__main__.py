"""
Benchmark Module Entry Point

Run with: python -m benchmark [preset] --<metrics> --<retriever>

Examples:
  python -m benchmark micro --hotpotqa --agentic-text2cypher
  python -m benchmark mini --ragas --chroma --graphrag
  python -m benchmark smoke --all-metrics --agentic-text2cypher
"""

from benchmark.run import main

if __name__ == "__main__":
    main()

