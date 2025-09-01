import os
from dotenv import load_dotenv

load_dotenv()

from retrievers.hybrid_cypher_retriever import HybridCypherRAGRetriever


def main():
    retriever = HybridCypherRAGRetriever()
    try:
        query = "banking services"
        print(f"Query: {query}")
        # Call underlying hybrid retriever to avoid LLM call
        raw = retriever.hybrid_retriever.search(query_text=query, top_k=5)
        records = getattr(raw, "records", [])
        print(f"records: {len(records)}")
        for i, rec in enumerate(records[:3], 1):
            node = rec.get("node")
            score = rec.get("score")
            neighbors = rec.get("neighbor_summaries")
            print({
                "i": i,
                "score": score,
                "node": str(node),
                "neighbors": neighbors[:5] if neighbors else None,
            })
    finally:
        retriever.close()


if __name__ == "__main__":
    main()


