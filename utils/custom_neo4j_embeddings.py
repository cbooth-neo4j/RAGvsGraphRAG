from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.llm import RateLimitHandler, rate_limit_handler


from utils.graph_rag_logger import setup_logging, get_logger
from dotenv import load_dotenv

from utils.llms import get_vertex_embeddings

load_dotenv()

setup_logging()
logger = get_logger(__name__)


class CustomVertexAIEmbeddings(Embedder):
    """
    Vertex AI embeddings class (LangChain-based).
    This class uses LangChain's VertexAIEmbeddings to generate vector embeddings for text data.

    Args:
        model_nm (str): The name of the Vertex AI text embedding model to use. Defaults to "text-embedding-005".
    """

    def __init__(
        self,
        model_nm: str = "text-embedding-005",
    ) -> None:
        super().__init__()
        # Initialize LangChain embedding model
        self.model = get_vertex_embeddings(model_name=model_nm)

    @rate_limit_handler
    def embed_query(
        self, text: str, task_type: str = "RETRIEVAL_QUERY", **kwargs: Any
    ) -> list[float]:
        """
        Generate embeddings for a given query using LangChain's VertexAIEmbeddings.

        Args:
            text (str): The text to generate an embedding for.
            task_type (str): Ignored for LangChain implementation, retained for API compatibility.
            **kwargs (Any): Additional keyword arguments (ignored by LangChain).
        """
        try:
            # LangChain returns a list[float] for a single text string
            embedding =  self.model.embed_query(text)
            # logger.debug(f'Query: {text}, Embedding: {embedding}')
            return embedding
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with LangChain VertexAI: {e}"
            ) from e

