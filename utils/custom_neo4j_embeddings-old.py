from typing import Any

from datasets.features.features import list_of_np_array_to_pyarrow_listarray
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.llm import rate_limit_handler
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from utils.llms import get_vertex_embeddings

from utils.graph_rag_logger import setup_logging, get_logger
from dotenv import load_dotenv

load_dotenv()

setup_logging()
logger = get_logger(__name__)

from neo4j_graphrag.embeddings.vertexai import VertexAIEmbeddings

class CustomVertexAIEmbeddings(Embedder):
    """
    Vertex AI embeddings class.
    This class uses the Vertex AI Python client to generate vector embeddings for text data.

    Args:
        model_nm (str): The name of the Vertex AI text embedding model to use. Defaults to "text-embedding-005".
    """

    def __init__(self, model_nm: str = "text-embedding-005") -> None:
        if TextEmbeddingModel is None:
            raise ImportError(
                """Could not import Vertex AI Python client.
                Please install it with `pip install "neo4j-graphrag[google]"`."""
            )
        self.model = get_vertex_embeddings(model_name=model_nm)


    # def embed_query(self, text: str, task_type: str = "RETRIEVAL_QUERY", **kwargs: Any) -> list[float]:
    #     """
    #     Generate embeddings for a given query using a Vertex AI text embedding model.
    #
    #     Args:
    #         text (str): The text to generate an embedding for.
    #         task_type (str): The type of the text embedding task. Defaults to "RETRIEVAL_QUERY". See https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#tasktype for a full list.
    #         **kwargs (Any): Additional keyword arguments to pass to the Vertex AI client's get_embeddings method.
    #     """
    #     # type annotation needed for mypy
    #     inputs: list[str | TextEmbeddingInput] = [TextEmbeddingInput(text, task_type)]
    #     embed_value =  self.model.embed_query(inputs)
    #     logger.debug(f'embed_query: : {embed_value} ')
    #     return embed_value

    @rate_limit_handler
    def embed_query(
            self, text: str, task_type: str = "RETRIEVAL_QUERY", **kwargs: Any
    ) -> list[float]:
        """
        Generate embeddings for a given query using a Vertex AI text embedding model.

        Args:
            text (str): The text to generate an embedding for.
            task_type (str): The type of the text embedding task. Defaults to "RETRIEVAL_QUERY". See https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#tasktype for a full list.
            **kwargs (Any): Additional keyword arguments to pass to the Vertex AI client's get_embeddings method.
        """
        try:
            # type annotation needed for mypy
            inputs: list[str | TextEmbeddingInput] = [
                TextEmbeddingInput(text, task_type)
            ]
            logger.debug(f'In embed_query: {text} and model: {self.model}')
            #embeddings = self.model.get_embeddings(inputs, **kwargs)
            TextEmbeddingModelFromLangChain()
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with VertexAI: {e}"
            ) from e


from dataclasses import dataclass
from typing import List


@dataclass
class TextEmbedding:
    values: List[float]


class TextEmbeddingModelFromLangChain:
    """Wraps LangChain VertexAIEmbeddings to behave like VertexAI's TextEmbeddingModel."""

    def __init__(self, embeddings):
        """
        Args:
            embeddings: an instance of langchain_google_vertexai.VertexAIEmbeddings
        """
        self.embeddings = embeddings

    def get_embeddings(self, texts: List[str]) -> List[TextEmbedding]:
        """Mimics vertexai.language_models.TextEmbeddingModel.get_embeddings()."""
        # LangChain's embed_documents handles list inputs
        vectors = self.embeddings.embed_documents(texts)
        return [TextEmbedding(values=v) for v in vectors]
