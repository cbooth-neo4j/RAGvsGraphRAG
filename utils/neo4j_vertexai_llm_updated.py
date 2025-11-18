from utils.llms import get_google_client

"""
VertexAI LLM implementation using the new google.genai SDK.

This class preserves compatibility with the neo4j_graphrag LLMInterface.
"""

import logging
from typing import Any, List, Optional, Sequence, Union

from google import genai
from google.genai import types

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.rate_limit import (
    RateLimitHandler,
    rate_limit_handler,
    async_rate_limit_handler,
)
from neo4j_graphrag.llm.types import (
    BaseMessage,
    LLMResponse,
    MessageList,
    ToolCall,
    ToolCallResponse,
)
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.tool import Tool


from utils.graph_rag_logger import setup_logging, get_logger
from dotenv import load_dotenv

load_dotenv()

setup_logging()
logger = get_logger(__name__)


class CustomVertexAILLM(LLMInterface):
    """Google Vertex AI (Gemini) implementation using the google.genai client."""

    def __init__(
        self,
        project: Optional[str] = None,
        location: str = "us-central1",
        credentials: Any = None,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        model_params: Optional[dict[str, Any]] = None,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = 2048,
        top_p: Optional[float] = 0.9,
        top_k: Optional[int] = None,
        stream: bool = False,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs,
    ):
        super().__init__(model_name=model, model_params=model_params, rate_limit_handler=rate_limit_handler)
        self.project = project
        self.location = location
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stream = stream
        logger.debug("In custom CustomVertexAILLM...")
        try:
            self.client = get_google_client()
            logger.info(f"Initialized google.genai client (model={self.model})")
        except Exception as e:
            raise LLMGenerationError(f"Failed to initialize GenAI client: {e}")

    @rate_limit_handler
    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[BaseMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Synchronous text generation call."""
        try:
            messages = self._prepare_messages(input, message_history, system_instruction)
            if self.stream:
                return self._stream_invoke(messages)
            return self._call_llm(messages)
        except Exception as e:
            logger.exception("Error in invoke()")
            raise LLMGenerationError(f"invoke() failed: {e}") from e

    @async_rate_limit_handler
    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[BaseMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Asynchronous text generation call."""
        try:
            messages = self._prepare_messages(input, message_history, system_instruction)
            if self.stream:
                return await self._astream_invoke(messages)
            return await self._acall_llm(messages)
        except Exception as e:
            logger.exception("Error in ainvoke()")
            raise LLMGenerationError(f"ainvoke() failed: {e}") from e


    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[BaseMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Invoke LLM with available tools."""
        try:
            response = self._call_llm(
                input,
                message_history=message_history,
                system_instruction=system_instruction,
                tools=tools,
            )
            return self._parse_tool_response(response)
        except Exception as e:
            logger.exception("Error in invoke_with_tools()")
            raise LLMGenerationError(f"invoke_with_tools() failed: {e}") from e

    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[BaseMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Async version of invoke_with_tools."""
        try:
            response = await self._acall_llm(
                input,
                message_history=message_history,
                system_instruction=system_instruction,
                tools=tools,
            )
            return self._parse_tool_response(response)
        except Exception as e:
            logger.exception("Error in ainvoke_with_tools()")
            raise LLMGenerationError(f"ainvoke_with_tools() failed: {e}") from e

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _build_config(self, tools: Optional[Sequence[Tool]] = None):
        cfg = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            cfg["top_k"] = self.top_k
        if tools:
            cfg["tools"] = [t.to_dict() for t in tools]
        return types.GenerateContentConfig(**cfg)

    def _prepare_messages(
        self,
        input: str,
        message_history: Optional[Union[List[BaseMessage], MessageHistory]],
        system_instruction: Optional[str],
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        if system_instruction:
            messages.append(BaseMessage(role="system", content=system_instruction))
        if message_history:
            for msg in message_history:
                messages.append(BaseMessage(role=msg.role, content=msg.content))
        messages.append(BaseMessage(role="user", content=input))
        return messages

    def _convert_messages_to_contents(self, messages: List[BaseMessage]):
        return [
            types.Content(role=m.role, parts=[types.Part(text=m.content)])
            for m in messages
        ]

    # ----------------------------------------------------------------------
    # Core generation logic
    # ----------------------------------------------------------------------

    def _call_llm(
        self,
        input: Union[str, List[BaseMessage]],
        message_history: Optional[Union[List[BaseMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[Sequence[Tool]] = None,
    ) -> LLMResponse:
        try:
            if isinstance(input, str):
                messages = self._prepare_messages(input, message_history, system_instruction)
            else:
                messages = input
            prompt = self._convert_messages_to_contents(messages)
            config = self._build_config(tools)
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
            logger.debug(f'_call_llm response: {response}')
            return LLMResponse(content=response.text)
        except Exception as e:
            logger.exception("Error in _call_llm()")
            raise LLMGenerationError(f"Error generating content (sync): {e}") from e

    async def _acall_llm(
        self,
        input: Union[str, List[BaseMessage]],
        message_history: Optional[Union[List[BaseMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[Sequence[Tool]] = None,
    ) -> LLMResponse:
        try:
            if isinstance(input, str):
                messages = self._prepare_messages(input, message_history, system_instruction)
            else:
                messages = input
            prompt = self._convert_messages_to_contents(messages)
            config = self._build_config(tools)
            response = await self.client.a_models.generate_content(
                model=self.model, contents=prompt, config=config
            )
            logger.debug(f'_acall_llm response: {response}')
            return LLMResponse(content=response.text)
        except Exception as e:
            logger.exception("Error in _acall_llm()")
            raise LLMGenerationError(f"Error generating content (async): {e}") from e

    # ----------------------------------------------------------------------
    # Streaming support
    # ----------------------------------------------------------------------

    def _stream_invoke(self, messages: List[BaseMessage]) -> LLMResponse:
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=self._convert_messages_to_contents(messages),
                config=self._build_config(),
            )
            chunks = []
            for event in stream:
                if event.candidates and event.candidates[0].content.parts:
                    chunks.append(event.candidates[0].content.parts[0].text)
            return LLMResponse(content="".join(chunks))
        except Exception as e:
            logger.exception("Error in _stream_invoke()")
            raise LLMGenerationError(f"Streaming invoke failed: {e}") from e

    async def _astream_invoke(self, messages: List[BaseMessage]) -> LLMResponse:
        try:
            stream = await self.client.a_models.generate_content_stream(
                model=self.model,
                contents=self._convert_messages_to_contents(messages),
                config=self._build_config(),
            )
            chunks = []
            async for event in stream:
                if event.candidates and event.candidates[0].content.parts:
                    chunks.append(event.candidates[0].content.parts[0].text)
            return LLMResponse(content="".join(chunks))
        except Exception as e:
            logger.exception("Error in _astream_invoke()")
            raise LLMGenerationError(f"Async streaming invoke failed: {e}") from e

    # ----------------------------------------------------------------------
    # Tool call response parsing
    # ----------------------------------------------------------------------

    def _parse_tool_response(self, response: LLMResponse) -> ToolCallResponse:
        """Translate a raw model response into a ToolCallResponse."""
        raw = getattr(response, "raw", None) or {}
        return ToolCallResponse(
            raw_response=raw,
            output_text=response.content,
            tool_calls=raw.get("tool_calls", []),
        )
