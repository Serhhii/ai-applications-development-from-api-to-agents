from google import genai
from google.genai import types

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class GeminiAIClient(AIClient):
    """
    Client for Google Gemini API using the official SDK.

    This implementation uses the official Google GenAI Python library to interact
    with Gemini models, providing both synchronous and streaming response capabilities.

    Attributes:
        _client (genai.Client): Google GenAI client instance.
        Inherits all other attributes from AIClient.
    """

    def __init__(self, endpoint: str, model_name: str, api_key: str, system_prompt: str):
        """
        Initialize the Gemini client with SDK.

        Args:
            endpoint (str): The Gemini API endpoint (for compatibility, not used by SDK).
            model_name (str): The Gemini model to use (e.g., 'gemini-3-flash-preview').
            api_key (str): The Google API key for authentication.
            system_prompt (str): The system instruction to guide the model's behavior.
        """
        super().__init__(endpoint=endpoint, model_name=model_name, api_key=api_key, system_prompt=system_prompt)
        self._client = genai.Client(api_key=api_key)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from Google's Gemini API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Note:
            Gemini uses 'system_instruction' parameter for system-level guidance.
            The response is printed to stdout before being returned.
        """
        max_tokens = kwargs.get("max_tokens", 1024)
        contents = [
            types.Content(
                role="user" if m.role == Role.USER else "model",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
        ]
        resp = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                max_output_tokens=max_tokens,
            ),
        )
        response_text = resp.text
        print(response_text)
        return Message(role=Role.ASSISTANT, content=response_text)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from Google's Gemini API.

        The response is streamed chunk-by-chunk, with each text chunk printed
        immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            Uses the async streaming interface provided by the Gemini SDK.
            Each chunk's text is printed to stdout as it arrives.
        """
        max_tokens = kwargs.get("max_tokens", 1024)
        contents = [
            types.Content(
                role="user" if m.role == Role.USER else "model",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
        ]
        full_response = ""
        async for chunk in self._client.aio.models.generate_content_stream(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                max_output_tokens=max_tokens,
            ),
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        print()
        return Message(role=Role.ASSISTANT, content=full_response)