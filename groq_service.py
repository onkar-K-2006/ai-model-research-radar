import json
import urllib.error
import urllib.request
from typing import Generator, List, Optional

try:
    from groq import Groq

    GROQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GROQ_AVAILABLE = False


def get_groq_api_key_from_dataflow() -> Optional[str]:
    """Retrieves the GROQ_API_KEY from Dataflow SDK secret storage."""
    try:
        from dataflow.dataflow import Dataflow

        dataflow = Dataflow()
        secret_val = dataflow.secret("GROQ_API_KEY")
        if secret_val and isinstance(secret_val, str) and secret_val.strip():
            return secret_val.strip()
    except Exception:
        pass

    return None


class GroqChatService:
    """Service wrapper for interacting with Groq LLM API with streaming support."""

    AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "openai/gpt-oss-120b",
    ]

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        if GROQ_AVAILABLE:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None

    def _stream_http(
        self,
        formatted_messages: List[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        """HTTP-based Server-Sent Events (SSE) streaming fallback using standard urllib."""
        payload = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "DataflowGroqClient/1.0",
        }
        req = urllib.request.Request(
            self.API_URL, data=data_bytes, headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line_bytes in resp:
                    line = line_bytes.decode("utf-8").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            choices = data_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            try:
                err_json = json.loads(error_body)
                err_msg = err_json.get("error", {}).get("message", error_body)
            except Exception:
                err_msg = error_body
            raise RuntimeError(f"Groq API HTTP {e.code} Error: {err_msg}")
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with Groq API: {str(e)}")

    def stream_chat_completion(
        self,
        messages: List[dict],
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Streams tokens from Groq Chat API."""
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        if self.client is not None:
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                return
            except Exception:
                # Fallback to HTTP SSE streaming if groq client call fails
                pass

        yield from self._stream_http(
            formatted_messages=formatted_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
