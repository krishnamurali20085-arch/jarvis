"""Provider adapters for OpenAI, Gemini, and Ollama chat APIs."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

import requests

from jarvis.config import Settings


class LLMError(RuntimeError):
    """Raised when a configured model provider cannot complete a chat request."""


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Any


@dataclass(frozen=True)
class AssistantTurn:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()


class ChatProvider(Protocol):
    name: str
    model: str

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn: ...


def _post_json(
    provider_name: str,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError(f"{provider_name} request failed due to a network error.") from exc
    if not response.ok:
        raise LLMError(
            f"{provider_name} request failed with HTTP {response.status_code}."
        )
    try:
        result = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise LLMError(f"{provider_name} returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise LLMError(f"{provider_name} returned an unexpected response.")
    return result


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _parse_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_openai_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
    if not isinstance(raw_calls, list):
        return ()
    calls = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        calls.append(
            ToolCall(
                call_id=str(raw.get("id") or uuid.uuid4().hex),
                name=function["name"],
                arguments=_parse_arguments(function.get("arguments", {})),
            )
        )
    return tuple(calls)


def _provider_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


class OpenAIProvider:
    name = "OpenAI"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is required when the OpenAI provider is selected.")
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.timeout = settings.request_timeout_seconds

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        normalized_messages = []
        for message in messages:
            item: dict[str, Any] = {
                "role": message["role"],
                "content": message.get("content", ""),
            }
            if message.get("tool_call_id"):
                item["tool_call_id"] = message["tool_call_id"]
                item["name"] = message["name"]
            if message.get("tool_calls"):
                item["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.arguments, ensure_ascii=False
                            ),
                        },
                    }
                    for call in message["tool_calls"]
                ]
            normalized_messages.append(item)

        payload = {
            "model": self.model,
            "messages": normalized_messages,
            "tools": _provider_tools(tools),
            "tool_choice": "auto",
            "max_completion_tokens": 1200,
        }
        result = _post_json(
            self.name,
            "https://api.openai.com/v1/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
        )
        try:
            message = result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI returned no assistant message.") from exc
        if not isinstance(message, dict):
            raise LLMError("OpenAI returned a malformed assistant message.")
        return AssistantTurn(
            text=_content_text(message.get("content")),
            tool_calls=_parse_openai_tool_calls(message.get("tool_calls")),
        )


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    supported = {
        "type",
        "description",
        "properties",
        "required",
        "enum",
        "items",
        "minimum",
        "maximum",
    }
    result = {key: value for key, value in schema.items() if key in supported}
    if isinstance(result.get("properties"), dict):
        result["properties"] = {
            name: _gemini_schema(value)
            for name, value in result["properties"].items()
            if isinstance(value, dict)
        }
    if isinstance(result.get("items"), dict):
        result["items"] = _gemini_schema(result["items"])
    return result


def _gemini_contents(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_instruction = ""
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system_instruction = str(message.get("content", ""))
            continue
        if role == "tool":
            try:
                tool_response = json.loads(message.get("content", "{}"))
            except json.JSONDecodeError:
                tool_response = {"result": message.get("content", "")}
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": message["name"],
                                "response": tool_response,
                            }
                        }
                    ],
                }
            )
            continue

        parts: list[dict[str, Any]] = []
        content = _content_text(message.get("content"))
        if content:
            parts.append({"text": content})
        if role == "assistant":
            for call in message.get("tool_calls", ()):
                parts.append(
                    {
                        "functionCall": {
                            "name": call.name,
                            "args": (
                                call.arguments
                                if isinstance(call.arguments, dict)
                                else {}
                            ),
                        }
                    }
                )
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    return system_instruction, contents


class GeminiProvider:
    name = "Gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is required when the Gemini provider is selected.")
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        self.timeout = settings.request_timeout_seconds

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        system_instruction, contents = _gemini_contents(messages)
        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": _gemini_schema(tool["parameters"]),
                        }
                        for tool in tools
                    ]
                }
            ],
            "toolConfig": {
                "functionCallingConfig": {"mode": "AUTO"},
            },
            "generationConfig": {"maxOutputTokens": 1200},
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(self.model, safe='')}:generateContent"
        )
        result = _post_json(
            self.name,
            url,
            payload,
            {"x-goog-api-key": self.api_key},
            self.timeout,
        )
        try:
            parts = result["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Gemini returned no assistant content.") from exc
        if not isinstance(parts, list):
            raise LLMError("Gemini returned malformed assistant content.")

        text_parts = []
        calls = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            function_call = part.get("functionCall")
            if isinstance(function_call, dict) and isinstance(
                function_call.get("name"), str
            ):
                calls.append(
                    ToolCall(
                        call_id=uuid.uuid4().hex,
                        name=function_call["name"],
                        arguments=function_call.get("args", {}),
                    )
                )
        return AssistantTurn(text="".join(text_parts), tool_calls=tuple(calls))


def _ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        item: dict[str, Any] = {
            "role": message["role"],
            "content": _content_text(message.get("content")),
        }
        if message.get("tool_call_id"):
            item["name"] = message["name"]
        if message.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "function": {
                        "name": call.name,
                        "arguments": (
                            call.arguments if isinstance(call.arguments, dict) else {}
                        ),
                    }
                }
                for call in message["tool_calls"]
            ]
        result.append(item)
    return result


class OllamaProvider:
    name = "Ollama"

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.timeout = max(settings.request_timeout_seconds, 120.0)

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        result = _post_json(
            self.name,
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": _ollama_messages(messages),
                "tools": _provider_tools(tools),
                "stream": False,
                "options": {"num_predict": 1200},
            },
            {},
            self.timeout,
        )
        message = result.get("message")
        if not isinstance(message, dict):
            raise LLMError("Ollama returned no assistant message.")
        return AssistantTurn(
            text=_content_text(message.get("content")),
            tool_calls=_parse_openai_tool_calls(message.get("tool_calls")),
        )


def create_provider(settings: Settings) -> ChatProvider:
    if settings.llm_provider == "openai":
        return OpenAIProvider(settings)
    if settings.llm_provider == "gemini":
        return GeminiProvider(settings)
    return OllamaProvider(settings)
