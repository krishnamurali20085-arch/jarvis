"""Tests for provider-neutral tool orchestration."""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import patch

from jarvis.brain import Brain, MAX_TOOL_CALLS_PER_TURN
from jarvis.config import get_settings
from jarvis.main import confirm_tool_action, run_cli
from jarvis.providers import (
    AssistantTurn,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    ToolCall,
)


class _SequenceProvider:
    name = "test"
    model = "test-model"

    def __init__(self, responses: list[AssistantTurn]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, Any]]] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        self.messages.append(messages)
        return self.responses.pop(0)


class _Response:
    ok = True
    status_code = 200

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def json(self) -> dict[str, Any]:
        return self.data


class _FakeBrain:
    def __init__(self) -> None:
        self.cleared = False

    def reply(self, user_text: str, *, confirm: Any = None) -> str:
        return f"reply to {user_text}"

    def clear_history(self) -> None:
        self.cleared = True


class BrainTests(unittest.TestCase):
    def test_confirmation_is_app_owned_and_tool_result_returns_to_model(self) -> None:
        settings = replace(get_settings(), system_prompt="system")
        provider = _SequenceProvider(
            [
                AssistantTurn(
                    "",
                    (ToolCall("call-1", "safe_action", {"item": "note"}),),
                ),
                AssistantTurn("Completed."),
            ]
        )
        executions: list[bool] = []

        def execute(name: str, arguments: Any, *, confirmed: bool = False) -> dict[str, Any]:
            executions.append(confirmed)
            if not confirmed:
                return {
                    "ok": False,
                    "error": {"code": "confirmation_required", "message": "confirm"},
                }
            return {"ok": True, "data": {"saved": True}}

        brain = Brain(
            settings,
            provider=provider,
            tool_schemas=[],
            tool_executor=execute,
        )
        result = brain.reply("save this", confirm=lambda _name, _args: True)

        self.assertEqual(result, "Completed.")
        self.assertEqual(executions, [False, True])
        self.assertEqual(provider.messages[1][-1]["role"], "tool")
        self.assertIn('"saved": true', provider.messages[1][-1]["content"])

    def test_tool_execution_is_capped_per_turn(self) -> None:
        settings = replace(get_settings(), system_prompt="system")
        calls = [
            AssistantTurn("", (ToolCall(f"call-{index}", "safe_action", {}),))
            for index in range(MAX_TOOL_CALLS_PER_TURN)
        ]
        provider = _SequenceProvider([*calls, AssistantTurn("Finished.")])
        executed: list[str] = []
        brain = Brain(
            settings,
            provider=provider,
            tool_schemas=[],
            tool_executor=lambda name, _args, **_kwargs: (
                executed.append(name) or {"ok": True}
            ),
        )

        self.assertEqual(brain.reply("do five things"), "Finished.")
        self.assertEqual(len(executed), MAX_TOOL_CALLS_PER_TURN)

    def test_openai_adapter_parses_function_calls_without_leaking_key(self) -> None:
        settings = replace(
            get_settings(),
            openai_api_key="not-a-real-key",
            openai_model="test-model",
        )
        provider = OpenAIProvider(settings)
        response = _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"Python"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        with patch("jarvis.providers.requests.post", return_value=response) as post:
            turn = provider.complete(
                [{"role": "user", "content": "search"}],
                [
                    {
                        "name": "web_search",
                        "description": "search",
                        "parameters": {"type": "object"},
                    }
                ],
            )

        self.assertEqual(turn.tool_calls[0].name, "web_search")
        self.assertEqual(turn.tool_calls[0].arguments, {"query": "Python"})
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer not-a-real-key",
        )

    def test_gemini_and_ollama_adapters_parse_normalized_turns(self) -> None:
        settings = replace(
            get_settings(),
            gemini_api_key="not-a-real-key",
            gemini_model="test-model",
        )
        gemini_response = _Response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "web_search",
                                        "args": {"query": "Python"},
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )
        with patch(
            "jarvis.providers.requests.post", return_value=gemini_response
        ) as post:
            gemini_turn = GeminiProvider(settings).complete(
                [{"role": "user", "content": "search"}], []
            )
        self.assertEqual(gemini_turn.tool_calls[0].arguments, {"query": "Python"})
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "not-a-real-key")

        ollama_settings = replace(
            settings,
            ollama_model="test-model",
            ollama_base_url="http://localhost:11434",
        )
        ollama_response = _Response(
            {
                "message": {
                    "content": "Searching.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": "Python"},
                            }
                        }
                    ],
                }
            }
        )
        with patch("jarvis.providers.requests.post", return_value=ollama_response):
            ollama_turn = OllamaProvider(ollama_settings).complete(
                [{"role": "user", "content": "search"}], []
            )
        self.assertEqual(ollama_turn.text, "Searching.")
        self.assertEqual(ollama_turn.tool_calls[0].name, "web_search")

    def test_ollama_uses_a_longer_timeout_for_local_generation(self) -> None:
        settings = replace(get_settings(), request_timeout_seconds=15)
        provider = OllamaProvider(settings)
        response = _Response({"message": {"content": "Ready."}})
        with patch("jarvis.providers.requests.post", return_value=response) as post:
            provider.complete([{"role": "user", "content": "hello"}], [])

        self.assertGreater(post.call_args.kwargs["timeout"], settings.request_timeout_seconds)

    def test_cli_discloses_web_access_and_confirmation_redacts_query_values(self) -> None:
        settings = replace(
            get_settings(),
            llm_provider="ollama",
            ollama_model="test-model",
        )
        brain = _FakeBrain()
        outputs: list[str] = []
        inputs = iter(["hello", "/clear", "/exit"])
        result = run_cli(
            brain,
            settings,
            input_fn=lambda _prompt: next(inputs),
            output_fn=outputs.append,
        )
        self.assertEqual(result, 0)
        self.assertTrue(brain.cleared)
        self.assertTrue(any("Web search" in item for item in outputs))

        confirmation_output: list[str] = []
        allowed = confirm_tool_action(
            "download_resource",
            {"url": "https://example.org/file.pdf?token=secret"},
            input_fn=lambda _prompt: "yes",
            output_fn=confirmation_output.append,
        )
        self.assertTrue(allowed)
        self.assertNotIn("secret", " ".join(confirmation_output))


if __name__ == "__main__":
    unittest.main()
