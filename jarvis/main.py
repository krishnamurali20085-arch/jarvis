"""Text CLI entry point for JARVIS."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from jarvis.brain import Brain, ConfirmationHandler
from jarvis.config import ConfigurationError, Settings, get_settings
from jarvis.providers import LLMError


class ConversationBrain(Protocol):
    def clear_history(self) -> None: ...

    def reply(
        self, user_text: str, *, confirm: ConfirmationHandler | None = None
    ) -> str: ...


def _safe_confirmation_preview(arguments: dict[str, Any]) -> str:
    preview = dict(arguments)
    url = preview.get("url")
    if isinstance(url, str):
        parsed = urlsplit(url)
        redacted_query = [
            (key, "[redacted]") for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        preview["url"] = urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urlencode(redacted_query), "")
        )
    serialized = json.dumps(preview, ensure_ascii=False, default=str)
    return serialized if len(serialized) <= 500 else serialized[:497] + "..."


def confirm_tool_action(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> bool:
    output_fn(f"JARVIS requests confirmation for {tool_name}.")
    output_fn(_safe_confirmation_preview(arguments))
    try:
        answer = input_fn("Allow this action? [y/N]: ").strip().casefold()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def run_cli(
    brain: ConversationBrain,
    settings: Settings,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    provider_names = {
        "openai": f"OpenAI ({settings.openai_model})",
        "gemini": f"Gemini ({settings.gemini_model})",
        "ollama": f"Ollama ({settings.ollama_model})",
    }
    output_fn(f"JARVIS is ready. Provider: {provider_names[settings.llm_provider]}.")
    if settings.llm_provider != "ollama":
        output_fn(
            "Privacy notice: prompts, conversation context, tool definitions, and "
            "tool results sent to this provider leave this computer."
        )
    output_fn(
        "Web search and webpage-reading tools contact public internet services "
        "even when Ollama is selected."
    )
    output_fn("Type /clear to reset this conversation or /exit to quit.")

    while True:
        try:
            message = input_fn("You> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nJARVIS session ended.")
            return 0
        command = message.strip().casefold()
        if command in {"/exit", "/quit"}:
            output_fn("JARVIS session ended.")
            return 0
        if command == "/clear":
            brain.clear_history()
            output_fn("Conversation cleared.")
            continue
        if not message.strip():
            continue

        try:
            response = brain.reply(
                message,
                confirm=lambda name, args: confirm_tool_action(
                    name, args, input_fn=input_fn, output_fn=output_fn
                ),
            )
        except (LLMError, ConfigurationError, ValueError) as exc:
            output_fn(f"JARVIS: I couldn't complete that request: {exc}")
            continue
        output_fn(f"JARVIS: {response}")


def main() -> int:
    try:
        settings = get_settings()
        brain = Brain(settings)
    except (ConfigurationError, LLMError) as exc:
        print(f"JARVIS could not start: {exc}", file=sys.stderr)
        return 1
    return run_cli(brain, settings)


if __name__ == "__main__":
    raise SystemExit(main())
