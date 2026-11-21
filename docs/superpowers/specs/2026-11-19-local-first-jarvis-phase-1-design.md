# Local-First JARVIS Assistant — Phase 1 Design

## Goal and scope

Build the foundation of a modular Python 3.10+ desktop assistant that can be
extended in separate increments. This design covers only Phase 1: configuration,
local memory, and a safe registry of system, web, and workspace-file tools.
LLM orchestration, conversation interfaces, voice interaction, and a desktop UI
are deferred to later phases.

Phase 2 will keep the assistant brain independent of its presentation layer so
the same conversation and tool-calling logic can serve both a keyboard interface
and a ChatGPT/Gemini-style desktop chat application. The desktop app shell and
its interaction design will be selected when Phase 2 is planned.

## Project layout

```text
jarvis/
  __init__.py
  config.py
  memory.py
  tools/
    __init__.py
    system_tools.py
    web_tools.py
    file_tools.py
.env.example
requirements.txt
.gitignore
tests/
```

The package boundaries are intentional: configuration and persistence have no
dependency on an LLM; tool implementations expose narrowly scoped Python
functions; and the registry provides the stable, JSON-compatible interface that
Phase 2 can map to provider-specific function-calling schemas.

## Components and contracts

- **Configuration:** Load `.env` values into validated typed settings. Keep API
  provider selection and credentials configurable without embedding secrets.
  Invalid settings must produce clear errors rather than silently selecting
  unsafe values.
- **Memory:** Use the standard-library SQLite module and a local database.
  Provide focused operations for user facts/preferences, notes, reminders, and
  to-do items. Initialize the schema safely and preserve data across process
  restarts.
- **Tool registry:** Define each tool with a stable name, description, JSON
  argument schema, and callable. Dispatch only registered functions; never
  interpret model output as shell commands or Python code.
- **System tools:** Use portable APIs for system metrics and narrowly scoped
  OS-aware operations for app/browser launching, screenshots, and supported
  hardware controls. Unsupported platform capabilities return an explicit
  result.
- **Web tools:** Provide live search and weather lookup with network failures
  reported as tool errors. URL opening is restricted to validated web URLs.
- **File tools:** Restrict reads, searches, writes, and organization to a
  configured workspace. Resolve and verify paths before access so traversal and
  symlink escapes cannot cross that boundary.

## Safety and errors

File overwrite and move/organization operations must not execute until a
trusted interaction layer has obtained explicit user confirmation. A tool
returns a confirmation-required result; it does not treat an LLM argument as
user approval and does not prompt from inside the tool. Destructive shell
commands are not exposed.

Tool and platform failures are returned explicitly with a useful message and
must not terminate the main application loop that will be added later. No broad
exception handling may turn an error into a success-shaped response.

## Dependencies and portability

Use Python's standard library for configuration parsing where practical,
SQLite, paths, and operating-system inspection. Add only focused dependencies
for portable hardware metrics and web access. OS-specific integrations are
isolated behind tool functions and documented; they must not prevent importing
the core package on another platform. `.env.example` documents settings without
containing real credentials.

## Verification

Phase 1 verification covers:

1. Importing the package and registry with valid example configuration.
2. Creating, closing, and reopening the SQLite database while retaining facts,
   notes, reminders, and to-do items.
3. Rejecting file paths outside the configured workspace, including traversal
   attempts.
4. Confirming that unregistered tools cannot dispatch and that
   confirmation-gated operations do not mutate files without trusted approval.
5. Reporting unsupported platform controls and web/network errors without
   uncaught exceptions.

Run the local tests with `python -m unittest discover -s tests -v`.

## Deferred work

Provider-specific LLM requests, multi-turn conversation history, model-driven
tool orchestration, CLI/app event loops, the desktop chat UI, wake-word
detection, speech-to-text, and text-to-speech are not part of this Phase 1
implementation.
