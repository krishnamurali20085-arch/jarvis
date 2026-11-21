# JARVIS Controlled Internet Access Design

**Status:** Approved by user

## Goal and scope

Give JARVIS useful public-web research and controlled download capabilities,
usable from a text CLI through the configured OpenAI, Gemini, or Ollama model.
This is the first bounded step toward broader agent behavior; capabilities are
added through explicit tools and policies, not an unrestricted grant of
network, shell, or filesystem access.

The initial milestone includes:

- A provider-neutral `brain.py` that manages conversation context and bounded
  calls to registered tools.
- A text `main.py` that displays answers, citations, provider/privacy status,
  and asks the user to confirm gated downloads.
- Public web search and bounded webpage reading, reusing the existing web tools.
- A controlled download operation with automatic RAG indexing for exact,
  license-verified registry entries and review staging for other sources.
- A verified source registry for exact, license-reviewed educational resources.

The desktop chat UI, voice loop, broad desktop autonomy, and model fine-tuning
are separate later subprojects. This milestone does not crawl the whole web or
download every dataset it can discover.

## Permission and trust model

The model has no direct sockets, browser scripting, arbitrary Python execution,
or shell access. It can request only registered tool calls. The application
validates every call and owns all user-confirmation decisions; model-supplied
arguments can never assert that the user approved an action.

Public search and bounded webpage reading do not require confirmation.
Downloads are handled as follows:

1. A source-registry entry identifies a specific resource or tightly bounded
   path, verified license and attribution, allowed media types, and size limit.
   Only a matching entry may download and index automatically.
2. Any other eligible public HTTPS URL requires explicit confirmation in the
   CLI. Its file is saved to staging for review and is not indexed.
3. Unknown or absent license information never permits indexing or training.
   Public accessibility is not treated as permission to reuse a dataset.
4. Model-requested downloads matching a verified registry entry may be
   automatically indexed into local RAG, but never executed, unpacked, or used
   for model-weight training. Other downloads remain in staging. The existing
   fixed-corpus Python seed command remains authorized.

`JARVIS_AUTO_INDEX_APPROVED_SOURCES` controls this behavior and defaults to
`true`; it applies only to exact verified registry entries. Setting it to
`false` sends even those downloads to staging without indexing.

Automatic eligibility is resource-specific, not blanket trust in a domain.
Unsupported formats, archives, installers, executables, local/private network
destinations, and redirect targets that fail validation are rejected.

## Runtime and data flow

`brain.py` selects the configured provider and supplies it with the registered
tool schemas. The model may request search, webpage reading, or download. The orchestrator
validates tool names and arguments, enforces a maximum of five tool calls per
assistant turn, executes permitted calls, and returns bounded results to the
model as untrusted evidence. A download matching a verified registry entry is
automatically indexed into the local RAG corpus when the setting is enabled;
all other downloads remain in staging for review. It must not treat fetched
page content as instructions. Responses that rely on fetched or indexed
content include source URLs.

The CLI mediates confirmation. A future desktop UI will reuse the same
confirmation contract and tool registry. With OpenAI or Gemini selected,
prompts, conversation context, and tool results sent to the model leave the
computer; the CLI must disclose the selected provider. Tool definitions sent
with requests also leave the computer. Ollama keeps model inference local,
while public web search still contacts the selected search service.

## CLI use and configuration

From the project root, run `.\.venv\Scripts\python.exe -m jarvis.main`. Select
`openai`, `gemini`, or `ollama` with `JARVIS_LLM_PROVIDER`; configure its API key
and model in `.env`, or ensure the selected Ollama service is running. Set
`JARVIS_AUTO_INDEX_APPROVED_SOURCES=false` to disable automatic indexing even
for verified registry entries. The default enables it only for those entries.

Non-registry downloads go under `data/download_staging/`, separate from
`data/educational/`. Verified registry downloads are validated in staging and,
on successful ingestion, retained with their indexable source files under
`data/educational/documents/`. A local manifest records the URL, retrieval
time, content hash, detected media type, source-registry entry if any, declared
license and attribution when known, and whether the download was automatic or
confirmed. If validation or indexing fails, the file remains in staging and
the assistant reports the error.

## Network and file limits

- Require HTTPS for fetched and downloaded URLs.
- Revalidate each redirect and reject localhost, loopback, link-local, private,
  and other non-public destinations.
- Reuse the configured request timeout and enforce a finite response-size
  limit of 25 MiB for this milestone.
- Allow only passive document/data formats: PDF, TXT, Markdown, HTML, CSV, and
  JSON. Validate response media type and file extension; reject archives and
  executable formats.
- Do not automatically extract archives or follow links found in downloaded
  documents.
- Surface timeout, HTTP, validation, and disk errors to the user without
  crashing the CLI loop. Never log API keys or authorization headers.

## Fine-tuning boundary

Downloading is not training. The current educational knowledge system uses
local RAG and does not change model weights. Fine-tuning remains deferred until
the user selects a local base model, hardware target, and appropriately licensed
curated Q&A data. Raw web pages, arbitrary scraped datasets, and unreviewed
downloads are not training inputs.

## Verification

Automated tests use mocked providers and network responses. They cover:

- Provider selection and conversation/tool orchestration.
- Tool-call budget enforcement and rejection of unknown tools.
- Confirmation required for non-registry downloads and impossible to spoof via
  model arguments.
- Redirect validation, including redirects to private/local destinations.
- Oversize, unsupported, mismatched-media-type, and failed downloads.
- Automatic indexing only for exact verified registry entries; other downloads
  remain staged and are not indexed.
- Existing offline RAG retrieval and source citations.

Acceptance requires the CLI to answer a current-information request using
registered web tools, cite fetched sources, automatically index an approved
registry resource when enabled, require confirmation and keep an unregistered
download isolated in staging, and leave model weights unchanged. There must be
no live-web dependency in the automated test suite.
