# Plan 3: OpenAI-compatible HTTP server

## Goal

Provide a persistent local HTTP server that accepts the simplest text-only OpenAI Chat Completions request shape. Discovery, inference and generated-token streaming remain deferred.

## Status

Implemented by Claude. Codex's initial review passed AST parsing, Black and Pyflakes for all 11 client Python files, empty-initializer checks, TOML parsing and whitespace checks. Claude reports compilation and Black passing for the latest startup, body-limit and error-handler fixes; independent revalidation of those fixes is pending. Claude also exercised address parsing as a pure function. Server startup, HTTP requests, declared installation and integration remain unverified. Discovery, inference and token streaming are not implemented.

## Requirements

- `python -m client` starts the server at `127.0.0.1:42731` without bootstrap addresses, model revision, a GPU or model downloads.
- Optional `EVERYGPU_HOST=host:port` overrides the listening address and port. Print the listening address at startup; report invalid configuration, occupied ports and startup failure explicitly rather than choosing another port. Ctrl+C shuts down cleanly without a traceback.
- `GET /v1/models` lists `olmoe` in OpenAI's model-list shape. Listing denotes the intended model, not available workers.
- `POST /v1/chat/completions` accepts a model name, a nonempty list of text messages with `system`, `user` and `assistant` roles, and optional boolean `stream` (default false). Conversation history is supported as request input.
- Reject malformed requests, unknown models and unsupported options, including tools, images, audio and generation settings. Reject request bodies over 1 MiB with HTTP 413. All errors, including unexpected failures (HTTP 500), use the OpenAI-style `error` object with `message`, `type`, `param` and `code`.
- Every valid chat request returns HTTP 501 with code `generation_not_implemented`, before opening a stream. Accept both `stream: true` and `stream: false`; fabricate no assistant text or token events.
- Install from a separate `client` dependency group containing only required server dependencies. No Torch, Safetensors, model or DHT imports are needed in the client.

## Workflow

```bash
uv sync --only-group client
uv run --group client python -m client
```

Once the client dependencies are installed, the direct entrypoint is `python -m client`.

```bash
EVERYGPU_HOST=0.0.0.0:42731 python -m client
```

This binds all network interfaces instead of loopback. Requests use the server machine's reachable address and configured port.

```http
POST /v1/chat/completions
Content-Type: application/json
```

```json
{
  "model": "olmoe",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": true
}
```

The request is validated, then returns the explicit not-implemented error. Invalid requests return HTTP 400; unknown models return HTTP 404. No request performs discovery or model execution.

The deferred root dependency-resolution blocker (`accelerate>=1.16`) can still prevent the documented `uv sync` command. Do not change deferred dependency constraints or claim a declared installation passes without verification.

## Steps

1. Add the client dependency group, minimal feature-organized package and server entrypoint.
2. Implement environment-based bind configuration, model listing, text request validation and consistent error responses.
3. Review the full diff, run formatting and static checks, and update the runbook and implementation status.

## Decisions

This plan sets up the HTTP boundary only. Accepting a streaming request does not establish streaming inference. Actual incremental output uses OpenAI-compatible server-sent events when execution is connected.

Before inference can work, the deferred work must join discovery using saved network configuration, inherit the network's pinned artifact revision, find compatible current candidate coverage, select and admit workers, and execute the request. No bootstrap or revision information belongs in callers' inference requests. These are prerequisites for Plan 4's end-to-end demonstration, not implemented behavior.

Do not add discovery inspection, bootstrap configuration, authentication, streaming token generators, tokenization, tensor loading, HTTP API features beyond the approved subset or extra CLI commands in this plan. Keep existing worker code unchanged unless a concrete server requirement makes a shared change necessary.

## Verification

- Format with Black and run static syntax, lint and diff checks appropriate to changed files.
- Review routes, request validation, response shapes, bind parsing, error handling and dependencies against the approved contract.
- Do not run the server, HTTP requests, fixture tests or integration tests; the user defers runtime verification. Do not add automated tests.
- Report static check results separately from unverified runtime, streaming, discovery and inference behavior. Do not download model weights, run GPUs, commit or change history.

## Learnings

### Codex (planner)

- Read prior plans' Learnings before starting a plan. Apply the recorded workflow and Python conventions to the first handoff and review.
- Present exact commands, defaults, request/response examples and verification limits, then wait for approval before implementation or handoff. Authorization to use Claude does not replace approval of the behavior contract.
- Separate the calling app from the endpoint operator. Callers send a model name and text messages; service configuration supplies network access, and discovery inherits the network's pinned revision. Do not require callers to manage bootstrap peers or artifact SHAs.
- Give one concrete default and one override mechanism. The approved address is `127.0.0.1:42731`, configurable through `EVERYGPU_HOST`; avoid competing example ports.
- State what the endpoint actually does. A server returning 501 has not joined EveryGPU or processed inference. Record deferred discovery, admission, execution and streaming explicitly as Plan 4 work.
- Distinguish accepting `stream: true` from delivering incremental tokens. Until execution exists, return the explicit error before opening an SSE stream; do not fabricate successful completions.
- Carry verification limits into the handoff and final report. Pure-function execution is separate from static checks; neither establishes server behavior. Label implementer-reported results and independent checks, and do not extend checks on an intermediate tree to later edits.

### Claude (implementer)

- Apply the Python rules in the first draft: module-based imports (including third-party), constants and data classes in `utils/`, rendering apart from error data, thin `__main__.py` with startup in the feature package. Review rounds were spent on these.
- Check JSON field types before set membership: list or object roles must produce a validation error rather than an unhandled TypeError.
- Validate address syntax strictly: require paired IPv6 brackets, reject unbracketed IPv6, and bracket IPv6 hosts in printed URLs.
- Preserve `HTTPException` headers (such as `Allow` on 405) when re-rendering errors.
- Close a pre-bound socket on every exit path (`with sock`), and bind it before `uvicorn` so port errors are explicit.
- Under a static-only mandate, do not install or run the server. A scratch venv for reading dependency signatures is acceptable; delete it, and state that declared minimum versions were not checked against older releases.
- When a peer asks to "ping" a review, send it with `ping send <exact registered name>`; list names with `ping list` first.
