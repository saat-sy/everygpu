# Plan 5: Repository logging

## Goal and status

Implemented, consolidated into monitoring/logging/ and reviewed by Codex; changes remain uncommitted. Make it possible to tell what the client, worker and model partitioner are doing, where an operation stalled, and why it failed.

## Implementation

The three Python entrypoints share standard-library logging under monitoring/logging/: constants.py holds constants, and logger.py holds configuration, formatting, context, timing and progress helpers. Package initializers stay empty. Application modules name their logger instances logger. Operational events use logging; command results remain on stdout. Client discovery, worker execution and streaming generation carry request/session context. Inspect the code as the implementation reference; the application state described in context.md does not cover these paths.

## Behavior

- Use Python's standard logging with one shared configuration used by all three entrypoints; no additional logging dependency. Configure handlers only at application startup.
- Default to INFO on stderr. Each event includes a UTC timestamp, level, module, readable message and applicable key=value context. Keep command results, verification reports and inspect output on stdout.
- Use one override, EVERYGPU_LOG_LEVEL, accepting DEBUG, INFO, WARNING, ERROR and CRITICAL. Reject invalid values before starting work.
- Keep third-party logs at WARNING by default, preserving their errors. DEBUG also enables libp2p diagnostics. Integrate Uvicorn with the configuration so events do not get duplicate handlers or conflicting formats.
- INFO shows operation starts and finishes, network joins, checkpoint pin, placement and loading summaries, worker readiness, route selection, admission, first generated token, generation completion and shutdown. Show elapsed time and counts where useful. Emit bounded loading progress so a long startup remains visible.
- DEBUG adds discovery decisions, per-worker admission, session activity, forward/relay steps and durations. Do not emit per-token or per-layer events at INFO. Durations measure observed operation time; do not introduce GPU synchronization or claim kernel timings.
- WARNING describes degraded operation, timeouts and cleanup failures; ERROR describes failed operations. Unexpected failures include tracebacks. Log an error once at the boundary responsible for reporting it; normal cancellation is not an error.
- Correlate a chat request with its session across the asyncio/Trio bridge using explicit context. Include the existing session ID in worker logs and peer/stage/step identifiers where available. Use existing execution metadata without changing the wire protocol.
- Do not log prompts, generated text, token IDs, tensors, credentials or raw request/frame bodies.

Example usage after implementation:

```sh
python -m worker --new-network
EVERYGPU_LOG_LEVEL=DEBUG python -m client --bootstrap <address>
EVERYGPU_LOG_LEVEL=DEBUG python -m model_partitioner verify ./partitions
```

Example event shape, with illustrative values:

```text
2026-11-01T21:40:00Z INFO client.network.admission worker admitted request_id=abc session_id=def peer=xyz stage=0 elapsed_ms=12
```

## Implementation and acceptance

1. Add shared configuration, initialize it in the entrypoints and integrate library logging. Convert operational prints while retaining deliberate command output.
2. Instrument startup, discovery, loading, HTTP request lifecycle, route admission, generation, worker execution and cleanup. For streaming requests, record completion, failure or disconnect when the stream actually ends, separately from HTTP headers being sent. Include final outcome, elapsed time and token counts.
3. Verify with disposable offline fixtures and captured stdout/stderr: default and DEBUG filtering, invalid configuration, no duplicate events, retained command output, session correlation across the thread bridge, and success/failure/disconnect cleanup. Inspect exception paths for payload leakage. Run Black, syntax and whitespace checks. Do not add test files, download models, run GPUs or perform uploads.

Done means all three entrypoints use consistent logging and the instrumented fixture flow explains startup through completion or failure. Report which checks ran and which real network/GPU paths remain unverified. Leave changes uncommitted and stop for review after this plan.

## Verification and limits

Codex independently reviewed the full diff and consolidated logging package. Ten synthetic ASGI cases passed against the consolidated code for success, expected/unexpected failures, body disconnect and streaming disconnect under ASGI 2.3/2.4. Filtering, invalid configuration, repeated setup and real Trio request/session context transfer passed. Black, Pyflakes and AST parsing passed across all 88 Python files. Helper references, logger naming in 26 application modules, empty initializers, obsolete-path removal and whitespace checks passed. Review fixtures used a disposable Python 3.12 environment, not the declared project installation. Focused source-extracted method fixtures also cover phase-start ordering and manifest/tokenizer cache suppression.

Claude reports additional local libp2p subprocess, CPU model-fixture, relay, partition and checksum checks; those were not independently rerun by Codex. Real GPU, Hub uploads, multi-machine networking and Windows/macOS behavior remain unverified. Libp2p can emit frequent DHT stream-read warnings at the approved WARNING baseline; their exact exception cause is unconfirmed.

## Learnings

- A streaming response runs after the endpoint returns: bind logging context around body delivery and explicit generator cleanup.
- Report text chunks separately from token counts; token totals are known only when generation supplies them.
- Exercise upload-body and stream disconnects, unexpected exception causes, and phase-start visibility; successful requests and static checks alone miss these logging defects.
