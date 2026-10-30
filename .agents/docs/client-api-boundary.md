# Client API and network boundary

## Decision

The client is a persistent HTTP service with a text-only OpenAI Chat Completions interface. Apps send a public model name and conversation messages to `/v1/chat/completions`. The endpoint operator supplies network access configuration; artifact revision handling belongs inside discovery. The intended model name is `olmoe`.

Plan 3 implements the HTTP boundary only. `/v1/models` lists the intended model without claiming worker availability. Valid chat requests, including `stream: true`, return HTTP 501 `generation_not_implemented` in the OpenAI error shape before any stream opens. There are no DHT lookups, model downloads, execution or token events.

Plan 4 must connect requests to EveryGPU discovery, inherit the network's pinned artifact revision, find compatible current coverage, select and admit workers, execute inference and return generated text. Streaming requests require OpenAI-compatible server-sent events carrying actual incremental output. Saved bootstrap configuration and execution details must be agreed before that implementation.

## Why

The calling app needs an inference API without understanding worker identities or artifact SHAs. Keeping those concerns inside the service lets the public request shape stay stable when the runtime is connected. An explicit error preserves the meaning of a successful chat response; worker listings and fake tokens cannot stand in for generated text.

Server startup, HTTP behavior and integration remain unverified. Accepting a streaming flag establishes request compatibility only.
