# Plan 4: First end-to-end P2P OLMoE inference

## Goal

One client discovers and selects workers, sends a prompt through partitioned OLMoE execution and receives generated text.

## Status

Planned. Implementation has not started. Uses Plan 2's worker records and Plan 3's HTTP server. Request-time discovery is deferred from Plan 3 and must be implemented before route selection and execution.

## Requirements

- Select a complete compatible stage route and expert placements from discovered candidates; confirm capacity directly with workers before execution.
- Keep the client responsible for its session and stage placement stable during generation. Workers independently accept or reject work.
- Execute prefill and autoregressive decode with per-session, per-layer KV state on stage workers.
- Use the model's learned router for expert selection. Execute local experts locally and remote expert calls directly, then combine results according to OLMoE's numerical operations.
- Send stage-boundary activations directly between workers. The client does not relay each activation or approve each expert call.
- Give tokenization, embeddings, the output head and sampling explicit owners. Preserve operation ordering, positions and request identities through the token loop.
- Return generated text to the requesting client and release session resources on completion or failure. A lost worker may fail the request explicitly; recovery is not required.

## Steps

1. Agree on the runnable end-to-end workflow, saved bootstrap configuration, worker assignments, transport/serialization, capacity confirmation and generation ownership. Check connectivity between the selected machines.
2. Connect the HTTP server to request-time discovery: inherit the network's pinned artifact revision, find current compatible coverage for every required unit, and report explicit failures for missing coverage. Add client-side route selection and worker admission to these discovery results.
3. Implement OLMoE stage/expert execution from the existing partitions, direct communication and the session's prefill/decode lifecycle.
4. Connect generated tokens to the HTTP API, including OpenAI-compatible server-sent events for streaming requests. Run a controlled numerical comparison and then the real distributed prompt-to-text demonstration in an approved environment.

## Decisions

Start with one active request under perfect conditions. The demonstration must use multiple workers, cross a stage boundary and execute selected experts remotely; two GPU workers are sufficient if assignments fit. This does not require separate GPU fleets for stages and experts.

Transport, reachable deployment and numerical tolerance must be settled before implementation. Callers send text conversations and a public model name; bootstrap configuration and artifact revision handling belong inside the service. Streaming API responses are required; streaming UI, multi-client optimization, replication, recovery, a benchmark suite and Muse execution are outside this plan.

## Verification

- Compare distributed prefill and multiple cached decode steps with equivalent reference execution using identical weights and inputs, documented dtype and numerical tolerance. Generated text alone is insufficient evidence of correctness.
- Send a real prompt from the client, discover and admit workers, and receive generated text from real OLMoE execution across multiple machines.
- Verify direct stage transfers and remote expert execution actually occur; confirm that session KV state is reused across decode steps.
- Verify incomplete coverage or rejected capacity prevents execution, and completion/failure releases reserved resources.
- Record the artifact revision, hardware, placement, commands and observed results. Keep fixture checks, real execution and remaining limitations distinct. Do not download or run real models in this checkout.

## Learnings

None recorded.
