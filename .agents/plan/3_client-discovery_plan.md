# Plan 3: Client tooling and request-time discovery

## Goal

A client accepts an OLMoE inference request, queries the DHT and reports candidate workers for the requested model pieces.

## Status

Planned. Implementation has not started. Requires Plan 2.

## Requirements

- Provide client tooling accepting a prompt and pinned model identity, with bootstrap access to the DHT.
- Discover workers when a request arrives rather than relying on a hard-coded worker inventory.
- Resolve candidate coverage for shared layers, all potentially selected routed experts and embedding/final units.
- Distinguish compatible, current advertisements from missing, expired or incompatible coverage. Show discovery results or an explicit discovery failure.
- Keep requesting inference independent of contributing GPU capacity. No central client-registration service is required.

## Steps

1. Agree on the client invocation, request input and discovery-result output before implementation.
2. Connect requests to DHT lookups using Plan 2's model-piece identities and records.
3. Report candidate coverage and missing pieces in a form the Plan 4 session selection can use.

## Decisions

This plan delivers discovery, not generated text. Candidate coverage does not reserve capacity or select the final execution route. Worker selection, direct admission and execution belong to Plan 4. The client need not download a global inventory of GPUs.

## Verification

- Submit a request and discover the compatible workers advertised through Plan 2.
- Verify a subsequent request observes refreshed discovery results when available workers change.
- Verify missing pieces, expired records and incompatible revisions do not appear as complete compatible coverage.
- Demonstrate the workflow with small fixtures; do not claim inference has run.

## Learnings

None recorded.
