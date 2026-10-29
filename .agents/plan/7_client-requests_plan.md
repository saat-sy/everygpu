# Plan 7: Client inference requests

## Goal

Run a client's inference request through the coordinator and workers and return the answer.

## Status

Planned. Implementation has not started.

## Requirements

- Accept a request from a registered client.
- Execute it through the configured worker route.
- Generate an answer and return it to the requesting client.
- Keep each request's generation state separate.

## Steps

1. Connect client requests to coordinator execution.
2. Complete the token-generation cycle across the workers.
3. Return the completed answer to the originating client.

## Decisions

All traffic passes through the coordinator. This plan requires one active request; streaming and multi-client queueing follow separately.

## Verification

- Submit a real client prompt and receive a generated answer using both remote GPUs.
- Verify the answer is delivered to the originating client.

## Learnings

None recorded.

