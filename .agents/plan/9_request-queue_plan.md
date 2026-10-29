# Plan 9: Multi-client request queue

## Goal

Serve multiple clients sequentially using a request queue.

## Status

Planned. Implementation has not started.

## Requirements

- Allow additional client requests to wait while one generation executes.
- Start the next queued request when the active request finishes.
- Keep request state and streamed output isolated by client.

## Steps

1. Add coordinator-side queueing.
2. Connect queued requests to the generation lifecycle.
3. Verify sequential service with multiple clients.

## Decisions

One generation is active at a time. Efficient simultaneous processing belongs to Plan 10.

## Verification

- Submit overlapping requests from at least two clients.
- Verify the second waits, then runs, and both receive their own complete streamed answers.

## Learnings

None recorded.

