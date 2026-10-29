# Plan 5: Distributed forward pass

## Goal

Run input from the coordinator through every worker stage and return the final output to the coordinator.

## Status

Planned. Implementation has not started.

## Requirements

- Execute the assigned layer ranges in order.
- Relay intermediate activations through the coordinator.
- Account for N stages while initially using two GPUs.

## Steps

1. Provide coordinator-triggered test input.
2. Execute the complete worker route.
3. Return the final stage output and verify correctness.

## Decisions

This plan covers a forward pass. A client and autoregressive text generation are introduced later.

## Verification

- Demonstrate a real forward pass through Colab and Kaggle.
- Compare the distributed output with an equivalent reference execution using an appropriate numerical tolerance.

## Learnings

None recorded.

