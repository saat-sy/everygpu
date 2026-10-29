# Plan 1: Model partitioning

## Goal

Partition OLMoE and Muse Spark Open into downloadable layer-range artifacts.

## Status

Planned. Implementation has not started.

## Requirements

- Support OLMoE and Muse Spark Open only.
- Partition by consecutive layer ranges and account for model components outside those ranges.
- Store all generated shards in the owner's Hugging Face account.
- Account for N stages rather than a fixed two-stage split.

## Steps

1. Prepare the project setup needed for partitioning.
2. Establish model partitioning and the information needed to load each range.
3. Produce and upload shards for the supported checkpoints.

## Decisions

Exact checkpoint IDs, destination repositories and runtime compatibility are resolved during implementation. Support is a target, not a validated capability.

## Verification

- Verify each supported checkpoint's artifacts cover the model without missing or duplicated weights.
- Verify uploaded shards can be retrieved and loaded for their assigned ranges.

## Learnings

None recorded.

