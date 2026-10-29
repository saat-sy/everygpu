# Plan 4: Connect workers and smoke-test loading

## Goal

Connect remote workers to the coordinator and verify coordinator-directed layer downloads.

## Status

Planned. Implementation has not started.

## Requirements

- The coordinator instructs each worker which model and layer range to load.
- Workers retrieve the corresponding Hugging Face artifacts and report readiness.
- Validate the setup on Colab and Kaggle.

## Steps

1. Connect workers to the coordinator.
2. Send layer-range assignments and observe loading.
3. Run and document the remote loading smoke test.

## Decisions

All worker communication goes through the coordinator. This plan verifies loading and readiness rather than a full forward pass.

## Verification

- Start both remote workers and assign their ranges from the coordinator.
- Verify each worker loads the requested range and the complete route becomes ready.

## Learnings

None recorded.

