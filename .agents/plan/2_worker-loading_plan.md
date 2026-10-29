# Plan 2: Worker setup and warm layers

## Goal

Run worker code on a remote GPU that downloads its assigned layer range and keeps it loaded.

## Status

Planned. Implementation has not started.

## Requirements

- Accept a model and consecutive layer-range assignment.
- Download assigned artifacts from the owner's Hugging Face account.
- Keep the assigned model portion loaded and ready for execution.
- Support the available Colab and Kaggle sessions.

## Steps

1. Provide a runnable worker for the remote GPU environment.
2. Accept an assignment and load its artifacts.
3. Expose enough worker state to verify the assigned range is warm.

## Decisions

Use manual assignments for this plan. Coordinator-driven assignments are verified in Plan 4. Warm means loaded and ready; GPU-session provisioning is not required.

## Verification

- Run the worker in each target environment.
- Verify the intended range is downloaded and remains loaded without downloading the entire model unnecessarily.

## Learnings

None recorded.

