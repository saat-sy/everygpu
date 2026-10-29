# Plan 3: Coordinator setup

## Goal

Set up the coordinator to own the execution path and instruct workers to execute their assigned stages.

## Status

Planned. Implementation has not started.

## Requirements

- Represent a default ordered path across N stages.
- Track stage assignments and worker readiness.
- Support dispatching stage inputs and receiving stage outputs.
- Relay all execution traffic through the coordinator.

## Steps

1. Establish the coordinator and its route configuration.
2. Represent worker assignments and readiness.
3. Provide coordinator-side stage dispatch and output handling.

## Decisions

Disaster recovery is deferred. Workers execute assigned layers on incoming model inputs; the coordinator does not need to provision GPU sessions. Live integration follows in Plan 4.

## Verification

- Verify default-path ordering and readiness handling with controlled worker responses.
- Verify dispatch follows the configured stage assignments.

## Learnings

None recorded.

