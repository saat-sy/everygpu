# Plan 6: Client registration

## Goal

Introduce a client that registers with the network through the coordinator.

## Status

Planned. Implementation has not started.

## Requirements

- The coordinator recognizes registered clients.
- The client can connect and receive registration confirmation.
- Clients submit work only and contribute no compute.

## Steps

1. Provide a runnable client.
2. Add coordinator-side client registration.
3. Verify registered clients are distinguishable.

## Decisions

Request execution follows in Plan 7. Registration does not imply a particular authentication system or user-account product.

## Verification

- Register two clients and verify each receives its own identity and confirmation.

## Learnings

None recorded.

