# Plan 2: Worker onboarding and DHT

## Goal

Onboard an OLMoE worker that loads assigned model pieces, joins a DHT and advertises what it can serve.

## Status

Planned. Implementation has not started.

## Requirements

- Build fresh worker tooling around the user's existing OLMoE partition artifacts; do not reuse the prototype.
- Accept a pinned artifact revision and explicit shared-layer, expert and applicable embedding/final-unit assignments. One worker can hold shared layers and experts together.
- Retrieve required files, verify them against the manifest and keep selected units loaded. Expert files contain 16 experts, but GPU loading can select individual experts.
- Join discovery through bootstrap peers and publish checkpoint-qualified model-piece records with peer identity, reachable endpoint and readiness.
- Refresh expiring advertisements and stop advertising availability on orderly shutdown. Advertise only pieces loaded and ready; records are not capacity guarantees.
- Keep weights and execution payloads outside the DHT. No central inference coordinator or permanent worker groups.

## Steps

1. Agree on the installation/startup workflow, assignment input, bootstrap configuration and readiness output before implementation. Settle the DHT library and minimum record schema.
2. Implement artifact retrieval, selective loading and worker readiness.
3. Implement DHT joining, piece advertisements, refresh and shutdown handling; provide a way to inspect onboarding results without the Plan 3 client.

## Decisions

Manual placement is sufficient. Automatic balancing, inference execution, client tooling and replication are outside this plan. Worker roles use the same program on rented machines and laptops. Exact CLI and networking choices remain to be agreed; the plan does not prescribe gRPC or QUIC.

## Verification

- Start multiple discovery peers and workers; look up advertised pieces from another peer without a central directory.
- Verify model revision and layer/expert identity distinguish records correctly.
- Verify selected units load and unrelated units are not unnecessarily loaded; account for whole-file downloads of grouped experts.
- Verify unready workers do not advertise service and expired records cease to be eligible.
- Use small fixtures on this checkout. Report real OLMoE GPU loading separately and run it only in an approved environment; fixture success does not establish real-model readiness.

## Learnings

None recorded.
