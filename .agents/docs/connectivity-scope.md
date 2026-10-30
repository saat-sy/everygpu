# Peer connectivity scope

## Decision

Plan 2 requires configured, reachable peer addresses and implements discovery. NAT traversal, hole punching, circuit relays and worker-to-worker execution communication remain Plan 4 work. Joining the DHT does not demonstrate that every device can exchange execution payloads.

## Why

- Two laptops on one LAN can use reachable local addresses when network policy permits. Across home routers, outbound connections to a reachable third peer can support coordination; successful hole punching permits a direct path, while failed direct attempts may need a relay.
- A bootstrap peer introduces devices, a hole-punch coordinator helps establish connections, and a relay forwards bytes. Roles can share a reachable machine but are not interchangeable guarantees.
- A relay needs bandwidth and reachability, not a GPU or a separate rental. Placement and scheduling remain independent of relaying.
- The user will not rent machines for tests. Local processes with direct links deliberately blocked can exercise relay behavior when implemented; they do not validate independent real home networks. Public relay hosting and detailed traversal mechanisms remain open.
- No permanent full mesh is required. Peers should connect to discovery neighbors and actual execution partners as needed.
