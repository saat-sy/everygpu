# libp2p discovery records

## Decision

Workers use py-libp2p (`libp2p==0.8.1`) for peer networking and the Kademlia DHT. Every reachable worker participates; storage responsibility follows key routing, not join order. Bootstrap peers introduce nodes and do not schedule work. The accepted replication policy uses the library default of up to 20 closest peers per record, with fewer in smaller networks.

Workers write three kinds of record:

- Network pin `/everygpu/network`: `{repo, revision}`, identical bytes from every writer.
- Piece provider records under `{repo}@{revision}/{unit id}`: one per provider, meaning "may serve".
- Peer status `/everygpu/peer/<peer id>`: `{repo, revision, addrs, ready, expires, seq}`, written only by that peer; highest `seq` wins.

A provider counts as serving only while its status is ready, unexpired and on the same revision. The worker announces pieces after verified loading, then writes the ready status. Refresh runs every 60 seconds, including network-pin refresh during loading; status expires after 180 seconds. Shutdown attempts an unavailable status for at most 15 seconds. Failed withdrawal falls back to expiry and must preserve the original error or interruption.

Ready denotes resident weights and discovery publication, not inference execution or reserved capacity. Announcements/value writes do not confirm receipt by every replica. Local lookup records may omit newer providers. Neither discovery nor metadata replication guarantees model coverage, execution connectivity or weight redundancy.

## Why

- Native Windows, macOS and Linux networking is the target. Select dependencies against that requirement; Hivemind's documented Windows path requires WSL. Another project's use of a library is not a selection criterion. Platform documentation does not substitute for an actual platform run.
- Provider records cannot be withdrawn and last 48 hours, so readiness and orderly shutdown live in the status record.
- Separate provider records let workers advertise the same piece without overwriting each other.
- The library drops announcements beyond 10 per 10 s per sender, so the worker raises that limit on its own DHT (private attribute; `libp2p==0.8.1` is pinned exactly).
- The library keeps disconnected peers in routing tables, costing later queries timeouts, so a network listener removes a peer when its last connection closes.
- Conflicting network pins are not detectable: a lookup returns one best value.
