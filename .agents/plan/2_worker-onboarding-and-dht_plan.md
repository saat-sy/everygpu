# Plan 2: Worker onboarding and DHT

## Goal

Provide a CLI that onboards a device, automatically selects and loads OLMoE pieces within its GPU memory budget, and advertises ready pieces through a DHT.

## Status

The `worker/` package implements the CLI below. Claude reports passing local py-libp2p discovery and stubbed onboarding checks on Linux aarch64. Codex independently checked cancellation, withdrawal failures and memory-limit validation with external operations stubbed. Native Windows/macOS, real artifact retrieval, GPU loading, public bootstrap deployment and multi-machine internet runs remain unverified. Installation blockers are documented and deferred by the user; Plan 2 is not verified end to end with its declared dependencies.

## Requirements

- Build fresh worker tooling around `saatwiksy/olmoe-partitions`; do not reuse the prototype. Support NVIDIA GPUs on personal laptops and rented machines through the same worker program. Assume devices remain available and offer full GPU use; VRAM is limited.
- Pin the artifact revision and keep workers in the same model network compatible. Include shared layers, individual experts, embedding and final units in placement. One worker can hold shared layers and experts together.
- Choose a conservative weight budget automatically, reserving memory for runtime buffers and applicable KV caches; allow an optional `--max-vram` limit. Full GPU use does not mean allocating all VRAM to weights.
- Join discovery before choosing pieces. Query checkpoint-qualified coverage, fill missing pieces first, prefer complete contiguous layers with their experts, and split expert placement when needed to fit. Add useful copies after complete coverage. Keep assignments stable after onboarding.
- Retrieve required files, verify them against the manifest and keep selected units loaded. Expert files contain 16 experts, but GPU loading can select individual experts.
- Publish checkpoint-qualified model-piece records with peer identity, reachable endpoint and readiness. Different workers must be able to advertise the same piece without overwriting one another.
- Refresh expiring advertisements and stop advertising availability on orderly shutdown. Advertise only pieces loaded and ready; records are not capacity guarantees.
- Keep weights and execution payloads outside the DHT. No central inference coordinator or permanent worker groups.
- Every reachable worker participates in the DHT. Use py-libp2p's default replication to up to 20 closest peers per directory record, with fewer when fewer peers are available. Storage responsibility follows DHT key routing rather than join order; participation is not limited to a fixed set of workers.
- The first worker explicitly starts a network; subsequent workers join through bootstrap addresses. Failed bootstrap must not silently create a separate network.
- Show GPU memory, pinned revision, selected pieces, download/loading progress, readiness and bootstrap addresses. Provide metadata inspection without requiring the Plan 3 inference client.

## CLI

Install the worker dependency group (Python 3.12–3.13; on Linux, `libgmp-dev` and a C compiler, because `fastecdsa` 2.3.2 required by `libp2p` has no Linux wheels):

```bash
uv sync --only-group worker
```

On Debian/Ubuntu, the native build prerequisites are installed with `sudo apt-get install build-essential python3-dev libgmp-dev`. The documented `uv sync` command is currently blocked by the root project's unrelated `model-partitioner-outputs` requirement `accelerate>=1.16`: the available index supplies only versions through 1.15.0, and uv resolves all groups when locking. Do not describe installation as passing until that constraint is resolved.

Start the first worker, then use its printed bootstrap address on other devices:

```bash
uv run --group worker python -m worker --new-network
uv run --group worker python -m worker --bootstrap <address>
```

Worker commands stay running; Ctrl+C withdraws availability and shuts down. The artifact repository is `saatwiksy/olmoe-partitions`. The first worker resolves and pins the revision; joiners read it from the network pin and fail if none is found. `--max-vram GIB` caps the GPU memory the worker plans to use. `--bootstrap` takes one or more addresses printed by any running worker.

```bash
uv run --group worker python -m worker inspect --bootstrap <address>
```

Inspection joins as a DHT client and reports discovered coverage, missing pieces and serving workers.

## Steps

1. Maintain the CLI, revision handoff, memory budgeting, placement policy and record schema below. Check the pinned DHT library against the project's Python and dependency requirements.
2. Implement network creation/joining, model metadata and coverage lookup with the library's default record replication. Add inspection so discovery can be checked independently of GPU loading.
3. Implement automatic piece selection, verified artifact retrieval and selective resident loading within the memory budget.
4. Connect readiness to advertisements, refresh and shutdown. Verify the complete onboarding workflow using small fixtures; report real GPU verification separately.

## Decisions

Plan 2 assumes configured, reachable peer addresses. NAT traversal, hole punching, relays and cross-NAT communication belong to Plan 4; none is implemented here. DHT participation does not establish home-network connectivity. The user will not rent machines for testing. A relay can share an existing reachable machine and needs no GPU; hosting and traversal details remain open. See [connectivity scope](../docs/connectivity-scope.md).

Concurrent joiners can observe the same missing pieces and choose overlapping assignments, delaying complete coverage. Coordination of placement attempts is deferred; initial placement validation uses sequential joins. DHT advertisements do not provide exclusive placement claims, and this plan does not introduce a network-wide lock.

py-libp2p (`libp2p==0.8.1`, dependency group `worker`, `requires-python <3.14` for that group) provides the host, transport and Kademlia DHT, selected for documented native Windows, macOS and Linux installation. Networking targets those three platforms natively; GPU loading is NVIDIA/CUDA only, and Apple Silicon/MPS is out of scope. torch 2.15 needs macOS 14 or later. The exact pin covers a private attribute the worker sets (see Publication). This does not adopt an inference runtime.

Current implementation (rationale in [discovery records](../docs/libp2p-discovery.md) and [weight placement](../docs/worker-placement.md)):

- Network pin: value record `/everygpu/network` holding `{repo, revision}` with identical bytes from every writer. The first worker stores it; every worker re-stores it from joining until shutdown, including during downloads and loading. A lookup returns one best value, so conflicting pins are not detected. Joiners and `inspect` fail when no valid pin is found.
- Piece records: Kademlia provider records under `{repo}@{revision}/{unit id}` with manifest unit ids (`embedding`, `layer:L`, `expert:L:E`, `final`). Every provider of a piece adds its own record, so providers never overwrite one another. Provider records last 48 hours, the library republishes them, and the protocol cannot withdraw them; they mean "may serve", never "ready".
- Peer status: value record `/everygpu/peer/<peer id>` holding `{repo, revision, addrs, ready, expires, seq}`, written only by that peer; the validator keeps the highest `seq`. A provider counts as serving a piece only while its status is ready, unexpired and on the same revision. Refresh every 60 s with 180 s expiry. A new peer id each run leaves records from earlier runs ineligible.
- Replication: use the library's default provider and value record placement on up to 20 (Kademlia k) peers closest to the key, found by key routing. This is the agreed policy; successful publication does not guarantee delivery to every target peer.
- Publication: after loading, the worker announces every chosen piece, then writes a ready status. READY requires each announcement to have been sent to at least one peer (a lone first worker holds its records itself) and the status write to finish without error. Announcements and value writes are unacknowledged, so delivery is not confirmed. Receivers drop ADD_PROVIDER beyond 10 per 10 s per sender by default; the worker raises that limit on its own DHT, which every peer runs. Refresh failures print a degraded-discovery warning and the worker keeps serving.
- Withdrawal: on every exit after the ready status was attempted, the worker writes `ready: false` best effort within 15 s, then closes the host. Failures/timeouts warn without hiding the original error or Ctrl+C. Exits during onboarding stop the refresh and close the host.
- Lookups: provider lookups are deduplicated by peer id and run with bounded concurrency. A node answers provider lookups from its local records first, so a long-running node's repeated lookups can miss newer providers; worker onboarding and `inspect` look up once per process.
- Routing table: the library adds every peer that queries a server, including clients; the worker removes a peer when its last connection closes, so departed peers do not cost later queries a timeout.
- Bootstrap: each address gets 15 s to connect; at least one must succeed, otherwise the command exits with an error.
- Memory: weight budget is 80% of `min(free VRAM, --max-vram)`, reserving 20% for runtime buffers and future KV caches. The reserve is an unmeasured heuristic, not an inference-capacity guarantee. Reject nonfinite and nonpositive limits before startup. The worker uses the first visible GPU; select others with `CUDA_VISIBLE_DEVICES`.
- Placement: units in model order, grouped as embedding, each layer's shared unit with its experts, and final. First, every group with no coverage at all is taken whole, in order, when it fits. Then remaining missing units are taken individually in model order, splitting a layer's experts at the budget edge. Only when coverage is complete are copies added, fewest existing copies first, then model order. Assignments are fixed for the process lifetime.
- Loading: whole files at the pinned revision through the Hugging Face cache, each file checked against the manifest sha256, then each selected unit's tensors checked and moved to the GPU. Revision resolution, manifest retrieval and loading run in threads that allow the command to cancel without waiting for those operations. Cancellation does not stop an in-flight thread's underlying I/O. Downloads of uncached files are refused when the cache filesystem lacks space. Pieces are announced only after every selected unit is resident.
- Listening: every interface address on a free port chosen at start; there is no port flag.

Automatic rebalancing of existing workers, inference execution, client route selection, capacity admission, weight/KV replication, recovery and Muse support are outside this plan. DHT metadata replication and additional weight copies chosen during onboarding are included. Execution transport is not prescribed.

## Verification

- Start a network and join multiple workers through bootstrap peers; look up advertised pieces from another peer without a central directory. Confirm failed bootstrap is explicit.
- Check record placement with one, two and at least four peers, including a publisher that joined after the first three. Distinguish authoritative replicas from optional lookup caches.
- Verify model revision and layer/expert identity distinguish records correctly.
- Verify separate worker advertisements coexist for the same piece, and inspection reports missing coverage.
- With sequential joins and differing simulated memory budgets, verify missing coverage is prioritized, complete contiguous layers are preferred when they fit, and assignments respect memory limits. Do not treat a DHT miss as proof of globally absent coverage.
- Verify selected units load and unrelated units are not unnecessarily loaded; account for whole-file downloads of grouped experts.
- Verify missing/corrupt artifacts cannot become ready, unready workers do not advertise service, orderly shutdown withdraws availability, and expired records cease to be eligible.
- Claude-reported results on this host (Linux aarch64, Python 3.12, scratch venv: libp2p 0.8.1, trio 0.34, protobuf 6.33.6, CPU torch 2.15; `fastecdsa` overridden to 4.0.0 because building 2.3.2 needs GMP headers this host lacks; real local peers, no GPU or Hub):
  - Dependency resolution of the worker group succeeds for Linux x86_64 and Windows x86_64 on Python 3.12–3.13; macOS arm64 resolves only with macOS 14 or later (torch). The declared group was not installed as-is anywhere.
  - An unreachable bootstrap address exits with an error; no network is created.
  - Holders per provider record: 1 with one peer, 2 with two, 5 of 5 with five peers, including 320 records announced by the fifth peer that joined last (20.5 s).
  - Rate-limit control: a receiver with the default limit kept 10 of 100 announcements; with the raised limit it kept 100 of 100.
  - A fresh client process found all 320 pieces (14.3 s); two providers of one piece coexist; another revision returns nothing; a provider without a ready status is excluded; withdrawal, a 4 s status expiry and stopping the refresh each remove that provider while another stays; a running refresh keeps a 4 s status valid at 8 s.
  - A departed client process is removed from the server routing table; without that removal it stayed listed.
  - CLI: missing or conflicting network flags and non-positive `--max-vram` exit 1; no GPU exits 2; dead bootstrap exits 1 for worker and `inspect`; joiner and `inspect` without a pin exit 1.
  - Stubbed onboarding (fake 8-unit manifest of equal sizes; GPU, Hub and loading stubbed): worker A with budget 4 units took embedding and layer 0 with both experts; worker B with budget 6 took the missing layer 1 group and final, then copies of embedding and layer 0's shared unit. `inspect` reported 8/8 with both workers and 4/8 after Ctrl+C on B. Ctrl+C during a 300 s stubbed load exited in 0.6 s. No worker port stayed open afterwards.
  - Not run: native Windows/macOS, the declared Linux install with GMP, real manifest/Hub retrieval, checksum failure paths, GPU loading, failed-announcement paths, multi-machine internet runs, public bootstrap deployment and NAT. Real OLMoE GPU loading needs an approved GPU environment.
- Use small fixtures on this checkout. Report real OLMoE GPU loading separately and run it only in an approved environment; fixture success does not establish real-model readiness.
- Independent installation review (Linux aarch64, Python 3.12): installing the declared `libp2p==0.8.1` in an isolated environment without overrides fails building its `fastecdsa==2.3.2` dependency because `gmp.h` is absent. `uv sync --only-group worker --dry-run` independently fails dependency resolution on `accelerate>=1.16` in the outputs group. No system packages or dependency constraints were changed; installation/import remain unverified with the declared dependencies and required build prerequisites together.
- Codex checks after the worker-wide refactor: all 24 worker Python files parse; Black, Pyflakes and whitespace checks pass for the four fix files. Function checks using Trio with external boundaries stubbed confirm prompt cancellation of revision/onboarding/inspection metadata fetches; original ValueError and KeyboardInterrupt outcomes survive successful, failed and timed-out withdrawal while host cleanup runs; nonfinite/nonpositive memory limits are rejected before dispatch and a valid limit is accepted. These are not GPU, Hub or multi-peer integration results.
- Completion requires the CLI onboarding and inspection workflow to work across multiple reachable peers within the stated scope. Document remaining GPU and home-network verification limits. Do not add automated tests unless requested.

## Learnings

### Claude (implementer)

- Check a library's real source and API before designing on it. py-libp2p provider records cannot be withdrawn, silently drop more than 10 announcements per 10 s per sender, and keep disconnected peers in routing tables; each needed a record-schema change or a workaround.
- Gate readiness on every step that makes the worker findable: a failed or unsent announcement must not become READY. Start refresh before the long load, and make shutdown cleanup unconditional with a bounded, shielded withdraw.
- Run negative controls for each workaround (default limit versus raised, with versus without the departed-peer listener); passing runs alone prove little. Separate measurement artifacts from real misses, such as asynchronous processing and expiry shorter than the operation.
- Test with real local peers and stub only GPU, Hub and manifest. Run client views as subprocesses, not in-process.
- Python rules apply to first drafts: no nested functions, module imports, one-line docstrings, no mutable globals. Give each file one responsibility (host lifecycle, record rules, advertising, discovery, placement order/selection/summary, loading budget/manifest/download/resident, one file per command); the user welcomes more files when each is independent.
- Name operations explicitly (`read_network_pin`, `find_coverage`); an object owns state its operations share (`Advertiser`), while stateless lookups stay functions. Drop forwarding helpers and generic task frameworks; keep bounded concurrency explicit.
- Report only what ran. Label the `fastecdsa` override in scratch environments, list unverified platforms and GPU paths, and rerun the CLI and two-worker flow after each refactor.
- Scratch hygiene: never `pkill -f` a pattern that appears in your own command line, use `"${S:?}"/*` for removals, remember background jobs ignore SIGINT unless the script restores it, and delete scratch environments afterwards.

### Codex (planner)

- Establish why devices join, their hardware limits and who chooses weights before startup commands. Separate worker weight placement from client request routing.
- Compare dependencies against native Windows/macOS/Linux needs before recommending them. Another project's choice is background, not justification; disclose platform constraints before delegation.
- Give implementers a standalone behavior contract and review the full diff, including untracked files. Label proposals, implemented behavior, agent-reported results and independently verified results separately.
- Distinguish DHT participants, bootstrap addresses, directory replicas and weight copies. Every reachable worker participates; the accepted library policy uses up to 20 storage peers per record.
- Apply YAGNI throughout the worker, not just the file named in feedback. Use clear responsibilities and explicit names; avoid generic frameworks and unnecessary forwarding layers.
- Explain networking through concrete device examples. A relay need not be rented or have a GPU, and DHT discovery does not prove execution connectivity. Keep NAT traversal/relays in Plan 4 and concurrent placement coordination deferred.
- Do not rent test machines or fix deferred dependency blockers without new authorization. Local peer simulations do not prove home-network reachability.
- Audit error paths and cancellation after refactors. Bound withdrawal, preserve original failures and check every blocking startup operation, including metadata fetches.
