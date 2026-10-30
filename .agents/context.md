# Context

## Project

EveryGPU pools GPUs across rented machines and personal laptops to serve a model that does not fit on an individual device. Saatwik Yajaman is building it solo for the Indie Developer track of the Meta Global AI Developer Hackathon.

A client requests an answer; a worker offers GPU execution. One machine can run both independently: its client can use other GPUs while its worker serves another client's request. Both device categories run the same worker program.

## Intended design

- Peer-to-peer discovery through a distributed hash table (DHT), with direct worker communication and no central inference coordinator.
- Each requesting client selects a complete route and confirms worker capacity for its own session. Workers independently admit work; no permanent worker groups.
- Hybrid layer and expert partitioning. Stage workers hold attention, norms, routers and applicable shared projections; expert workers hold routed experts. These roles can coexist on one GPU, and local experts execute locally.
- Start the fresh runtime with the user's existing OLMoE partitions. The eventual target is Muse Spark Open with source NVFP4/BF16 weights, scales and metadata preserved. Do not reuse the old prototype or adopt another system's runtime by implication.
- Workers keep assigned weights resident. The client coordinates the session without relaying every activation or approving each expert call. Stage workers dispatch selected expert calls and forward stage outputs directly.
- Explain the perfect-condition architecture first. Reliability direction is replication of expert weights and attention KV state where protection is needed; baseline implementation does not require recovery.
- Immutable weight hosting on Hugging Face or object storage can coexist with decentralized scheduling. P2P weight distribution is a separate possible feature.

Discovery uses py-libp2p; execution transport and serialization remain open. Persistent gRPC connections are a suggestion; QUIC is an alternative requiring measurement, not a P2P requirement. Embedding, tokenization, output-head and sampling ownership remain open. Workers choose resident weight placement at onboarding; clients select execution routes among ready workers; the model's learned router selects experts per token.

The P2P architecture sections below contain the full design handoff, open decisions and conditional capacity/network estimates. These describe intended behavior and reasoning, not implemented or benchmarked results. Comparisons to other systems are background. Keep explanations anchored to the current question rather than introducing implementation checklists.

## Plans and worktrees

The numbered files in `.agents/plan/` define the current roadmap, each intended for its own worktree:

1. [Model partitioning](plan/1_model-partitioning_plan.md): existing artifact preparation and verification work.
2. [Worker onboarding and DHT](plan/2_worker-onboarding-and-dht_plan.md): join discovery, choose and load OLMoE pieces automatically, and advertise readiness.
3. [Client tooling and discovery](plan/3_client-discovery_plan.md): query the DHT when a request arrives and identify compatible candidate coverage.
4. [First P2P inference](plan/4_p2p-inference_plan.md): select and admit workers, execute OLMoE directly across peers and return generated text.

The roadmap stops at the first end-to-end OLMoE inference. Muse execution, multi-client optimization and replication remain future directions without implementation plans. Execute one numbered plan at a time, report its result and wait for explicit approval before starting the next. Approval to edit the roadmap does not authorize implementation.

## State

Current branch `onboarding` maps to [Plan 2](plan/2_worker-onboarding-and-dht_plan.md). The worker CLI is implemented. Claude reports local discovery and stubbed two-worker onboarding/inspection/shutdown checks on Linux aarch64 with `fastecdsa` overridden to 4.0.0. Codex independently checked cancellation, withdrawal failure/timeout handling and finite memory limits with external operations stubbed. Native Windows/macOS, real artifact retrieval, checksum failure paths, GPU loading, public bootstrap deployment and multi-machine internet runs remain unverified. Plan 2 is not verified end to end with its declared dependencies. Plans 3–4 are not started.

Installation fixes are deferred by the user: declared `fastecdsa==2.3.2` fails to build here without GMP headers; `uv sync --only-group worker` also fails because uv resolves the outputs group's unavailable `accelerate>=1.16` constraint. Do not silently override dependencies or treat scratch-environment results as a successful declared installation.

Plan 2 assumes reachable peer addresses. NAT traversal, hole punching, relays and cross-NAT communication stay in Plan 4. The user will not rent machines for tests; no connectivity extension was implemented. See [connectivity scope](docs/connectivity-scope.md).

The user reports real OLMoE structural verification and bitwise-identical original/reassembled outputs on Colab. Codex independently checked all 87 declared files (82 partition files, 5 assets) in `saatwiksy/olmoe-partitions` at `be38d577a642e2c3ed0c79127899a59f261fb615`: sizes and hashes match. The manifest identifies source `allenai/OLMoE-1B-7B-0924` at `6d84c48581ece794365f2b8e9cfb043c68ade9c5`. Remote partition payloads were not downloaded or loaded.

Plan 1's Muse partitioning, weight verification and upload remain unverified against the real checkpoint. Real Muse keys, artifacts, RAM requirements, tokenization and numerical execution compatibility remain unresolved. The user defers fresh remote unit loading and Muse model-output execution, cannot currently perform the large-GPU test and wants to move past it. Do not treat that test as a completed prerequisite or block architecture discussion on it.

The runtime starting point is the existing OLMoE artifacts plus a fresh P2P implementation. Initial functional goal: one client sends a prompt through partitioned workers and receives generated text. Two GPU workers with some deliberately remote experts are a suggested deployment if assignments fit, not a validated placement. Distributed inference and benchmarks remain unvalidated.

## Worker runbook

Layout of `worker/`: `cli.py` parser and exit codes; `commands/` (serve, onboard, report); `discovery/` (node, records, advertiser, discovery); `placement/` (order, selection, summary); `loading/` (budget, manifest, download, resident); `utils/` (constants, models).

Intended commands below require resolving the deferred installation blockers. The worker group targets Python 3.12–3.13; Linux needs GMP headers and a C compiler.

```bash
uv sync --only-group worker
uv run --group worker python -m worker --new-network
uv run --group worker python -m worker --bootstrap <address>
uv run --group worker python -m worker inspect --bootstrap <address>
```

The first worker pins `saatwiksy/olmoe-partitions` to an immutable revision; joiners inherit it from DHT metadata. Workers use the first visible CUDA GPU (`CUDA_VISIBLE_DEVICES` selects another), with weight budget `0.8 * min(free VRAM, --max-vram GiB)`. The optional limit must be positive and finite. CUDA/NVIDIA is the GPU scope; documented macOS networking does not include Apple Silicon GPU support.

Placement fills missing coverage, preferring whole uncovered layers before individual gaps; copies follow only once selected and observed units cover the model. Weights stay resident. Expert downloads group 16 experts, but loading is per expert. OLMoE weights total about 12.89 GiB before runtime overhead. Details: [placement](docs/worker-placement.md).

Every reachable worker participates in the DHT. Accepted metadata replication uses the library default of up to 20 closest peers. Provider records mean "may serve"; a separate ready status expires after 180 seconds and refreshes every 60. Ready means verified resident weights and discovery publication, not implemented inference or reserved capacity. Ctrl+C withdraws availability best effort; failures/timeouts warn and preserve the original outcome. Startup I/O allows cancellation without waiting for worker threads. Details: [discovery records](docs/libp2p-discovery.md).

## Model Partitioner runbook

Layout of `model_partitioner/`: `cli.py` entrypoint; `partition/` (adapters, source, partitioner); `verification/` (verifier, output_check); `artifact/` (loader, hub); `utils/` (constants, models, tensor_utils).

Install `model-partitioner` for preparation and weight verification, or `model-partitioner-outputs` for model-output execution (`uv sync --only-group GROUP`). Authenticate with `hf auth login`; Muse also needs approved Hub access.

Disk holds the cached source and partitions: about 28 GB for OLMoE or 1.07 TB for Muse, plus software and free-space headroom. Model-output verification adds a temporary reassembled checkpoint: about 42 GB total for OLMoE or 1.6 TB for Muse before overhead. Muse peak RAM is unmeasured; 64 GB is a target to investigate, not a validated requirement.

```
python -m model_partitioner partition --model allenai/OLMoE-1B-7B-0924 --output ./partitions
python -m model_partitioner verify ./partitions
python -m model_partitioner upload ./partitions --repo you/olmoe-partitions
```

- `partition --model` takes exactly `allenai/OLMoE-1B-7B-0924` or `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`. It resolves the repo's current commit SHA, validates the config, then downloads weights at that SHA into the Hugging Face cache (reused on later runs), checks every shard against the Hub's sha256 and records repo and SHA in the manifest. The default output is `./partitions/<model name>`, which must be empty or new.
- `verify` finds the original through the manifest's repo and SHA in the cache (downloading what is missing). Default: checksums, coverage, exact reconstruction, no model execution or GPU. Optional mutually exclusive `--cpu` and `--gpu` compare model outputs using the selected hardware. Muse CPU output execution is unsupported; its GPU path needs compatible NVFP4 hardware and remains unverified.
- `upload` checks local files, creates the repo as private if missing (an existing repo keeps its visibility), uploads only the files the manifest declares, and checks them against the manifest at the resulting revision. The result is an EveryGPU artifact, not a `from_pretrained` checkpoint. Large folders are split across several commits, and an interrupted upload is rerun from the start.
- Exit 0: all requested checks passed. 1: a check or command failed. 2: a requested check could not run (no GPU, source unavailable, unsupported hardware). Usage errors exit 1.

## Collaboration

Codex plans and reviews; Claude implements when delegation is authorized. Follow the Codex collaboration agreement in `housekeeping.md` and the separate planner Learnings in Plan 1. The user expects to understand the concrete result before implementation, then receive complete follow-through within the agreed scope. Individual feedback is not relayed to Claude without authorization.

## P2P architecture handoff

### Status and scope

The agreed direction is DHT discovery, direct worker communication and client-owned session coordination, with hybrid layer/expert partitioning and no permanent worker groups. Start with the user's OLMoE partitions and a fresh runtime; Muse Spark Open NVFP4/BF16 is the eventual target. The user wants to develop their own design. External systems are background references, not runtime choices.

This document records intended architecture and reasoning supplied by the user. Mechanisms described as suggestions and theoretical calculations are not implemented or benchmarked behavior. Understand the perfect-condition architecture first; failures and recovery are subsequent concerns.

### Discovery and admission

A DHT maps checkpoint-qualified model pieces (shared layer or layer/expert pair) to candidate workers. Advertisements identify peers, reachable addresses and compatible loaded pieces; they may include capacity and performance information. Records expire and refresh, so discovery is best effort. A client need not inventory every GPU.

Each DHT node stores part of the directory records and maintains contacts that help searches reach the relevant peers. Records are normally replicated. A process can advertise several GPU resources, so DHT peer count need not equal GPU count. For example, a lookup can ask who serves a checkpoint's shared layer 5 or layer 5 expert 23. The directory contains metadata, not weights, activations or KV caches.

Known reachable peers or bootstrap addresses introduce new peers. Multiple bootstrap peers need not become schedulers. Restricted networks may need traversal or relays; QUIC alone does not guarantee connectivity. As with trackerless BitTorrent, discovery finds peers and separate connections carry data. Model execution still has ordered dependencies and session state.

Each client selects compatible workers covering the shared layers in order and all experts the model might select. Selection can consider memory, execution speed and connection quality. Direct capacity confirmation is needed because advertisements can be stale and clients compete. Workers accept or reject requests independently. Reservation granularity, expiration and fairness remain open.

Discovery does not solve placement; placement does not guarantee execution capacity. Groups can be assembled per session without persisting between requests. Attention placement should initially remain stable for a session because its KV caches reside there.

### Execution and session state

A client submits a prompt, selects the route and confirms capacity. Tokenization and embedding produce model inputs; selected workers execute layers in order, and the output computation produces a token. Tokenization, embedding, output-head and sampling ownership must be explicit in the runtime and remain undecided, including client versus final-worker sampling.

A stage worker holds attention, norms, the router and applicable shared projections for its layers. It performs the model's shared operations, updates per-request KV state, dispatches selected expert computations, combines their results and completes the layer. The exact normalization, residual and projection ordering must follow the model implementation.

The learned router selects experts for each token. The stage worker groups calls by destination, executes local experts locally and sends remote activations to expert workers. Those workers return numerical results, which the stage worker combines using routing weights. Experts are identified by both layer and expert ID: expert 7 in layer 2 has different weights from expert 7 in layer 3. Standard feed-forward expert workers need no attention KV cache. Stage and expert roles belong to the same worker program and can coexist on a GPU.

A worker repeats for its assigned layers, then sends activations directly to the next stage. For example, A can hold shared layers 0–3 and B shared layers 4–7 while other workers hold their experts and the remaining model. The client need not relay activations or approve expert calls. The output computation produces a token that returns to the client; the generation loop feeds the token into the next model traversal, with ownership still to be settled. Prefill processes prompt positions and establishes caches. Each decode token traverses the model again; KV caching avoids recomputing earlier keys and values but does not remove the new token's computations. Weights remain resident rather than moving per call.

Persistent connections can carry multiple steps and requests. Session and operation identities must keep responses, execution positions and caches separate. No full mesh is required: peers connect to discovery neighbors and actual execution partners.

Different clients independently choose routes through the same pool. Requests can share weights and connections, while execution positions and KV state remain separate unless explicit prefix sharing exists. Worker queues can interleave session operations without simultaneous kernels. Admission of ten requests implies neither tenfold throughput nor a speed target.

### Artifacts and devices

The user's partitioner already exists. Shared layer units contain attention, norms, routers and applicable shared projections. Each expert file groups 16 routed experts; the manifest addresses experts individually and maps them to tensors. Downloading a file fetches the whole group, while GPU loading can select a subset. Download grouping does not dictate runtime placement.

Source precision is preserved without quantization: OLMoE BF16 Safetensors and eventual Muse mixed NVFP4/BF16, including scales and metadata. Preserving packed tensors does not establish kernel compatibility or resident VRAM use. Muse real-checkpoint and execution compatibility remain unverified.

Standard OLMoE-1B-7B-0924 supplies 16 layers, 64 experts per layer and eight selected experts per token. It establishes discovery, communication, sessions and scheduling at smaller scale. Success does not validate Muse's numerical backend, quantization or memory footprint.

Rented GPUs and laptops use the same worker program. Roles depend on available memory, compute and communication rather than device category. A laptop can run both client and worker; a rented machine can run only a worker. Alice can submit work to others while her GPU helps Bob. Rentals usually provide dedicated memory and sustained operation. Laptops also face foreground applications, power, cooling, variable links and owner pauses. Even the perfect-condition scenario must account for memory, compute and communication differences. Full rental coverage plus contributor capacity is a possible deployment policy, not a protocol requirement; dedicated machines can fail too.

### Reliability direction

The perfect-condition baseline needs no recovery. An expert replacement needs compatible exact weights and the interrupted call's input, retained until completion. Retries must not apply duplicate or late responses twice.

An attention replacement additionally needs KV state and a consistent session position. A backup can receive an initial cache copy and incremental updates. Synchronous replication waits for the required acknowledgment; asynchronous replication can overlap compute but may require replay from a safely acknowledged position.

Two total KV copies double protected cache storage without automatically duplicating weights or reserving takeover compute. A backup can serve other work while holding state; fast takeover needs capacity. Detection, rerouting and replay still cost time. Replicas need separate failure domains. Neither replicated weights nor replicated KV alone guarantees failover capacity or equal speed after failure.

### Conditional Muse memory estimates

These transferred calculations are unvalidated budgets, not procurement advice. Assumptions: 537 decimal GB of weights (about 500.12 GiB); ten independent text requests; 8,192 input plus up to 1,024 generated tokens each; BF16 KV; all required weights resident; no baseline backups; 20% of total VRAM reserved for runtime and buffers.

Using 16 KV heads and head dimension 64, one position per layer uses `2 × 16 × 64 × 2 = 4,096` bytes. Assuming 16 full-attention layers and 46 sliding-window layers capped at 2,048 positions, each request uses 0.5625 GiB full-attention cache plus 0.359375 GiB sliding cache, totaling 0.921875 GiB. Ten requests use 9.21875 GiB. This requires eviction of expired sliding entries and excludes padding and temporary allocations.

Weights plus caches total about 509.34 GiB; dividing by 0.80 yields 636.67 GiB planned aggregate capacity.

| Configuration | Conditional count |
| - | -: |
| Only 24 GiB rentals | 27 |
| Only 8 GiB laptops | 80 |
| Only 16 GiB laptops | 40 |
| Equal counts of 8 and 16 GiB laptops | 27 of each |
| 40 eight-GiB laptops plus 24 GiB rentals | 14 rentals |
| 60 eight-GiB laptops plus 24 GiB rentals | 7 rentals |

Full stated VRAM availability is assumed. Uneven shards, conversion and per-device allocations can increase counts. With duplicate KV only and the same allowance: 28 rentals, 82 eight-GiB laptops or 41 sixteen-GiB laptops. With two complete weight/cache copies: 54 rentals, 160 eight-GiB laptops or 80 sixteen-GiB laptops. These are memory budgets, not failover guarantees.

### Conditional network and cost estimates

A naive all-remote layout sends a separate BF16 4,096-wide input and output for each of eight experts in each of 62 layers.

| Work | Aggregate expert payload |
| - | -: |
| One token through all layers | 7.75 MiB |
| One decode step across ten requests | 77.5 MiB |
| Prefill for ten 8,192-token prompts | 620 GiB |
| 1,024 decode steps for ten requests | 77.5 GiB |
| Combined workload | 697.5 GiB |

Destination grouping, remote combination and local execution can reduce traffic. Protocol overhead, stage transfers and backups are excluded. Eight calls within a layer can overlap, but 62 layers impose sequential expert exchanges. Hypothetical effective round-trip delays of 10 ms or 30 ms per layer give network waiting alone of 0.62 or 1.86 seconds per token: ceilings of about 1.61 or 0.54 tokens/s before compute, transfer serialization and queueing. These are illustrative bounds, not expected speeds.

A shared 1 Gbit/s bottleneck carrying 77.5 MiB needs at least about 0.65 seconds. Independent paths can transfer concurrently; aggregate traffic is not every link's load. A shared 1 Gbit/s source needs an ideal 71.6 minutes to distribute 537 GB total; caching and independent source paths change startup behavior.

At the discussion's advertised A5000 starting rate of $0.27/GPU-hour, 27 rentals cost $7.29/hour, 14 cost $3.78/hour and 54 cost $14.58/hour. This is a conditional arithmetic example, not a current price check. Availability and rates vary; storage, startup, charged transfers and laptop electricity are excluded.

### Open decisions

- Transport and serialization; NAT traversal and relays.
- Concurrent onboarding placement coordination, rebalancing and load-aware weight copies.
- Reservation granularity, expiration, admission and fairness.
- Tokenization, embedding, output-head and sampling ownership.
- Muse kernels and actual resident memory.
- Cache replication protocol, consistent recovery boundaries and replay.
- Trust, authentication and acceptance of worker outputs.
- Measured speed targets and concurrency approach.

### References

The user's descriptions and decisions are authoritative for intended design. These links are reference material; this context update does not independently verify their contents or the estimates above.

- [User OLMoE partitions](https://huggingface.co/saatwiksy/olmoe-partitions/tree/main)
- [OLMoE configuration](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json)
- [BitTorrent DHT specification](https://www.bittorrent.org/beps/bep_0005.html)
- [Muse overview](https://dev.meta.ai/docs/muse-spark-open)
- [Muse settings and attention windows](https://dev.meta.ai/docs/muse-spark-open/settings)
- [Muse model card](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4/blob/main/README.md)
- [Muse architecture](https://huggingface.co/docs/transformers/en/model_doc/muse_spark)
- [Runpod A5000 pricing](https://www.runpod.io/gpu-models/rtx-a5000)
