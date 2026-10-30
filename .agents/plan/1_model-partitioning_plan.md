# Plan 1: Model partitioning

## Goal

An offline Model Partitioner that turns an OLMoE or Muse Spark Open (NVFP4) checkpoint into independently loadable Safetensors units plus a versioned manifest, verifies reassembly, and is published to the owner's Hugging Face account. Run once per model version, never on worker registration.

## Status

Implementation is available. The user reports real OLMoE structural and GPU output verification passed on Colab; its upload is independently verified at Hub file-integrity level. Current work is Muse partitioning, CPU weight verification and upload on a rented machine. Fresh remote unit loading is deferred by the user; Muse output execution is deferred for cost.

## Requirements

- Support OLMoE (`allenai/OLMoE-1B-7B-0924`) and Muse Spark Open NVFP4 (`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`) only. Pin an immutable source revision (commit SHA).
- PyTorch and Safetensors only; no custom checkpoint format. Preserve dtypes, quantized payloads (packed U8, FP8 scales, FP32 scales) and metadata byte for byte; never dequantize or requantize.
- Unit roles: `embedding`, `layer` (everything in a decoder layer except experts: attention, norms, router, shared/dense projections), `expert` (one expert of one layer; depends on its `layer` unit), `final` (final norm, output head), `perception` (vision/audio/projector).
- Consecutive whole-layer ranges are assembled from a layer unit plus its expert units; experts are also individually loadable. Distributed expert execution is out of scope.
- Experts may be per-expert checkpoint keys or fused tensors with a leading expert dimension. Fused tensors are split into contiguous expert slices recorded in the manifest. Any other layout is rejected as unsupported.
- Bounded memory: at most one output file's tensors in memory at a time. Classification and planning use only tensor names, shapes and dtypes read from local shard headers; `partition` prints the planned units, files and sizes before writing.
- Manifest (versioned): source repo and revision, config, tokenizer/processor/quantization assets with checksums, per-shard safetensors metadata, units (role, layer, expert, dependencies, file), tensors (source key, slice, shape, dtype, bytes, sha256), file checksums. Layer count, expert count and tying are read from the stored config. A manifest with a different schema version is rejected, not read.
- CLI: `partition --model REPO [--output DIR]`, `verify DIR [--cpu | --gpu]` and `upload DIR --repo REPO`, plus loading helpers for the artifacts. `partition` pins the repo's current commit SHA, downloads through the Hugging Face cache and verifies shard hashes against the Hub. `verify` locates the original from the manifest's repo and SHA. `upload` publishes only manifest-declared files and checks them at the resulting revision. Assignment-driven unit retrieval belongs to worker loading.
- No worker service, coordinator, client API, streaming, queueing or serving.

## Steps

1. Project setup: `model_partitioner/` package with `model-partitioner` (extraction, Hub checks) and `model-partitioner-outputs` (adds Transformers for output equivalence) dependency groups in `pyproject.toml`.
2. Core: shared partitioner, manifest models and loader, with one adapter per model family (tensor classification, config reading, execution-compatibility check).
3. Verification: coverage, checksums, exact tensor reconstruction, model-output equivalence with documented tolerance.
4. OLMoE first, end to end: tiny fixtures, then the real checkpoint. Report before moving on.
5. Muse adapter: packed quantized tensors, scales, perception components. Synthetic NVFP4-style fixture (U8, F8_E4M3, F32); real headers and checkpoint as access and resources allow. OLMoE passing does not establish Muse support.

## Decisions

- Expert files hold 16 experts so Muse's 15,872 experts stay in about a thousand files; each expert remains a separately addressable unit inside its file. See `docs/unit-granularity.md` and `docs/pinned-source-and-publishing.md` for why.
- Model-output equivalence loads the original and a reassembled checkpoint through Transformers with identical inputs, eval mode and dtype. Muse NVFP4 output equivalence additionally needs ModelOpt-capable kernels and Blackwell hardware; without them it is reported unverified, and byte-exact reconstruction is the only claim.
- Each adapter accepts only an explicit list of layer, embedding and final tensor names, and `partition` checks that config `model_type` matches the family; anything else fails loudly. Muse's list comes from the public Transformers modeling code (including quantization suffixes); the checkpoint is gated, so it is unchecked against real keys.
- Muse's root file `l4_200k_base` (3.6 MB, no extension, likely a tokenizer ranks file; unconfirmed because the Muse tokenizer config is gated) is preserved byte for byte as an asset. Whether the artifact is self-contained for tokenization stays unverified.
- `partition` and the reassembled-checkpoint scratch space refuse non-empty or overlapping destinations. Every source shard must match the Hub's sha256 at the pinned commit; the manifest records that repo and commit.

## Verification

- Coverage: every source tensor is covered exactly once (slices tile the expert dimension); no extra tensors.
- Every tensor and file checksum matches; every source tensor is reconstructed byte-exactly.
- OLMoE original and reassembled logits agree within tolerance.
- User-reported real OLMoE run on Colab: `verify` passed file checksums, coverage and exact reconstruction; `verify --gpu` also passed output equivalence on the tested inputs (max abs logits diff 0, bitwise equal, device auto). The run was not independently executed; GPU model and exact commands were not supplied.
- OLMoE upload, inspected independently by Codex through the Hugging Face API: `saatwiksy/olmoe-partitions` at revision `be38d577a642e2c3ed0c79127899a59f261fb615`, built from `allenai/OLMoE-1B-7B-0924` at `6d84c48581ece794365f2b8e9cfb043c68ade9c5`. It has 1,042 units, 82 partition files (13,838,715,664 bytes) and 5 assets. All 87 declared files are present with matching sizes and hashes: partition files against Hub LFS sha256, small non-LFS assets downloaded and hashed. Partition payloads were not downloaded and no remote unit was loaded.
- Pending: real Muse checkpoint compatibility, partitioning, CPU weight verification and upload. Inspect accessible real headers before a large download or rental recommendation; 64 GB RAM is an unverified target, not a measured minimum.
- Deferred by the user: fresh retrieval/loading of a remote unit and Muse model-output execution. Plain `verify` requires no GPU; `--cpu` executes the model on CPU and is not a synonym for structural checks.
- `upload` compares every declared file at the uploaded revision with the manifest (needs destination repo and token). Retrieving an uploaded artifact and loading its units with the loader is not automated.

## Learnings

### Claude (implementer)

- Before building a CLI, propose exact commands, flags, defaults and examples and wait for approval. The first build exposed seven subcommands and many flags where three commands were wanted. Prefer the fewest arguments: positional where obvious, and derive revision, source and family from the Hub, config or manifest.
- Build only what was asked: no extra files, modules, aliases, frameworks or standing rules. Record a rule in housekeeping only when the user asks.
- When told to remove something, remove all of it in one pass: flags, helpers, imports, constants, docs and dependency groups. Grep for stale references.
- Answer a direct question with the true state before acting; never imply completion. Say what is built, run and unverified. "Ack me" means acknowledge, then wait.
- Follow an interruption immediately. Never retry a declined command verbatim.
- This host is for code and fixtures. Do not download models or run real checkpoints, GPUs or uploads here, and ask first. Delete accidental downloads.
- Clean up before reporting: replaced files, temporary installs, caches and scratch.
- Lay out the package by feature with a shared `utils/` from the start, following large-repo norms, instead of a flat folder.
- Check real library APIs with `inspect` or `--help` before use. Mocks hid a missing `HfApi.upload_large_folder` and an unimported `huggingface_hub.errors`. Add unmocked offline checks, such as `HF_HUB_OFFLINE=1` with a fake cache.
- Build fail-closed from the start: verify every shard and asset, validate arguments and destinations before expensive work, report disk and network needs before a download, and refuse unsafe states. Review found each of these late.
- Run negative controls (corrupt a byte, drop a tensor, go offline, delete a file); passing runs alone prove little.
- After each milestone, ping Codex (`ping list` for names) with changed definitions, checks run and unverified paths. Do not claim Muse, NVFP4, GPU or Hub behavior without a real run, and label user-reported results as user-reported.

### Codex (planner)

- Start with what the user will actually run and receive. Agree on installation, command examples, defaults, outputs and the definition of done before implementation; a technically valid core alone is not the requested end-to-end result.
- Ask about consequential product choices early. Download behavior, model selection, output location, verification modes and upload workflow need explicit agreement; inferred requirements cause implementation and cleanup loops.
- Simplicity means few decisions for the user, not an arbitrary number of commands or flags. Accept repository IDs, automate revision tracking and cache lookup, make output paths optional, and keep partition, verify and upload as clear operations.
- Delegate the agreed result in one self-contained handoff. Include project purpose, current plan, approved examples, constraints, verification and exclusions so a fresh implementer session does not need the user to repeat context. Do not add speculative features to the handoff.
- Finish routine follow-through without waiting for repeated prompts: send authorized work, review it, relay defects and verify fixes. Pause for unresolved requirements, new scope and the next numbered plan; respect requests to keep feedback individual.
- Honor response-only requests: "ack only" means acknowledge and stop; "show me the arguments" means explain the current interface before proposing or implementing changes. Label proposals, implemented behavior and verified results distinctly.
- Audit behavior, not just code size or successful fixtures. Check manifest relationships, missing/corrupt inputs, provenance, destination safety and actual dependency APIs. Keep shared constants/data structures consistent without mechanical file merging or speculative frameworks.
- Separate artifact correctness from full-model execution. Report exactly which model, inputs and stages passed; fixture success does not establish real Muse support, output equivalence or distributed inference. Record user-deferred checks explicitly.
- Give environment-specific runnable instructions using the current working tree. Account for unpushed code, hardware, disk copies and dependency compatibility. Present memory/time figures as estimates until measured; inspect real tensor headers before advising a costly rental.
