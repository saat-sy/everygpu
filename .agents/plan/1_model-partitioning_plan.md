# Plan 1: Model partitioning

## Goal

An offline Model Partitioner that turns an OLMoE or Muse Spark Open (NVFP4) checkpoint into independently loadable Safetensors units plus a versioned manifest, verifies reassembly, and is published to the owner's Hugging Face account. Run once per model version, never on worker registration.

## Status

Code complete and checked on local fixtures. Real-checkpoint runs, uploads and Muse validation remain; see Verification.

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

- Expert files hold `--experts-per-file` experts (default 16) so Muse's 15,872 experts stay in about a thousand files; each expert remains a separately addressable unit inside its file.
- Model-output equivalence loads the original and a reassembled checkpoint through Transformers with identical inputs, eval mode and dtype. Muse NVFP4 output equivalence additionally needs ModelOpt-capable kernels and Blackwell hardware; without them it is reported unverified, and byte-exact reconstruction is the only claim.
- Each adapter accepts only an explicit list of layer, embedding and final tensor names, and `partition` checks that config `model_type` matches the family; anything else fails loudly. Muse's list comes from the public Transformers modeling code (including quantization suffixes); the checkpoint is gated, so it is unchecked against real keys.
- Muse's root file `l4_200k_base` (3.6 MB, no extension, likely a tokenizer ranks file; unconfirmed because the Muse tokenizer config is gated) is preserved byte for byte as an asset. Whether the artifact is self-contained for tokenization stays unverified.
- `partition` and the reassembled-checkpoint scratch space refuse non-empty or overlapping destinations. Every source shard must match the Hub's sha256 at the pinned commit; the manifest records that repo and commit.

## Verification

- Coverage: every source tensor is covered exactly once (slices tile the expert dimension); no extra tensors.
- Every tensor and file checksum matches; every source tensor is reconstructed byte-exactly.
- OLMoE original and reassembled logits agree within tolerance.
- Real OLMoE and Muse runs and uploads are done by the owner on a larger machine; this checkout validates code on local fixtures only.
- `upload` compares every declared file at the uploaded revision with the manifest (needs destination repo and token). Retrieving an uploaded artifact and loading its units with the loader is not automated.

## Learnings

- Agree on concrete command examples, arguments, defaults and behavior with the user before implementation. Treat unresolved product choices as questions, not requirements. Give Claude only the agreed scope; avoid speculative features, files and abstractions. Routine implementation and review fixes can proceed within that scope without another approval.
