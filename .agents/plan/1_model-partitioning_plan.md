# Plan 1: Model partitioning

## Goal

An offline Model Partitioner that turns an OLMoE or Muse Spark Open (NVFP4) checkpoint into independently loadable Safetensors units plus a versioned manifest, verifies reassembly, and uploads the result to the owner's Hugging Face account. Run once per model version, never on worker registration.

## Status

Code complete and checked on local fixtures. Real-checkpoint runs, uploads and Muse validation remain; see Verification.

## Requirements

- Support OLMoE (`allenai/OLMoE-1B-7B-0924`) and Muse Spark Open NVFP4 (`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`) only. Pin an immutable source revision (commit SHA).
- PyTorch and Safetensors only; no custom checkpoint format. Preserve dtypes, quantized payloads (packed U8, FP8 scales, FP32 scales) and metadata byte for byte; never dequantize or requantize.
- Unit roles: `embedding`, `layer` (everything in a decoder layer except experts: attention, norms, router, shared/dense projections), `expert` (one expert of one layer; depends on its `layer` unit), `final` (final norm, output head), `perception` (vision/audio/projector).
- Consecutive whole-layer ranges are assembled from a layer unit plus its expert units; experts are also individually loadable. Distributed expert execution is out of scope.
- Experts may be per-expert checkpoint keys or fused tensors with a leading expert dimension. Fused tensors are split into contiguous expert slices recorded in the manifest. Any other layout is rejected as unsupported.
- Bounded memory: at most one output file's tensors in memory at a time. Classification and planning use only tensor names, shapes and dtypes, read from local shard headers or remotely via range reads (`inspect`), so a remote checkpoint is validated without downloading it.
- Manifest (versioned): source repo and revision, config, tokenizer/processor/quantization assets with checksums, per-shard safetensors metadata, units (role, layer, expert, dependencies, file), tensors (source key, slice, shape, dtype, bytes, sha256), file checksums. Layer count, expert count and tying are read from the stored config. A manifest with a different schema version is rejected, not read.
- Commands: `inspect`, `partition`, `verify`, `upload`, `verify-upload`, plus loading helpers for the artifacts. The pinned source revision is obtained with the standard `hf download`; assignment-driven unit retrieval belongs to worker loading. Upload takes an explicit destination repo; none is assumed.
- No worker service, coordinator, client API, streaming, queueing or serving.

## Steps

1. Project setup: `model_partitioner/` package with `model-partitioner` (extraction, upload) and `model-partitioner-outputs` (adds Transformers for output equivalence) dependency groups in `pyproject.toml`.
2. Core: shared partitioner, manifest models and loader, with one adapter per model family (tensor classification, config reading, execution-compatibility check).
3. Verification: coverage, checksums, exact tensor reconstruction, model-output equivalence with documented tolerance.
4. OLMoE first, end to end: tiny fixtures, then the real checkpoint. Report before moving on.
5. Muse adapter: packed quantized tensors, scales, perception components. Synthetic NVFP4-style fixture (U8, F8_E4M3, F32); real headers and checkpoint as access and resources allow. OLMoE passing does not establish Muse support.

## Decisions

- Expert files hold `--experts-per-file` experts (default 16) so Muse's 15,872 experts stay in about a thousand files; each expert remains a separately addressable unit inside its file.
- Model-output equivalence loads the original and a reassembled checkpoint through Transformers with identical inputs, eval mode and dtype. Muse NVFP4 output equivalence additionally needs ModelOpt-capable kernels and Blackwell hardware; without them it is reported unverified, and byte-exact reconstruction is the only claim.
- Muse tensor-name patterns come from the public Transformers modeling code; the checkpoint itself is gated, so unknown names fail loudly until validated.

## Verification

- Coverage: every source tensor is covered exactly once (slices tile the expert dimension); no extra tensors.
- Every tensor and file checksum matches; every source tensor is reconstructed byte-exactly.
- OLMoE original and reassembled logits agree within tolerance.
- Real OLMoE and Muse runs and uploads are done by the owner on a larger machine; this checkout validates code on local fixtures only.
- Upload is checked against Hub file checksums; uploaded artifacts can be retrieved and loaded for assigned units (needs destination repo and token).

## Learnings

None recorded.
