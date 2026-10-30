# Context

## Project

EveryGPU pools GPUs across machines to run models that do not fit on one participating device. Saatwik Yajaman is building it solo for the Indie Developer track of the Meta Global AI Developer Hackathon.

The long-term vision combines dedicated GPU workers with user devices that submit requests and contribute compute. The current clients submit requests only.

## Intended design

- Support OLMoE and Muse Spark Open partitioning. Exact checkpoints and runtime compatibility need verification during implementation.
- Publish all model shards to the owner's Hugging Face account.
- Assign workers consecutive layer ranges. Workers download their assignments and keep them loaded.
- Start execution with OLMoE on Colab and Kaggle. Account for N stages.
- The coordinator owns the default execution path, layer assignments and stage dispatch.
- All traffic passes through the coordinator, including intermediate activations.
- Introduce clients after a distributed forward pass works.
- Add client inference, token streaming, sequential queueing and then efficient multi-client processing in that order.
- Disaster recovery is deferred.
- The concurrency approach and benchmark scope remain TBD until their plans are reached.

Workers execute model layers on incoming inputs. Intermediate stages exchange activations; token IDs enter the model and the final output supports token sampling.

## Plans and worktrees

The `init` branch prepares agent context and the eleven plan documents. It contains no application implementation.

The eleven numbered files in `.agents/plan/` describe the features, each intended for its own worktree. Execute one numbered plan at a time, report its result and wait for explicit approval before starting the next.

## State

Current branch `model_partitioning` executes `.agents/plan/1_model-partitioning_plan.md`. Plans 2 to 11 are not started.

The user reports real OLMoE structural verification and bitwise-identical original/reassembled outputs on Colab. Codex independently checked all 87 declared files (82 partition files, 5 assets) in `saatwiksy/olmoe-partitions` at `be38d577a642e2c3ed0c79127899a59f261fb615`: sizes and hashes match. The manifest identifies source `allenai/OLMoE-1B-7B-0924` at `6d84c48581ece794365f2b8e9cfb043c68ade9c5`. Remote partition payloads were not downloaded or loaded.

Current focus: Muse NVFP4 partitioning, CPU weight verification and upload on a rented GCloud machine. Real Muse keys, artifacts, RAM requirements and tokenization compatibility remain unverified; inspect real headers before the full download. The user defers fresh remote unit loading and Muse model-output execution. Distributed inference and benchmarks remain unvalidated.

The implementation direction is a fresh build. The OLMoE prototype is reference material only.

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
