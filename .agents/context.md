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

Plans 2 to 11 are written but not started. Plan 1 (model partitioning) is implemented in `model_partitioner/` and checked on tiny local fixtures only: a random OLMoE with per-expert keys, and synthetic Muse-style NVFP4 layouts (fused and per-expert). No real OLMoE or Muse checkpoint has been partitioned, no shard has been uploaded, and Muse output equivalence is unverified (needs `nvidia-modelopt` and a Blackwell GPU). Muse tensor-name patterns come from the public Transformers code; the checkpoint is gated and its real keys are unchecked. No inference or benchmark has been validated.

The implementation direction is a fresh build. The OLMoE prototype is reference material only.

## Model Partitioner runbook

Layout of `model_partitioner/`: `cli.py` entrypoint; `partition/` (adapters, source, partitioner); `verification/` (verifier, output_check); `artifact/` (loader, hub); `utils/` (constants, models, tensor_utils).

Run on a larger machine. Disk: OLMoE about 45 GB; Muse about 537 GB for the cached original, 537 GB for the partitions and, for `verify --cpu/--gpu`, a third 537 GB reassembled checkpoint (scratch next to the partitions, removed afterwards): roughly 1.6 TB. Install only the needed dependency group: `model-partitioner`, or `model-partitioner-outputs` for output checks (`uv sync --only-group model-partitioner-outputs`, or `pip install --group ...` with pip 25.1+; use a CUDA torch index for GPUs). Authenticate with `hf auth login`; Muse also needs its license accepted on the Hub.

```
python -m model_partitioner partition --model allenai/OLMoE-1B-7B-0924 [--output ./partitions]
python -m model_partitioner verify ./partitions [--cpu | --gpu]
python -m model_partitioner upload ./partitions --repo you/olmoe-partitions
```

- `partition --model` takes exactly `allenai/OLMoE-1B-7B-0924` or `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`. It resolves the repo's current commit SHA, validates the config, then downloads weights at that SHA into the Hugging Face cache (reused on later runs), checks every shard against the Hub's sha256 and records repo and SHA in the manifest. The default output is `./partitions/<model name>`, which must be empty or new.
- `verify` finds the original through the manifest's repo and SHA in the cache (downloading what is missing). Default: checksums, coverage, exact reconstruction, no GPU. `--cpu` or `--gpu` (automatic placement over all GPUs) also compare original and reassembled outputs. Muse NVFP4 output checks run only with `--gpu` on Blackwell with `nvidia-modelopt`.
- `upload` checks local files, creates the repo as private if missing (an existing repo keeps its visibility), uploads only the files the manifest declares, and checks them against the manifest at the resulting revision. The result is an EveryGPU artifact, not a `from_pretrained` checkpoint. Large folders are split across several commits, and an interrupted upload is rerun from the start.
- Exit 0: all requested checks passed. 1: a check or command failed. 2: a requested check could not run (no GPU, source unavailable, unsupported hardware); argparse usage errors also exit 2.

## Collaboration

Use simple explanations and concrete request traces. Keep plan documents focused on goals, requirements, steps, decisions and verification rather than prescribing implementation details. Distinguish agreed requirements from proposals and measured results from assumptions.
