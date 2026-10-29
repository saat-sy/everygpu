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

Run on a larger machine (OLMoE needs about 45 GB of disk; Muse about 537 GB source, 537 GB partitions and, for `verify --outputs`, a third 537 GB reassembled checkpoint: roughly 1.6 TB, plus any download cache). Install only its dependency group: `model-partitioner` for inspect, partition, upload and structural verify; `model-partitioner-outputs` also for `verify --outputs` (`uv sync --only-group model-partitioner-outputs`, or `pip install --group model-partitioner-outputs` with pip 25.1+; use a CUDA torch index for GPUs). Then run `python -m model_partitioner` from the repo root.

```
# headers only, no weights (Muse needs HF_TOKEN with the license accepted)
python -m model_partitioner inspect --family olmoe --repo-id allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5
hf download allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --local-dir SRC
python -m model_partitioner partition --family olmoe --source SRC --repo-id allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --out ART --check-hub-hashes
python -m model_partitioner verify --artifact ART --source SRC --outputs WORK [--device cuda:0 | --device-map auto --max-memory 0=70GiB,cpu=200GiB]
python -m model_partitioner upload --artifact ART --dest-repo OWNER/REPO     # prints the uploaded revision
python -m model_partitioner verify-upload --artifact ART --dest-repo OWNER/REPO --revision SHA --load-unit expert:0:3 --load-dir DIR
```

`verify` exits 0 when every check passes, 1 when any fails and 2 when a requested check could not run (for example Muse output equivalence without Blackwell hardware).

## Collaboration

Use simple explanations and concrete request traces. Keep plan documents focused on goals, requirements, steps, decisions and verification rather than prescribing implementation details. Distinguish agreed requirements from proposals and measured results from assumptions.
