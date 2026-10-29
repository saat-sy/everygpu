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

All eleven plans are written. Implementation has not started. No inference, model compatibility, shard upload or benchmark has been validated in this checkout.

The implementation direction is a fresh build. The OLMoE prototype is reference material only.

## Collaboration

Use simple explanations and concrete request traces. Keep plan documents focused on goals, requirements, steps, decisions and verification rather than prescribing implementation details. Distinguish agreed requirements from proposals and measured results from assumptions.
