# Pinned source and publishing

## Decision

`partition --model` resolves the repo's current commit SHA once and uses it for the config, the weights and the manifest. `verify` finds the original from the manifest's repo and SHA through the Hugging Face cache. `upload` sends only the files the manifest declares and checks them at the resulting revision.

## Why

- The user chooses a model repository and destination; revision pinning and source-cache lookup are internal details, not required CLI arguments.
- Separate partition, verify and upload commands let the user inspect local results and choose weight-only or model-output verification before publication. Preparing and checking weights does not require full-model GPU execution.
- A moving branch would let the original change under a finished artifact; a SHA makes verification repeatable.
- Every source shard must match the Hub's sha256, and a missing hash fails, so provenance is never silently skipped.
- Publishing only declared files keeps source weights, cache and scratch out of the repo.
- Repos are created private; an existing repo keeps its visibility.
- The uploaded repo is an EveryGPU artifact for the worker loader, not a `from_pretrained` checkpoint.
