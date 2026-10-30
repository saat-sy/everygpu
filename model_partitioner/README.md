# Model Partitioner cheat sheet

Run from the repo root with Python 3.12+ and `uv` installed.

## Setup

```bash
uv sync --only-group model-partitioner
source .venv/bin/activate
hf auth login
```

Use a Hugging Face write token for uploads. Muse also requires approved model access.

## OLMoE

```bash
python -m model_partitioner partition --model allenai/OLMoE-1B-7B-0924 --output ./partitions/olmoe
python -m model_partitioner verify ./partitions/olmoe
python -m model_partitioner upload ./partitions/olmoe --repo saatwiksy/olmoe-partitions
```

## Muse NVFP4

```bash
python -m model_partitioner partition --model meta-models/Muse-Spark-Open-816B-A42B-NVFP4 --output ./partitions/muse
python -m model_partitioner verify ./partitions/muse
python -m model_partitioner upload ./partitions/muse --repo saatwiksy/muse-partitions
```

`partition` downloads automatically. Output directories must be new or empty. Omit `--output` to use `./partitions/<model name>/`. Change upload repo names as needed; new repos are private.

Plain `verify` checks weights without a GPU or model execution. For optional output comparison, install `uv sync --only-group model-partitioner-outputs`, then add `--cpu` or `--gpu` to `verify`. Muse CPU output execution is unsupported; its GPU path remains unverified.

Keep space for both the downloaded model and partitions: about 28 GB for OLMoE or 1.07 TB for Muse, plus overhead. Output comparison temporarily adds another model copy. Muse real-checkpoint compatibility and peak RAM remain unverified.

Exit codes: `0` passed, `1` failed, `2` could not verify. Upload after verification passes.
