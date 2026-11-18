# Model Partitioner cheat sheet

One command downloads or reads a model, partitions it, verifies it and optionally uploads it. Run from the repo root.

```bash
# 1. Hub download, partition, structural verify, upload (Muse is the default model)
python -m model_partitioner --model meta-models/Muse-Spark-Open-816B-A42B-NVFP4 --upload saatwiksy/muse-partitions
# 2. Local source, partition, structural verify, upload
python -m model_partitioner --source-dir /models/muse --upload saatwiksy/muse-partitions
# 3. Local source plus GPU output comparison before upload
python -m model_partitioner --source-dir /models/muse --verify gpu --upload saatwiksy/muse-partitions
# 4. OLMoE plus CPU output comparison, no upload
python -m model_partitioner --model allenai/OLMoE-1B-7B-0924 --verify cpu
# 5. Local partition and structural verify only
python -m model_partitioner --source-dir /models/muse
# 6. Existing partitions, structural verify only
python -m model_partitioner --verify-only ./partitions/muse --source-dir /models/muse
# 7. Existing partitions plus GPU comparison
python -m model_partitioner --verify-only ./partitions/muse --source-dir /models/muse --verify gpu
# 8. Existing partitions: verify, then upload
python -m model_partitioner --verify-only ./partitions/muse --source-dir /models/muse --upload saatwiksy/muse-partitions
```

## Flags

| Flag | Meaning |
|---|---|
| `--model REPO` | Hub repo; selects the adapter (default Muse). With `--source-dir` it only selects the adapter. |
| `--source-dir DIR` | Local checkpoint (`config.json`, `.safetensors` shards, index) instead of a download. |
| `--output DIR` | New or empty partition directory (default `./partitions/<model name>`). |
| `--verify cpu\|gpu` | Additional model-output comparison. Structural verification (checksums, tensor coverage, exact reconstruction) always runs. Muse supports only `gpu`. |
| `--upload REPO` | Upload only if every requested check passes. New repos are private; existing visibility is unchanged. |
| `--verify-only DIR` | Verify existing partitions; never downloads the latest or repartitions. Model, config and revision come from the manifest. |

Contradictory flags are rejected (`--verify-only` with `--output`, or a `--model` that differs from the manifest). Hub-sourced artifacts may omit `--source-dir` with `--verify-only`: the exact pinned commit is fetched. Locally sourced artifacts require it.

## Setup

Prerequisites: Python 3.12+ and, only when dependencies are missing, `uv` on `PATH`. For Hub access or upload run `hf auth login` or set `HF_TOKEN` (write token for upload; Muse also needs approved access).

The command checks the current interpreter first. If `torch`, `safetensors` and `huggingface_hub` (plus `transformers` and `accelerate` with `--verify`) are present it runs there. Otherwise it reuses or creates `./.venv-model-partitioner` with `uv venv --no-python-downloads` and runs `uv pip install --group model-partitioner` (`model-partitioner-outputs` with `--verify`), then re-runs itself in that environment under the same report. Interpreters are never downloaded and no other dependency groups are resolved. The interpreter must satisfy `requires-python` from `pyproject.toml`, and every package of the chosen dependency group must import and meet its pyproject version constraint (read from `pyproject.toml`, only simple `>=`, `==` style bounds). That is an import and version check, not a test of every API. A missing, outdated or broken environment uses the isolated setup, or the run stops early with a report. If setup cannot run, the run stops and the report holds the diagnostic.

## Checks before expensive work

Argument and mode validity (for example Muse with `--verify cpu`), disjoint source, partitions and report paths including symlinks and ancestors, empty output, writable locations, remote config and source access (gated models included), local source existence, config model match and complete shards, free disk per filesystem (cached weights are not counted as new download; the partition and output-check scratch needs are summed), GPU compatibility for `--verify gpu`, and with `--upload` the token, destination and write permission. The upload check uses the Hub's write check for existing repos; for a repo that does not exist yet it checks token role and namespace, which does not prove create permission until upload. Tensor names, shapes and dtypes of Hub sources are checked after download. Auth and destination are checked again right before upload.

## Output

A summary is followed by numbered stages (only those that apply): Preflight checks, Prepare environment, Check access and resources, Fetch source, Validate source weights, Partition, Verify, Upload, Report. Known totals (shards hashed, files written, tensors reconstructed) show a counter bar; other work shows elapsed time. Hugging Face download and upload progress is its own. On a terminal indicators update in place; logs and non-TTY output are plain lines. The run ends with `PASSED`, `FAILED`, `INCOMPLETE` or `INTERRUPTED`, elapsed time and report paths. The uploaded URL is printed only after upload and remote integrity both pass.

Exit codes: `0` passed, `1` failed, `2` incomplete (a check could not verify), `128+signal` interrupted.

## Upload gate

Upload happens only if file checksums, coverage, exact reconstruction and any `--verify` comparison all PASS. A failed, skipped or unverified required check blocks it and the reason is printed. After upload, every declared file's size and sha256 on the Hub is compared with the manifest; a mismatch fails the run. Remote changes are not rolled back.

## Reports

Every run writes `./reports/<UTC timestamp>/` with `run.log`, `report.json` and `report.md`, outside the source and partitions. The report holds the requested configuration, model and source identity, executed and skipped stages with reasons, durations and exit codes, every check result, UTC start, end and elapsed time, host CPU, RAM, platform, Python and dependency versions, free disk space, manifest unit, file and byte counts, upload result and remote integrity outcome, and peak RAM. Peak RAM is the largest peak resident memory of this process (kept across the setup re-exec on Linux) or of any single waited child, where `resource` is supported; it is not per stage and not GPU memory. Total elapsed time spans the whole run including setup. Reports are saved after every stage and on setup, preflight, verification and upload errors, Ctrl+C and SIGTERM. A SIGKILL or machine loss cannot produce a final report; the last saved stage state remains. Tokens and the environment are never logged. Hugging Face's own progress output goes to the terminal, not `run.log`.

## Limits

Disk: partitions are about the size of the weights, so download plus partitions need about 28 GB combined for OLMoE (about 14 GB each) or 1.07 TB combined for Muse (about 533 GB each). `--verify` temporarily adds another copy (about 14 GB or 533 GB), giving about 42 GB or 1.60 TB combined, plus overhead; cached weights are not counted as new download. Muse output comparison needs a Blackwell GPU and `nvidia-modelopt`. Muse real-checkpoint compatibility and peak RAM remain unverified. Setup targets POSIX layouts (`.venv-model-partitioner/bin/python`).
