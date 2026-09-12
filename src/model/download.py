"""Download model artifacts from the configured Hugging Face repository."""

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

import config


def download_files(files, local_dir: str | Path = "."):
    destination = Path(local_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for filename in files:
        print(f"Downloading {filename}...")
        hf_hub_download(
            repo_id=config.MODEL_REPOSITORY,
            filename=filename,
            local_dir=destination,
        )


def download_coordinator(local_dir: str | Path = "."):
    download_files(config.COORDINATOR_FILES, local_dir)


def download_stage(stage: int, local_dir: str | Path = "."):
    download_files([f"stage-{stage}.safetensors", "config.json"], local_dir)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "coordinator"
    download_coordinator() if target == "coordinator" else download_stage(int(target))


if __name__ == "__main__":
    main()
