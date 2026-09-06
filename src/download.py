import sys

from huggingface_hub import hf_hub_download

import config


def download_files(files):
    for filename in files:
        print(f"Downloading {filename}...")
        hf_hub_download(
            repo_id=config.MODEL_REPOSITORY,
            filename=filename,
            local_dir=".",
        )


def download_coordinator():
    download_files(config.COORDINATOR_FILES)


def download_stage(stage: int):
    download_files([f"stage-{stage}.safetensors", "config.json"])


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "coordinator"
    download_coordinator() if target == "coordinator" else download_stage(int(target))


if __name__ == "__main__":
    main()
