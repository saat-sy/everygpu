import sys

from huggingface_hub import hf_hub_download

import constants


def download_files(files):
    for f in files:
        print(f"Downloading {f}...")
        hf_hub_download(
            repo_id=constants.MODEL_REPOSITORY,
            filename=f,
            local_dir=".",
        )


def download_laptop():
    download_files(constants.LAPTOP_FILES)


def download_stage(stage: int):
    download_files([f"stage-{stage}.safetensors", "config.json"])


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "laptop"
    download_laptop() if target == "laptop" else download_stage(int(target))
