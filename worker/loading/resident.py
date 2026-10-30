"""Selected units resident in GPU memory."""

import torch

from model_partitioner.artifact import loader
from model_partitioner.utils import models
from worker.loading import download
from worker.utils import constants


def load(
    repo: str, revision: str, manifest: models.Manifest, unit_ids: list[str]
) -> dict[str, dict[str, torch.Tensor]]:
    """Download and verify the files, then keep only the selected units on the GPU."""
    units = {unit.id: unit for unit in manifest.units}
    files = sorted({units[u].file for u in unit_ids})
    root = download.download_verified(repo, revision, manifest, files)
    resident = {}
    with loader.Artifact(root) as artifact:
        for i, unit_id in enumerate(unit_ids, 1):
            resident[unit_id] = {
                name: tensor.to(constants.DEVICE)
                for name, tensor in artifact.load_unit(unit_id).items()
            }
            print(f"\rloaded {i}/{len(unit_ids)} units", end="", flush=True)
    torch.cuda.synchronize()
    print()
    return resident
