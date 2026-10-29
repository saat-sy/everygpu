"""Loads units and tensors from a partitioned artifact directory."""
import contextlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from model_partitioner import constants, models, tensor_utils


class ChecksumMismatch(ValueError):
    """Raised when a loaded tensor does not match its manifest checksum."""


def read_manifest(artifact_dir: Path) -> models.Manifest:
    text = (Path(artifact_dir) / constants.MANIFEST_NAME).read_text()
    return models.Manifest.from_dict(json.loads(text))


class Artifact:
    """A partitioned artifact: units load independently of one another."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = read_manifest(self.root)
        self.units = {unit.id: unit for unit in self.manifest.units}
        self.pieces = {piece.name: piece for piece in self.manifest.tensors}
        self.pieces_by_key = defaultdict(list)
        for piece in self.manifest.tensors:
            self.pieces_by_key[piece.source_key].append(piece)
        self._stack = contextlib.ExitStack()
        self._handles: dict[str, object] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def _handle(self, file: str):
        if file not in self._handles:
            self._handles[file] = self._stack.enter_context(
                safe_open(self.root / file, framework='pt'))
        return self._handles[file]

    def load_piece(self, name: str, verify: bool = True) -> torch.Tensor:
        piece = self.pieces[name]
        tensor = self._handle(piece.file).get_tensor(name)
        if verify and tensor_utils.tensor_sha256(tensor) != piece.sha256:
            raise ChecksumMismatch(f'{name} in {piece.file}')
        return tensor

    def load_unit(self, unit_id: str, verify: bool = True) -> dict[str, torch.Tensor]:
        """Shard-tensor name to tensor for one unit; dependencies are not loaded."""
        return {name: self.load_piece(name, verify) for name in self.units[unit_id].tensors}

    def tensor(self, key: str, verify: bool = True) -> torch.Tensor:
        """Rebuild one original checkpoint tensor from its pieces."""
        pieces = sorted(self.pieces_by_key[key], key=lambda piece: piece.slice_start or 0)
        if not pieces:
            raise KeyError(key)
        if pieces[0].slice_start is None:
            return self.load_piece(pieces[0].name, verify)
        edges = [(piece.slice_start, piece.slice_stop) for piece in pieces]
        expected = [(i, i + 1) for i in range(self.manifest.source_tensors[key].shape[0])]
        if edges != expected:
            raise ValueError(f'{key}: slices do not tile the expert dimension')
        return torch.cat([self.load_piece(piece.name, verify) for piece in pieces], dim=0)

    def export_checkpoint(self, out_dir: Path) -> None:
        """Write original-named shards, index and assets that Transformers can load."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        weight_map, shard, shard_bytes, names = {}, {}, 0, []
        for key, info in self.manifest.source_tensors.items():
            shard[key] = self.tensor(key)
            shard_bytes += info.nbytes
            if shard_bytes >= constants.REASSEMBLED_SHARD_BYTES:
                names.append(self._flush(out_dir, shard, weight_map, len(names)))
                shard, shard_bytes = {}, 0
        if shard:
            names.append(self._flush(out_dir, shard, weight_map, len(names)))
        total = sum(info.nbytes for info in self.manifest.source_tensors.values())
        index = {'metadata': {'total_size': total}, 'weight_map': weight_map}
        (out_dir / constants.INDEX_NAME).write_text(json.dumps(index))
        for name in self.manifest.assets:
            shutil.copy2(self.root / constants.ASSETS_DIR / name, out_dir / name)

    @staticmethod
    def _flush(out_dir, shard, weight_map, number) -> str:
        name = f'model-{number:05d}.safetensors'
        save_file(shard, out_dir / name, metadata={'format': 'pt'})
        weight_map.update({key: name for key in shard})
        return name
