"""Readers for source checkpoints: header inspection and tensor reads."""
import contextlib
import fnmatch
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from model_partitioner import constants, models, tensor_utils


def spec_from_slice(key: str, file: str, tensor_slice) -> models.TensorSpec:
    shape = tuple(tensor_slice.get_shape())
    dtype = tensor_slice.get_dtype()
    nbytes = tensor_utils.numel(shape) * constants.DTYPE_BYTES[dtype]
    return models.TensorSpec(key, file, shape, dtype, nbytes)


class LocalSource:
    """A local checkpoint directory of .safetensors files plus config."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._stack = contextlib.ExitStack()
        self._handles: dict[str, object] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stack.close()

    @property
    def files(self) -> list[str]:
        return sorted(path.name for path in self.root.glob('*.safetensors'))

    def config(self) -> dict:
        return json.loads((self.root / 'config.json').read_text())

    def _handle(self, file: str):
        if file not in self._handles:
            self._handles[file] = self._stack.enter_context(
                safe_open(self.root / file, framework='pt'))
        return self._handles[file]

    def _check_index(self, specs: list[models.TensorSpec]) -> None:
        path = self.root / constants.INDEX_NAME
        if not path.is_file():
            if self.files != ['model.safetensors']:
                raise models.IncompleteSource(
                    f'{self.files or "no shards"}: need {constants.INDEX_NAME} or a single model.safetensors')
            return
        index = json.loads(path.read_text())
        expected = {}
        for key, file in index['weight_map'].items():
            expected.setdefault(file, set()).add(key)
        if missing := set(expected) - set(self.files):
            raise models.IncompleteSource(f'shards listed in index but absent: {sorted(missing)}')
        if extra := set(self.files) - set(expected):
            raise models.IncompleteSource(f'shards on disk but not in index: {sorted(extra)}')
        found = {}
        for spec in specs:
            found.setdefault(spec.file, set()).add(spec.key)
        for file, keys in expected.items():
            if keys != found[file]:
                diff = sorted(keys ^ found[file])[:5]
                raise models.IncompleteSource(f'{file}: tensors differ from index, e.g. {diff}')
        total = index.get('metadata', {}).get('total_size')
        if total is not None and total != sum(spec.nbytes for spec in specs):
            raise models.IncompleteSource('index total_size differs from shard tensor bytes')

    def specs(self) -> list[models.TensorSpec]:
        specs = []
        for file in self.files:
            handle = self._handle(file)
            specs.extend(spec_from_slice(key, file, handle.get_slice(key))
                         for key in handle.keys())
        keys = [spec.key for spec in specs]
        if len(keys) != len(set(keys)):
            raise models.UnsupportedLayout('duplicate tensor keys across shards')
        self._check_index(specs)
        return sorted(specs, key=lambda spec: spec.key)

    def shard_metadata(self, file: str) -> dict[str, str]:
        return dict(self._handle(file).metadata() or {})

    def read(self, spec: models.TensorSpec, start: int | None = None,
             stop: int | None = None) -> torch.Tensor:
        handle = self._handle(spec.file)
        if start is None:
            return handle.get_tensor(spec.key)
        return handle.get_slice(spec.key)[start:stop]

    def asset_paths(self) -> list[Path]:
        names = {path.name for path in self.root.iterdir() if path.is_file()}
        matched = {name for name in names
                   if any(fnmatch.fnmatch(name, pat) for pat in constants.ASSET_PATTERNS)}
        return [self.root / name for name in sorted(matched)]


def copy_assets(source: LocalSource, dest: Path) -> dict[str, models.FileInfo]:
    dest.mkdir(parents=True, exist_ok=True)
    infos = {}
    for path in source.asset_paths():
        shutil.copy2(path, dest / path.name)
        infos[path.name] = models.FileInfo(path.stat().st_size, tensor_utils.file_sha256(path))
    return infos
