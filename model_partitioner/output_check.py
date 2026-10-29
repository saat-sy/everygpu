"""Checks original and reassembled models give equivalent outputs."""
import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

from model_partitioner import adapters, constants, loader, verifier

_NAME = 'model output equivalence'


@dataclass(frozen=True)
class RunOptions:
    """Where and how models run; device_map shards one model across devices."""
    device: str = 'cpu'
    device_map: str | None = None
    max_memory: dict | None = None


def _dtype(adapter: adapters.Adapter, config: dict):
    for section in (config, adapter.text_config(config)):
        for field in ('dtype', 'torch_dtype'):
            if section.get(field):
                return section[field]
    return 'auto'


def _logits(model_dir: Path, adapter: adapters.Adapter, config: dict, ids: torch.Tensor,
            options: RunOptions) -> torch.Tensor:
    model_class = getattr(transformers, adapter.model_class)
    kwargs = {'dtype': _dtype(adapter, config)}
    if options.device_map:
        kwargs.update(device_map=options.device_map, max_memory=options.max_memory)
    model = model_class.from_pretrained(model_dir, **kwargs)
    if not options.device_map:
        model.to(options.device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ids.to(model.device)).logits.float().cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def check_outputs(artifact_dir: Path, source_dir: Path, workdir: Path,
                  options: RunOptions = RunOptions()) -> verifier.CheckResult:
    workdir = Path(workdir)
    with loader.Artifact(artifact_dir) as artifact:
        manifest = artifact.manifest
        adapter = adapters.get_adapter(manifest.family)
        if blocker := adapter.output_blocker(manifest.config):
            return verifier.CheckResult(_NAME, verifier.UNVERIFIED, blocker)
        generator = torch.Generator().manual_seed(constants.OUTPUT_CHECK_SEED)
        ids = torch.randint(0, adapter.vocab_size(manifest.config),
                            (constants.OUTPUT_CHECK_BATCH, constants.OUTPUT_CHECK_SEQ_LEN),
                            generator=generator)
        reference = _logits(source_dir, adapter, manifest.config, ids, options)
        reassembled_dir = workdir / 'reassembled'
        artifact.export_checkpoint(reassembled_dir)
    candidate = _logits(reassembled_dir, adapter, manifest.config, ids, options)
    diff = (reference - candidate).abs().max().item()
    close = torch.allclose(reference, candidate, atol=constants.LOGITS_ATOL,
                           rtol=constants.LOGITS_RTOL)
    detail = (f'max abs logits diff {diff:.3e} (atol {constants.LOGITS_ATOL}, '
              f'rtol {constants.LOGITS_RTOL}), bitwise equal: {bool(diff == 0)}, '
              f'device {options.device_map or options.device}')
    return verifier.CheckResult(_NAME, verifier.PASSED if close else verifier.FAILED, detail)
