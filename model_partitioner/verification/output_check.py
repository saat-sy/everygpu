"""Checks original and reassembled models give equivalent outputs."""

import gc
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import transformers

from model_partitioner.utils import constants, models
from model_partitioner.artifact import hub, loader
from model_partitioner.partition import adapters


def _dtype(adapter: adapters.Adapter, config: dict):
    for section in (config, adapter.text_config(config)):
        for field in ("dtype", "torch_dtype"):
            if section.get(field):
                return section[field]
    return "auto"


def _logits(
    model_dir: Path,
    adapter: adapters.Adapter,
    config: dict,
    ids: torch.Tensor,
    options: models.RunOptions,
) -> torch.Tensor:
    model_class = getattr(transformers, adapter.model_class)
    kwargs = {"dtype": _dtype(adapter, config)}
    if options.device_map:
        kwargs["device_map"] = options.device_map
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


def _arch_runs(arch: str, capability: tuple[int, int]) -> bool:
    kind, _, number = arch.partition("_")
    major, minor = divmod(int(number.rstrip("af")), 10)
    if kind == "sm":
        return major == capability[0] and minor <= capability[1]
    return (major, minor) <= capability


def _gpu_blocker() -> str | None:
    if not torch.cuda.is_available():
        return "GPU requested but CUDA is not available"
    archs = torch.cuda.get_arch_list()
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        if not any(_arch_runs(arch, capability) for arch in archs):
            return (
                f"cuda:{index} (capability {capability}) is not supported "
                f"by this PyTorch build ({archs})"
            )
    return None


def _scratch_dir(artifact_dir: Path, needed: int) -> Path:
    """A fresh directory next to the artifact, never an existing one."""
    base = Path(artifact_dir).resolve().parent
    free = shutil.disk_usage(base).free
    if free < needed:
        raise OSError(
            f"scratch needs about {needed:,} bytes but {base} has {free:,} free"
        )
    return Path(tempfile.mkdtemp(prefix=".verify-scratch-", dir=base))


def _unverified(reason: str) -> models.CheckResult:
    return models.CheckResult(constants.OUTPUT_CHECK_NAME, constants.UNVERIFIED, reason)


def check_outputs(
    artifact_dir: Path, options: models.RunOptions = models.RunOptions()
) -> models.CheckResult:
    """Compare logits of the pinned source and a checkpoint reassembled in scratch space."""
    gpu = options.device_map is not None or options.device.startswith("cuda")
    if gpu and (blocker := _gpu_blocker()):
        return _unverified(blocker)
    with loader.Artifact(artifact_dir) as artifact:
        manifest = artifact.manifest
        adapter = adapters.get_adapter(manifest.family)
        if blocker := adapter.output_blocker(manifest.config, gpu):
            return _unverified(blocker)
        needed = sum(info.nbytes for info in manifest.source_tensors.values())
        try:
            source_dir = hub.fetch_source(manifest.repo_id, manifest.revision)
            scratch = _scratch_dir(artifact_dir, needed)
        except (ValueError, OSError) as error:
            return _unverified(str(error))
        print(
            f"scratch: {scratch} (about {needed:,} bytes)", file=sys.stderr, flush=True
        )
        try:
            return _compare(artifact, source_dir, adapter, options, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _compare(
    artifact: loader.Artifact,
    source_dir: Path,
    adapter: adapters.Adapter,
    options: models.RunOptions,
    scratch: Path,
) -> models.CheckResult:
    config = artifact.manifest.config
    generator = torch.Generator().manual_seed(constants.OUTPUT_CHECK_SEED)
    ids = torch.randint(
        0,
        adapter.vocab_size(config),
        (constants.OUTPUT_CHECK_BATCH, constants.OUTPUT_CHECK_SEQ_LEN),
        generator=generator,
    )
    reference = _logits(source_dir, adapter, config, ids, options)
    reassembled_dir = scratch / "reassembled"
    artifact.export_checkpoint(reassembled_dir)
    candidate = _logits(reassembled_dir, adapter, config, ids, options)
    diff = (reference - candidate).abs().max().item()
    close = torch.allclose(
        reference, candidate, atol=constants.LOGITS_ATOL, rtol=constants.LOGITS_RTOL
    )
    detail = (
        f"max abs logits diff {diff:.3e} (atol {constants.LOGITS_ATOL}, "
        f"rtol {constants.LOGITS_RTOL}), bitwise equal: {bool(diff == 0)}, "
        f"device {options.device_map or options.device}"
    )
    return models.CheckResult(
        constants.OUTPUT_CHECK_NAME,
        constants.PASSED if close else constants.FAILED,
        detail,
    )
