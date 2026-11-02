"""Builds and validates a RunConfig and derives its stage list. Stdlib only."""

import json
from pathlib import Path

from model_partitioner.partition import adapters
from model_partitioner.utils import constants, models, path_utils


class ConfigError(ValueError):
    """The requested combination of options cannot run."""


def _peek_manifest(artifact_dir: Path) -> dict:
    path = artifact_dir / constants.MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text())
        if manifest["schema_version"] != constants.SCHEMA_VERSION:
            raise ConfigError(
                f"{path}: manifest schema {manifest['schema_version']} is not "
                f"readable by this tool (schema {constants.SCHEMA_VERSION})"
            )
        return {"repo_id": manifest["repo_id"], "revision": manifest["revision"]}
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, ConfigError):
            raise
        raise ConfigError(f"cannot read {path}: {error}") from error


def report_overlap(report_dir: Path, **options) -> str | None:
    """Describe a user path that overlaps the report directory, if any."""
    for name in ("source_dir", "output", "verify_only"):
        path = options.get(name)
        if path and path_utils.overlaps(report_dir, path):
            return f"report directory {report_dir} overlaps --{name.replace('_', '-')} {path}"
    return None


def _check_paths(cfg: models.RunConfig) -> None:
    artifact = cfg.artifact_dir
    pairs = [("report directory", cfg.report_dir, "partitions", artifact)]
    if cfg.source_dir:
        pairs += [
            ("report directory", cfg.report_dir, "source", cfg.source_dir),
            ("partitions", artifact, "source", cfg.source_dir),
        ]
    for first_name, first, second_name, second in pairs:
        if path_utils.overlaps(first, second):
            raise ConfigError(f"{first_name} {first} overlaps {second_name} {second}")
    if cfg.verify_only is None:
        try:
            path_utils.require_empty_dir(artifact)
        except ValueError as error:
            raise ConfigError(str(error)) from error
        if not path_utils.is_writable(artifact):
            raise ConfigError(f"output {artifact} is not writable")


def resolve(
    report_dir: Path,
    model: str | None = None,
    source_dir: Path | None = None,
    output: Path | None = None,
    verify_only: Path | None = None,
    verify: str | None = None,
    upload: str | None = None,
) -> models.RunConfig:
    """Validate options without touching the network; raise ConfigError if unusable."""
    if verify_only:
        if output:
            raise ConfigError(
                "--verify-only reads existing partitions; remove --output"
            )
        if not Path(verify_only).is_dir():
            raise ConfigError(f"--verify-only {verify_only} is not a directory")
        manifest = _peek_manifest(Path(verify_only))
        if model and model != manifest["repo_id"]:
            raise ConfigError(
                f"--model {model} differs from the artifact's model {manifest['repo_id']}"
            )
        model = manifest["repo_id"]
        if source_dir is None and constants.LOCAL_REVISION_PATTERN.fullmatch(
            manifest["revision"]
        ):
            raise ConfigError(
                "artifact was built from a local source; --source-dir is required"
            )
    model = model or constants.DEFAULT_MODEL
    try:
        adapter = adapters.adapter_for_repo(model)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if source_dir and not Path(source_dir).is_dir():
        raise ConfigError(f"--source-dir {source_dir} is not a directory")
    if verify == constants.VERIFY_CPU and (
        blocker := adapter.output_blocker({}, False)
    ):
        raise ConfigError(blocker)
    if upload and not constants.REPO_ID_PATTERN.fullmatch(upload):
        raise ConfigError(f"--upload {upload!r} must look like namespace/name")
    if not verify_only:
        output = output or Path(constants.PARTITIONS_DIR) / model.split("/")[-1]
    cfg = models.RunConfig(
        model=model,
        report_dir=Path(report_dir),
        source_dir=Path(source_dir) if source_dir else None,
        output=Path(output) if output else None,
        verify_only=Path(verify_only) if verify_only else None,
        verify=verify,
        upload=upload,
    )
    _check_paths(cfg)
    return cfg


def skipped_stages(cfg: models.RunConfig) -> dict[str, str]:
    skipped = {}
    if cfg.source_dir:
        skipped[constants.STAGE_FETCH] = "local --source-dir"
    if cfg.verify_only:
        reason = "--verify-only uses existing partitions"
        skipped[constants.STAGE_VALIDATE] = reason
        skipped[constants.STAGE_PARTITION] = reason
    if not cfg.upload:
        skipped[constants.STAGE_UPLOAD] = "no --upload"
    return skipped


def stage_names(cfg: models.RunConfig) -> list[str]:
    skipped = skipped_stages(cfg)
    return [
        name
        for name in (
            constants.STAGE_PREFLIGHT,
            constants.STAGE_ENVIRONMENT,
            constants.STAGE_ACCESS,
            constants.STAGE_FETCH,
            constants.STAGE_VALIDATE,
            constants.STAGE_PARTITION,
            constants.STAGE_VERIFY,
            constants.STAGE_UPLOAD,
            constants.STAGE_REPORT,
        )
        if name not in skipped
    ]
