"""Command-line entrypoint: partition, verify and upload."""

import argparse
import json
import sys
from pathlib import Path

from model_partitioner.utils import constants, models, tensor_utils
from model_partitioner.artifact import hub, loader
from model_partitioner.partition import adapters, partitioner, source
from model_partitioner.verification import verifier


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-partitioner")
    sub = parser.add_subparsers(dest="command", required=True)

    supported = ", ".join(a.repo_id for a in adapters.ADAPTERS.values())
    part = sub.add_parser(
        "partition",
        help="download a model and write Safetensors units and manifest",
        description="Pins the current commit, downloads it to the Hugging Face cache and writes partitions.",
    )
    part.add_argument(
        "--model", required=True, help=f"Hugging Face repo ID: {supported}"
    )
    part.add_argument(
        "--output",
        type=Path,
        help="empty or new directory (default: ./partitions/<model name>)",
    )

    check = sub.add_parser(
        "verify",
        help="check partitions against the original checkpoint",
        description="Checksums, coverage and exact reconstruction against the pinned original; no GPU by default.",
    )
    check.add_argument("partitions", type=Path, help="directory written by partition")
    mode = check.add_mutually_exclusive_group()
    mode.add_argument(
        "--cpu", action="store_true", help="also compare model outputs on the CPU"
    )
    mode.add_argument(
        "--gpu",
        action="store_true",
        help="also compare model outputs on the GPU(s), placed automatically",
    )

    upload = sub.add_parser(
        "upload",
        help="publish partitions to a Hugging Face model repo",
        description="Uploads the manifest-declared files, creating a private repo if missing, then verifies them.",
    )
    upload.add_argument("partitions", type=Path, help="directory written by partition")
    upload.add_argument(
        "--repo", required=True, help="destination, e.g. you/olmoe-partitions"
    )
    return parser


def _partition(args) -> int:
    adapter = adapters.adapter_for_repo(args.model)
    output = args.output or Path("partitions") / args.model.split("/")[1]
    tensor_utils.require_empty_dir(output)
    revision = hub.resolve_revision(args.model)
    config_path = hub.fetch_config(args.model, revision)
    adapter.check_config(json.loads(config_path.read_text()))
    partitioner.check_destination(config_path.parent, output)
    print(f"{args.model} @ {revision}")
    files = hub.remote_files(args.model, revision)
    _check_space(files, output)
    with source.LocalSource(hub.fetch_source(args.model, revision)) as src:
        print(partitioner.format_summary(partitioner.plan_source(src, adapter)))
        manifest = partitioner.partition(
            src,
            adapter,
            output,
            args.model,
            revision,
            {name: info.sha256 for name, info in files.items()},
        )
    size = sum(info.size for info in manifest.files.values())
    print(f"wrote {len(manifest.files)} files, {size:,} bytes to {output}")
    return constants.EXIT_OK


def _check_space(files: dict[str, models.FileInfo], output: Path) -> None:
    """Report the download and output sizes; refuse if the output cannot fit."""
    weights = sum(i.size for n, i in files.items() if n.endswith(".safetensors"))
    free = tensor_utils.free_bytes(output)
    print(
        f"download: {weights:,} bytes into the Hugging Face cache (skipped if cached)\n"
        f"output:   about {weights:,} bytes in {output} ({free:,} free there)"
    )
    if free < weights:
        raise ValueError(
            f"{output} needs about {weights:,} bytes but has {free:,} free"
        )


def _verify(args) -> int:
    results = verifier.verify_structure(args.partitions)
    if args.cpu or args.gpu:
        if any(r.status == constants.FAILED for r in results):
            results.append(
                models.CheckResult(
                    constants.OUTPUT_CHECK_NAME,
                    constants.UNVERIFIED,
                    "skipped: structural checks failed",
                )
            )
        else:
            from model_partitioner.verification import output_check

            options = (
                models.RunOptions("cuda", "auto") if args.gpu else models.RunOptions()
            )
            results.append(output_check.check_outputs(args.partitions, options))
    for result in results:
        print(f"{result.status:11} {result.name} {result.detail}")
    statuses = {result.status for result in results}
    if constants.FAILED in statuses:
        return constants.EXIT_FAILED
    if constants.UNVERIFIED in statuses:
        return constants.EXIT_INCOMPLETE
    return constants.EXIT_OK


def _upload(args) -> int:
    with loader.Artifact(args.partitions) as artifact:
        local = verifier.check_files(artifact)
    print(f"{local.status:11} {local.name} {local.detail}")
    if local.status != constants.PASSED:
        return constants.EXIT_FAILED
    created = hub.ensure_repo(args.repo)
    print(
        f"{'created private repo' if created else 'existing repo, visibility unchanged'}: "
        f"{args.repo}"
    )
    revision = hub.upload_artifact(args.partitions, args.repo)
    problems = hub.check_upload(args.partitions, args.repo, revision)
    if problems:
        print("\n".join(problems))
        return constants.EXIT_FAILED
    print(
        f"uploaded {args.repo} at revision {revision}; declared files match the manifest"
    )
    return constants.EXIT_OK


def main() -> None:
    try:
        args = _parser().parse_args()
    except SystemExit as usage:
        sys.exit(usage.code and constants.EXIT_FAILED)
    handlers = {"partition": _partition, "verify": _verify, "upload": _upload}
    try:
        status = handlers[args.command](args)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        status = constants.EXIT_FAILED
    sys.exit(status)
