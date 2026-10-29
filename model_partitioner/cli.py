"""Command-line entrypoint: inspect, partition, verify, upload, verify-upload."""
import argparse
import json
import sys
from pathlib import Path

from model_partitioner import adapters, constants, hub, partitioner, source, verifier


EXIT_FAILED = 1
EXIT_INCOMPLETE = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='model-partitioner')
    sub = parser.add_subparsers(dest='command', required=True)

    inspect = sub.add_parser('inspect', help='plan from headers only; no weights are read')
    inspect.add_argument('--family', required=True, choices=sorted(adapters.ADAPTERS))
    inspect.add_argument('--source', type=Path, help='local checkpoint directory')
    inspect.add_argument('--repo-id')
    inspect.add_argument('--revision')
    inspect.add_argument('--experts-per-file', type=int, default=constants.DEFAULT_EXPERTS_PER_FILE)

    part = sub.add_parser('partition', help='write Safetensors units and manifest')
    part.add_argument('--family', required=True, choices=sorted(adapters.ADAPTERS))
    part.add_argument('--source', type=Path, required=True)
    part.add_argument('--repo-id', required=True)
    part.add_argument('--revision', required=True)
    part.add_argument('--out', type=Path, required=True)
    part.add_argument('--experts-per-file', type=int, default=constants.DEFAULT_EXPERTS_PER_FILE)
    part.add_argument('--check-hub-hashes', action='store_true',
                      help='compare source shard sha256 with Hub LFS hashes')

    check = sub.add_parser('verify', help='check an artifact against its source')
    check.add_argument('--artifact', type=Path, required=True)
    check.add_argument('--source', type=Path, required=True)
    check.add_argument('--outputs', type=Path, metavar='WORKDIR',
                       help='also compare model outputs, using WORKDIR for scratch files')
    check.add_argument('--device', default='cpu', help='device for --outputs, e.g. cuda:0')
    check.add_argument('--device-map', help='e.g. auto, to shard the model across devices')
    check.add_argument('--max-memory', help='per-device limits, e.g. 0=70GiB,1=70GiB,cpu=200GiB')

    upload = sub.add_parser('upload', help='upload an artifact to an explicit repo')
    upload.add_argument('--artifact', type=Path, required=True)
    upload.add_argument('--dest-repo', required=True)
    upload.add_argument('--public', action='store_true')

    verify_up = sub.add_parser('verify-upload', help='check an uploaded revision against the manifest')
    verify_up.add_argument('--artifact', type=Path, required=True)
    verify_up.add_argument('--dest-repo', required=True)
    verify_up.add_argument('--revision', required=True)
    verify_up.add_argument('--load-unit', help='also download and load this unit, e.g. expert:0:3')
    verify_up.add_argument('--load-dir', type=Path, help='download directory for --load-unit')

    return parser


def _parse_max_memory(text: str | None) -> dict | None:
    if not text:
        return None
    pairs = (item.split('=') for item in text.split(','))
    return {(int(key) if key.isdigit() else key): value for key, value in pairs}


def _inspect(args) -> int:
    adapter = adapters.get_adapter(args.family)
    if args.source:
        with source.LocalSource(args.source) as src:
            specs, config = src.specs(), src.config()
    else:
        specs = hub.remote_specs(args.repo_id, args.revision)
        config = hub.remote_config(args.repo_id, args.revision)
    units = partitioner.plan(specs, adapter, config, args.experts_per_file)
    files = {unit.file for unit in units}
    print(json.dumps({'tensors': len(specs), 'files': len(files),
                      'roles': partitioner.summarize(units)}, indent=1))
    return 0


def _partition(args) -> int:
    expected = hub.lfs_hashes(args.repo_id, args.revision) if args.check_hub_hashes else None
    with source.LocalSource(args.source) as src:
        manifest = partitioner.partition(
            src, adapters.get_adapter(args.family), args.out, args.repo_id, args.revision,
            args.experts_per_file, expected)
    size = sum(info.size for info in manifest.files.values())
    print(f'{len(manifest.units)} units, {len(manifest.files)} files, {size} bytes')
    return 0


def _verify(args) -> int:
    results = verifier.verify_structure(args.artifact, args.source)
    if args.outputs:
        from model_partitioner import output_check
        args.outputs.mkdir(parents=True, exist_ok=True)
        options = output_check.RunOptions(args.device, args.device_map,
                                          _parse_max_memory(args.max_memory))
        results.append(output_check.check_outputs(args.artifact, args.source, args.outputs,
                                                  options))
    for result in results:
        print(f'{result.status:11} {result.name} {result.detail}')
    statuses = {result.status for result in results}
    if verifier.FAILED in statuses:
        return EXIT_FAILED
    return EXIT_INCOMPLETE if verifier.UNVERIFIED in statuses else 0


def _upload(args) -> int:
    revision = hub.upload_artifact(args.artifact, args.dest_repo, private=not args.public)
    print(f'uploaded revision {revision}')
    return _report_upload(args.artifact, args.dest_repo, revision)


def _verify_upload(args) -> int:
    status = _report_upload(args.artifact, args.dest_repo, args.revision)
    if args.load_unit and status == 0:
        count = hub.check_load(args.dest_repo, args.revision, args.load_unit, args.load_dir)
        print(f'loaded {count} tensors for {args.load_unit} and its dependencies')
    return status


def _report_upload(artifact: Path, repo: str, revision: str) -> int:
    problems = hub.check_upload(artifact, repo, revision)
    print('all declared files match the manifest' if not problems else '\n'.join(problems))
    return EXIT_FAILED if problems else 0


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == 'verify-upload' and bool(args.load_unit) != bool(args.load_dir):
        parser.error('--load-unit and --load-dir go together')
    handlers = {'inspect': _inspect, 'partition': _partition, 'verify': _verify,
                'upload': _upload, 'verify-upload': _verify_upload}
    sys.exit(handlers[args.command](args))
