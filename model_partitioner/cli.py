"""Command-line entrypoint: python -m model_partitioner [options]."""

import argparse
import signal
import sys
from pathlib import Path

from model_partitioner.utils import constants, models
from model_partitioner.workflow import launcher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m model_partitioner",
        description="Download or read a model, partition it, verify it and optionally upload it.",
    )
    parser.add_argument(
        "--model",
        help=f"Hugging Face repo ID; selects the model adapter (default {constants.DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="local checkpoint directory instead of a download",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="new or empty partition directory (default ./partitions/<model name>)",
    )
    parser.add_argument(
        "--verify",
        choices=(constants.VERIFY_CPU, constants.VERIFY_GPU),
        help="additionally compare model outputs; structural checks always run",
    )
    parser.add_argument(
        "--upload",
        metavar="REPO",
        help="upload to this Hub repo only if every check passes",
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        metavar="DIR",
        help="verify existing partitions (never repartitions); model and source come from the manifest",
    )
    return parser


def _terminate(signum, _frame) -> None:
    raise models.Interrupted(signum)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    try:
        args = _parser().parse_args(argv)
    except SystemExit as usage:
        sys.exit(usage.code and constants.EXIT_FAILED)
    signal.signal(signal.SIGTERM, _terminate)
    sys.exit(launcher.launch(argv, **vars(args)))
