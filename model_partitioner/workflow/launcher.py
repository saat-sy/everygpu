"""Runs the parent-side stages, then hands over to the dependency-ready workflow."""

import os
import sys
from pathlib import Path

from model_partitioner.utils import constants, models, progress
from model_partitioner.workflow import config, environment, report


def _summary(tracker: progress.Progress, cfg: models.RunConfig) -> None:
    source = cfg.source_dir or "Hugging Face (pinned commit)"
    lines = [
        f"model:    {cfg.model}",
        f"source:   {source}",
        f"output:   {cfg.artifact_dir}" + (" (existing)" if cfg.verify_only else ""),
        "verify:   structural"
        + (f" + {cfg.verify} output comparison" if cfg.verify else ""),
        f"upload:   {cfg.upload or 'no'}",
        f"report:   {cfg.report_dir}",
    ]
    tracker.say("\n".join(lines))


def launch(argv: list[str], **options) -> int:
    """Validate, prepare the environment, run the workflow; return the exit code."""
    resumed = constants.RUN_DIR_ENV in os.environ
    directory = (
        Path(os.environ[constants.RUN_DIR_ENV])
        if resumed
        else report.unique_dir(Path(constants.REPORTS_DIR))
    )
    if not resumed and (clash := config.report_overlap(directory, **options)):
        print(f"error: {clash}", file=sys.stderr)
        return constants.EXIT_FAILED
    tracker = progress.Progress(sys.stdout, directory / constants.LOG_NAME)
    try:
        if resumed:
            run = report.Report.resume(directory, tracker)
        else:
            directory.parent.mkdir(parents=True, exist_ok=True)
            run = report.Report.create(directory, options, tracker)
    except OSError as error:
        print(
            f"error: cannot create report directory {directory}: {error}",
            file=sys.stderr,
        )
        return constants.EXIT_FAILED
    try:
        cfg = config.resolve(report_dir=directory, **options)
    except config.ConfigError as error:
        run.fail_stage(constants.STAGE_PREFLIGHT, error)
        return run.finish(constants.FAILED, error)
    run.plan(cfg, config.stage_names(cfg), config.skipped_stages(cfg))
    try:
        if not resumed:
            _summary(tracker, cfg)
            with run.stage(constants.STAGE_PREFLIGHT):
                tracker.say(
                    "  PASSED      arguments, modes, paths and output locations"
                )
            with run.stage(constants.STAGE_ENVIRONMENT):
                python = environment.prepare(cfg, tracker)
            if python:
                environment.reexec(python, argv, directory)
        elif unusable := environment.problems(cfg):
            joined = "; ".join(unusable)
            raise models.StageError(f"prepared environment is not usable: {joined}")
        from model_partitioner.workflow import pipeline

        return pipeline.Workflow(cfg, run, tracker).run()
    except BaseException as error:
        return run.finish(report.outcome_of(error), error)
