"""Run state: stage records, JSON and Markdown reports, host facts. Stdlib only."""

import contextlib
import dataclasses
import datetime
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

from model_partitioner.utils import constants, models, path_utils, progress

_DEPENDENCIES = (
    "torch",
    "safetensors",
    "huggingface_hub",
    "transformers",
    "accelerate",
)
_ERROR_LIMIT = 2000


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def unique_dir(root: Path) -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / stamp
    suffix = 0
    while path.exists():
        suffix += 1
        path = Path(root) / f"{stamp}-{suffix}"
    return path


def outcome_of(error: BaseException) -> str:
    if isinstance(error, (KeyboardInterrupt, models.Interrupted)):
        return constants.INTERRUPTED
    if isinstance(error, models.StageError):
        return error.outcome
    return constants.FAILED


def _exit_code(outcome: str, error: BaseException | None) -> int:
    if outcome == constants.INTERRUPTED:
        signum = getattr(error, "signum", 2)
        return constants.EXIT_SIGNAL_BASE + signum
    return {
        constants.PASSED: constants.EXIT_OK,
        constants.INCOMPLETE: constants.EXIT_INCOMPLETE,
    }.get(outcome, constants.EXIT_FAILED)


def _host() -> dict:
    cpu = ""
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    ram = None
    with contextlib.suppress(OSError, ValueError, AttributeError):
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    versions = {}
    for name in _DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "dependencies": versions,
    }


def _peak_rss() -> dict:
    try:
        import resource
    except ImportError:
        return {"supported": False}
    unit = 1 if sys.platform == "darwin" else 1024
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return {
        "supported": True,
        "scope": "largest peak resident set of this process or any one waited child, whole run, not per stage, not GPU memory",
        "bytes": max(own, children) * unit,
    }


class Report:
    """Owns the run directory, stage records and saved report files."""

    def __init__(self, directory: Path, data: dict, tracker: progress.Progress):
        self.directory = Path(directory)
        self.data = data
        self.progress = tracker
        self.names: list[str] = []
        self._started = data["started_unix"]

    @classmethod
    def create(cls, directory: Path, options: dict, tracker: progress.Progress):
        Path(directory).mkdir(parents=True, exist_ok=False)
        data = {
            "started": _now(),
            "started_unix": time.time(),
            "outcome": constants.RUNNING,
            "requested": {k: str(v) if v else v for k, v in options.items()},
            "stages": [],
            "skipped_stages": {},
            "checks": [],
            "manifest": None,
            "upload": None,
            "identity": {},
            "host": _host(),
            "disk": {},
            "report_dir": str(Path(directory).resolve()),
            "log": str((Path(directory) / constants.LOG_NAME).resolve()),
        }
        return cls(directory, data, tracker)

    @classmethod
    def resume(cls, directory: Path, tracker: progress.Progress):
        data = json.loads((Path(directory) / constants.REPORT_JSON).read_text())
        report = cls(directory, data, tracker)
        report.data["host"] = _host()
        return report

    def plan(self, cfg: models.RunConfig, names: list[str], skipped: dict) -> None:
        self.names = names
        self.data["skipped_stages"] = skipped
        self.data["config"] = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in dataclasses.asdict(cfg).items()
        }

    def record_check(self, name: str, status: str, detail: str = "") -> None:
        self.data["checks"].append({"name": name, "status": status, "detail": detail})
        self.progress.say(f"  {status:11} {name} {detail}".rstrip())

    def _begin(self, name: str) -> dict:
        number = self.names.index(name) + 1 if name in self.names else None
        label = f"[{number}/{len(self.names)}] " if number else ""
        self.progress.say(f"\n{label}{name}")
        record = {
            "name": name,
            "number": number,
            "status": constants.RUNNING,
            "started": _now(),
            "seconds": None,
            "exit_code": None,
            "detail": "",
        }
        self.data["stages"].append(record)
        self.save()
        return record

    def _end(self, record: dict, started: float, error: BaseException | None) -> None:
        outcome = outcome_of(error) if error else constants.PASSED
        record.update(
            status=outcome,
            seconds=round(time.monotonic() - started, 2),
            exit_code=_exit_code(outcome, error),
            detail=str(error)[:_ERROR_LIMIT] if error else "",
        )
        self.progress.say(
            f"  {outcome} in {progress.format_seconds(record['seconds'])}"
            + (f": {record['detail']}" if error else "")
        )
        self.save()

    @contextlib.contextmanager
    def stage(self, name: str):
        record, started = self._begin(name), time.monotonic()
        try:
            yield record
        except BaseException as error:
            self._end(record, started, error)
            raise
        self._end(record, started, None)

    def fail_stage(self, name: str, error: BaseException) -> None:
        record = self._begin(name)
        self._end(record, time.monotonic(), error)

    def save(self) -> None:
        for name, text in (
            (constants.REPORT_JSON, json.dumps(self.data, indent=1, default=str)),
            (constants.REPORT_MD, render_markdown(self.data)),
        ):
            path = self.directory / name
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(text)
            temp.replace(path)

    def finish(self, outcome: str, error: BaseException | None = None) -> int:
        """Record the outcome, save reports, print the closing summary; return exit code."""
        done = {s["name"] for s in self.data["stages"]}
        for name in self.names:
            if name not in done and name != constants.STAGE_REPORT:
                self.data["skipped_stages"][
                    name
                ] = "not run: an earlier stage did not pass"
        if error is not None:
            self.data["error"] = str(error)[:_ERROR_LIMIT]
            if outcome == constants.FAILED and not isinstance(
                error, (ValueError, OSError, models.StageError)
            ):
                self.progress.record(traceback.format_exc())
        elapsed = time.time() - self._started
        self.data.update(
            outcome=outcome,
            exit_code=_exit_code(outcome, error),
            ended=_now(),
            elapsed_seconds=round(elapsed, 2),
            peak_rss=_peak_rss(),
        )
        self.data["disk"] = {
            str(p): shutil.disk_usage(path_utils.existing_ancestor(p)).free
            for p in {self.directory, *self._known_paths()}
        }
        try:
            with self.stage(constants.STAGE_REPORT):
                self.data.update(outcome=outcome)
        except OSError as problem:
            self.progress.say(f"cannot save report: {problem}")
        self._summary(outcome, elapsed)
        return self.data["exit_code"]

    def _known_paths(self) -> list[Path]:
        config = self.data.get("config", {})
        return [
            Path(config[k])
            for k in ("source_dir", "output", "verify_only")
            if config.get(k)
        ]

    def _summary(self, outcome: str, elapsed: float) -> None:
        say = self.progress.say
        say(f"\n{outcome}  elapsed {progress.format_seconds(elapsed)}")
        say(f"report: {self.directory / constants.REPORT_JSON}")
        say(f"        {self.directory / constants.REPORT_MD}")
        say(f"log:    {self.directory / constants.LOG_NAME}")
        upload = self.data.get("upload") or {}
        if upload.get("verified"):
            say(f"uploaded: {upload['url']}")
        elif upload.get("status"):
            say(f"upload: {upload['status']} {upload.get('reason', '')}".rstrip())


def render_markdown(data: dict) -> str:
    lines = [
        f"# Model partitioner run: {data['outcome']}",
        "",
        f"- started {data['started']}, ended {data.get('ended', '-')}, "
        f"elapsed {data.get('elapsed_seconds', '-')} s",
        f"- model {data.get('identity', {}).get('model', '-')}, "
        f"source {data.get('identity', {}).get('source', '-')}, "
        f"revision {data.get('identity', {}).get('revision', '-')}",
        f"- log {data['log']}",
        "",
        "## Stages",
        "",
        "| # | Stage | Status | Seconds | Exit | Detail |",
        "|---|---|---|---|---|---|",
    ]
    for s in data["stages"]:
        lines.append(
            f"| {s['number'] or ''} | {s['name']} | {s['status']} | {s['seconds']} "
            f"| {s['exit_code']} | {s['detail']} |"
        )
    for name, reason in data["skipped_stages"].items():
        lines.append(f"| | {name} | SKIPPED | | | {reason} |")
    lines += ["", "## Checks", ""]
    lines += [f"- {c['status']}: {c['name']} {c['detail']}" for c in data["checks"]]
    for title, key in (
        ("Manifest", "manifest"),
        ("Upload", "upload"),
        ("Requested", "requested"),
        ("Host", "host"),
        ("Disk free (bytes)", "disk"),
        ("Peak memory", "peak_rss"),
    ):
        if data.get(key):
            lines += [
                "",
                f"## {title}",
                "",
                "```json",
                json.dumps(data[key], indent=1),
                "```",
            ]
    if data.get("error"):
        lines += ["", "## Error", "", data["error"]]
    return "\n".join(lines) + "\n"
