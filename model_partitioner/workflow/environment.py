"""Finds or prepares the Python environment with the partitioner dependencies. Stdlib only."""

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from model_partitioner.utils import constants, models, progress

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = REPO_ROOT / constants.ENV_DIR_NAME

_PROBE = """
import importlib, importlib.metadata as metadata, json, platform, sys
try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
except ImportError as error:
    print(json.dumps(["packaging is not importable: %s" % error])); sys.exit(0)
spec = json.loads(sys.argv[1])
found = []
python = platform.python_version()
if not SpecifierSet(spec["python"]).contains(python, prereleases=True):
    found.append("Python %s does not satisfy %s" % (python, spec["python"]))
for text in spec["requirements"]:
    requirement = Requirement(text)
    try:
        importlib.import_module(requirement.name.replace("-", "_"))
        version = metadata.version(requirement.name)
    except Exception as error:
        found.append("%s not importable (%s: %s)" % (requirement.name, type(error).__name__, error))
        continue
    if not requirement.specifier.contains(version, prereleases=True):
        found.append("%s %s does not satisfy %s" % (requirement.name, version, requirement.specifier))
print(json.dumps(found))
"""


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / constants.PYPROJECT_NAME).read_text())


def _group(cfg: models.RunConfig) -> str:
    return constants.OUTPUTS_GROUP if cfg.verify else constants.CORE_GROUP


def _requirements(groups: dict, name: str) -> list[str]:
    found = []
    for item in groups[name]:
        if isinstance(item, dict):
            found += _requirements(groups, item["include-group"])
        else:
            found.append(item)
    return found


def problems(cfg: models.RunConfig, python: Path | None = None) -> list[str]:
    """Why `python` (default: this one) cannot run the workflow; empty if it can."""
    project = _pyproject()
    interpreter = str(python or sys.executable)
    spec = {
        "python": project["project"]["requires-python"],
        "requirements": _requirements(project["dependency-groups"], _group(cfg)),
    }
    done = subprocess.run(
        [interpreter, "-I", "-c", _PROBE, json.dumps(spec)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(done.stdout.splitlines()[-1])
    except (IndexError, ValueError):
        return [f"{interpreter} cannot be probed: {done.stderr.strip()[-200:]}"]


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_group_gone(process: subprocess.Popen, seconds: float) -> bool:
    """Reap the direct child and wait until no process of its group remains."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _group_alive(process.pid):
            return True
        time.sleep(0.05)
    return False


def _stop(process: subprocess.Popen) -> None:
    """SIGTERM the whole group, then SIGKILL survivors after the grace period."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    if _wait_group_gone(process, constants.TERMINATE_GRACE_SECONDS):
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    _wait_group_gone(process, constants.TERMINATE_GRACE_SECONDS)
    process.wait()


def _run(command: list[str], tracker: progress.Progress) -> None:
    tracker.say("$ " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        for line in process.stdout:
            tracker.say("  " + line.rstrip())
        code = process.wait()
    except BaseException:
        _stop(process)
        raise
    finally:
        process.stdout.close()
    if code:
        raise models.StageError(f"{command[0]} exited with status {code}")


def prepare(cfg: models.RunConfig, tracker: progress.Progress) -> Path | None:
    """Return None to continue here, or the interpreter to re-run under (needs uv to set up)."""
    if not (absent := problems(cfg)):
        tracker.say(f"environment ok: {sys.executable}")
        return None
    tracker.say(f"{sys.executable}: {'; '.join(absent)}")
    python = ENV_DIR / "bin" / "python"
    if python.exists() and not problems(cfg, python):
        tracker.say(f"reusing environment {ENV_DIR}")
        return python
    if any(item.startswith("Python ") for item in absent):
        raise models.StageError(f"unsupported interpreter: {'; '.join(absent)}")
    uv = shutil.which("uv")
    if uv is None:
        raise models.StageError(
            "uv is required to prepare dependencies: install it from "
            f"https://docs.astral.sh/uv/ ({'; '.join(absent)})"
        )
    if not os.access(REPO_ROOT, os.W_OK):
        raise models.StageError(f"{REPO_ROOT} is not writable; cannot create {ENV_DIR}")
    if not python.exists():
        no_downloads = "--no-python-downloads"
        _run(
            [uv, "venv", no_downloads, "--python", sys.executable, str(ENV_DIR)],
            tracker,
        )
    _run(
        [uv, "pip", "install", "--python", str(python), "--group", _group(cfg)], tracker
    )
    if still := problems(cfg, python):
        raise models.StageError(
            f"environment {ENV_DIR} is not usable: {'; '.join(still)}"
        )
    return python


def reexec(python: Path, argv: list[str], run_dir: Path) -> None:
    """Replace this process with the prepared interpreter; never returns."""
    env = dict(os.environ)
    env[constants.RUN_DIR_ENV] = str(run_dir.resolve())
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")])
    )
    sys.stdout.flush()
    os.execve(str(python), [str(python), "-m", "model_partitioner", *argv], env)
