"""Console and log progress: counters, activity timers, plain lines. Stdlib only."""

import contextlib
import datetime
import threading
import time
from pathlib import Path

from model_partitioner.utils import constants

_CLEAR_LINE = "\r\x1b[2K"


def format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s" if minutes else f"{seconds:.1f}s"


class Task:
    """One running activity: a counter when `total` is known, else a timer."""

    def __init__(self, label: str, total: int | None):
        self.label, self.total, self.done = label, total, 0
        self.started = time.monotonic()

    def advance(self, count: int = 1) -> None:
        self.done += count

    def elapsed(self) -> str:
        return format_seconds(time.monotonic() - self.started)

    def plain(self) -> str:
        count = f" {self.done}/{self.total}" if self.total else ""
        return f"{self.label}:{count} ({self.elapsed()} elapsed)"

    def live(self) -> str:
        if not self.total:
            return f"{self.label} ... {self.elapsed()}"
        filled = constants.BAR_WIDTH * self.done // self.total
        bar = "#" * filled + "-" * (constants.BAR_WIDTH - filled)
        return f"{self.label} [{bar}] {self.done}/{self.total} {self.elapsed()}"


class Progress:
    """Writes lines to `stream` (None = silent) and a plain-text `log`."""

    def __init__(self, stream=None, log: Path | None = None):
        self.stream = stream
        self.log = Path(log) if log else None
        self.tty = bool(stream and stream.isatty())
        self._lock = threading.Lock()
        self._live: Task | None = None

    def record(self, line: str) -> None:
        if self.log:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
            with open(self.log, "a") as handle:
                handle.write(f"{stamp} {line}\n")

    def say(self, line: str) -> None:
        with self._lock:
            self.record(line)
            if self.stream:
                prefix = _CLEAR_LINE if self.tty and self._live else ""
                print(prefix + line, file=self.stream, flush=True)

    def _draw(self) -> None:
        with self._lock:
            if self._live and self.stream:
                print(
                    _CLEAR_LINE + self._live.live(),
                    end="",
                    file=self.stream,
                    flush=True,
                )

    def _tick(self, task: Task, stop: threading.Event) -> None:
        period = (
            constants.TTY_REFRESH_SECONDS
            if self.tty
            else constants.PLAIN_PROGRESS_SECONDS
        )
        while not stop.wait(period):
            if self.tty:
                self._draw()
            else:
                self.say(task.plain())

    @contextlib.contextmanager
    def task(self, label: str, total: int | None = None):
        """Counter (known total) or activity timer (no total) until the block ends."""
        task, stop = Task(label, total), threading.Event()
        self.say(f"{label}: started" + (f" ({total} items)" if total else ""))
        self._live = task
        thread = threading.Thread(target=self._tick, args=(task, stop), daemon=True)
        thread.start()
        try:
            yield task
        finally:
            stop.set()
            thread.join()
            self._live = None
            with self._lock:
                if self.tty and self.stream:
                    print(_CLEAR_LINE, end="", file=self.stream, flush=True)
            self.say(task.plain().replace(" elapsed)", " total)"))
