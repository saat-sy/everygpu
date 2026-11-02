"""Bounded rolling-baseline outlier detector for comparable operations."""

import collections
import statistics
import threading

from monitoring.telemetry import constants
from monitoring.telemetry import models


class SlowDetector:
    """Flags a value above a multiple of the median of earlier values in the same group."""

    def __init__(self):
        self._groups: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

    def observe(self, group: tuple, value: float) -> models.Outlier | None:
        """Compare against preceding observations only, then add the value to the window."""
        with self._lock:
            window = self._groups.get(group)
            if window is None:
                window = self._groups[group] = collections.deque(
                    maxlen=constants.BASELINE_WINDOW
                )
                if len(self._groups) > constants.BASELINE_MAX_GROUPS:
                    self._groups.popitem(last=False)
            self._groups.move_to_end(group)
            outlier = None
            if len(window) >= constants.BASELINE_WARMUP:
                median = statistics.median(window)
                threshold = max(
                    median * constants.BASELINE_FACTOR, constants.BASELINE_MIN_SECONDS
                )
                if value > threshold:
                    outlier = models.Outlier(value, median, threshold, len(window))
            window.append(value)
            return outlier
