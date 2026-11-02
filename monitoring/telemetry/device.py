"""Device telemetry: CUDA-event intervals around a stage step and periodic NVML device gauges."""

import contextlib
import functools
import logging

from monitoring.telemetry import constants
from monitoring.telemetry import lifecycle

logger = logging.getLogger(__name__)

_PCI_FORMAT = "{:08X}:{:02X}:{:02X}.0"
_NVML_CLOCK_SM, _NVML_CLOCK_MEM, _NVML_TEMPERATURE_GPU = 1, 2, 0
_PCI_FIELDS = ("pci_domain_id", "pci_bus_id", "pci_device_id")


class CudaTimer:
    """Records CUDA events around device phases and reads them once each has completed."""

    def __init__(self, event_factory):
        self._event = event_factory
        self._pairs: dict[str, tuple] = {}

    @contextlib.contextmanager
    def interval(self, name: str):
        start, end = self._event(enable_timing=True), self._event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pairs[name] = (start, end)

    def read(self) -> dict[str, float]:
        """Seconds per phase; waits only on each phase's own end event, never the whole device."""
        seconds = {}
        for name, (start, end) in self._pairs.items():
            end.synchronize()
            seconds[name] = start.elapsed_time(end) / 1000
        return seconds


class _NullTimer:
    def interval(self, name: str):
        return contextlib.nullcontext()

    def read(self) -> dict[str, float]:
        return {}


def timer(torch_module, device):
    """A CUDA-event timer when telemetry is on and the stage runs on CUDA, else a no-op."""
    if lifecycle.active() is None or device.type != "cuda":
        return _NullTimer()
    return CudaTimer(torch_module.cuda.Event)


class _Sampler:
    """Torch and NVML readings for one device; a failing field is simply absent."""

    def __init__(self, nvml, torch_module, handle, index: int, attributes: dict):
        self.nvml, self.torch = nvml, torch_module
        self.handle, self.index, self.attributes = handle, index, attributes

    def observe(self, read, options):
        from opentelemetry import metrics as otel_metrics

        try:
            return [otel_metrics.Observation(read(self), self.attributes)]
        except Exception:
            return []


def _utilization(s):
    return s.nvml.nvmlDeviceGetUtilizationRates(s.handle).gpu / 100


def _memory_used(s):
    return s.nvml.nvmlDeviceGetMemoryInfo(s.handle).used


def _memory_total(s):
    return s.nvml.nvmlDeviceGetMemoryInfo(s.handle).total


def _temperature(s):
    return s.nvml.nvmlDeviceGetTemperature(s.handle, _NVML_TEMPERATURE_GPU)


def _power(s):
    return s.nvml.nvmlDeviceGetPowerUsage(s.handle) / 1000


def _clock_sm(s):
    return s.nvml.nvmlDeviceGetClockInfo(s.handle, _NVML_CLOCK_SM)


def _clock_memory(s):
    return s.nvml.nvmlDeviceGetClockInfo(s.handle, _NVML_CLOCK_MEM)


def _torch_allocated(s):
    return s.torch.cuda.memory_allocated(s.index)


def _torch_reserved(s):
    return s.torch.cuda.memory_reserved(s.index)


# name, unit, reader
_NVML_GAUGES = (
    ("device.utilization", "1", _utilization),
    ("device.memory.used", "By", _memory_used),
    ("device.memory.total", "By", _memory_total),
    ("device.temperature", "Cel", _temperature),
    ("device.power", "W", _power),
    ("device.clock.sm", "MHz", _clock_sm),
    ("device.clock.memory", "MHz", _clock_memory),
)
_TORCH_GAUGES = (
    ("device.torch.allocated", "By", _torch_allocated),
    ("device.torch.reserved", "By", _torch_reserved),
)


def register() -> None:
    """Identify the current CUDA device and sample it; allocator gauges work without NVML."""
    state = lifecycle.active()
    if state is None:
        return
    try:
        import torch as torch_module

        index = torch_module.cuda.current_device()
        props = torch_module.cuda.get_device_properties(index)
        uuid, pci = _identity(props)
    except Exception as error:
        logger.warning("device identity unavailable: %s", type(error).__name__)
        return
    attributes = {"gpu.name": props.name}
    if uuid:
        attributes["gpu.uuid"] = uuid
    if pci:
        attributes["gpu.pci_bus_id"] = pci
    lifecycle.identify(**attributes)
    sampler = _Sampler(None, torch_module, None, index, attributes)
    _create(state, sampler, _TORCH_GAUGES)
    handle = _open_nvml(state, sampler, uuid, pci)
    if handle is not None:
        _create(state, sampler, _NVML_GAUGES)


def _open_nvml(state, sampler: _Sampler, uuid: str, pci: str):
    """Initialise NVML and find the device; releases the init on any failure."""
    try:
        import pynvml as nvml

        nvml.nvmlInit()
    except Exception as error:
        logger.warning("nvml sampling unavailable: %s", type(error).__name__)
        return None
    try:
        sampler.handle = _handle(nvml, uuid, pci)
    except Exception as error:
        nvml.nvmlShutdown()
        logger.warning("nvml sampling unavailable: %s", type(error).__name__)
        return None
    sampler.nvml = nvml
    state.closers.append(nvml.nvmlShutdown)
    return sampler.handle


def _create(state, sampler: _Sampler, gauges) -> None:
    for name, unit, read in gauges:
        state.meter.create_observable_gauge(
            constants.METRIC_PREFIX + name,
            callbacks=[functools.partial(sampler.observe, read)],
            unit=unit,
        )


def _identity(props) -> tuple[str, str]:
    uuid = str(getattr(props, "uuid", "") or "")
    if uuid and not uuid.startswith(("GPU-", "MIG-")):
        uuid = "GPU-" + uuid
    fields = [getattr(props, name, None) for name in _PCI_FIELDS]
    pci = _PCI_FORMAT.format(*fields) if all(type(f) is int for f in fields) else ""
    return uuid, pci


def _handle(nvml, uuid: str, pci: str):
    if uuid:
        try:
            return nvml.nvmlDeviceGetHandleByUUID(uuid)
        except Exception:
            if not pci:
                raise
    return nvml.nvmlDeviceGetHandleByPciBusId(pci)
