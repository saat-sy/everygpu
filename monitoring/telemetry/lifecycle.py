"""Telemetry lifecycle: environment-driven setup, the one process-level state holder and bounded shutdown."""

import contextvars
import logging
import os
import threading
import time
import uuid

from monitoring.telemetry import constants
from monitoring.telemetry import detector
from monitoring.telemetry import models

logger = logging.getLogger(__name__)


_ACTIVE: contextvars.ContextVar[models.Active | None] = contextvars.ContextVar(
    "everygpu_telemetry", default=None
)


class _Throttle(logging.Filter):
    """Lets one sanitized exporter problem per logger through each interval."""

    def __init__(self):
        super().__init__()
        self._last: dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING or not record.name.startswith(
            constants.OTEL_LOGGER
        ):
            return True
        now = time.monotonic()
        if now - self._last.get(record.name, -constants.FAILURE_LOG_SECONDS) < (
            constants.FAILURE_LOG_SECONDS
        ):
            return False
        self._last[record.name] = now
        kind = record.exc_info[0].__name__ if record.exc_info else ""
        record.msg, record.args, record.exc_info = (
            f"telemetry export problem {kind}".rstrip(),
            (),
            None,
        )
        return True


def active() -> models.Active | None:
    """The live state in this context, or None while telemetry is disabled."""
    return _ACTIVE.get()


def configure(service: str) -> bool:
    """Enable telemetry when the OTLP endpoint is set; never raises. Returns whether it is on."""
    if _ACTIVE.get() is not None or not os.environ.get(constants.ENDPOINT_ENV):
        return _ACTIVE.get() is not None
    try:
        _ACTIVE.set(_build(service))
    except Exception as error:
        logger.warning("telemetry disabled: %s", type(error).__name__)
        return False
    throttle = _Throttle()
    for handler in logging.getLogger().handlers:
        handler.addFilter(throttle)
    logger.info("telemetry enabled service=%s", service)
    return True


def identify(**attributes) -> None:
    """Attach process identity, such as peer or device, to later local spans."""
    if (state := _ACTIVE.get()) is not None:
        state.identity.update(attributes)


def shutdown() -> None:
    """Flush and stop exporters within a bounded time; safe when disabled."""
    state = _ACTIVE.get()
    _ACTIVE.set(None)
    if state is None:
        return
    threads = [
        threading.Thread(target=_stop, args=(provider,), daemon=True)
        for provider in (state.tracer_provider, state.meter_provider)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + constants.SHUTDOWN_SECONDS
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        logger.warning("telemetry shutdown abandoned after a bounded wait")
    for close in state.closers:
        close()


def _stop(provider) -> None:
    try:
        provider.shutdown()
    except Exception as error:
        logger.warning("telemetry shutdown failed: %s", type(error).__name__)


def _build(service: str) -> models.Active:
    from opentelemetry.exporter.otlp.proto.http import metric_exporter
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk import metrics as sdk_metrics
    from opentelemetry.sdk import resources
    from opentelemetry.sdk import trace as sdk_trace
    from opentelemetry.sdk.metrics import export as metrics_export
    from opentelemetry.sdk.trace import export as trace_export

    resource = resources.Resource.create(
        {"service.name": service, "service.instance.id": uuid.uuid4().hex}
    )
    processor = trace_export.BatchSpanProcessor(
        trace_exporter.OTLPSpanExporter(timeout=constants.EXPORT_TIMEOUT_SECONDS),
        max_queue_size=constants.SPAN_QUEUE_SIZE,
        max_export_batch_size=constants.SPAN_BATCH_SIZE,
        schedule_delay_millis=constants.SPAN_DELAY_MILLIS,
        export_timeout_millis=constants.EXPORT_TIMEOUT_SECONDS * 1000,
    )
    tracer_provider = sdk_trace.TracerProvider(
        resource=resource, shutdown_on_exit=False
    )
    tracer_provider.add_span_processor(processor)
    meter_provider = None
    try:
        reader = metrics_export.PeriodicExportingMetricReader(
            metric_exporter.OTLPMetricExporter(
                timeout=constants.EXPORT_TIMEOUT_SECONDS
            ),
            export_interval_millis=constants.METRIC_INTERVAL_MILLIS,
            export_timeout_millis=constants.EXPORT_TIMEOUT_SECONDS * 1000,
        )
        meter_provider = sdk_metrics.MeterProvider(
            [reader], resource=resource, shutdown_on_exit=False
        )
        meter = meter_provider.get_meter(constants.SCOPE_NAME)
        return models.Active(
            tracer_provider.get_tracer(constants.SCOPE_NAME),
            meter,
            _instruments(meter),
            tracer_provider,
            meter_provider,
            detector.SlowDetector(),
        )
    except Exception:
        for provider in (tracer_provider, meter_provider):
            if provider is not None:
                provider.shutdown()
        raise


def _instruments(meter) -> dict:
    created = {}
    for key, (kind, unit, description, buckets) in constants.METRICS.items():
        name = constants.METRIC_PREFIX + key
        if kind == constants.HISTOGRAM:
            created[key] = meter.create_histogram(
                name,
                unit=unit,
                description=description,
                explicit_bucket_boundaries_advisory=buckets,
            )
        elif kind == constants.UPDOWN:
            created[key] = meter.create_up_down_counter(name, unit, description)
        else:
            created[key] = meter.create_counter(name, unit, description)
    return created
