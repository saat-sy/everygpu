"""OpenTelemetry setup and instruments for pipeline metrics."""

import os

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from . import constants

_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

request_duration = meter.create_histogram(
    constants.REQUEST_DURATION_METRIC,
    unit=constants.MILLISECONDS_UNIT,
)
request_token_latency = meter.create_histogram(
    constants.REQUEST_TOKEN_LATENCY_METRIC,
    unit=constants.MILLISECONDS_UNIT,
)
request_tokens = meter.create_histogram(
    constants.REQUEST_TOKENS_METRIC,
    unit=constants.TOKENS_UNIT,
)
request_throughput = meter.create_histogram(
    constants.REQUEST_THROUGHPUT_METRIC,
    unit=constants.TOKENS_PER_SECOND_UNIT,
)
server_duration = meter.create_histogram(
    constants.SERVER_DURATION_METRIC,
    unit=constants.MILLISECONDS_UNIT,
)
stage_gpu_duration = meter.create_histogram(
    constants.STAGE_GPU_DURATION_METRIC,
    unit=constants.MILLISECONDS_UNIT,
)
stage_processing_duration = meter.create_histogram(
    constants.STAGE_PROCESSING_DURATION_METRIC,
    unit=constants.MILLISECONDS_UNIT,
)
stage_gpu_memory_reserved = meter.create_gauge(
    constants.STAGE_GPU_MEMORY_RESERVED_METRIC,
    unit=constants.BYTES_UNIT,
)
edge_duration = meter.create_histogram(
    constants.EDGE_DURATION_METRIC,
    unit=constants.MILLISECONDS_UNIT,
)
edge_payload_size = meter.create_histogram(
    constants.EDGE_PAYLOAD_SIZE_METRIC,
    unit=constants.BYTES_UNIT,
)


def configure_telemetry() -> None:
    """Configure OTLP exporters once for this process."""
    global _meter_provider, _tracer_provider

    if _tracer_provider is not None:
        return

    resource = Resource.create(
        {
            SERVICE_NAME: os.getenv(
                "OTEL_SERVICE_NAME",
                constants.DEFAULT_OTEL_SERVICE_NAME,
            )
        }
    )

    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(_tracer_provider)

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    _meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(_meter_provider)


def shutdown_telemetry() -> None:
    """Flush pending telemetry and stop exporter workers."""
    if _meter_provider is not None:
        _meter_provider.shutdown()
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
