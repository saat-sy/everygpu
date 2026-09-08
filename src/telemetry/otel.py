"""OpenTelemetry tracing setup for pipeline profiles."""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import resources
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

import config

_tracer_provider: TracerProvider | None = None

tracer = trace.get_tracer(__name__)


def configure_telemetry() -> None:
    """Configure the OTLP trace exporter once for this process."""
    global _tracer_provider

    if _tracer_provider is not None:
        return

    resource = resources.Resource.create(
        {
            resources.SERVICE_NAME: config.TEMPO_SERVICE_NAME,
        }
    )

    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(_tracer_provider)


def shutdown_telemetry() -> None:
    """Flush pending telemetry and stop exporter workers."""
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
