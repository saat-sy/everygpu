"""Tracing facade: spans, bounded wire carriers and log correlation; every call is a no-op while disabled."""

import contextlib

from monitoring.telemetry import constants
from monitoring.telemetry import lifecycle


class _NullSpan:
    """Stands in for a span while telemetry is disabled."""

    def set_attribute(self, key, value) -> None:
        pass

    def add_event(self, name, attributes=None) -> None:
        pass

    def set_status(self, status) -> None:
        pass

    def end(self) -> None:
        pass


NULL = _NullSpan()


def start(name: str, attributes: dict | None = None, *, parent=None, kind="internal"):
    """A started span that is not made current; the caller ends it."""
    state = lifecycle.active()
    if state is None:
        return NULL
    from opentelemetry import trace

    return state.tracer.start_span(
        name,
        context=parent,
        kind=getattr(trace.SpanKind, kind.upper()),
        attributes={**state.identity, **(attributes or {})},
        record_exception=False,
        set_status_on_exception=False,
    )


@contextlib.contextmanager
def span(name: str, attributes: dict | None = None, *, parent=None, kind="internal"):
    """A span that is current inside the block and ended on exit; errors keep only their type."""
    started = start(name, attributes, parent=parent, kind=kind)
    if started is NULL:
        yield NULL
        return
    from opentelemetry import trace

    with trace.use_span(
        started,
        end_on_exit=False,
        record_exception=False,
        set_status_on_exception=False,
    ):
        try:
            yield started
        except BaseException as error:
            fail(started, error)
            raise
        finally:
            started.end()


def fail(span_, error: BaseException) -> None:
    """Mark a span failed by exception type only; cancellation is not a failure."""
    name = type(error).__name__
    if name in constants.CANCELLATION_NAMES:
        span_.set_attribute("everygpu.cancelled", True)
        return
    mark_error(span_, name)


def mark_error(span_, name: str) -> None:
    """Set the span's error status to a bounded name, never a message."""
    if span_ is not NULL:
        from opentelemetry import trace

        span_.set_status(trace.Status(trace.StatusCode.ERROR, name))
    span_.set_attribute("error.type", name)


def context_of(span_):
    """The context in which `span_` is current, or None while disabled."""
    if span_ is NULL:
        return None
    from opentelemetry import trace

    return trace.set_span_in_context(span_)


def current():
    """The current context to carry across a thread or loop boundary, or None."""
    if lifecycle.active() is None:
        return None
    from opentelemetry import context

    return context.get_current()


@contextlib.contextmanager
def activate(context_):
    """Make a carried context current for the block; same task only."""
    if context_ is None:
        yield
        return
    from opentelemetry import context

    token = context.attach(context_)
    try:
        yield
    finally:
        context.detach(token)


def clean(raw) -> dict | None:
    """A bounded traceparent carrier from untrusted metadata, or None."""
    if not isinstance(raw, dict):
        return None
    parent, state = raw.get("traceparent"), raw.get("tracestate")
    if not isinstance(parent, str) or len(parent) > constants.MAX_TRACEPARENT_CHARS:
        return None
    result = {"traceparent": parent}
    if isinstance(state, str) and len(state) <= constants.MAX_TRACESTATE_CHARS:
        result["tracestate"] = state
    return result


def carrier(incoming: dict | None = None) -> dict | None:
    """The carrier for the current span, else the incoming one so downstream timing stays on."""
    if lifecycle.active() is not None:
        from opentelemetry.trace.propagation import tracecontext

        found: dict = {}
        tracecontext.TraceContextTextMapPropagator().inject(found)
        if found := clean(found):
            return found
    return incoming


def extract(raw):
    """The context a carrier describes, or None when absent, malformed or disabled."""
    cleaned = clean(raw)
    if cleaned is None or lifecycle.active() is None:
        return None
    from opentelemetry.trace.propagation import tracecontext

    context_ = tracecontext.TraceContextTextMapPropagator().extract(cleaned)
    from opentelemetry import trace

    valid = trace.get_current_span(context_).get_span_context().is_valid
    return context_ if valid else None


def log_fields() -> tuple[str, str] | None:
    """The current trace and span ids as hex for log correlation, or None."""
    if lifecycle.active() is None:
        return None
    from opentelemetry import trace

    context_ = trace.get_current_span().get_span_context()
    if not context_.is_valid:
        return None
    return format(context_.trace_id, "032x"), format(context_.span_id, "016x")
