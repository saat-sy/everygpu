"""Telemetry data structures: peer-reported timing, route communication totals and outliers."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Downstream:
    """Compact communication totals for the edges below a worker, summed without per-hop detail."""

    communication_seconds: float = 0.0
    edges: int = 0
    incomplete: int = 0

    def wire(self) -> dict:
        return {
            "communication_seconds": self.communication_seconds,
            "edges": self.edges,
            "incomplete": self.incomplete,
        }


@dataclasses.dataclass(frozen=True)
class Timing:
    """A validated peer-reported handling time and the totals below that peer."""

    handling_seconds: float
    downstream: Downstream


@dataclasses.dataclass(frozen=True)
class Outlier:
    """A value above its comparable baseline."""

    value: float
    median: float
    threshold: float
    samples: int


@dataclasses.dataclass(frozen=True)
class Edge:
    """A directed sender-to-receiver exchange and its comparable-state labels."""

    kind: str
    source: str
    target: str
    phase: str
    connection: str
    step: int


@dataclasses.dataclass
class Active:
    """Live providers and instruments; exists only while telemetry is enabled."""

    tracer: object
    meter: object
    instruments: dict
    tracer_provider: object
    meter_provider: object
    slow: object
    identity: dict = dataclasses.field(default_factory=dict)
    closers: list = dataclasses.field(default_factory=list)
