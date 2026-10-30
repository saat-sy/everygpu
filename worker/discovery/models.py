"""Discovery data structures: a ready peer's dialable addresses and execution endpoint."""

import dataclasses

from worker.execution import models as execution_models


@dataclasses.dataclass(frozen=True)
class PeerStatus:
    addrs: list[str]
    execution: execution_models.Endpoint | None = None
