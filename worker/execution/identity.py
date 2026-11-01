"""Admission identity: the pin, peer and executable stage a session expects of a worker."""

import dataclasses

from worker.execution import models


def build(
    repo: str, revision: str, peer: str, stage: models.StageDescriptor | None
) -> dict:
    """The identity a worker presents, and a client expects, at admission."""
    return {
        "repo": repo,
        "revision": revision,
        "peer": peer,
        "stage": stage and dataclasses.asdict(stage),
    }
