"""DHT record rules shared by advertising and discovery: validation and key names."""

import json

from libp2p.records.validator import Validator

from model_partitioner.utils import constants as partitioner_constants
from worker.utils import constants


class RecordValidator(Validator):
    """Library adapter: accepts only well-formed EveryGPU records; newest status sequence wins."""

    def validate(self, key: str, value: bytes) -> None:
        data = json.loads(value)
        if not _names_artifact(data):
            raise ValueError(f"{key}: malformed record")
        if _is_peer_key(key) and not _has_status_fields(data):
            raise ValueError(f"{key}: malformed status")
        if key != constants.NETWORK_KEY and not _is_peer_key(key):
            raise ValueError(f"{key}: unknown record")

    def select(self, key: str, values: list[bytes]) -> int:
        if key == constants.NETWORK_KEY:
            return 0
        seqs = [json.loads(value)["seq"] for value in values]
        return seqs.index(max(seqs))


def piece_key(repo: str, revision: str, unit_id: str) -> str:
    """Provider-record key for one unit of one artifact revision."""
    return f"{repo}@{revision}/{unit_id}"


def peer_key(peer_id: str) -> str:
    """Value-record key for one peer's status."""
    return constants.PEER_KEY_PREFIX + peer_id


def _is_peer_key(key: str) -> bool:
    return key.startswith(constants.PEER_KEY_PREFIX)


def _names_artifact(data) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("repo"), str)
        and partitioner_constants.REVISION_PATTERN.fullmatch(str(data.get("revision")))
        is not None
    )


def _has_status_fields(data: dict) -> bool:
    return (
        isinstance(data.get("ready"), bool)
        and isinstance(data.get("seq"), int)
        and isinstance(data.get("expires"), (int, float))
        and isinstance(data.get("addrs"), list)
    )
