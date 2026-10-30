"""Data structures for specs, plans and the manifest. No I/O."""

import dataclasses
from dataclasses import dataclass, field

from model_partitioner.utils import constants


class UnsupportedLayout(ValueError):
    """Raised when a checkpoint tensor does not fit a supported layout."""


class IncompleteSource(ValueError):
    """Raised when a source checkpoint is missing shards, tensors, layers or experts."""


class ChecksumMismatch(ValueError):
    """Raised when a loaded tensor does not match its manifest checksum."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class RunOptions:
    """Where and how models run; device_map shards one model across devices."""

    device: str = "cpu"
    device_map: str | None = None


@dataclass(frozen=True)
class TensorSpec:
    key: str
    file: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class Placement:
    role: str
    layer: int | None = None
    expert: int | None = None
    component: str | None = None
    fused_experts: bool = False


@dataclass
class PlannedPiece:
    spec: TensorSpec
    name: str
    start: int | None = None
    stop: int | None = None


@dataclass
class PlannedUnit:
    id: str
    role: str
    file: str
    layer: int | None = None
    expert: int | None = None
    component: str | None = None
    depends_on: list[str] = field(default_factory=list)
    pieces: list[PlannedPiece] = field(default_factory=list)


@dataclass
class Piece:
    name: str
    unit: str
    file: str
    source_key: str
    shape: list[int]
    dtype: str
    nbytes: int
    sha256: str
    slice_start: int | None = None
    slice_stop: int | None = None


@dataclass
class Unit:
    id: str
    role: str
    file: str
    layer: int | None
    expert: int | None
    component: str | None
    depends_on: list[str]
    tensors: list[str]


@dataclass
class FileInfo:
    size: int
    sha256: str


@dataclass
class SourceFile:
    name: str
    size: int
    sha256: str
    metadata: dict[str, str]


@dataclass
class SourceTensor:
    shape: list[int]
    dtype: str
    nbytes: int
    file: str


@dataclass
class Manifest:
    family: str
    repo_id: str
    revision: str
    config: dict
    experts_per_file: int
    source_files: list[SourceFile]
    source_tensors: dict[str, SourceTensor]
    assets: dict[str, FileInfo]
    files: dict[str, FileInfo]
    units: list[Unit]
    tensors: list[Piece]
    schema_version: int = constants.SCHEMA_VERSION

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        if data["schema_version"] != constants.SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema {data['schema_version']} is not readable by this tool "
                f"(schema {constants.SCHEMA_VERSION}); re-run partition"
            )
        data = dict(data)
        data["source_files"] = [SourceFile(**f) for f in data["source_files"]]
        data["source_tensors"] = {
            k: SourceTensor(**v) for k, v in data["source_tensors"].items()
        }
        data["assets"] = {k: FileInfo(**v) for k, v in data["assets"].items()}
        data["files"] = {k: FileInfo(**v) for k, v in data["files"].items()}
        data["units"] = [Unit(**u) for u in data["units"]]
        data["tensors"] = [Piece(**p) for p in data["tensors"]]
        return cls(**data)
