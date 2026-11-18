"""Shared constants and defaults for the model partitioner."""

import re
from types import MappingProxyType

SCHEMA_VERSION = 2
MANIFEST_NAME = "manifest.json"
ASSETS_DIR = "assets"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
DEFAULT_EXPERTS_PER_FILE = 16
HASH_CHUNK_BYTES = 1 << 24
REASSEMBLED_SHARD_BYTES = 4 << 30
SLICE_SEPARATOR = "::"
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
LOCAL_REVISION_PREFIX = "local-sha256:"
LOCAL_REVISION_PATTERN = re.compile(LOCAL_REVISION_PREFIX + r"[0-9a-f]{64}")
OUTPUT_CHECK_NAME = "model output equivalence"

ROLE_EMBEDDING = "embedding"
ROLE_LAYER = "layer"
ROLE_EXPERT = "expert"
ROLE_FINAL = "final"
ROLE_PERCEPTION = "perception"

PASSED = "PASSED"
FAILED = "FAILED"
UNVERIFIED = "UNVERIFIED"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2
EXIT_SIGNAL_BASE = 128

INCOMPLETE = "INCOMPLETE"
INTERRUPTED = "INTERRUPTED"
NOT_RUN = "NOT RUN"
RUNNING = "RUNNING"

DEFAULT_MODEL = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
VERIFY_CPU = "cpu"
VERIFY_GPU = "gpu"
PARTITIONS_DIR = "partitions"
REPORTS_DIR = "reports"
LOG_NAME = "run.log"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
RUN_DIR_ENV = "MODEL_PARTITIONER_RUN_DIR"
PYPROJECT_NAME = "pyproject.toml"
TERMINATE_GRACE_SECONDS = 10
ENV_DIR_NAME = ".venv-model-partitioner"
CORE_GROUP = "model-partitioner"
OUTPUTS_GROUP = "model-partitioner-outputs"
REPO_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
DISK_MARGIN_BYTES = 1 << 30
TTY_REFRESH_SECONDS = 1.0
PLAIN_PROGRESS_SECONDS = 15.0
BAR_WIDTH = 24
HUB_URL = "https://huggingface.co/"
CHECK_FILES = "file checksums"
CHECK_COVERAGE = "coverage"
CHECK_RECONSTRUCTION = "exact tensor reconstruction"

STAGE_PREFLIGHT = "Preflight checks"
STAGE_ENVIRONMENT = "Prepare environment"
STAGE_ACCESS = "Check access and resources"
STAGE_FETCH = "Fetch source"
STAGE_VALIDATE = "Validate source weights"
STAGE_PARTITION = "Partition"
STAGE_VERIFY = "Verify"
STAGE_UPLOAD = "Upload"
STAGE_REPORT = "Report"

SAFETENSORS_METADATA = MappingProxyType({"format": "pt"})

OUTPUT_CHECK_SEED = 0
OUTPUT_CHECK_BATCH = 2
OUTPUT_CHECK_SEQ_LEN = 16
LOGITS_ATOL = 1e-4
LOGITS_RTOL = 1e-4

ASSET_PATTERNS = (
    "config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "chat_template*",
    "processor_config.json",
    "preprocessor_config.json",
    "vocab*",
    "merges.txt",
    "l4_200k_base",
)

DTYPE_BYTES = MappingProxyType(
    {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "F16": 2,
        "BF16": 2,
        "I16": 2,
        "U16": 2,
        "F32": 4,
        "I32": 4,
        "U32": 4,
        "F64": 8,
        "I64": 8,
        "U64": 8,
    }
)
