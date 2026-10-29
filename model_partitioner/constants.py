"""Shared constants and defaults for the model partitioner."""
from types import MappingProxyType

SCHEMA_VERSION = 2
MANIFEST_NAME = 'manifest.json'
ASSETS_DIR = 'assets'
INDEX_NAME = 'model.safetensors.index.json'
DEFAULT_EXPERTS_PER_FILE = 16
HASH_CHUNK_BYTES = 1 << 24
REASSEMBLED_SHARD_BYTES = 4 << 30
SLICE_SEPARATOR = '::'
REVISION_LENGTH = 40

ROLE_EMBEDDING = 'embedding'
ROLE_LAYER = 'layer'
ROLE_EXPERT = 'expert'
ROLE_FINAL = 'final'
ROLE_PERCEPTION = 'perception'

OUTPUT_CHECK_SEED = 0
OUTPUT_CHECK_BATCH = 2
OUTPUT_CHECK_SEQ_LEN = 16
LOGITS_ATOL = 1e-4
LOGITS_RTOL = 1e-4

ASSET_PATTERNS = (
    'config.json',
    'generation_config.json',
    'hf_quant_config.json',
    'tokenizer*',
    'special_tokens_map.json',
    'chat_template*',
    'processor_config.json',
    'preprocessor_config.json',
    'vocab*',
    'merges.txt',
)

DTYPE_BYTES = MappingProxyType({
    'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
    'F16': 2, 'BF16': 2, 'I16': 2, 'U16': 2,
    'F32': 4, 'I32': 4, 'U32': 4,
    'F64': 8, 'I64': 8, 'U64': 8,
})
