CLIENT_ENDPOINT_ID = "client"
SERVER_ENDPOINT_ID = "server"

NANOSECONDS_PER_MILLISECOND = 1_000_000
MILLISECONDS_PER_SECOND = 1_000

DEFAULT_OTEL_SERVICE_NAME = "server"

REQUEST_DURATION_METRIC = "pipeline.request.duration"
REQUEST_TOKEN_LATENCY_METRIC = "pipeline.request.token_latency"
REQUEST_TOKENS_METRIC = "pipeline.request.tokens"
REQUEST_THROUGHPUT_METRIC = "pipeline.request.throughput"
SERVER_DURATION_METRIC = "pipeline.server.duration"
STAGE_GPU_DURATION_METRIC = "pipeline.stage.gpu.duration"
STAGE_PROCESSING_DURATION_METRIC = "pipeline.stage.processing.duration"
STAGE_GPU_MEMORY_RESERVED_METRIC = "pipeline.stage.gpu.memory.reserved"
EDGE_DURATION_METRIC = "pipeline.edge.duration"
EDGE_PAYLOAD_SIZE_METRIC = "pipeline.edge.payload.size"

MILLISECONDS_UNIT = "ms"
BYTES_UNIT = "By"
TOKENS_UNIT = "{token}"
TOKENS_PER_SECOND_UNIT = "{token}/s"

MODEL_ATTRIBUTE = "model"
TIMING_ATTRIBUTE = "timing"
TOKEN_TYPE_ATTRIBUTE = "token_type"
SERVER_PHASE_ATTRIBUTE = "server_phase"
NODE_ID_ATTRIBUTE = "node_id"
STAGE_ID_ATTRIBUTE = "stage_id"
DEVICE_ATTRIBUTE = "device"
LAYER_START_ATTRIBUTE = "layer_start"
LAYER_END_ATTRIBUTE = "layer_end"
PHASE_ATTRIBUTE = "phase"
SOURCE_NODE_ID_ATTRIBUTE = "source_node_id"
TARGET_NODE_ID_ATTRIBUTE = "target_node_id"
MESSAGE_TYPE_ATTRIBUTE = "message_type"
