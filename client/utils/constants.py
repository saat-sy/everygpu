"""Constants shared by the client server modules."""

HOST_ENV = "EVERYGPU_HOST"
DEFAULT_ADDRESS = "127.0.0.1:42731"
MAX_BODY_BYTES = 1024 * 1024
MODEL_ID = "olmoe"
MODEL_OWNER = "everygpu"
ROLES = frozenset({"system", "user", "assistant"})
REQUEST_FIELDS = frozenset({"model", "messages", "stream"})
MESSAGE_FIELDS = frozenset({"role", "content"})
INVALID_REQUEST = "invalid_request_error"
SERVER_ERROR = "server_error"
NOT_IMPLEMENTED_CODE = "generation_not_implemented"
NOT_IMPLEMENTED_MESSAGE = "Generation is not implemented yet."
INTERNAL_ERROR_MESSAGE = "Internal server error."
