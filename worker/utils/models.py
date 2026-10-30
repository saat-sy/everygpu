"""Worker error types. No I/O."""


class NoGPU(RuntimeError):
    """Raised when no CUDA device is visible."""
