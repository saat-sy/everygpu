"""Generation data structures: the pinned tokenizer and the end-of-response record. No I/O."""

import dataclasses

import tokenizers


@dataclasses.dataclass(frozen=True)
class Tokenizer:
    """The tokenizer and generation settings read from the pinned artifact revision."""

    tokenizer: tokenizers.Tokenizer
    eos: frozenset[int]
    template: str | None
    bos_token: str
    eos_token: str


@dataclasses.dataclass(frozen=True)
class Finish:
    """The last item of a response: why it ended and the token counts."""

    reason: str
    prompt_tokens: int
    completion_tokens: int
