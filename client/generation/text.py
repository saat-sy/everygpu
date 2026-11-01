"""Prompt rendering, tokenization and incremental detokenization. Pure and CPU-bound."""

import jinja2
import jinja2.sandbox

from client.generation import models
from client.utils import constants
from client.utils import models as client_models


def encode_chat(tokenizer: models.Tokenizer, messages: list[dict]) -> list[int]:
    """Prompt token ids; the message order and roles are preserved."""
    return tokenizer.tokenizer.encode(
        _render(tokenizer, messages), add_special_tokens=False
    ).ids


def _render(tokenizer: models.Tokenizer, messages: list[dict]) -> str:
    if tokenizer.template is None:
        turns = "".join(
            f"{constants.ROLE_LABELS[m['role']]}: {m['content']}\n" for m in messages
        )
        return f"{turns}{constants.ROLE_LABELS['assistant']}:"
    environment = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )
    environment.globals["raise_exception"] = _reject
    try:
        return environment.from_string(tokenizer.template).render(
            messages=messages,
            add_generation_prompt=True,
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
        )
    except jinja2.TemplateError as error:
        raise client_models.ApiError(
            400, f"The chat template rejected these messages: {error}", param="messages"
        ) from None


def _reject(message: str):
    raise jinja2.TemplateError(message)


class Detokenizer:
    """Turns generated ids into text deltas, holding back bytes of an unfinished character."""

    def __init__(self, tokenizer: models.Tokenizer):
        self._decode = tokenizer.tokenizer.decode
        self._ids: list[int] = []
        self._prefix = 0
        self._read = 0

    def add(self, token_id: int) -> str:
        """The text newly completed by this token; empty while a character is partial."""
        self._ids.append(token_id)
        seen = self._text(self._prefix, self._read)
        text = self._text(self._prefix, len(self._ids))
        if len(text) > len(seen) and text.startswith(seen) and not text.endswith("�"):
            self._prefix, self._read = self._read, len(self._ids)
            return text[len(seen) :]
        return ""

    def flush(self) -> str:
        """Whatever remains at the end of the response, even if its last character is partial."""
        seen = self._text(self._prefix, self._read)
        text = self._text(self._prefix, len(self._ids))
        self._read = len(self._ids)
        return text[len(seen) :]

    def _text(self, start: int, stop: int) -> str:
        return self._decode(self._ids[start:stop], skip_special_tokens=True)
