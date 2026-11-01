"""Loads the tokenizer and generation settings from the pinned artifact, verified against its manifest."""

import hashlib
import json
import pathlib

import huggingface_hub
import tokenizers

from client.generation import models
from client.network import models as network_models
from client.utils import constants


def load(pinned: network_models.Checkpoint) -> models.Tokenizer:
    """Blocking: fetch the three small assets at the pinned revision and check size and sha256."""
    tokenizer_path = _fetch(pinned, constants.TOKENIZER_FILE)
    tokenizer_config = _json(_fetch(pinned, constants.TOKENIZER_CONFIG_FILE))
    generation_config = _json(_fetch(pinned, constants.GENERATION_CONFIG_FILE))
    try:
        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    except Exception as error:
        raise network_models.RouteError(
            f"the pinned tokenizer is unreadable: {error}"
        ) from None
    vocab = pinned.config["vocab_size"]
    if tokenizer.get_vocab_size(with_added_tokens=True) > vocab:
        raise network_models.RouteError(
            "the pinned tokenizer exceeds the model vocabulary"
        )
    template = tokenizer_config.get("chat_template") or None
    if template is not None and not isinstance(template, str):
        raise network_models.RouteError(
            "the pinned chat template is not a single string"
        )
    return models.Tokenizer(
        tokenizer,
        _eos(
            generation_config.get("eos_token_id", pinned.config.get("eos_token_id")),
            vocab,
        ),
        template,
        _token_text(tokenizer_config.get("bos_token")),
        _token_text(tokenizer_config.get("eos_token")),
    )


def _fetch(pinned: network_models.Checkpoint, name: str) -> pathlib.Path:
    info = pinned.assets.get(name)
    if info is None:
        raise network_models.RouteError(f"the pinned artifact has no {name}")
    try:
        path = pathlib.Path(
            huggingface_hub.hf_hub_download(
                pinned.repo, constants.ASSET_PREFIX + name, revision=pinned.revision
            )
        )
        with open(path, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except Exception as error:
        raise network_models.RouteError(
            f"cannot fetch {name}: {type(error).__name__}: {error}"
        ) from None
    if path.stat().st_size != info.size or digest != info.sha256:
        raise network_models.RouteError(f"{name} differs from the pinned manifest")
    return path


def _json(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except ValueError:
        raise network_models.RouteError(f"{path.name} is not valid JSON") from None
    if not isinstance(value, dict):
        raise network_models.RouteError(f"{path.name} is not a JSON object")
    return value


def _eos(value, vocab: int) -> frozenset[int]:
    ids = value if isinstance(value, list) else [value]
    if not ids or not all(type(i) is int and 0 <= i < vocab for i in ids):
        raise network_models.RouteError(
            "the pinned artifact has no valid end-of-sequence id"
        )
    return frozenset(ids)


def _token_text(value) -> str:
    if isinstance(value, dict):
        value = value.get("content")
    return value if isinstance(value, str) else ""
