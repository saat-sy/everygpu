"""Execution data structures: wire frames, endpoints, sessions, model config and request errors."""

import dataclasses

import torch
import trio

_DEFAULT_RMS_NORM_EPS = 1e-5


class ProtocolError(ValueError):
    """A frame violates the wire format; the connection cannot be resynchronised."""


class RequestError(Exception):
    """A well-formed request the worker refuses; reported to the sender as an error response."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclasses.dataclass(frozen=True)
class Frame:
    msg_type: int
    meta: dict
    payload: bytes = b""


@dataclasses.dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    def __str__(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{host}:{self.port}"


@dataclasses.dataclass
class Session:
    session_id: str
    owner: int
    state: dict = dataclasses.field(default_factory=dict)
    closed: bool = False
    lock: trio.Lock = dataclasses.field(default_factory=trio.Lock)


class StageError(ValueError):
    """The resident units or config cannot form an executable stage."""


@dataclasses.dataclass(frozen=True)
class ForwardRequest:
    position: int
    temperature: float
    seq: int
    token_ids: list[int] | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None


@dataclasses.dataclass(frozen=True)
class Config:
    hidden: int
    heads: int
    head_dim: int
    layers: int
    experts: int
    top_k: int
    vocab: int
    max_positions: int
    eps: float
    rope_theta: float
    norm_topk: bool
    clip_qkv: float | None

    @classmethod
    def from_dict(cls, config: dict) -> "Config":
        unsupported = {
            "model_type": config.get("model_type") != "olmoe",
            "hidden_act": config.get("hidden_act") != "silu",
            "attention_bias": bool(config.get("attention_bias")),
            "rope_scaling": bool(config.get("rope_scaling")),
            "num_key_value_heads": config.get("num_key_value_heads")
            != config["num_attention_heads"],
        }
        if bad := [name for name, flagged in unsupported.items() if flagged]:
            raise StageError(f"unsupported OLMoE config: {bad}")
        heads = config["num_attention_heads"]
        return cls(
            hidden=config["hidden_size"],
            heads=heads,
            head_dim=config["hidden_size"] // heads,
            layers=config["num_hidden_layers"],
            experts=config["num_experts"],
            top_k=config["num_experts_per_tok"],
            vocab=config["vocab_size"],
            max_positions=config["max_position_embeddings"],
            eps=config.get("rms_norm_eps", _DEFAULT_RMS_NORM_EPS),
            rope_theta=config.get("rope_theta", 10000.0),
            norm_topk=config["norm_topk_prob"],
            clip_qkv=config.get("clip_qkv"),
        )


@dataclasses.dataclass(frozen=True)
class Layer:
    input_norm: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    post_norm: torch.Tensor
    router: torch.Tensor
    experts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
