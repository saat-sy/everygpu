"""Stage assembly: turns resident units into an executable embedding, layer range and final."""

import torch
import torch.nn.functional as F

from worker.execution import models
from worker.execution import stage_models
from worker.execution import olmoe

_MIN_TEMPERATURE = 1e-6


class Stage:
    """The components a worker executes: optional embedding, contiguous layers, optional final."""

    def __init__(self, config: dict, weights: dict[str, dict[str, torch.Tensor]]):
        self.cfg = cfg = models.Config.from_dict(config)
        tensors = {name: t for unit in weights.values() for name, t in unit.items()}
        indices = _layer_indices(weights, cfg)
        self.layers = [_layer(cfg, tensors, weights, i) for i in indices]
        self.has_embedding = "embedding" in weights
        self.has_final = "final" in weights
        _check_topology(cfg, indices, self.has_embedding, self.has_final)
        self.embedding = tensors.get("model.embed_tokens.weight")
        self.final_norm = tensors.get("model.norm.weight")
        self.lm_head = tensors.get("lm_head.weight")
        if (self.has_embedding and self.embedding is None) or (
            self.has_final and (self.final_norm is None or self.lm_head is None)
        ):
            raise models.StageError("embedding or final unit lacks its tensors")
        self.first = indices[0] if indices else 0 if self.has_embedding else cfg.layers
        self.stop = self.first + len(self.layers)
        self.completes = self.has_final and self.stop == cfg.layers
        self.descriptor = models.StageDescriptor(
            self.first,
            self.stop,
            cfg.layers,
            cfg.experts,
            self.has_embedding,
            self.has_final,
        )
        probe = self.layers[0].q if self.layers else self.embedding
        probe = self.lm_head if probe is None else probe
        self.dtype, self.device = probe.dtype, probe.device

    def embed(self, token_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor([token_ids], device=self.device)
        return F.embedding(ids, self.embedding)

    def run_layers(self, hidden: torch.Tensor, position: int, caches: dict):
        extended = {}
        for index, w in zip(range(self.first, self.stop), self.layers):
            hidden, extended[index] = olmoe.decoder_layer(
                self.cfg, w, hidden, position, caches.get(index)
            )
        return hidden, extended

    def sample(self, hidden: torch.Tensor, temperature: float) -> int:
        last = olmoe.rms_norm(hidden[:, -1], self.final_norm, self.cfg.eps)
        logits = F.linear(last, self.lm_head).float()[0]
        if temperature < _MIN_TEMPERATURE:
            return int(logits.argmax())
        return int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))


def _layer_indices(weights: dict, cfg: models.Config) -> list[int]:
    indices = sorted(int(u.split(":")[1]) for u in weights if u.startswith("layer:"))
    if any(not 0 <= i < cfg.layers for i in indices):
        raise models.StageError(f"layer index outside 0..{cfg.layers - 1}: {indices}")
    if indices and indices != list(range(indices[0], indices[-1] + 1)):
        raise models.StageError(f"layers are not contiguous: {indices}")
    for unit in weights:
        if not unit.startswith("expert:"):
            continue
        _, layer, expert = unit.split(":")
        if int(layer) not in indices or int(expert) >= cfg.experts:
            raise models.StageError(f"expert unit {unit} has no matching layer or slot")
    return indices


def _check_topology(cfg: models.Config, indices, has_embedding, has_final) -> None:
    if indices:
        if has_embedding and indices[0] != 0:
            raise models.StageError("embedding stage must start at layer 0")
        if has_final and indices[-1] != cfg.layers - 1:
            raise models.StageError("final stage must end at the last layer")
    elif has_embedding == has_final:
        raise models.StageError(
            "a stage without layers holds exactly one of embedding or final"
        )


def _layer(
    cfg: models.Config, tensors: dict, weights: dict, i: int
) -> stage_models.Layer:
    p = f"model.layers.{i}."
    missing = [e for e in range(cfg.experts) if f"expert:{i}:{e}" not in weights]
    if missing:
        raise models.StageError(
            f"layer {i} lacks {len(missing)} of {cfg.experts} experts (first: {missing[:3]})"
        )
    try:
        return stage_models.Layer(
            input_norm=tensors[p + "input_layernorm.weight"],
            q=tensors[p + "self_attn.q_proj.weight"],
            k=tensors[p + "self_attn.k_proj.weight"],
            v=tensors[p + "self_attn.v_proj.weight"],
            o=tensors[p + "self_attn.o_proj.weight"],
            q_norm=tensors[p + "self_attn.q_norm.weight"],
            k_norm=tensors[p + "self_attn.k_norm.weight"],
            post_norm=tensors[p + "post_attention_layernorm.weight"],
            router=tensors[p + "mlp.gate.weight"],
            experts=[
                tuple(
                    tensors[f"{p}mlp.experts.{e}.{n}_proj.weight"]
                    for n in ("gate", "up", "down")
                )
                for e in range(cfg.experts)
            ],
        )
    except KeyError as error:
        raise models.StageError(f"layer {i} lacks tensor {error.args[0]}") from None
