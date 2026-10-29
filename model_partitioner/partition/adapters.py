"""Model-family adapters: tensor classification and config access."""

import importlib.util
import re

import torch

from model_partitioner.utils import constants, models

_EXPERT_KEY = re.compile(r"^mlp\.experts\.(\d+)\.")
_FUSED_EXPERT_KEY = re.compile(r"^mlp\.experts\.[^.\d][^.]*")


class Adapter:
    family: str
    repo_id: str
    model_type: str
    layer_prefix: str
    layer_tensors: re.Pattern
    embedding_keys: frozenset[str]
    final_keys: frozenset[str]
    perception_prefixes: dict[str, str] = {}
    model_class: str = "AutoModelForCausalLM"

    def check_config(self, config: dict) -> None:
        if config.get("model_type") != self.model_type:
            raise models.UnsupportedLayout(
                f"config model_type {config.get('model_type')!r} is not "
                f"{self.model_type!r} ({self.repo_id})"
            )

    def text_config(self, config: dict) -> dict:
        return config

    def num_layers(self, config: dict) -> int:
        return self.text_config(config)["num_hidden_layers"]

    def num_experts(self, config: dict) -> int:
        return self.text_config(config)["num_experts"]

    def sparse_layers(self, config: dict) -> set[int]:
        text = self.text_config(config)
        types = text.get("mlp_layer_types")
        if types:
            return {i for i, kind in enumerate(types) if kind == "sparse"}
        return set(range(self.num_layers(config)))

    def vocab_size(self, config: dict) -> int:
        return self.text_config(config)["vocab_size"]

    def output_blocker(self, config: dict, gpu: bool) -> str | None:
        return None

    def classify(self, key: str) -> models.Placement:
        if key.startswith(self.layer_prefix):
            return self._classify_layer(key)
        if key in self.embedding_keys:
            return models.Placement(constants.ROLE_EMBEDDING)
        if key in self.final_keys:
            return models.Placement(constants.ROLE_FINAL)
        for prefix, component in self.perception_prefixes.items():
            if key.startswith(prefix):
                return models.Placement(constants.ROLE_PERCEPTION, component=component)
        raise models.UnsupportedLayout(f"{self.family}: unrecognized tensor {key}")

    def _classify_layer(self, key: str) -> models.Placement:
        index, _, rest = key[len(self.layer_prefix) :].partition(".")
        if index.isdigit():
            layer = int(index)
            if self.layer_tensors.fullmatch(rest):
                return models.Placement(constants.ROLE_LAYER, layer)
            if match := _EXPERT_KEY.match(rest):
                return models.Placement(
                    constants.ROLE_EXPERT, layer, int(match.group(1))
                )
            if _FUSED_EXPERT_KEY.match(rest):
                return models.Placement(
                    constants.ROLE_EXPERT, layer, fused_experts=True
                )
        raise models.UnsupportedLayout(f"{self.family}: unrecognized tensor {key}")


class OlmoeAdapter(Adapter):
    family = "olmoe"
    repo_id = "allenai/OLMoE-1B-7B-0924"
    model_type = "olmoe"
    layer_prefix = "model.layers."
    layer_tensors = re.compile(
        r"(input_layernorm|post_attention_layernorm)\.weight"
        r"|self_attn\.([qkvo]_proj|[qk]_norm)\.weight"
        r"|mlp\.gate\.weight"
    )
    embedding_keys = frozenset({"model.embed_tokens.weight"})
    final_keys = frozenset({"model.norm.weight", "lm_head.weight"})


_QUANTIZED = r"(weight|weight_scale|weight_scale_2|input_scale)"


class MuseSparkAdapter(Adapter):
    family = "muse_spark"
    repo_id = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
    model_type = "muse_spark"
    layer_prefix = "model.language_model.layers."
    layer_tensors = re.compile(
        rf"self_attn\.([qkvo]|gate)_proj\.{_QUANTIZED}"
        r"|(input|pre_feedforward|post_feedforward)_layernorm\.weight"
        r"|post_(attention|feedforward)_residual_gate\.gate"
        r"|mlp\.gate\.(weight|e_score_correction_bias)"
        rf"|mlp\.(pre_expert_proj|post_expert_proj|gate_proj|up_proj|down_proj)\.{_QUANTIZED}"
        r"|mlp\.pre_expert_norm\.weight"
        r"|mlp\.experts\.post_expert_norm\.weight"
    )
    embedding_keys = frozenset({"model.language_model.embed_tokens.weight"})
    final_keys = frozenset({"model.language_model.norm.weight", "lm_head.weight"})
    perception_prefixes = {
        "model.vision_tower.": "vision_tower",
        "model.vision_adapter.": "vision_adapter",
        "model.vision_projection.": "vision_projection",
        "model.perception_emb_norm.": "perception_emb_norm",
        "model.audio_tower.": "audio_tower",
    }
    model_class = "MuseSparkForConditionalGeneration"

    def text_config(self, config: dict) -> dict:
        return config["text_config"]

    def num_experts(self, config: dict) -> int:
        return self.text_config(config)["num_local_experts"]

    def output_blocker(self, config: dict, gpu: bool) -> str | None:
        if not gpu:
            return "NVFP4 output verification is only supported with --gpu"
        if importlib.util.find_spec("modelopt") is None:
            return "NVFP4 inference needs nvidia-modelopt, which is not installed"
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
            return "NVFP4 inference needs a Blackwell GPU (compute capability 10+)"
        return None


ADAPTERS = {adapter.family: adapter for adapter in (OlmoeAdapter(), MuseSparkAdapter())}


def get_adapter(family: str) -> Adapter:
    if family not in ADAPTERS:
        raise ValueError(
            f"unsupported family {family!r}; choose from {sorted(ADAPTERS)}"
        )
    return ADAPTERS[family]


def adapter_for_repo(repo_id: str) -> Adapter:
    for adapter in ADAPTERS.values():
        if adapter.repo_id == repo_id:
            return adapter
    supported = sorted(adapter.repo_id for adapter in ADAPTERS.values())
    raise ValueError(f"unsupported model {repo_id!r}; supported: {supported}")
