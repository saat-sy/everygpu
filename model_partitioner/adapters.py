"""Model-family adapters: tensor classification and config access."""
import importlib.util
import re

import torch

from model_partitioner import constants, models

_EXPERT_KEY = re.compile(r'^mlp\.experts\.(\d+)\.')
_FUSED_EXPERT_KEY = re.compile(r'^mlp\.experts\.[^.\d][^.]*')


class Adapter:
    family: str
    layer_prefix: str
    embedding_prefixes: tuple[str, ...]
    final_prefixes: tuple[str, ...]
    perception_prefixes: dict[str, str] = {}
    model_class: str = 'AutoModelForCausalLM'

    def text_config(self, config: dict) -> dict:
        return config

    def num_layers(self, config: dict) -> int:
        return self.text_config(config)['num_hidden_layers']

    def num_experts(self, config: dict) -> int:
        return self.text_config(config)['num_experts']

    def sparse_layers(self, config: dict) -> set[int]:
        text = self.text_config(config)
        types = text.get('mlp_layer_types')
        if types:
            return {i for i, kind in enumerate(types) if kind == 'sparse'}
        return set(range(self.num_layers(config)))

    def vocab_size(self, config: dict) -> int:
        return self.text_config(config)['vocab_size']

    def output_blocker(self, config: dict) -> str | None:
        return None

    def classify(self, key: str) -> models.Placement:
        if key.startswith(self.layer_prefix):
            return self._classify_layer(key)
        for prefix in self.embedding_prefixes:
            if key.startswith(prefix):
                return models.Placement(constants.ROLE_EMBEDDING)
        for prefix in self.final_prefixes:
            if key.startswith(prefix):
                return models.Placement(constants.ROLE_FINAL)
        for prefix, component in self.perception_prefixes.items():
            if key.startswith(prefix):
                return models.Placement(constants.ROLE_PERCEPTION, component=component)
        raise models.UnsupportedLayout(f'{self.family}: unrecognized tensor {key}')

    def _classify_layer(self, key: str) -> models.Placement:
        index, _, rest = key[len(self.layer_prefix):].partition('.')
        if not index.isdigit():
            raise models.UnsupportedLayout(f'{self.family}: unrecognized tensor {key}')
        layer = int(index)
        if match := _EXPERT_KEY.match(rest):
            return models.Placement(constants.ROLE_EXPERT, layer, int(match.group(1)))
        if _FUSED_EXPERT_KEY.match(rest):
            return models.Placement(constants.ROLE_EXPERT, layer, fused_experts=True)
        return models.Placement(constants.ROLE_LAYER, layer)


class OlmoeAdapter(Adapter):
    family = 'olmoe'
    layer_prefix = 'model.layers.'
    embedding_prefixes = ('model.embed_tokens.',)
    final_prefixes = ('model.norm.', 'lm_head.')


class MuseSparkAdapter(Adapter):
    family = 'muse_spark'
    layer_prefix = 'model.language_model.layers.'
    embedding_prefixes = ('model.language_model.embed_tokens.',)
    final_prefixes = ('model.language_model.norm.', 'lm_head.')
    perception_prefixes = {
        'model.vision_tower.': 'vision_tower',
        'model.vision_adapter.': 'vision_adapter',
        'model.vision_projection.': 'vision_projection',
        'model.perception_emb_norm.': 'perception_emb_norm',
        'model.audio_tower.': 'audio_tower',
    }
    model_class = 'MuseSparkForConditionalGeneration'

    def text_config(self, config: dict) -> dict:
        return config['text_config']

    def num_experts(self, config: dict) -> int:
        return self.text_config(config)['num_local_experts']

    def output_blocker(self, config: dict) -> str | None:
        if importlib.util.find_spec('modelopt') is None:
            return 'NVFP4 inference needs nvidia-modelopt, which is not installed'
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
            return 'NVFP4 inference needs a Blackwell GPU (compute capability 10+)'
        return None


ADAPTERS = {adapter.family: adapter for adapter in (OlmoeAdapter(), MuseSparkAdapter())}


def get_adapter(family: str) -> Adapter:
    if family not in ADAPTERS:
        raise ValueError(f'unsupported family {family!r}; choose from {sorted(ADAPTERS)}')
    return ADAPTERS[family]
