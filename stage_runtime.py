from pathlib import Path
from time import perf_counter_ns
from typing import TypedDict

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeModel, OlmoeRotaryEmbedding


class StageDefinition(TypedDict):
    id: int
    layers: tuple[int, int]
    embeddings: bool
    output_head: bool


class PassthroughDecoder(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


class StageRuntime(nn.Module):
    def __init__(
        self,
        config: OlmoeConfig,
        stage: StageDefinition,
        device: str | torch.device,
    ):
        super().__init__()
        self.stage = stage
        self.device = torch.device(device)
        self.lm_head: nn.Linear | None = None

        with torch.device("meta"):
            self.model = OlmoeModel(config)
            if stage["output_head"]:
                self.lm_head = nn.Linear(
                    config.hidden_size, config.vocab_size, bias=False
                )

        first_layer, last_layer = stage["layers"]
        for layer in range(config.num_hidden_layers):
            if layer < first_layer or layer > last_layer:
                self.model.layers[layer] = PassthroughDecoder()

        if not stage["embeddings"]:
            setattr(self.model, "embed_tokens", nn.Identity())
        if not stage["output_head"]:
            setattr(self.model, "norm", nn.Identity())

    def load_weights(self, checkpoint: Path):
        weights = load_file(str(checkpoint), device=str(self.device))
        self.load_state_dict(weights, strict=True, assign=True)
        self.model.rotary_emb = OlmoeRotaryEmbedding(
            config=self.model.config, device=self.device
        )
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward_tokens(self, input_ids, attention_mask=None):
        if not self.stage["embeddings"]:
            raise ValueError("This stage does not accept token IDs")

        input_ids = torch.as_tensor(
            input_ids, dtype=torch.long, device=self.device
        )
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        if attention_mask is not None:
            attention_mask = torch.as_tensor(
                attention_mask, dtype=torch.long, device=self.device
            )
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    @torch.inference_mode()
    def forward_hidden(self, hidden_states):
        lm_head = self.lm_head
        if not self.stage["output_head"] or lm_head is None:
            raise ValueError("This stage does not have an output head")

        hidden_states = hidden_states.to(self.device)
        hidden_states = self.model(
            inputs_embeds=hidden_states,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, -1:, :]
        return lm_head(hidden_states)

    @torch.inference_mode()
    def sample_token(self, hidden_states):
        logits = self.forward_hidden(hidden_states)[0, -1].float()
        return int(torch.argmax(logits).item())

    @torch.inference_mode()
    def timed_forward_tokens(self, input_ids, attention_mask=None):
        return self._measure_gpu(
            lambda: self.forward_tokens(input_ids, attention_mask=attention_mask)
        )

    @torch.inference_mode()
    def timed_sample_token(self, hidden_states):
        logits, gpu_ms = self._measure_gpu(
            lambda: self.forward_hidden(hidden_states)
        )
        sample_start = perf_counter_ns()
        token_id = int(torch.argmax(logits[0, -1].float()).item())
        sample_ms = (perf_counter_ns() - sample_start) / 1_000_000
        return token_id, gpu_ms, sample_ms

    def gpu_memory(self) -> tuple[int, int]:
        if self.device.type != "cuda":
            return 0, 0
        return (
            torch.cuda.memory_allocated(self.device),
            torch.cuda.memory_reserved(self.device),
        )

    def _measure_gpu(self, operation):
        if self.device.type != "cuda":
            start_ns = perf_counter_ns()
            result = operation()
            return result, (perf_counter_ns() - start_ns) / 1_000_000

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(self.device):
            start.record()
            result = operation()
            end.record()
            end.synchronize()
        return result, start.elapsed_time(end)


def load_stage(
    stage_id: int,
    model_dir: str | Path = ".",
    device: str | torch.device | None = None,
) -> StageRuntime:
    model_dir = Path(model_dir)
    checkpoint = model_dir / f"stage-{stage_id}.safetensors"
    with safe_open(str(checkpoint), framework="pt", device="cpu") as f:
        keys = list(f.keys())

    layers = sorted(
        {int(key.split(".")[2]) for key in keys if key.startswith("model.layers.")}
    )
    if not layers:
        raise ValueError(f"No transformer layers found in {checkpoint}")

    stage: StageDefinition = {
        "id": stage_id,
        "layers": (layers[0], layers[-1]),
        "embeddings": "model.embed_tokens.weight" in keys,
        "output_head": "lm_head.weight" in keys,
    }
    config = OlmoeConfig.from_json_file(str(model_dir / "config.json"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    runtime = StageRuntime(config, stage, device)
    runtime.load_weights(checkpoint)
    return runtime
