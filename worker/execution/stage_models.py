"""Tensor-bearing stage data structures. No I/O."""

import dataclasses

import torch


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
