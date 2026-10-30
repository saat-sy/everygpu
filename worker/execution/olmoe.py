"""OLMoE forward kernels as stateless functions, mirroring transformers modeling_olmoe numerics."""

import torch
import torch.nn.functional as F

from worker.execution import models


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return weight * normed.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope(cfg: models.Config, position: int, length: int, like: torch.Tensor):
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (
            torch.arange(0, cfg.head_dim, 2, dtype=torch.float, device=like.device)
            / cfg.head_dim
        )
    )
    steps = torch.arange(position, position + length, device=like.device).float()
    freqs = steps[:, None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(like.dtype), emb.sin().to(like.dtype)


def attention(
    cfg: models.Config, w: models.Layer, x: torch.Tensor, position: int, cache
):
    """Causal self-attention over [1, seq, hidden]; returns output and the extended (k, v)."""
    _, seq, _ = x.shape
    q = rms_norm(F.linear(x, w.q), w.q_norm, cfg.eps)
    k = rms_norm(F.linear(x, w.k), w.k_norm, cfg.eps)
    v = F.linear(x, w.v)
    if cfg.clip_qkv is not None:
        q, k, v = (t.clamp(-cfg.clip_qkv, cfg.clip_qkv) for t in (q, k, v))
    q, k, v = (
        t.view(1, seq, cfg.heads, cfg.head_dim).transpose(1, 2) for t in (q, k, v)
    )
    cos, sin = _rope(cfg, position, seq, x)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    if cache is not None:
        k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
    scores = torch.matmul(q, k.transpose(2, 3)) * cfg.head_dim**-0.5
    visible = (
        torch.arange(k.shape[2], device=x.device)
        <= (position + torch.arange(seq, device=x.device))[:, None]
    )
    scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v).transpose(1, 2).reshape(1, seq, -1)
    return F.linear(out, w.o), (k, v)


def moe(cfg: models.Config, w: models.Layer, x: torch.Tensor) -> torch.Tensor:
    """Top-k routed experts, all local; each expert output is scaled by its router weight."""
    flat = x.reshape(-1, cfg.hidden)
    logits = F.linear(flat, w.router)
    weights, chosen = torch.topk(
        F.softmax(logits, dim=-1, dtype=torch.float), cfg.top_k
    )
    if cfg.norm_topk:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(logits.dtype)
    out = torch.zeros_like(flat)
    for expert in chosen.unique().tolist():
        token, slot = torch.where(chosen == expert)
        gate, up, down = w.experts[expert]
        state = flat[token]
        hidden = F.linear(F.silu(F.linear(state, gate)) * F.linear(state, up), down)
        out.index_add_(0, token, (hidden * weights[token, slot, None]).to(out.dtype))
    return out.view_as(x)


def decoder_layer(
    cfg: models.Config, w: models.Layer, x: torch.Tensor, position: int, cache
):
    attended, cache = attention(
        cfg, w, rms_norm(x, w.input_norm, cfg.eps), position, cache
    )
    x = x + attended
    return x + moe(cfg, w, rms_norm(x, w.post_norm, cfg.eps)), cache
