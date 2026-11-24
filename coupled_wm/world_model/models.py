# SPDX-License-Identifier: Apache-2.0
"""DreamerV4 world model: block-causal video tokenizer (encoder + decoder) and dynamics.

The global world model of the paper. The tokenizer compresses each 256x256 RGB frame (plus
the 50-d proprioception) into ``n_latents`` tokens of width ``bottleneck_dim`` (tanh-bounded);
the dynamics transformer predicts the clean latents of frame ``t`` from noisy latents,
the action ``action[t]`` and the history (shortcut / flow-matching objective with
``num_tau_levels`` noise levels), and optionally the reward from an agent token.

Conventions
-----------
* Frames are float RGB in [0, 1], shape ``(B, T, 3, H, W)``.
* ``action[t]`` is consumed together with frame ``t`` (no shift): the dynamics input at
  position ``t`` is ``[act_proj(action[t]), noise level, registers, z_t]``.
* Attention is spatial within a frame and causal over time every ``temporal_every``-th layer.

Architecture configs
--------------------
:func:`build_tokenizer` and :func:`build_dynamics` take a flat dict. These dicts are the
``model`` blocks of the YAML configs and the ``architecture`` block of a pretrained dir's
``config.json``. Missing keys default to the paper's G1 values (:data:`G1_TOKENIZER_ARCH`,
:data:`G1_DYNAMICS_ARCH`); unknown keys raise. Saved configs are always complete.

Tokenizer keys: ``image_size`` [H, W], ``patch_size``, ``in_channels``, ``d_model``,
``n_layers``, ``num_heads_q``, ``num_heads_kv_latent`` (GQA key/value heads of the latent
stream, in the encoder and in the decoder; the patch/readout stream is plain MHA),
``temporal_every``, ``n_latents``, ``bottleneck_dim``, ``use_proprio``, ``proprio_dim``,
``num_proprio`` (proprio tokens per frame; must be 1), ``seq_len`` (RoPE table length = max
frame positions), ``context_len`` (encoder temporal sliding window, 0 = off), ``dropout``,
``mae_max_mask_prob`` (training-time MAE patch dropout).

Dynamics keys: ``action_dim``, ``num_latents``, ``latent_dim``, ``d_model``, ``num_layers``,
``num_heads``, ``num_registers``, ``num_tau_levels`` (power of two), ``temporal_every``,
``mlp_ratio``, ``seq_len``, ``context_len`` (temporal sliding window in frames, 0 = off; it
binds when more than ``context_len`` frames are visible, also through a KV cache),
``dropout``, ``use_reward_head``, ``mtp_length``, ``reward_hidden_dim``,
``reward_num_buckets``, ``reward_symlog_range``.

The two schemas keep the paper's constructor names, so shared quantities are spelled
differently: the dynamics' ``num_latents`` / ``latent_dim`` must equal the tokenizer's
``n_latents`` / ``bottleneck_dim`` (checked when they are paired), and ``num_layers`` /
``num_heads`` are the dynamics' counterparts of ``n_layers`` / ``num_heads_q``.

``seq_len``, ``context_len``, ``dropout`` (and the tokenizer's ``mae_max_mask_prob``) do not
change any weight and may be overridden at load time (:data:`TOKENIZER_RUNTIME_KEYS`,
:data:`DYNAMICS_RUNTIME_KEYS`). Every other key must match the weights: head counts and
``temporal_every`` change the function without changing any parameter shape, so they are
never inferred from a state dict.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# MATH is last: on GPU, flash / memory-efficient attention are chosen whenever eligible.
_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
# Most sequences in one SDPA call: larger batches are split into calls of at most this many (same result,
# the sequences are independent). FlashAttention's backward fails with an illegal memory access on a B200
# (torch 2.9.1, CUDA 12.8) from about 16k sequences per call on (24 heads of 64, four-frame sequences:
# 16,768 sequences fail, 8,384 pass). A temporal layer attends one sequence per spatial token of each clip.
_SDPA_MAX_BATCH = 8192

TOKENIZER_MODEL_TYPE = "dreamerv4_tokenizer"
DYNAMICS_MODEL_TYPE = "dreamerv4_dynamics"


# --------------------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------------------

class SymlogTwoHotHead(nn.Module):
    """Linear logits over ``num_buckets`` evenly spaced bins in symlog space."""

    def __init__(self, input_dim: int, num_buckets: int = 255, min_val: float = -20.0, max_val: float = 20.0):
        super().__init__()
        self.num_buckets = num_buckets
        self.min_val = min_val
        self.max_val = max_val
        self.linear = nn.Linear(input_dim, num_buckets)
        self.register_buffer("buckets", torch.linspace(min_val, max_val, num_buckets))

    def forward(self, x):
        return self.linear(x)

    def to_symlog(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    def get_targets(self, rewards):
        """Two-hot targets: ``(low, low_weight, high, high_weight)`` for rewards of any shape."""
        y = self.to_symlog(rewards)
        width = (self.max_val - self.min_val) / (self.num_buckets - 1)
        indices = ((y - self.min_val) / width).clamp(0, self.num_buckets - 1)
        low = indices.floor().long()
        high = indices.ceil().long()
        low_weight = high.float() - indices
        high_weight = indices - low.float()
        # A value exactly on a bucket centre has low == high: put all the weight on it.
        exact = low == high
        low_weight[exact] = 1.0
        high_weight[exact] = 0.0
        return low, low_weight, high, high_weight


class RewardMTPHead(nn.Module):
    """Multi-token-prediction reward head: one symlog two-hot head per future step."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, mtp_length: int = 8, num_buckets: int = 255,
                 min_val: float = -20.0, max_val: float = 20.0):
        super().__init__()
        self.mtp_length = mtp_length
        self.hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.heads = nn.ModuleList(
            [SymlogTwoHotHead(hidden_dim, num_buckets, min_val, max_val) for _ in range(mtp_length)]
        )

    def forward(self, h_t):
        """``h_t`` (B, T, D) -> logits (B, T, mtp_length, num_buckets)."""
        x = self.hidden(h_t)
        return torch.stack([head(x) for head in self.heads], dim=2)


# --------------------------------------------------------------------------------------
# Attention building blocks
# --------------------------------------------------------------------------------------

class KVCache:
    """Rolling key/value buffer ``[B, H, max_seq_len, D]``; the oldest entries are dropped when full."""

    def __init__(self, max_seq_len: int, batch_size: int, num_heads: int, head_dim: int,
                 device: torch.device, dtype: torch.dtype):
        self.max_seq_len = max_seq_len
        self.curr_len = 0
        self.k_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append ``[B, H, S_new, D]`` keys/values and return the valid history."""
        S_new = k_new.shape[2]
        if S_new > self.max_seq_len:
            raise ValueError(f"cannot write {S_new} frames into a KV cache of {self.max_seq_len} frames")
        if self.curr_len + S_new > self.max_seq_len:
            shift = (self.curr_len + S_new) - self.max_seq_len
            self.k_cache = torch.roll(self.k_cache, shifts=-shift, dims=2)
            self.v_cache = torch.roll(self.v_cache, shifts=-shift, dims=2)
            self.curr_len -= shift
        self.k_cache[:, :, self.curr_len:self.curr_len + S_new, :] = k_new
        self.v_cache[:, :, self.curr_len:self.curr_len + S_new, :] = v_new
        self.curr_len += S_new
        return self.k_cache[:, :, :self.curr_len, :], self.v_cache[:, :, :self.curr_len, :]

    def reset(self):
        self.curr_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


def _kv_cache_bytes(specs, max_seq_len: int, dtype: torch.dtype) -> int:
    """Bytes of the keys and values of KV caches with ``(rows, heads, head_dim)`` specs (None = none)."""
    return sum(2 * math.prod(spec) for spec in specs if spec is not None) * max_seq_len * dtype.itemsize


class FeedForwardSwiGLU(nn.Module):
    """``down(dropout(SiLU(up(x)) * gate(x)))``; hidden size defaults to ``8 * d_model // 3``."""

    def __init__(self, d_model: int, d_hidden: int | None = None, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        if d_hidden is None:
            d_hidden = (8 * d_model) // 3
        self.up = nn.Linear(d_model, d_hidden, bias=bias)
        self.gate = nn.Linear(d_model, d_hidden, bias=bias)
        self.down = nn.Linear(d_hidden, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.silu(self.up(x))
        g = self.gate(x)
        h = self.drop(u * g)
        return self.down(h)


class MHA_GQA(nn.Module):
    """Multi-head attention with grouped-query attention and interleaved-pair RoPE.

    ``q_src [..., Sq, d_model]``, ``kv_src [..., Skv, d_model]`` (equal leading dims) ->
    ``[..., Sq, d_model]``. ``num_heads_kv`` (G) divides ``num_heads_q`` (Hq). RoPE uses a
    precomputed table of ``max_seq_len`` positions; with a :class:`KVCache` the new keys are
    rotated at ``kv_position_ids`` before they are cached.
    """

    def __init__(self, d_model: int, num_heads_q: int, num_heads_kv: int, head_dim: int, dropout: float = 0.0,
                 max_seq_len: int = 8192, rope_base: float = 10000.0):
        super().__init__()
        assert d_model == num_heads_q * head_dim, "d_model must equal Hq * Dh"
        assert num_heads_q % num_heads_kv == 0, "Hq must be divisible by G (num_heads_kv)"
        assert head_dim % 2 == 0, "RoPE rotates pairs: head_dim must be even"
        self.d_model = d_model
        self.Hq = num_heads_q
        self.G = num_heads_kv
        self.Dh = head_dim
        self.Wq = nn.Linear(d_model, self.Hq * self.Dh, bias=False)
        self.Wk = nn.Linear(d_model, self.G * self.Dh, bias=False)
        self.Wv = nn.Linear(d_model, self.G * self.Dh, bias=False)
        self.out = nn.Linear(self.Hq * self.Dh, d_model, bias=False)
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.init_nonpersistent_buffers()

    def init_nonpersistent_buffers(self, device: torch.device | str | None = None) -> None:
        """(Re)compute the RoPE cos/sin tables in fp32 on ``device`` (default device if None)."""
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.Dh, 2, device=device).float() / self.Dh))
        positions = torch.arange(self.max_seq_len, dtype=inv_freq.dtype, device=inv_freq.device)
        freqs = torch.outer(positions, inv_freq)  # [S, Dh/2]
        # Each frequency is repeated for its (x_2i, x_2i+1) pair: [S, Dh].
        self.register_buffer("cos_cache", freqs.cos().repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer("sin_cache", freqs.sin().repeat_interleave(2, dim=-1), persistent=False)

    def _apply_rope(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rotate ``x [..., heads, S, Dh]`` at positions ``0..S-1`` or at ``position_ids [..., S]``."""
        seq_len = x.shape[-2]
        if position_ids is None:
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
        else:
            cos = self.cos_cache[position_ids]
            sin = self.sin_cache[position_ids]
        cos = cos.to(x.dtype).unsqueeze(-3)  # broadcast over heads
        sin = sin.to(x.dtype).unsqueeze(-3)
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """``(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)``."""
        x = x.unflatten(-1, (-1, 2))
        return torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(-2)

    def _proj_q(self, x):  # [..., Sq, d] -> [..., Hq, Sq, Dh]
        q = self.Wq(x)
        q = q.view(*q.shape[:-1], self.Hq, self.Dh)
        return q.transpose(-3, -2)

    def _proj_kv(self, x):  # [..., Skv, d] -> [..., G, Skv, Dh] for k and v
        k = self.Wk(x)
        k = k.view(*k.shape[:-1], self.G, self.Dh).transpose(-3, -2)
        v = self.Wv(x)
        v = v.view(*v.shape[:-1], self.G, self.Dh).transpose(-3, -2)
        return k, v

    def forward(self, q_src, kv_src, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False,
                q_position_ids: Optional[torch.Tensor] = None, kv_position_ids: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None, update_cache: bool = True):
        q = self._proj_q(q_src)
        k, v = self._proj_kv(kv_src)

        if kv_cache is not None:
            # Rotate the new keys at their absolute positions, then attend to [cache, new].
            k = self._apply_rope(k, position_ids=kv_position_ids)
            if update_cache:
                k, v = kv_cache.update(k, v)
            else:
                # Read-only: concatenate without writing to the cache.
                k = torch.cat([kv_cache.k_cache[:, :, :kv_cache.curr_len, :], k], dim=2)
                v = torch.cat([kv_cache.v_cache[:, :, :kv_cache.curr_len, :], v], dim=2)
            q = self._apply_rope(q, position_ids=q_position_ids)
        else:
            q = self._apply_rope(q, position_ids=q_position_ids)
            k = self._apply_rope(k, position_ids=kv_position_ids)

        effective_causal = False if q.shape[-2] == 1 else is_causal

        # SDPA takes 4D inputs: merge leading dims ([B, T, H, S, Dh] -> [B*T, H, S, Dh]).
        prefix_shape = q.shape[:-3]
        merge = len(prefix_shape) > 1
        if merge:
            n = math.prod(prefix_shape)
            q = q.reshape(n, *q.shape[-3:])
            k = k.reshape(n, *k.shape[-3:])
            v = v.reshape(n, *v.shape[-3:])
        dropout_p = self.dropout if self.training else 0.0
        with sdpa_kernel(backends=_SDPA_BACKENDS):
            outs = []
            for s in range(0, q.shape[0], _SDPA_MAX_BATCH):  # one call unless the batch is very large
                e = s + _SDPA_MAX_BATCH
                mask = attn_mask
                if mask is not None and mask.dim() == 4 and mask.shape[0] == q.shape[0] > 1:
                    mask = mask[s:e]  # a per-sequence mask; a broadcast one is shared
                outs.append(F.scaled_dot_product_attention(
                    query=q[s:e], key=k[s:e], value=v[s:e],
                    attn_mask=mask, is_causal=effective_causal,
                    dropout_p=dropout_p,
                    enable_gqa=(self.G != self.Hq),
                ))
            out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        if merge:
            out = out.view(*prefix_shape, *out.shape[-3:])

        out = out.transpose(-3, -2).contiguous()  # [..., Sq, Hq, Dh]
        out = out.view(*out.shape[:-2], self.Hq * self.Dh)
        return self.out(out)


def _is_temporal(i: int, temporal_every: int) -> bool:
    """Tokenizer layer schedule: alternate when ``temporal_every == 1``, else every k-th layer."""
    return (i % 2 == 1) if temporal_every == 1 else (((i + 1) % temporal_every) == 0)


def _is_last(i: int, n_layers: int, temporal_every: int) -> bool:
    """Tokenizer layers without a patch/latent stream output: the last, and the last spatial
    layer before a final temporal layer (their outputs are never read)."""
    return (i == n_layers - 1) or (i == n_layers - 2 and (n_layers % temporal_every) == 0)


# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------

class BlockCausalEncoderLayer(nn.Module):
    """Encoder layer, spatial-only or temporal-only.

    Spatial: latents attend to {latents, patches, proprio}; patches attend to patches.
    Temporal: every token attends causally to its own history (optionally within a window of
    ``context_len`` frames).
    """

    def __init__(self, d_model: int, num_heads_q: int, num_heads_kv: int, seq_len: int, num_latents: int,
                 num_patches: int, dropout: float = 0.0, temporal: bool = False, is_last: bool = False,
                 num_proprio: int = 0, context_len: int = 0):
        super().__init__()
        Dh = d_model // num_heads_q
        self.temporal = temporal
        self.is_last = is_last
        self.context_len = context_len
        # Submodules are created in the paper's order, so a seeded init draws the same weights.
        self.attn_latent = MHA_GQA(
            d_model, num_heads_q, num_heads_kv, Dh, dropout=dropout,
            max_seq_len=seq_len if temporal else num_patches + num_latents + num_proprio,
        )
        if not is_last:
            # The patch stream uses standard MHA (G = Hq).
            self.attn_patch = MHA_GQA(d_model, num_heads_q, num_heads_q, Dh, dropout=dropout,
                                      max_seq_len=seq_len if temporal else num_patches)
        self.ln_L_q = nn.RMSNorm(d_model)
        self.ln_L_kv = nn.RMSNorm(d_model)
        if not is_last:
            self.ln_P_q = nn.RMSNorm(d_model)
            self.ln_P_kv = nn.RMSNorm(d_model)
        self.ffn_L = FeedForwardSwiGLU(d_model, None, dropout)
        if not is_last:
            self.ffn_P = FeedForwardSwiGLU(d_model, None, dropout)
        self.ln_L_ff = nn.RMSNorm(d_model)
        if not is_last:
            self.ln_P_ff = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _spatial(self, P, L, Prop=None):
        union_sp = torch.cat([L, P] if Prop is None else [L, P, Prop], dim=2)
        L_att = self.attn_latent(self.ln_L_q(L), self.ln_L_kv(union_sp), None, False)
        L_out = L + self.dropout(L_att)
        L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
        if self.is_last:
            return None, L_out
        P_att = self.attn_patch(self.ln_P_q(P), self.ln_P_kv(P), attn_mask=None, is_causal=False)
        P_out = P + self.dropout(P_att)
        P_out = P_out + self.dropout(self.ffn_P(self.ln_P_ff(P_out)))
        return P_out, L_out

    def _temporal(self, P, L):
        T = L.shape[1]
        if self.context_len > 0 and T > self.context_len:
            # Attend iff 0 <= query - key < context_len. The explicit mask already encodes
            # causality, so is_causal is off.
            idx = torch.arange(T, device=L.device)
            dist = idx.view(-1, 1) - idx.view(1, -1)
            keep = (dist >= 0) & (dist < self.context_len)
            t_mask = torch.full((T, T), -1e9, device=L.device, dtype=L.dtype)
            t_mask.masked_fill_(keep, 0.0)
            t_causal = False
        else:
            t_mask, t_causal = None, True

        L = L.permute(0, 2, 1, 3)  # [B, Nl, T, d]
        if self.is_last:
            P_out = None
        else:
            P = P.permute(0, 2, 1, 3)  # [B, Np, T, d]
            P_att = self.attn_patch(self.ln_P_q(P), self.ln_P_kv(P), attn_mask=t_mask, is_causal=t_causal)
            P_out = P + self.dropout(P_att)
            P_out = P_out + self.dropout(self.ffn_P(self.ln_P_ff(P_out)))
            P_out = P_out.permute(0, 2, 1, 3)
        L_att = self.attn_latent(self.ln_L_q(L), self.ln_L_kv(L), attn_mask=t_mask, is_causal=t_causal)
        L_out = L + self.dropout(L_att)
        L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
        return P_out, L_out.permute(0, 2, 1, 3)

    def forward(self, P: torch.Tensor, L: torch.Tensor, Prop: Optional[torch.Tensor] = None):
        """``P [B, T, Np, d]``, ``L [B, T, Nl, d]``, ``Prop [B, T, 1, d]`` (spatial layers only)."""
        if self.temporal:
            return self._temporal(P, L)
        return self._spatial(P, L, Prop=Prop)


class DreamerV4Encoder(nn.Module):
    """Block-causal tokenizer encoder.

    Patches (+ one proprio token) and ``n_latents`` learned latent tokens per frame go through
    alternating spatial / temporal layers; the latents are read out through a linear
    down-projection and a tanh bottleneck (computed in fp32).
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        d_model: int,
        n_layers: int,
        num_heads_q: int,
        num_heads_kv_latent: int,
        seq_len: int,
        dropout: float = 0.0,
        n_latents: int = 256,
        bottleneck_dim: int = 16,
        temporal_every: int = 1,
        context_len: int = 0,
        in_channels: int = 3,
        mae_max_mask_prob: float = 0.9,
        use_proprio: bool = False,
        proprio_dim: int = 0,
    ):
        super().__init__()
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.H, self.W = H, W
        self.P = patch_size
        self.Np = (H // patch_size) * (W // patch_size)
        self.Nl = n_latents
        self.d = d_model
        self.mae_max_mask_prob = mae_max_mask_prob

        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.latent_tokens = nn.Parameter(torch.randn(n_latents, d_model) / math.sqrt(d_model))

        self.layers = nn.ModuleList([
            BlockCausalEncoderLayer(
                d_model=d_model,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv_latent,
                seq_len=seq_len,
                num_latents=self.Nl,
                num_patches=self.Np,
                dropout=dropout,
                temporal=_is_temporal(i, temporal_every),
                is_last=_is_last(i, n_layers, temporal_every),
                num_proprio=1 if use_proprio else 0,
                context_len=context_len if _is_temporal(i, temporal_every) else 0,
            )
            for i in range(n_layers)
        ])
        self.down_proj = nn.Linear(d_model, bottleneck_dim)

        self.use_proprio = use_proprio
        if use_proprio:
            assert proprio_dim > 0, "proprio_dim must be > 0 when use_proprio=True"
            self.proprio_proj = nn.Linear(proprio_dim, d_model)
            self.mask_proprio_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
            nn.init.trunc_normal_(self.mask_proprio_token, std=0.02)
            self.proprio_mask_prob = 0.5  # training: each frame's proprio is masked with p = 0.5

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, C, H, W]`` -> patch tokens ``[B, T, Np, d]``."""
        B, T, C, H, W = x.shape
        tok = self.patch_embed(x.view(B * T, C, H, W))
        tok = tok.flatten(2).transpose(1, 2)
        return tok.view(B, T, self.Np, self.d)

    def forward(self, video: torch.Tensor, proprio: Optional[torch.Tensor] = None):
        """``video [B, T, C, H, W]``, ``proprio [B, T, proprio_dim]``.

        Returns ``(P_enc, L_enc, Z)`` with ``Z [B, T, Nl, bottleneck_dim]`` (``P_enc`` is None
        when the last layer has no patch stream).
        """
        B, T, C, H, W = video.shape
        assert (H, W) == (self.H, self.W)
        P = self.patchify(video)

        # MAE patch dropout: per-frame rate p ~ U(0, mae_max_mask_prob), per-patch Bernoulli(p).
        if self.training and self.mae_max_mask_prob > 0.0:
            B, T, Np, D = P.shape
            drop_p = torch.rand(B, T, device=P.device, dtype=P.dtype) * self.mae_max_mask_prob
            rand = torch.rand(B, T, Np, device=P.device, dtype=P.dtype)
            mask = rand < drop_p.unsqueeze(-1)
            P = torch.where(mask.unsqueeze(-1), self.mask_token.to(P.dtype), P)

        Prop = None
        if self.use_proprio and proprio is not None:
            Prop = self.proprio_proj(proprio).unsqueeze(2)  # [B, T, 1, d]
            if self.training:
                prop_mask = torch.rand(B, T, device=Prop.device, dtype=Prop.dtype) < self.proprio_mask_prob
                Prop = torch.where(prop_mask[:, :, None, None], self.mask_proprio_token.to(Prop.dtype), Prop)

        L0 = self.latent_tokens[None, None, :, :].expand(B, T, self.Nl, self.d).contiguous()
        P_enc, L_enc = P, L0
        for layer in self.layers:
            P_enc, L_enc = layer(P_enc, L_enc, Prop=None if layer.temporal else Prop)

        # tanh bottleneck in fp32: under bf16, tanh saturates to exactly +-1 for |x| > ~4 and
        # its gradient becomes zero, which can collapse the code. Cast back afterwards.
        pre_tanh = self.down_proj(L_enc).float()
        Z_fp32 = torch.tanh(pre_tanh)
        Z = Z_fp32.to(L_enc.dtype)

        if self.training:
            # Saturation diagnostics read by the tokenizer training loop.
            with torch.no_grad():
                self._last_pretanh_abs_mean = pre_tanh.detach().abs().mean()
                self._last_z_sat_frac = (Z_fp32.detach().abs() > 0.99).float().mean()

        return P_enc, L_enc, Z


class BlockCausalDecoderLayer(nn.Module):
    """Decoder layer, spatial-only or temporal-only.

    Spatial: latents attend to latents; readout tokens (patches + proprio) attend to
    {latents, readouts}, with image readouts blind to proprio readouts and vice versa.
    Temporal: causal over each token's own history, optionally through a :class:`KVCache`.
    """

    def __init__(self, d_model: int, num_heads_q: int, num_heads_kv_latent: int, seq_len: int, num_latents: int,
                 num_patches: int, dropout: float = 0.0, temporal: bool = False, is_last: bool = False,
                 use_proprio: bool = False, num_proprio: int = 1):
        super().__init__()
        Dh = d_model // num_heads_q
        self.temporal = temporal
        self.is_last = is_last
        self.use_proprio = use_proprio
        self.num_proprio = num_proprio if use_proprio else 0
        spatial_kv_len = num_latents + num_patches + self.num_proprio

        if not is_last:
            self.attn_latent = MHA_GQA(d_model, num_heads_q, num_heads_kv_latent, Dh, dropout=dropout,
                                       max_seq_len=seq_len if temporal else num_latents)
        self.attn_patch = MHA_GQA(d_model, num_heads_q, num_heads_q, Dh, dropout=dropout,
                                  max_seq_len=seq_len if temporal else spatial_kv_len)
        if not is_last:
            self.ln_L_q = nn.RMSNorm(d_model)
            self.ln_L_kv = nn.RMSNorm(d_model)
        self.ln_P_q = nn.RMSNorm(d_model)
        self.ln_P_kv = nn.RMSNorm(d_model)
        if not is_last:
            self.ffn_L = FeedForwardSwiGLU(d_model, None, dropout)
        self.ffn_P = FeedForwardSwiGLU(d_model, None, dropout)
        if not is_last:
            self.ln_L_ff = nn.RMSNorm(d_model)
        self.ln_P_ff = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if use_proprio:
            # Rows: [image readouts | proprio readouts]; columns: [latents | image | proprio].
            Nr = num_patches + num_proprio
            Nkv = num_latents + num_patches + num_proprio
            mask = torch.zeros(Nr, Nkv)
            mask[:num_patches, num_latents + num_patches:] = -1e9
            mask[num_patches:, num_latents:num_latents + num_patches] = -1e9
            self.register_buffer("proprio_spatial_mask", mask, persistent=True)

    def _spatial(self, R, L):
        union_sp = torch.cat([L, R], dim=2)
        if self.is_last:
            L_out = None
        else:
            L_att = self.attn_latent(self.ln_L_q(L), self.ln_L_kv(L), attn_mask=None, is_causal=False)
            L_out = L + self.dropout(L_att)
            L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
        spatial_mask = self.proprio_spatial_mask.to(dtype=R.dtype) if self.use_proprio else None
        R_att = self.attn_patch(self.ln_P_q(R), self.ln_P_kv(union_sp), attn_mask=spatial_mask, is_causal=False)
        R_out = R + self.dropout(R_att)
        R_out = R_out + self.dropout(self.ffn_P(self.ln_P_ff(R_out)))
        return R_out, L_out

    def _temporal(self, R, L, kv_cache_l, kv_cache_r, position_ids, update_cache):
        B, T, Nr, d = R.shape
        if T > 1 and kv_cache_r is not None and kv_cache_r.curr_len > 0:
            # is_causal with Sq != Skv aligns the mask top-left: the new frames would only see
            # the oldest cache entries. Multi-frame prefill is valid on an empty cache only.
            raise RuntimeError("decoder: a multi-frame forward_step needs an empty KV cache (reset it first)")

        R_flat = R.transpose(1, 2).reshape(B * Nr, T, d)
        R_att = self.attn_patch(self.ln_P_q(R_flat), self.ln_P_kv(R_flat), is_causal=True, kv_cache=kv_cache_r,
                                q_position_ids=position_ids, kv_position_ids=position_ids, update_cache=update_cache)
        R_out = R_flat + self.dropout(R_att)
        R_out = R_out + self.dropout(self.ffn_P(self.ln_P_ff(R_out)))
        R_out = R_out.view(B, Nr, T, d).transpose(1, 2)

        if self.is_last:
            return R_out, None
        Nl = L.shape[2]
        L_flat = L.transpose(1, 2).reshape(B * Nl, T, d)
        L_att = self.attn_latent(self.ln_L_q(L_flat), self.ln_L_kv(L_flat), is_causal=True, kv_cache=kv_cache_l,
                                 q_position_ids=position_ids, kv_position_ids=position_ids, update_cache=update_cache)
        L_out = L_flat + self.dropout(L_att)
        L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
        return R_out, L_out.view(B, Nl, T, d).transpose(1, 2)

    def forward(self, R: torch.Tensor, L: torch.Tensor, kv_cache_l: Optional[KVCache] = None,
                kv_cache_r: Optional[KVCache] = None, position_ids: Optional[torch.Tensor] = None,
                update_cache: bool = True):
        """``R [B, T, Nr, d]`` readout tokens, ``L [B, T, Nl, d]`` latents."""
        if self.temporal:
            return self._temporal(R, L, kv_cache_l, kv_cache_r, position_ids, update_cache)
        return self._spatial(R, L)


class DreamerV4Decoder(nn.Module):
    """Block-causal tokenizer decoder: latents ``Z [B, T, Nl, d_b]`` -> frames (+ proprio).

    ``forward`` decodes a whole clip; ``init_cache`` + ``forward_step`` decode frame by frame
    with KV caches in the temporal layers (a multi-frame ``forward_step`` prefills them).
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        d_model: int,
        n_layers: int,
        num_heads_q: int,
        num_heads_kv_latent: int,
        bottleneck_dim: int,
        seq_len: int,
        dropout: float = 0.0,
        n_latents: int = 256,
        in_channels: int = 3,
        temporal_every: int = 1,
        use_proprio: bool = False,
        proprio_dim: int = 0,
        num_proprio: int = 1,
    ):
        super().__init__()
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.H, self.W = H, W
        self.P = patch_size
        self.Np = (H // patch_size) * (W // patch_size)
        self.Nl = n_latents
        self.d = d_model
        self.d_b = bottleneck_dim
        self.C = in_channels
        self.use_proprio = use_proprio
        self.num_proprio = num_proprio if use_proprio else 0

        if use_proprio:
            assert proprio_dim > 0, "proprio_dim must be > 0 when use_proprio=True"
            self.proprio_dim = proprio_dim
            self.readout_proprio_token = nn.Parameter(torch.randn(num_proprio, d_model) / math.sqrt(d_model))
            self.proprio_head = nn.Linear(d_model, proprio_dim)

        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.readout_tokens = nn.Parameter(torch.randn(self.Np, d_model) / math.sqrt(d_model))
        self.layers = nn.ModuleList([
            BlockCausalDecoderLayer(
                d_model=d_model,
                num_heads_q=num_heads_q,
                num_heads_kv_latent=num_heads_kv_latent,
                seq_len=seq_len,
                num_latents=self.Nl,
                num_patches=self.Np,
                dropout=dropout,
                temporal=_is_temporal(i, temporal_every),
                is_last=_is_last(i, n_layers, temporal_every),
                use_proprio=use_proprio,
                num_proprio=num_proprio,
            )
            for i in range(n_layers)
        ])
        self.patch_head = nn.Linear(d_model, (patch_size ** 2) * in_channels)

    def _build_readout(self, B: int, T: int) -> torch.Tensor:
        R0 = self.readout_tokens[None, None].expand(B, T, self.Np, self.d).contiguous()
        if self.use_proprio:
            R0_prop = self.readout_proprio_token[None, None].expand(B, T, self.num_proprio, self.d).contiguous()
            R0 = torch.cat([R0, R0_prop], dim=2)
        return R0

    def _decode_output(self, R_dec: torch.Tensor, B: int, T: int):
        """Readouts -> ``(R_dec, x_hat [B, T, C, H, W])`` (+ ``prop_hat [B, T, proprio_dim]``)."""
        R_img = R_dec[:, :, :self.Np, :]
        patches = self.patch_head(R_img).view(B, T, self.Np, self.P, self.P, self.C)
        Hp, Wp = self.H // self.P, self.W // self.P
        patches = patches.view(B, T, Hp, Wp, self.P, self.P, self.C).permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x_hat = patches.view(B, T, self.H, self.W, self.C).permute(0, 1, 4, 2, 3).contiguous()
        if self.use_proprio:
            prop_hat = self.proprio_head(R_dec[:, :, self.Np:, :].squeeze(2))
            return R_dec, x_hat, prop_hat
        return R_dec, x_hat

    def forward(self, Z: torch.Tensor):
        """Decode a whole clip ``Z [B, T, Nl, d_b]``."""
        B, T, Nl, d_b = Z.shape
        assert Nl == self.Nl and d_b == self.d_b
        L = self.up_proj(Z)
        R_dec, L_dec = self._build_readout(B, T), L
        # Compile boundary: fusing _build_readout into layer 0's kernel caused bf16 NaNs.
        torch._dynamo.graph_break()
        for layer in self.layers:
            R_dec, L_dec = layer(R_dec, L_dec)
        return self._decode_output(R_dec, B, T)

    def _cache_layout(self, batch_size: int) -> list:
        """Per layer, ``(latent, readout)`` cache shapes ``(rows, kv_heads, head_dim)`` or None."""
        readout_rows = batch_size * (self.Np + self.num_proprio)
        layout = []
        for layer in self.layers:
            if not layer.temporal:
                layout.append((None, None))
                continue
            latent = None if layer.is_last else (batch_size * self.Nl, layer.attn_latent.G, layer.attn_latent.Dh)
            layout.append((latent, (readout_rows, layer.attn_patch.G, layer.attn_patch.Dh)))
        return layout

    def kv_cache_bytes(self, batch_size: int, max_seq_len: int, dtype: torch.dtype | None = None) -> int:
        """Bytes :meth:`init_cache` allocates (keys + values; ``dtype`` defaults to the weights')."""
        specs = [spec for pair in self._cache_layout(batch_size) for spec in pair]
        return _kv_cache_bytes(specs, max_seq_len, dtype or self.up_proj.weight.dtype)

    def init_cache(self, batch_size: int, device: torch.device, max_seq_len: int):
        """Allocate KV caches (latent and readout streams) for every temporal layer."""
        dtype = self.up_proj.weight.dtype

        def make(spec):
            return None if spec is None else KVCache(max_seq_len, *spec, device, dtype)

        layout = self._cache_layout(batch_size)
        self.caches_l = [make(latent) for latent, _ in layout]
        self.caches_r = [make(readout) for _, readout in layout]

    def forward_step(self, Z: torch.Tensor, start_step_idx: int, update_cache: bool = True):
        """Decode ``Z [B, T, Nl, d_b]`` at absolute frame positions ``start_step_idx + 0..T-1``."""
        B, T, Nl, d_b = Z.shape
        L = self.up_proj(Z)
        R_dec, L_dec = self._build_readout(B, T), L
        pos_ids = torch.arange(start_step_idx, start_step_idx + T, device=Z.device, dtype=torch.long).unsqueeze(0)
        for i, layer in enumerate(self.layers):
            if layer.temporal:
                R_dec, L_dec = layer(R_dec, L_dec, kv_cache_l=self.caches_l[i], kv_cache_r=self.caches_r[i],
                                     position_ids=pos_ids, update_cache=update_cache)
            else:
                R_dec, L_dec = layer(R_dec, L_dec)
        return self._decode_output(R_dec, B, T)


# --------------------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------------------

class BlockCausalDynamicsLayer(nn.Module):
    """Dynamics layer over ``[B, T, N, D]``: spatial (within a frame, optional mask) or
    temporal (each token over its own history, within a sliding window of ``context_len``
    frames; KV cache for frame-by-frame generation)."""

    def __init__(self, d_model, num_heads, seq_len, num_tokens, context_len: int = 0, dropout=0.0, mlp_ratio=4.0,
                 temporal=False):
        super().__init__()
        self.d_model = d_model
        self.temporal = temporal
        self.context_len = context_len
        self.norm1 = nn.RMSNorm(d_model, eps=1e-6)
        self.attn = MHA_GQA(d_model=d_model, num_heads_q=num_heads, num_heads_kv=num_heads,
                            head_dim=d_model // num_heads, dropout=dropout,
                            max_seq_len=seq_len if temporal else num_tokens)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-6)
        self.ffn = FeedForwardSwiGLU(d_model, int(d_model * mlp_ratio * 2 / 3), dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, kv_cache=None, position_ids=None, update_cache=True):
        B, T, N, D = x.shape
        skip_x = x
        x_norm = self.norm1(x)

        if self.temporal:
            x_flat = x_norm.permute(0, 2, 1, 3).reshape(-1, T, self.d_model)  # [B*N, T, D]
            if kv_cache is None and self.context_len > 0 and T > self.context_len:
                # Sliding window: query j attends key i iff 0 <= j - i < context_len.
                idx = torch.arange(T, device=x.device)
                dist = idx.view(-1, 1) - idx.view(1, -1)
                keep = (dist >= 0) & (dist < self.context_len)
                temp_mask = torch.full((T, T), -1e9, device=x.device, dtype=x.dtype)
                temp_mask.masked_fill_(keep, 0.0)
                x_attn = self.attn(x_flat, x_flat, attn_mask=temp_mask, is_causal=False,
                                   q_position_ids=position_ids, kv_position_ids=position_ids)
            elif kv_cache is None:
                x_attn = self.attn(x_flat, x_flat, is_causal=True, q_position_ids=position_ids,
                                   kv_position_ids=position_ids)
            else:
                x_attn = self.attn(x_flat, x_flat, attn_mask=self._cached_mask(kv_cache, T, update_cache, x),
                                   is_causal=False, kv_cache=kv_cache, q_position_ids=position_ids,
                                   kv_position_ids=position_ids, update_cache=update_cache)
            x_attn = x_attn.view(B, N, T, D).permute(0, 2, 1, 3)
        else:
            x_flat = x_norm.view(-1, N, self.d_model)  # [B*T, N, D]
            x_attn = self.attn(x_flat, x_flat, attn_mask=attn_mask, is_causal=False).view(B, T, N, D)

        x = skip_x + self.dropout(x_attn)
        return x + self.dropout(self.ffn(self.norm2(x)))

    def _cached_mask(self, kv_cache: KVCache, T: int, update_cache: bool, x: torch.Tensor) -> torch.Tensor:
        """Additive mask ``[1, 1, T, cache + T]`` for a cached pass over ``T`` new frames.

        The keys are consecutive frames ending with the new ones (the cache holds the most
        recent frames). ``is_causal`` would align the mask top-left (the new queries would see
        only the oldest cache entries), so the mask is explicit: causal among the new frames,
        and the uncached forward's sliding window of ``context_len`` frames over the cache.
        """
        if update_cache and T > kv_cache.max_seq_len:
            raise ValueError(f"a cache-writing pass over {T} frames does not fit the KV cache "
                             f"({kv_cache.max_seq_len} frames)")
        total_len = min(kv_cache.curr_len + T, kv_cache.max_seq_len) if update_cache else kv_cache.curr_len + T
        cache_len = total_len - T
        chunk_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
        cache_mask = torch.zeros((T, cache_len), device=x.device, dtype=x.dtype)
        mask = torch.cat([cache_mask, chunk_mask], dim=1)
        if 0 < self.context_len < total_len:
            # New frame q is key cache_len + q: hide the keys context_len or more frames back.
            dist = ((cache_len + torch.arange(T, device=x.device)).view(-1, 1)
                    - torch.arange(total_len, device=x.device).view(1, -1))
            mask = mask.masked_fill(dist >= self.context_len, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)


class DreamerV4Dynamics(nn.Module):
    """Shortcut / flow-matching dynamics transformer over tokenizer latents.

    Tokens per frame: ``[action, noise level (tau, step), registers, latents]`` plus, with
    ``use_reward_head``, an agent token that reads the world tokens but is hidden from them;
    its output feeds the multi-token reward head. ``forward`` runs a whole clip;
    ``init_cache`` + ``forward_step`` generate frame by frame.
    """

    def __init__(
        self,
        action_dim: int,
        num_latents: int,
        latent_dim: int,
        d_model: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        num_registers: int = 4,
        seq_len: int = 128,
        context_len: int = 96,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        num_tau_levels: int = 128,
        temporal_every: int = 4,
        use_reward_head: bool = False,
        mtp_length: int = 8,
        reward_hidden_dim: int = 512,
        reward_num_buckets: int = 255,
        reward_symlog_range: Tuple[float, float] = (-20.0, 20.0),
    ):
        super().__init__()
        num_step_levels = int(math.log2(num_tau_levels)) + 1
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_registers = num_registers
        self.use_reward_head = use_reward_head
        self.has_agent_token = use_reward_head

        if self.has_agent_token:
            self.agent_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        else:
            self.register_parameter("agent_token", None)
        if use_reward_head:
            self.reward_head = RewardMTPHead(input_dim=d_model, hidden_dim=reward_hidden_dim, mtp_length=mtp_length,
                                             num_buckets=reward_num_buckets, min_val=reward_symlog_range[0],
                                             max_val=reward_symlog_range[1])
        else:
            self.reward_head = None

        self.act_proj = nn.Linear(action_dim, d_model)
        self.sigma_embed = nn.Embedding(num_tau_levels, d_model // 2)
        self.step_embed = nn.Embedding(num_step_levels, d_model // 2)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.register_tokens = nn.Parameter(torch.randn(num_registers, d_model) * 0.02)

        self.num_world_tokens = 1 + 1 + num_registers + num_latents  # action, noise, registers, latents
        self.total_tokens = self.num_world_tokens + (1 if self.has_agent_token else 0)

        layers = []
        for i in range(num_layers):
            is_temporal = (i + 1) % temporal_every == 0
            layers.append(BlockCausalDynamicsLayer(
                d_model=d_model, num_heads=num_heads, seq_len=seq_len,
                context_len=context_len if is_temporal else 0, num_tokens=self.total_tokens,
                dropout=dropout, mlp_ratio=mlp_ratio, temporal=is_temporal,
            ))
        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Linear(d_model, latent_dim)
        self.init_nonpersistent_buffers()

    def init_nonpersistent_buffers(self, device: torch.device | str | None = None) -> None:
        """Spatial mask hiding the agent token (last column) from the world tokens."""
        if self.has_agent_token:
            mask = torch.zeros(self.total_tokens, self.total_tokens, dtype=torch.float32, device=device)
            mask[:-1, -1] = -1e9
            self.register_buffer("spatial_mask", mask, persistent=False)

    def _embed(self, action, noisy_z, sigma_idx, step_idx):
        B, T = noisy_z.shape[:2]
        emb_act = self.act_proj(action).unsqueeze(2)
        emb_noise = torch.cat([self.sigma_embed(sigma_idx), self.step_embed(step_idx)], dim=-1).unsqueeze(2)
        emb_reg = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        emb_z = self.z_proj(noisy_z)
        x = torch.cat([emb_act, emb_noise, emb_reg, emb_z], dim=2)
        if self.has_agent_token:
            agent_part = self.agent_token.view(1, 1, 1, self.d_model).expand(B, T, -1, -1)
            x = torch.cat([x, agent_part], dim=2)
        return x

    def _spatial_layer(self, layer, x):
        if self.has_agent_token:
            return layer(x, attn_mask=self.spatial_mask.to(dtype=x.dtype))
        return layer(x)

    def _readout(self, x):
        """``-> (pred_z [B, T, Nl, latent_dim], reward logits [B, T, mtp, buckets] or None)``."""
        if self.has_agent_token:
            pred_z = self.out_proj(x[:, :, :-1, :][:, :, -self.num_latents:, :])
            pred_rewards = self.reward_head(x[:, :, -1, :])
        else:
            pred_z = self.out_proj(x[:, :, -self.num_latents:, :])
            pred_rewards = None
        return pred_z, pred_rewards

    def forward(self, action: torch.Tensor, noisy_z: torch.Tensor, sigma_idx: torch.Tensor, step_idx: torch.Tensor):
        """``action [B, T, A]``, ``noisy_z [B, T, Nl, latent_dim]``, ``sigma_idx``/``step_idx [B, T]``
        -> ``(pred_z, reward logits or None)``."""
        x = self._embed(action, noisy_z, sigma_idx, step_idx)
        for layer in self.layers:
            x = layer(x) if layer.temporal else self._spatial_layer(layer, x)
        return self._readout(x)

    def _cache_layout(self, batch_size: int) -> list:
        """Per layer, the cache shape ``(rows, kv_heads, head_dim)`` or None."""
        return [(batch_size * self.total_tokens, layer.attn.G, layer.attn.Dh) if layer.temporal else None
                for layer in self.layers]

    def kv_cache_bytes(self, batch_size: int, max_seq_len: int, dtype: torch.dtype | None = None) -> int:
        """Bytes :meth:`init_cache` allocates (keys + values; ``dtype`` defaults to the weights')."""
        return _kv_cache_bytes(self._cache_layout(batch_size), max_seq_len, dtype or self.act_proj.weight.dtype)

    def init_cache(self, batch_size: int, device: torch.device, max_seq_len: int):
        """Allocate a KV cache of ``max_seq_len`` frames for every temporal layer.

        The cached passes apply the same ``context_len`` window as :meth:`forward`, so the
        cache may be longer than the window.
        """
        dtype = self.act_proj.weight.dtype
        self.caches = [None if spec is None else KVCache(max_seq_len, *spec, device, dtype)
                       for spec in self._cache_layout(batch_size)]

    def forward_step(self, action: torch.Tensor, noisy_z: torch.Tensor, sigma_idx: torch.Tensor,
                     step_idx: torch.Tensor, start_step_idx: int, update_cache: bool = True):
        """Cached forward at absolute frame positions ``start_step_idx + 0..T-1``.

        ``update_cache=False`` attends to [cache, current frames] without writing (refinement
        passes); ``update_cache=True`` appends the frames (prefill, cache-writing pass).
        Returns ``(pred_z, reward logits or None)``.
        """
        x = self._embed(action, noisy_z, sigma_idx, step_idx)
        T = x.shape[1]
        pos_ids = torch.arange(start_step_idx, start_step_idx + T, device=x.device, dtype=torch.long).unsqueeze(0)
        for i, layer in enumerate(self.layers):
            if layer.temporal:
                x = layer(x, kv_cache=self.caches[i], position_ids=pos_ids, update_cache=update_cache)
            else:
                x = self._spatial_layer(layer, x)
        return self._readout(x)


# --------------------------------------------------------------------------------------
# Architecture configs
# --------------------------------------------------------------------------------------

G1_TOKENIZER_ARCH: dict[str, Any] = {
    "image_size": [256, 256],
    "patch_size": 16,
    "in_channels": 3,
    "d_model": 1024,
    "n_layers": 16,
    "num_heads_q": 16,
    "num_heads_kv_latent": 16,
    "temporal_every": 4,
    "n_latents": 256,
    "bottleneck_dim": 32,
    "use_proprio": True,
    "proprio_dim": 50,
    "num_proprio": 1,
    "seq_len": 64,
    "context_len": 0,
    "dropout": 0.0,
    "mae_max_mask_prob": 0.9,
}
"""Paper G1 tokenizer (trained on 64-frame clips, no temporal window)."""

G1_DYNAMICS_ARCH: dict[str, Any] = {
    "action_dim": 22,
    "num_latents": 256,
    "latent_dim": 32,
    "d_model": 1536,
    "num_layers": 24,
    "num_heads": 24,
    "num_registers": 4,
    "num_tau_levels": 128,
    "temporal_every": 4,
    "mlp_ratio": 4.0,
    "seq_len": 128,
    "context_len": 96,
    "dropout": 0.0,
    "use_reward_head": True,
    "mtp_length": 8,
    "reward_hidden_dim": 512,
    "reward_num_buckets": 255,
    "reward_symlog_range": [-20.0, 20.0],
}
"""Paper G1 dynamics (backbone stage on 32- and 128-frame clips, reward stage on 32)."""

G1_CONTROL_HZ = 10
"""Frame / control rate of the G1 data."""

G1_PREPROCESSING: dict[str, Any] = {
    "image_size": [256, 256],
    "image_format": "float RGB in [0, 1], (T, 3, H, W)",
    "h5_image_key": "images",
    "h5_image_format": "uint8 (T, H, W, 3)",
    "h5_color_order": "bgr",
    "proprio_dim": 50,
    "h5_proprio_key": "proprio",
    "proprio_normalization": None,
    "h5_action_key": "commands",
}
"""Tokenizer input contract of the G1 data (BGR on disk, flipped to RGB on load; raw proprio)."""

G1_INFERENCE: dict[str, Any] = {
    "num_inference_steps": 4,
    "d_min_idx": 7,
    "seq_cor_tau_idx": 12,
    "init_context_len": 32,
    "kv_cache_len": 32,
    "reward_mtp_index": 0,
}
"""Constants of the G1 environment's generation loop: Euler steps per frame, the finest
shortcut step index (``d_min_idx`` = log2 num_tau_levels = 7, a derived value recorded for
reference), the noise level at which frames are written to the KV cache (``seq_cor_tau_idx``:
the generated frames and the prefilled context), the number of real context frames, the KV cache length
(frames) and the reward MTP head used as the step reward."""

G1_ACTION: dict[str, Any] = {
    "dim": 22,
    "representation": "absolute command per dim (h5 key 'commands'): base velocity vx, vy, yaw rate (0-2), "
                      "upper-body joint positions (3-19), grips in [0, 1] (20-21); consumed with the frame of the "
                      "same index",
    "base_velocity_dims": [0, 1, 2],
    "base_velocity_names": ["vx", "vy", "yaw_rate"],
    "upper_body_dims": list(range(3, 20)),
    "grip_dims": [20, 21],
    "grip_range": [0.0, 1.0],
    "scale": [0.3, 0.4, 1.5, 0.65, 0.52, 0.52, 1.8, 2.5, 1.4, 1.6, 1.4, 1.4,
              1.7, 1.7, 1.97, 1.97, 1.61, 1.61, 1.61, 1.61, 1.0, 1.0],
}
"""G1 action layout. ``scale`` is the symmetric envelope (about the per-dim max |command| of the
G1 data) the environment clamps absolute commands to; grips are also clamped to ``grip_range``."""

TOKENIZER_RUNTIME_KEYS = frozenset({"seq_len", "context_len", "dropout", "mae_max_mask_prob"})
DYNAMICS_RUNTIME_KEYS = frozenset({"seq_len", "context_len", "dropout"})

_INT, _FLOAT, _BOOL = "int", "float", "bool"
_TOKENIZER_TYPES = {
    "patch_size": _INT, "in_channels": _INT, "d_model": _INT, "n_layers": _INT, "num_heads_q": _INT,
    "num_heads_kv_latent": _INT, "temporal_every": _INT, "n_latents": _INT, "bottleneck_dim": _INT,
    "use_proprio": _BOOL, "proprio_dim": _INT, "num_proprio": _INT, "seq_len": _INT, "context_len": _INT,
    "dropout": _FLOAT, "mae_max_mask_prob": _FLOAT,
}
_DYNAMICS_TYPES = {
    "action_dim": _INT, "num_latents": _INT, "latent_dim": _INT, "d_model": _INT, "num_layers": _INT,
    "num_heads": _INT, "num_registers": _INT, "num_tau_levels": _INT, "temporal_every": _INT, "mlp_ratio": _FLOAT,
    "seq_len": _INT, "context_len": _INT, "dropout": _FLOAT, "use_reward_head": _BOOL, "mtp_length": _INT,
    "reward_hidden_dim": _INT, "reward_num_buckets": _INT,
}


def _coerce(name: str, value: Any, kind: str) -> Any:
    if kind == _BOOL:
        if not isinstance(value, bool):
            raise TypeError(f"architecture key {name!r} must be a bool, got {value!r}")
        return value
    if isinstance(value, bool):
        raise TypeError(f"architecture key {name!r} must be a number, got {value!r}")
    if kind == _INT:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise TypeError(f"architecture key {name!r} must be an int, got {value!r}")
        return value
    if not isinstance(value, (int, float)):
        raise TypeError(f"architecture key {name!r} must be a number, got {value!r}")
    return float(value)


def _pair(name: str, value: Any, kind: str) -> list:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value, value]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"architecture key {name!r} must be a pair, got {value!r}")
    return [_coerce(name, v, kind) for v in value]


def _normalize(cfg: Mapping[str, Any], defaults: Mapping[str, Any], types: Mapping[str, str], what: str,
               complete: bool) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        raise TypeError(f"{what} architecture must be a mapping, got {type(cfg).__name__}")
    unknown = sorted(set(cfg) - set(defaults))
    if unknown:
        raise KeyError(f"unknown {what} architecture key(s) {unknown}; valid keys: {sorted(defaults)}")
    if complete:
        missing = sorted(set(defaults) - set(cfg))
        if missing:
            raise KeyError(f"{what} architecture is missing key(s) {missing}")
    out = {}
    for key, default in defaults.items():
        value = copy.deepcopy(cfg.get(key, default))
        out[key] = _coerce(key, value, types[key]) if key in types else value
    return out


def tokenizer_arch(cfg: Mapping[str, Any] | None = None, *, complete: bool = False) -> dict[str, Any]:
    """Validate a tokenizer architecture dict and return it complete, in canonical order.

    Missing keys take the paper G1 values unless ``complete`` is set (then they raise).
    """
    a = _normalize(cfg or {}, G1_TOKENIZER_ARCH, _TOKENIZER_TYPES, "tokenizer", complete)
    a["image_size"] = _pair("image_size", a["image_size"], _INT)
    H, W = a["image_size"]
    if H % a["patch_size"] or W % a["patch_size"]:
        raise ValueError(f"image_size {a['image_size']} is not divisible by patch_size {a['patch_size']}")
    if a["d_model"] % a["num_heads_q"] or (a["d_model"] // a["num_heads_q"]) % 2:
        raise ValueError("d_model must be num_heads_q * an even head_dim")
    if a["num_heads_q"] % a["num_heads_kv_latent"]:
        raise ValueError("num_heads_q must be divisible by num_heads_kv_latent")
    n, k = a["n_layers"], a["temporal_every"]
    if n < 1 or k < 1:
        raise ValueError("n_layers and temporal_every must be >= 1")
    if n >= 2 and _is_last(n - 2, n, k) and not _is_temporal(n - 1, k):
        # Layer n-2 drops the patch (encoder) / latent (decoder) stream that the spatial layer
        # n-1 still reads. Only temporal_every == 1 with an odd n_layers >= 3 does this.
        raise ValueError(f"n_layers={n} with temporal_every={k}: layer {n - 2} drops the patch/latent stream that "
                         f"the spatial layer {n - 1} needs; with temporal_every=1 use an even n_layers")
    if a["use_proprio"] and (a["proprio_dim"] < 1 or a["num_proprio"] != 1):
        raise ValueError("use_proprio needs proprio_dim >= 1 and num_proprio == 1 (the encoder embeds one proprio "
                         "token per frame and the decoder reads one out)")
    if a["context_len"] < 0 or a["seq_len"] < 1:
        raise ValueError("context_len must be >= 0 and seq_len >= 1")
    return a


def dynamics_arch(cfg: Mapping[str, Any] | None = None, *, complete: bool = False) -> dict[str, Any]:
    """Validate a dynamics architecture dict and return it complete, in canonical order.

    Missing keys take the paper G1 values unless ``complete`` is set (then they raise).
    """
    a = _normalize(cfg or {}, G1_DYNAMICS_ARCH, _DYNAMICS_TYPES, "dynamics", complete)
    a["reward_symlog_range"] = _pair("reward_symlog_range", a["reward_symlog_range"], _FLOAT)
    if a["d_model"] % a["num_heads"] or (a["d_model"] // a["num_heads"]) % 2 or a["d_model"] % 2:
        raise ValueError("d_model must be even and num_heads * an even head_dim")
    n = a["num_tau_levels"]
    if n < 2 or n & (n - 1):
        raise ValueError(f"num_tau_levels must be a power of two (shortcut step sizes), got {n}")
    if a["num_layers"] < 1 or a["temporal_every"] < 1:
        raise ValueError("num_layers and temporal_every must be >= 1")
    if a["context_len"] < 0 or a["seq_len"] < 1:
        raise ValueError("context_len must be >= 0 and seq_len >= 1")
    lo, hi = a["reward_symlog_range"]
    if not lo < hi or a["reward_num_buckets"] < 2:
        raise ValueError("reward_symlog_range must be increasing and reward_num_buckets >= 2")
    return a


def build_tokenizer(cfg: Mapping[str, Any] | None = None) -> Tuple[DreamerV4Encoder, DreamerV4Decoder]:
    """Build ``(encoder, decoder)`` from a tokenizer architecture dict (see module docstring)."""
    a = tokenizer_arch(cfg)
    common = dict(
        image_size=tuple(a["image_size"]), patch_size=a["patch_size"], d_model=a["d_model"],
        n_layers=a["n_layers"], num_heads_q=a["num_heads_q"], num_heads_kv_latent=a["num_heads_kv_latent"],
        seq_len=a["seq_len"], dropout=a["dropout"], n_latents=a["n_latents"], bottleneck_dim=a["bottleneck_dim"],
        temporal_every=a["temporal_every"], in_channels=a["in_channels"], use_proprio=a["use_proprio"],
        proprio_dim=a["proprio_dim"] if a["use_proprio"] else 0,
    )
    encoder = DreamerV4Encoder(**common, context_len=a["context_len"], mae_max_mask_prob=a["mae_max_mask_prob"])
    decoder = DreamerV4Decoder(**common, num_proprio=a["num_proprio"])
    return encoder, decoder


def build_dynamics(cfg: Mapping[str, Any] | None = None) -> DreamerV4Dynamics:
    """Build the dynamics model from a dynamics architecture dict (see module docstring)."""
    a = dynamics_arch(cfg)
    return DreamerV4Dynamics(
        action_dim=a["action_dim"], num_latents=a["num_latents"], latent_dim=a["latent_dim"],
        d_model=a["d_model"], num_layers=a["num_layers"], num_heads=a["num_heads"],
        num_registers=a["num_registers"], seq_len=a["seq_len"], context_len=a["context_len"],
        dropout=a["dropout"], mlp_ratio=a["mlp_ratio"], num_tau_levels=a["num_tau_levels"],
        temporal_every=a["temporal_every"], use_reward_head=a["use_reward_head"], mtp_length=a["mtp_length"],
        reward_hidden_dim=a["reward_hidden_dim"], reward_num_buckets=a["reward_num_buckets"],
        reward_symlog_range=tuple(a["reward_symlog_range"]),
    )


def init_nonpersistent_buffers(module: nn.Module, device: torch.device | str = "cpu") -> None:
    """Recompute every non-persistent buffer of ``module`` (RoPE tables, spatial masks).

    Needed after building on the meta device: those buffers are not in any state dict.
    """
    for m in module.modules():
        if isinstance(m, (MHA_GQA, DreamerV4Dynamics)):
            m.init_nonpersistent_buffers(device)


__all__ = [
    "DYNAMICS_MODEL_TYPE", "DYNAMICS_RUNTIME_KEYS", "DreamerV4Decoder", "DreamerV4Dynamics", "DreamerV4Encoder",
    "G1_ACTION", "G1_CONTROL_HZ", "G1_DYNAMICS_ARCH", "G1_INFERENCE", "G1_PREPROCESSING", "G1_TOKENIZER_ARCH",
    "KVCache", "RewardMTPHead", "SymlogTwoHotHead", "TOKENIZER_MODEL_TYPE", "TOKENIZER_RUNTIME_KEYS",
    "build_dynamics", "build_tokenizer", "dynamics_arch", "init_nonpersistent_buffers", "tokenizer_arch",
]
