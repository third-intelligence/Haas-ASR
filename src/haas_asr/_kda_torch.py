"""Pure-PyTorch port of ``fla.layers.kda.KimiDeltaAttention`` (fla 0.5.2), for inference.

fla's kernels are written in Triton, which needs a CUDA GPU. This port runs the same
math with plain PyTorch ops, so the model also runs on the CPU (and on macOS).

Only the configuration that Haas uses is ported: ``use_short_conv=True``, with the
L2 norm of q and k, the gate and the beta sigmoid computed "in kernel",
``state_v_first=True``, no lower bound and no negative eigenvalues. Parameter names
and shapes match fla's, so the state dict of a ``.nemo`` checkpoint loads unchanged.

The cache contract is fla's too: per layer, ``recurrent_state`` is ``[B, HV, V, K]``
and ``conv_state`` is a tuple of three ``[B, D, W]`` tensors holding the last W
inputs (before activation) of each short convolution.

The recurrence runs one step at a time, like fla's ``fused_recurrent`` kernel, which
is what fla itself uses for inference on up to 64 steps. Decoding feeds one label per
step, so this is also the fast path here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ShortConvolution(nn.Conv1d):
    """Depthwise causal convolution + SiLU with a rolling input cache (fla's, ``silu``).

    Subclasses ``nn.Conv1d`` only for its ``weight`` and ``bias``, named and shaped as
    in fla. ``forward`` pads causally with the cache instead of Conv1d's padding.
    """

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias)

    def forward(
        self, x: torch.Tensor, cache: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]; cache: [B, D, W] or None.
        batch_size, _, dim = x.shape
        width = self.kernel_size[0]
        xt = x.transpose(1, 2)  # [B, D, T]
        if cache is None:
            cache = x.new_zeros(batch_size, dim, width)
        cache = cache.to(xt.dtype)
        # The last W - 1 inputs seen so far, followed by the new ones.
        y = F.conv1d(torch.cat([cache[..., 1:], xt], dim=-1), self.weight, self.bias, groups=dim)
        new_cache = torch.cat([cache, xt], dim=-1)[..., -width:]
        return F.silu(y).transpose(1, 2), new_cache


class GatedRMSNorm(nn.Module):
    """fla's ``FusedRMSNormGated(activation="sigmoid")``: ``rmsnorm(x) * w * sigmoid(g)``."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float() * torch.sigmoid(g.float())).to(dtype)


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gated delta rule with a gate per key dimension, one step at a time.

    Same inputs as fla's ``fused_recurrent_kda`` with ``use_qk_l2norm_in_kernel``,
    ``use_gate_in_kernel``, ``use_beta_sigmoid_in_kernel`` and ``state_v_first``:

    - q, k: ``[B, T, H, K]``; v: ``[B, T, HV, V]``
    - g (before the gate): ``[B, T, HV, K]``; beta (before the sigmoid): ``[B, T, HV]``
    - state: ``[B, HV, V, K]``

    Computes in float32 and returns the output in v's dtype and the state in float32.
    """
    dtype = v.dtype
    batch_size, steps, num_heads, head_k_dim = q.shape
    num_v_heads, head_v_dim = v.shape[2], v.shape[3]
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))

    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * head_k_dim**-0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if (group := num_v_heads // num_heads) > 1:
        q = q.repeat_interleave(group, dim=2)
        k = k.repeat_interleave(group, dim=2)
    g = g + dt_bias.float().view(num_v_heads, head_k_dim)
    decay = (-A_log.float().view(num_v_heads, 1).exp() * F.softplus(g)).exp()
    beta = torch.sigmoid(beta)

    shape = (batch_size, num_v_heads, head_v_dim, head_k_dim)
    if initial_state is None:
        state = q.new_zeros(shape)
    else:
        state = initial_state.reshape(shape).float().clone()
    o = q.new_empty(batch_size, steps, num_v_heads, head_v_dim)
    for t in range(steps):
        state = state * decay[:, t, :, None, :]
        k_t = k[:, t, :, :, None]  # [B, HV, K, 1]
        delta = (v[:, t] - (state @ k_t).squeeze(-1)) * beta[:, t, :, None]  # [B, HV, V]
        state = state + delta[..., None] * k[:, t, :, None, :]
        o[:, t] = (state @ q[:, t, :, :, None]).squeeze(-1)
    return o.to(dtype), state


class TorchKimiDeltaAttention(nn.Module):
    """Drop-in for fla's ``KimiDeltaAttention`` in the configuration Haas uses."""

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: str = "chunk",  # accepted for fla's signature; there is one kernel here
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        if not use_short_conv or allow_neg_eigval or safe_gate or lower_bound is not None:
            raise NotImplementedError("only the KDA configuration that Haas uses is ported")
        self.hidden_size = hidden_size
        self.conv_size = conv_size
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias)
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias)
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, conv_bias)

        self.gate_dim = self.num_v_heads * self.head_k_dim
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        # Overwritten by the checkpoint; fla's random initialisation is not needed.
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.gate_dim))
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = GatedRMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ):
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported (Haas does not use it)")
        batch_size, steps, _ = hidden_states.shape
        last = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last is not None and last.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last["conv_state"]

        x = hidden_states
        q, conv_q = self.q_conv1d(self.q_proj(x), conv_q)
        k, conv_k = self.k_conv1d(self.k_proj(x), conv_k)
        v, conv_v = self.v_conv1d(self.v_proj(x), conv_v)
        g = self.f_proj(x).view(batch_size, steps, self.num_v_heads, self.head_k_dim)
        beta = self.b_proj(x)
        q = q.view(batch_size, steps, self.num_heads, self.head_k_dim)
        k = k.view(batch_size, steps, self.num_heads, self.head_k_dim)
        v = v.view(batch_size, steps, self.num_v_heads, self.head_v_dim)

        state = last.get("recurrent_state") if last is not None else None
        o, state = recurrent_kda(q, k, v, g, beta, self.A_log, self.dt_bias, state)

        if past_key_values is not None and use_cache:
            past_key_values.update(
                layer_idx=self.layer_idx,
                recurrent_state=state,
                conv_state=(conv_q, conv_k, conv_v),
                offset=steps,
            )

        gate = self.g_proj(x).view(batch_size, steps, self.num_v_heads, self.head_v_dim)
        o = self.o_norm(o, gate).reshape(batch_size, steps, self.value_dim)
        return self.o_proj(o), None, past_key_values
