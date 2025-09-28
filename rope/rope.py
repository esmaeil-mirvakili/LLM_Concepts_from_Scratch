from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RotaryPositionalEmbedding:
    def __init__(self, head_dim, rot_dim: Optional[int] = None, theta: float = 10_000.0, device=None, dtype=None):
        self.head_dim = head_dim
        self.rot_dim = rot_dim
        self.theta = theta
        self.device = device
        self.rope_device = device
        self.rope_dtype = dtype
        self.inv_freq = get_inv_freq(
            head_dim=head_dim,
            rot_dim=rot_dim,
            theta=theta,
            device=device,
            dtype=dtype
        )
    def apply(self, q, k, pos_ids, pos_scale: float = 1.0, inv_freq_scale=1.0):
        if pos_scale != 1.0:
            pos_ids = (pos_ids.to(torch.float32) / pos_scale).to(q.dtype)
        cos, sin = get_rope_angles(pos_ids, inv_freq=self.inv_freq * inv_freq_scale)
        cos = cos.unsqueeze(1).to(q.dtype)  # (B,1,T,rotary_dim)
        sin = sin.unsqueeze(1).to(q.dtype)
        return apply_rope(q, k, cos, sin, rot_dim=self.rot_dim)


def get_inv_freq(
    head_dim: int,
    rot_dim: Optional[int] = None,
    theta: float = 10_000.0,
    device=None,
    dtype=None,
):
    if rot_dim is None:
        rot_dim = head_dim
    assert rot_dim % 2 == 0, "Rotational dim should be dividable by two."
    pos_ids = torch.arange(0, rot_dim, 2, device=device, dtype=torch.float32)
    inv_freq = theta ** (-pos_ids / rot_dim)
    return inv_freq.to(dtype=dtype or torch.float32)


def get_rope_angles(pos_ids: Tensor, inv_freq: Tensor):
    # pos_ids: (B, T)
    # inv_freq: (rot_dim / 2)
    # angles: (B, T, rot_dim / 2)
    angles = torch.einsum("ij,k->ijk", pos_ids.to(inv_freq.dtype), inv_freq)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return cos, sin


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, rot_dim: int):
    q1, q2 = q[..., :rot_dim], q[..., rot_dim:]
    k1, k2 = k[..., :rot_dim], k[..., rot_dim:]
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    # even/odd interleave
    q1_even, q1_odd = q1[..., ::2], q1[..., 1::2]
    k1_even, k1_odd = k1[..., ::2], k1[..., 1::2]

    q1_rot = torch.stack(
        (q1_even * cos - q1_odd * sin, q1_even * sin + q1_odd * cos), dim=-1
    ).flatten(-2)
    k1_rot = torch.stack(
        (k1_even * cos - k1_odd * sin, k1_even * sin + k1_odd * cos), dim=-1
    ).flatten(-2)
    q_out = torch.cat((q1_rot, q2), dim=-1)
    k_out = torch.cat((k1_rot, k2), dim=-1)
    return q_out, k_out


def make_pos_ids(batch_sizes: Tensor, past_sizes: Tensor):
    # batch_sizes: (B)
    # past_sizes: (B)
    T = int(batch_sizes.max().item())
    ids = torch.arange(T, device=past_sizes.device)
    pos = ids.unsqueeze(0) + past_sizes.unsqueeze(1)
    return pos[:, :T]
