import pytest
import torch


from rope.rope import (
    RotaryPositionalEmbedding,
    apply_rope,
    get_inv_freq,
    get_rope_angles,
    make_pos_ids,
)


def test_get_inv_freq_defaults_and_dtype():
    head_dim = 8
    inv_freq = get_inv_freq(head_dim=head_dim)
    assert inv_freq.shape == (head_dim // 2,)
    assert inv_freq.dtype == torch.float32

    expected = 10_000.0 ** (
        -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    )
    torch.testing.assert_close(inv_freq, expected)

    custom = get_inv_freq(
        head_dim=head_dim,
        rot_dim=4,
        theta=1_000.0,
        dtype=torch.float64,
    )
    assert custom.shape == (2,)
    assert custom.dtype == torch.float64
    expected_custom = (
        1_000.0
        ** (-torch.arange(0, 4, 2, dtype=torch.float32) / 4)
    ).to(torch.float64)
    torch.testing.assert_close(custom, expected_custom)


def test_get_rope_angles_matches_manual_trig():
    pos_ids = torch.tensor([[0.0, 5.0], [3.0, 7.0]])
    inv_freq = torch.tensor([1.0, 2.0])

    cos, sin = get_rope_angles(pos_ids, inv_freq)

    angles = torch.einsum("bt,d->btd", pos_ids.to(inv_freq.dtype), inv_freq)
    torch.testing.assert_close(cos, torch.cos(angles))
    torch.testing.assert_close(sin, torch.sin(angles))


def test_apply_rope_agrees_with_complex_rotation():
    torch.manual_seed(0)
    B, H, T, head_dim, rot_dim = 2, 3, 4, 8, 6
    q = torch.randn(B, H, T, head_dim)
    k = torch.randn(B, H, T, head_dim)

    inv_freq = get_inv_freq(head_dim=head_dim, rot_dim=rot_dim)
    pos_ids = torch.arange(T).repeat(B, 1)
    cos, sin = get_rope_angles(pos_ids, inv_freq)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_out, k_out = apply_rope(q, k, cos, sin, rot_dim=rot_dim)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert q_out.dtype == q.dtype
    assert k_out.dtype == k.dtype

    def _expected_rot(x):
        x_rot = x[..., :rot_dim]
        pairs = x_rot.reshape(*x_rot.shape[:-1], rot_dim // 2, 2)
        complex_vals = torch.view_as_complex(pairs)
        angles = torch.einsum(
            "bt,d->btd", pos_ids.to(inv_freq.dtype), inv_freq
        ).unsqueeze(1)
        rotator = torch.complex(torch.cos(angles), torch.sin(angles))
        rotated = torch.view_as_real(complex_vals * rotator)
        return rotated.reshape(*x_rot.shape)

    expected_q = _expected_rot(q)
    expected_k = _expected_rot(k)

    torch.testing.assert_close(q_out[..., :rot_dim], expected_q)
    torch.testing.assert_close(k_out[..., :rot_dim], expected_k)
    torch.testing.assert_close(q_out[..., rot_dim:], q[..., rot_dim:])
    torch.testing.assert_close(k_out[..., rot_dim:], k[..., rot_dim:])


@pytest.mark.parametrize("pos_scale, inv_scale", [(2.0, 1.0), (0.5, 0.75)])
def test_rotary_embedding_apply_respects_scaling(pos_scale, inv_scale):
    torch.manual_seed(0)
    rope = RotaryPositionalEmbedding(head_dim=8, rot_dim=6)
    B, H, T = 1, 2, 5
    q = torch.randn(B, H, T, rope.head_dim)
    k = torch.randn(B, H, T, rope.head_dim)
    pos_ids = torch.arange(T).repeat(B, 1)

    out_q, out_k = rope.apply(
        q, k, pos_ids, pos_scale=pos_scale, inv_freq_scale=inv_scale
    )

    scaled_pos = (pos_ids.to(torch.float32) / pos_scale).to(q.dtype)
    cos, sin = get_rope_angles(
        scaled_pos, inv_freq=rope.inv_freq * inv_scale
    )
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    exp_q, exp_k = apply_rope(q, k, cos, sin, rot_dim=rope.rot_dim)

    torch.testing.assert_close(out_q, exp_q)
    torch.testing.assert_close(out_k, exp_k)


def test_make_pos_ids_handles_batch_offsets():
    batch_sizes = torch.tensor([5, 3, 4])
    past_sizes = torch.tensor([0, 10, 2])

    pos = make_pos_ids(batch_sizes, past_sizes)

    assert pos.shape == (3, batch_sizes.max().item())
    assert pos.dtype == torch.long
    assert pos.device == past_sizes.device

    torch.testing.assert_close(pos[0, : batch_sizes[0]], torch.arange(5))
    torch.testing.assert_close(pos[1, : batch_sizes[1]], torch.arange(10, 13))
    torch.testing.assert_close(pos[2, : batch_sizes[2]], torch.arange(2, 6))
