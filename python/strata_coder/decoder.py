from typing import Dict, Optional
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


@torch.no_grad()
def unpack_sc_lut_indices(packed: torch.Tensor, n_tiles: int, bits: int) -> torch.Tensor:
    """Unpacks uint8 bytes back into lookup table indices [R, n_tiles]."""
    R = packed.shape[0]
    if bits == 10:
        p = packed.view(R, n_tiles // 4, 5).to(torch.int32)
        i0 = p[:, :, 0] | ((p[:, :, 1] & 0x03) << 8)
        i1 = (p[:, :, 1] >> 2) | ((p[:, :, 2] & 0x0F) << 6)
        i2 = (p[:, :, 2] >> 4) | ((p[:, :, 3] & 0x3F) << 4)
        i3 = (p[:, :, 3] >> 6) | (p[:, :, 4] << 2)
        return torch.stack([i0, i1, i2, i3], dim=-1).reshape(R, n_tiles)
    if bits == 9:
        p = packed.view(R, n_tiles // 8, 9).to(torch.int64)
        lo = (
            p[:, :, 0]
            | (p[:, :, 1] << 8)
            | (p[:, :, 2] << 16)
            | (p[:, :, 3] << 24)
            | (p[:, :, 4] << 32)
            | (p[:, :, 5] << 40)
            | (p[:, :, 6] << 48)
            | (p[:, :, 7] << 56)
        )
        i0 = (lo & 0x1FF).to(torch.int32)
        i1 = ((lo >> 9) & 0x1FF).to(torch.int32)
        i2 = ((lo >> 18) & 0x1FF).to(torch.int32)
        i3 = ((lo >> 27) & 0x1FF).to(torch.int32)
        i4 = ((lo >> 36) & 0x1FF).to(torch.int32)
        i5 = ((lo >> 45) & 0x1FF).to(torch.int32)
        i6 = ((lo >> 54) & 0x1FF).to(torch.int32)
        i7 = (((lo >> 63) & 0x01) | ((p[:, :, 8] & 0xFF) << 1)).to(torch.int32)
        return torch.stack([i0, i1, i2, i3, i4, i5, i6, i7], dim=-1).reshape(R, n_tiles)
    if bits == 6:
        p = packed.view(R, n_tiles // 4, 3).to(torch.int32)
        w24 = p[:, :, 0] | (p[:, :, 1] << 8) | (p[:, :, 2] << 16)
        i0 = w24 & 0x3F
        i1 = (w24 >> 6) & 0x3F
        i2 = (w24 >> 12) & 0x3F
        i3 = (w24 >> 18) & 0x3F
        return torch.stack([i0, i1, i2, i3], dim=-1).reshape(R, n_tiles)
    raise ValueError(f"Unsupported bit width: {bits}")


def resolve_optimal_dtype(
    device: torch.device,
    requested_dtype: torch.dtype = torch.bfloat16,
    force_arch_sm: int = 0,
) -> torch.dtype:
    """
    Auto-selects FP16 on Volta / Turing GPUs (sm_70 / sm_75, e.g. Tesla V100 SXM2)
    where hardware BF16 Tensor Cores are unavailable, while keeping BF16 on sm_80+.
    """
    if requested_dtype != torch.bfloat16:
        return requested_dtype
    if force_arch_sm > 0:
        return torch.float16 if force_arch_sm < 80 else torch.bfloat16
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            return torch.float16
    return torch.bfloat16


@torch.no_grad()
def decode_sc_lut_matrix(
    idx_or_packed: torch.Tensor,
    scales: torch.Tensor,
    u_svd: torch.Tensor,
    v_svd: torch.Tensor,
    lut: torch.Tensor,
    q_basis: torch.Tensor,
    bits: int = 0,
    out_dtype: torch.dtype = torch.bfloat16,
    force_arch_sm: int = 0,
) -> torch.Tensor:
    """Decoder for Strata-Coder Mixed-Bit MoE matrices (with automatic V100 sm_70 FP16 support)."""
    target_dtype = resolve_optimal_dtype(idx_or_packed.device, out_dtype, force_arch_sm=force_arch_sm)
    R, C = u_svd.shape[0], v_svd.shape[1]
    tile = lut.shape[1]
    n_tiles = C // tile
    idx = unpack_sc_lut_indices(idx_or_packed, n_tiles, bits) if bits > 0 else idx_or_packed
    w_rot = lut.float()[idx.long()].reshape(R, scales.shape[1], -1) * scales.float().unsqueeze(-1)
    w_main = (w_rot.reshape(R, C // 64, 64) @ q_basis.float()).reshape(R, C)
    w_out = w_main + (u_svd.float() @ v_svd.float())
    if target_dtype == torch.float16:
        w_out = w_out.clamp(-65504.0, 65504.0)
    return w_out.to(target_dtype)


if _HAS_TRITON:
    @triton.jit
    def _sc_lut_10bit_dotlut_splitk_kernel(
        packed_ptr,
        scales_ptr,
        dotlut_ptr,
        out_ptr,
        R: tl.constexpr,
        N_blocks: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_t = tl.program_id(2)
        r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r_offs < R

        packed_row_stride = N_blocks * 5
        scales_row_stride = N_blocks
        n_tiles = N_blocks * 4
        dotlut_t_base = dotlut_ptr + pid_t * n_tiles * 1024
        b_start = pid_s * BLOCKS_PER_SPLIT
        acc = tl.zeros([BLOCK_R], dtype=tl.float32)

        for sb in tl.static_range(0, BLOCKS_PER_SPLIT):
            b = b_start + sb
            p_base = packed_ptr + r_offs * packed_row_stride + b * 5
            b0 = tl.load(p_base + 0, mask=r_mask, other=0).to(tl.int32)
            b1 = tl.load(p_base + 1, mask=r_mask, other=0).to(tl.int32)
            b2 = tl.load(p_base + 2, mask=r_mask, other=0).to(tl.int32)
            b3 = tl.load(p_base + 3, mask=r_mask, other=0).to(tl.int32)
            b4 = tl.load(p_base + 4, mask=r_mask, other=0).to(tl.int32)

            i0 = b0 | ((b1 & 0x03) << 8)
            i1 = (b1 >> 2) | ((b2 & 0x0F) << 6)
            i2 = (b2 >> 4) | ((b3 & 0x3F) << 4)
            i3 = (b3 >> 6) | (b4 << 2)

            d_base = dotlut_t_base + (b * 4) * 1024
            v0 = tl.load(d_base + 0 * 1024 + i0, mask=r_mask, other=0.0)
            v1 = tl.load(d_base + 1 * 1024 + i1, mask=r_mask, other=0.0)
            v2 = tl.load(d_base + 2 * 1024 + i2, mask=r_mask, other=0.0)
            v3 = tl.load(d_base + 3 * 1024 + i3, mask=r_mask, other=0.0)

            sc = tl.load(scales_ptr + r_offs * scales_row_stride + b, mask=r_mask, other=0.0).to(tl.float32)
            acc += (v0 + v1 + v2 + v3) * sc

        n_splits = N_blocks // BLOCKS_PER_SPLIT
        out_idx = (pid_t * n_splits + pid_s) * R + r_offs
        tl.store(out_ptr + out_idx, acc, mask=r_mask)

    @triton.jit
    def _sc_lut_10bit_w13_dotlut_splitk_kernel(
        w1_packed_ptr,
        w1_scales_ptr,
        w3_packed_ptr,
        w3_scales_ptr,
        dotlut_ptr,
        out1_ptr,
        out3_ptr,
        R: tl.constexpr,
        N_blocks: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_t = tl.program_id(2)
        r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r_offs < R

        packed_row_stride = N_blocks * 5
        scales_row_stride = N_blocks
        n_tiles = N_blocks * 4
        dotlut_t_base = dotlut_ptr + pid_t * n_tiles * 1024
        b_start = pid_s * BLOCKS_PER_SPLIT

        acc1 = tl.zeros([BLOCK_R], dtype=tl.float32)
        acc3 = tl.zeros([BLOCK_R], dtype=tl.float32)

        for sb in tl.static_range(0, BLOCKS_PER_SPLIT):
            b = b_start + sb
            d_base = dotlut_t_base + (b * 4) * 1024

            p1 = w1_packed_ptr + r_offs * packed_row_stride + b * 5
            b10 = tl.load(p1 + 0, mask=r_mask, other=0).to(tl.int32)
            b11 = tl.load(p1 + 1, mask=r_mask, other=0).to(tl.int32)
            b12 = tl.load(p1 + 2, mask=r_mask, other=0).to(tl.int32)
            b13 = tl.load(p1 + 3, mask=r_mask, other=0).to(tl.int32)
            b14 = tl.load(p1 + 4, mask=r_mask, other=0).to(tl.int32)
            i10 = b10 | ((b11 & 0x03) << 8)
            i11 = (b11 >> 2) | ((b12 & 0x0F) << 6)
            i12 = (b12 >> 4) | ((b13 & 0x3F) << 4)
            i13 = (b13 >> 6) | (b14 << 2)
            v10 = tl.load(d_base + 0 * 1024 + i10, mask=r_mask, other=0.0)
            v11 = tl.load(d_base + 1 * 1024 + i11, mask=r_mask, other=0.0)
            v12 = tl.load(d_base + 2 * 1024 + i12, mask=r_mask, other=0.0)
            v13 = tl.load(d_base + 3 * 1024 + i13, mask=r_mask, other=0.0)
            sc1 = tl.load(w1_scales_ptr + r_offs * scales_row_stride + b, mask=r_mask, other=0.0).to(tl.float32)
            acc1 += (v10 + v11 + v12 + v13) * sc1

            p3 = w3_packed_ptr + r_offs * packed_row_stride + b * 5
            b30 = tl.load(p3 + 0, mask=r_mask, other=0).to(tl.int32)
            b31 = tl.load(p3 + 1, mask=r_mask, other=0).to(tl.int32)
            b32 = tl.load(p3 + 2, mask=r_mask, other=0).to(tl.int32)
            b33 = tl.load(p3 + 3, mask=r_mask, other=0).to(tl.int32)
            b34 = tl.load(p3 + 4, mask=r_mask, other=0).to(tl.int32)
            i30 = b30 | ((b31 & 0x03) << 8)
            i31 = (b31 >> 2) | ((b32 & 0x0F) << 6)
            i32 = (b32 >> 4) | ((b33 & 0x3F) << 4)
            i33 = (b33 >> 6) | (b34 << 2)
            v30 = tl.load(d_base + 0 * 1024 + i30, mask=r_mask, other=0.0)
            v31 = tl.load(d_base + 1 * 1024 + i31, mask=r_mask, other=0.0)
            v32 = tl.load(d_base + 2 * 1024 + i32, mask=r_mask, other=0.0)
            v33 = tl.load(d_base + 3 * 1024 + i33, mask=r_mask, other=0.0)
            sc3 = tl.load(w3_scales_ptr + r_offs * scales_row_stride + b, mask=r_mask, other=0.0).to(tl.float32)
            acc3 += (v30 + v31 + v32 + v33) * sc3

        n_splits = N_blocks // BLOCKS_PER_SPLIT
        out_idx = (pid_t * n_splits + pid_s) * R + r_offs
        tl.store(out1_ptr + out_idx, acc1, mask=r_mask)
        tl.store(out3_ptr + out_idx, acc3, mask=r_mask)

    @triton.jit
    def _sc_lut_9bit_dotlut_splitk_kernel(
        packed_ptr,
        scales_ptr,
        dotlut_ptr,
        out_ptr,
        R: tl.constexpr,
        N_blocks: tl.constexpr,
        BLOCKS_PER_SCALE: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_t = tl.program_id(2)
        r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r_offs < R

        packed_row_stride = N_blocks * 9
        scales_row_stride = N_blocks // BLOCKS_PER_SCALE
        n_tiles = N_blocks * 8
        dotlut_t_base = dotlut_ptr + pid_t * n_tiles * 512
        b_start = pid_s * BLOCKS_PER_SPLIT
        acc = tl.zeros([BLOCK_R], dtype=tl.float32)

        for sb in tl.static_range(0, BLOCKS_PER_SPLIT):
            b = b_start + sb
            p_base = packed_ptr + r_offs * packed_row_stride + b * 9
            b0 = tl.load(p_base + 0, mask=r_mask, other=0).to(tl.uint64)
            b1 = tl.load(p_base + 1, mask=r_mask, other=0).to(tl.uint64)
            b2 = tl.load(p_base + 2, mask=r_mask, other=0).to(tl.uint64)
            b3 = tl.load(p_base + 3, mask=r_mask, other=0).to(tl.uint64)
            b4 = tl.load(p_base + 4, mask=r_mask, other=0).to(tl.uint64)
            b5 = tl.load(p_base + 5, mask=r_mask, other=0).to(tl.uint64)
            b6 = tl.load(p_base + 6, mask=r_mask, other=0).to(tl.uint64)
            b7 = tl.load(p_base + 7, mask=r_mask, other=0).to(tl.uint64)
            b8 = tl.load(p_base + 8, mask=r_mask, other=0).to(tl.int32)

            lo = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
            i0 = (lo & 0x1FF).to(tl.int32)
            i1 = ((lo >> 9) & 0x1FF).to(tl.int32)
            i2 = ((lo >> 18) & 0x1FF).to(tl.int32)
            i3 = ((lo >> 27) & 0x1FF).to(tl.int32)
            i4 = ((lo >> 36) & 0x1FF).to(tl.int32)
            i5 = ((lo >> 45) & 0x1FF).to(tl.int32)
            i6 = ((lo >> 54) & 0x1FF).to(tl.int32)
            i7 = ((lo >> 63).to(tl.int32) & 0x01) | ((b8 & 0xFF) << 1)

            d_base = dotlut_t_base + (b * 8) * 512
            v0 = tl.load(d_base + 0 * 512 + i0, mask=r_mask, other=0.0)
            v1 = tl.load(d_base + 1 * 512 + i1, mask=r_mask, other=0.0)
            v2 = tl.load(d_base + 2 * 512 + i2, mask=r_mask, other=0.0)
            v3 = tl.load(d_base + 3 * 512 + i3, mask=r_mask, other=0.0)
            v4 = tl.load(d_base + 4 * 512 + i4, mask=r_mask, other=0.0)
            v5 = tl.load(d_base + 5 * 512 + i5, mask=r_mask, other=0.0)
            v6 = tl.load(d_base + 6 * 512 + i6, mask=r_mask, other=0.0)
            v7 = tl.load(d_base + 7 * 512 + i7, mask=r_mask, other=0.0)

            sc = tl.load(scales_ptr + r_offs * scales_row_stride + (b // BLOCKS_PER_SCALE), mask=r_mask, other=0.0).to(tl.float32)
            acc += (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7) * sc

        n_splits = N_blocks // BLOCKS_PER_SPLIT
        out_idx = (pid_t * n_splits + pid_s) * R + r_offs
        tl.store(out_ptr + out_idx, acc, mask=r_mask)

    @triton.jit
    def _sc_lut_9bit_w13_dotlut_splitk_kernel(
        w1_packed_ptr,
        w1_scales_ptr,
        w3_packed_ptr,
        w3_scales_ptr,
        dotlut_ptr,
        out1_ptr,
        out3_ptr,
        R: tl.constexpr,
        N_blocks: tl.constexpr,
        BLOCKS_PER_SCALE: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_t = tl.program_id(2)
        r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r_offs < R

        packed_row_stride = N_blocks * 9
        scales_row_stride = N_blocks // BLOCKS_PER_SCALE
        n_tiles = N_blocks * 8
        dotlut_t_base = dotlut_ptr + pid_t * n_tiles * 512
        b_start = pid_s * BLOCKS_PER_SPLIT

        acc1 = tl.zeros([BLOCK_R], dtype=tl.float32)
        acc3 = tl.zeros([BLOCK_R], dtype=tl.float32)

        for sb in tl.static_range(0, BLOCKS_PER_SPLIT):
            b = b_start + sb
            d_base = dotlut_t_base + (b * 8) * 512
            sc_idx = b // BLOCKS_PER_SCALE

            p1 = w1_packed_ptr + r_offs * packed_row_stride + b * 9
            b10 = tl.load(p1 + 0, mask=r_mask, other=0).to(tl.uint64)
            b11 = tl.load(p1 + 1, mask=r_mask, other=0).to(tl.uint64)
            b12 = tl.load(p1 + 2, mask=r_mask, other=0).to(tl.uint64)
            b13 = tl.load(p1 + 3, mask=r_mask, other=0).to(tl.uint64)
            b14 = tl.load(p1 + 4, mask=r_mask, other=0).to(tl.uint64)
            b15 = tl.load(p1 + 5, mask=r_mask, other=0).to(tl.uint64)
            b16 = tl.load(p1 + 6, mask=r_mask, other=0).to(tl.uint64)
            b17 = tl.load(p1 + 7, mask=r_mask, other=0).to(tl.uint64)
            b18 = tl.load(p1 + 8, mask=r_mask, other=0).to(tl.int32)
            lo1 = b10 | (b11 << 8) | (b12 << 16) | (b13 << 24) | (b14 << 32) | (b15 << 40) | (b16 << 48) | (b17 << 56)
            v10 = tl.load(d_base + 0 * 512 + (lo1 & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v11 = tl.load(d_base + 1 * 512 + ((lo1 >> 9) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v12 = tl.load(d_base + 2 * 512 + ((lo1 >> 18) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v13 = tl.load(d_base + 3 * 512 + ((lo1 >> 27) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v14 = tl.load(d_base + 4 * 512 + ((lo1 >> 36) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v15 = tl.load(d_base + 5 * 512 + ((lo1 >> 45) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v16 = tl.load(d_base + 6 * 512 + ((lo1 >> 54) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v17 = tl.load(d_base + 7 * 512 + (((lo1 >> 63).to(tl.int32) & 0x01) | ((b18 & 0xFF) << 1)), mask=r_mask, other=0.0)
            sc1 = tl.load(w1_scales_ptr + r_offs * scales_row_stride + sc_idx, mask=r_mask, other=0.0).to(tl.float32)
            acc1 += (v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17) * sc1

            p3 = w3_packed_ptr + r_offs * packed_row_stride + b * 9
            b30 = tl.load(p3 + 0, mask=r_mask, other=0).to(tl.uint64)
            b31 = tl.load(p3 + 1, mask=r_mask, other=0).to(tl.uint64)
            b32 = tl.load(p3 + 2, mask=r_mask, other=0).to(tl.uint64)
            b33 = tl.load(p3 + 3, mask=r_mask, other=0).to(tl.uint64)
            b34 = tl.load(p3 + 4, mask=r_mask, other=0).to(tl.uint64)
            b35 = tl.load(p3 + 5, mask=r_mask, other=0).to(tl.uint64)
            b36 = tl.load(p3 + 6, mask=r_mask, other=0).to(tl.uint64)
            b37 = tl.load(p3 + 7, mask=r_mask, other=0).to(tl.uint64)
            b38 = tl.load(p3 + 8, mask=r_mask, other=0).to(tl.int32)
            lo3 = b30 | (b31 << 8) | (b32 << 16) | (b33 << 24) | (b34 << 32) | (b35 << 40) | (b36 << 48) | (b37 << 56)
            v30 = tl.load(d_base + 0 * 512 + (lo3 & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v31 = tl.load(d_base + 1 * 512 + ((lo3 >> 9) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v32 = tl.load(d_base + 2 * 512 + ((lo3 >> 18) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v33 = tl.load(d_base + 3 * 512 + ((lo3 >> 27) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v34 = tl.load(d_base + 4 * 512 + ((lo3 >> 36) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v35 = tl.load(d_base + 5 * 512 + ((lo3 >> 45) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v36 = tl.load(d_base + 6 * 512 + ((lo3 >> 54) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
            v37 = tl.load(d_base + 7 * 512 + (((lo3 >> 63).to(tl.int32) & 0x01) | ((b38 & 0xFF) << 1)), mask=r_mask, other=0.0)
            sc3 = tl.load(w3_scales_ptr + r_offs * scales_row_stride + sc_idx, mask=r_mask, other=0.0).to(tl.float32)
            acc3 += (v30 + v31 + v32 + v33 + v34 + v35 + v36 + v37) * sc3

        n_splits = N_blocks // BLOCKS_PER_SPLIT
        out_idx = (pid_t * n_splits + pid_s) * R + r_offs
        tl.store(out1_ptr + out_idx, acc1, mask=r_mask)
        tl.store(out3_ptr + out_idx, acc3, mask=r_mask)


@torch.no_grad()
def build_dotlut(x: torch.Tensor, lut: torch.Tensor, q_basis: torch.Tensor) -> torch.Tensor:
    """
    Rotates input x [T, C] by Q_basis^T and precomputes the codebook dot-product table
    DotLUT = x_rot_tiles @ lut.T of shape [T, C // tile, N_centroids] in float32.
    """
    T, C = x.shape
    tile = lut.shape[1]
    x_rot = (x.float().view(T, C // 64, 64) @ q_basis.float().T).reshape(T, C // tile, tile)
    return (x_rot @ lut.float().T).contiguous()


@torch.no_grad()
def sc_lut_linear(
    x: torch.Tensor,
    idx_packed: torch.Tensor,
    scales: torch.Tensor,
    u_svd: torch.Tensor,
    v_svd: torch.Tensor,
    lut: torch.Tensor,
    q_basis: torch.Tensor,
    bits: int,
    dotlut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Directly evaluates x @ W^T = (x Q_basis^T) W_rot^T + (x V_svd^T) U_svd^T
    using the Split-K Fused DotLUT Triton kernel without materializing W [R, C].
    """
    xf = x.float()
    T, C = xf.shape
    R = u_svd.shape[0]
    if dotlut is None:
        dotlut = build_dotlut(xf, lut, q_basis)

    if _HAS_TRITON and x.is_cuda and bits in (9, 10):
        if scales.dtype != torch.float16:
            scales = scales.to(torch.float16)
        BLOCK_R = 64
        if bits == 10:
            n_blocks = idx_packed.shape[1] // 5
            bps = 8
            n_splits = n_blocks // bps
            partial = torch.empty((T, n_splits, R), device=x.device, dtype=torch.float32)
            grid = (triton.cdiv(R, BLOCK_R), n_splits, T)
            _sc_lut_10bit_dotlut_splitk_kernel[grid](
                idx_packed,
                scales,
                dotlut,
                partial,
                R=R,
                N_blocks=n_blocks,
                BLOCKS_PER_SPLIT=bps,
                BLOCK_R=BLOCK_R,
                num_warps=4,
            )
            out = partial.sum(dim=1)
        else:
            n_blocks = idx_packed.shape[1] // 9
            blocks_per_scale = n_blocks // scales.shape[1]
            bps = 4 * blocks_per_scale
            n_splits = n_blocks // bps
            partial = torch.empty((T, n_splits, R), device=x.device, dtype=torch.float32)
            grid = (triton.cdiv(R, BLOCK_R), n_splits, T)
            _sc_lut_9bit_dotlut_splitk_kernel[grid](
                idx_packed,
                scales,
                dotlut,
                partial,
                R=R,
                N_blocks=n_blocks,
                BLOCKS_PER_SCALE=blocks_per_scale,
                BLOCKS_PER_SPLIT=bps,
                BLOCK_R=BLOCK_R,
                num_warps=4,
            )
            out = partial.sum(dim=1)
    else:
        tile = lut.shape[1]
        n_tiles = C // tile
        idx = unpack_sc_lut_indices(idx_packed, n_tiles, bits).long()
        n_groups = scales.shape[1]
        tiles_per_group = n_tiles // n_groups
        # Gather dotlut [T, n_tiles, N_centroids] along indices [R, n_tiles]
        gathered = torch.gather(
            dotlut.unsqueeze(1).expand(T, R, n_tiles, -1),
            dim=3,
            index=idx.unsqueeze(0).unsqueeze(-1).expand(T, R, n_tiles, 1),
        ).squeeze(-1)
        out = (gathered.view(T, R, n_groups, tiles_per_group).sum(dim=-1) * scales.float().unsqueeze(0)).sum(dim=-1)

    return out + (xf @ v_svd.float().T) @ u_svd.float().T


@torch.no_grad()
def sc_lut_expert_mlp(
    z_e: torch.Tensor,
    raw: Dict[str, torch.Tensor],
    lut: torch.Tensor,
    q_basis: torch.Tensor,
    bits: int,
    dotlut_z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused MoE Expert MLP: evaluates w2(SiTU(w1(z_e), w3(z_e))) directly from compressed
    expert tensors using the Split-K Fused W1+W3 DotLUT Triton kernel + low-rank SVD.
    """
    zf = z_e.float()
    T = zf.shape[0]
    if dotlut_z is None:
        dotlut_z = build_dotlut(zf, lut, q_basis)

    w1_p, w1_s, w1_u, w1_v = raw["w1.idx_packed"], raw["w1.scales"], raw["w1.u_svd"], raw["w1.v_svd"]
    w3_p, w3_s, w3_u, w3_v = raw["w3.idx_packed"], raw["w3.scales"], raw["w3.u_svd"], raw["w3.v_svd"]
    R = w1_u.shape[0]

    if _HAS_TRITON and z_e.is_cuda and bits in (9, 10):
        if w1_s.dtype != torch.float16:
            w1_s = w1_s.to(torch.float16)
        if w3_s.dtype != torch.float16:
            w3_s = w3_s.to(torch.float16)
        BLOCK_R = 64
        if bits == 10:
            n_blocks = w1_p.shape[1] // 5
            bps = 8
            n_splits = n_blocks // bps
            p1 = torch.empty((T, n_splits, R), device=z_e.device, dtype=torch.float32)
            p3 = torch.empty((T, n_splits, R), device=z_e.device, dtype=torch.float32)
            grid = (triton.cdiv(R, BLOCK_R), n_splits, T)
            _sc_lut_10bit_w13_dotlut_splitk_kernel[grid](
                w1_p, w1_s, w3_p, w3_s, dotlut_z, p1, p3,
                R=R, N_blocks=n_blocks, BLOCKS_PER_SPLIT=bps, BLOCK_R=BLOCK_R, num_warps=4,
            )
            g1 = p1.sum(dim=1) + (zf @ w1_v.float().T) @ w1_u.float().T
            u3 = p3.sum(dim=1) + (zf @ w3_v.float().T) @ w3_u.float().T
        else:
            n_blocks = w1_p.shape[1] // 9
            blocks_per_scale = n_blocks // w1_s.shape[1]
            bps = 4 * blocks_per_scale
            n_splits = n_blocks // bps
            p1 = torch.empty((T, n_splits, R), device=z_e.device, dtype=torch.float32)
            p3 = torch.empty((T, n_splits, R), device=z_e.device, dtype=torch.float32)
            grid = (triton.cdiv(R, BLOCK_R), n_splits, T)
            _sc_lut_9bit_w13_dotlut_splitk_kernel[grid](
                w1_p, w1_s, w3_p, w3_s, dotlut_z, p1, p3,
                R=R, N_blocks=n_blocks, BLOCKS_PER_SCALE=blocks_per_scale,
                BLOCKS_PER_SPLIT=bps, BLOCK_R=BLOCK_R, num_warps=4,
            )
            g1 = p1.sum(dim=1) + (zf @ w1_v.float().T) @ w1_u.float().T
            u3 = p3.sum(dim=1) + (zf @ w3_v.float().T) @ w3_u.float().T
    else:
        g1 = sc_lut_linear(zf, w1_p, w1_s, w1_u, w1_v, lut, q_basis, bits, dotlut=dotlut_z)
        u3 = sc_lut_linear(zf, w3_p, w3_s, w3_u, w3_v, lut, q_basis, bits, dotlut=dotlut_z)

    h = (4.0 * torch.tanh(g1 / 4.0) * torch.sigmoid(g1)) * (25.0 * torch.tanh(u3 / 25.0))
    return sc_lut_linear(
        h,
        raw["w2.idx_packed"],
        raw["w2.scales"],
        raw["w2.u_svd"],
        raw["w2.v_svd"],
        lut,
        q_basis,
        bits,
    )


if _HAS_TRITON:
    @triton.jit
    def _fused_situ_kernel(g_ptr, u_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0)
        tg = 2.0 * tl.sigmoid(g * 0.5) - 1.0
        sg = tl.sigmoid(g)
        tu = 2.0 * tl.sigmoid(u * 0.08) - 1.0
        tl.store(out_ptr + offs, (4.0 * tg * sg) * (25.0 * tu), mask=mask)

    @triton.jit
    def _batched_svd_v_kernel(
        x_ptr,
        ptrs_ptr,
        v1_col: tl.constexpr,
        v3_col: tl.constexpr,
        out_v1_ptr,
        out_v3_ptr,
        C: tl.constexpr,
        SHARED_X: tl.constexpr,
        DUAL: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_e = tl.program_id(1)
        rank = tl.load(ptrs_ptr + pid_e * 14 + 13).to(tl.int32)
        if pid_r >= rank:
            if pid_r < 16:
                tl.store(out_v1_ptr + pid_e * 16 + pid_r, 0.0)
                if DUAL:
                    tl.store(out_v3_ptr + pid_e * 16 + pid_r, 0.0)
            return
        v1_base = tl.load(ptrs_ptr + pid_e * 14 + v1_col).to(tl.pointer_type(tl.float16)) + pid_r * C
        if DUAL:
            v3_base = tl.load(ptrs_ptr + pid_e * 14 + v3_col).to(tl.pointer_type(tl.float16)) + pid_r * C
        x_base = x_ptr if SHARED_X else (x_ptr + pid_e * C)
        acc1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        acc3 = tl.zeros([BLOCK_C], dtype=tl.float32)
        for c0 in range(0, C, BLOCK_C):
            c_offs = c0 + tl.arange(0, BLOCK_C)
            mask = c_offs < C
            xv = tl.load(x_base + c_offs, mask=mask, other=0.0).to(tl.float32)
            w1 = tl.load(v1_base + c_offs, mask=mask, other=0.0).to(tl.float32)
            acc1 += xv * w1
            if DUAL:
                w3 = tl.load(v3_base + c_offs, mask=mask, other=0.0).to(tl.float32)
                acc3 += xv * w3
        tl.store(out_v1_ptr + pid_e * 16 + pid_r, tl.sum(acc1, axis=0))
        if DUAL:
            tl.store(out_v3_ptr + pid_e * 16 + pid_r, tl.sum(acc3, axis=0))

    @triton.jit
    def _batched_svd_u_kernel(
        v1_out_ptr,
        v3_out_ptr,
        ptrs_ptr,
        u1_col: tl.constexpr,
        u3_col: tl.constexpr,
        out1_ptr,
        out3_ptr,
        R: tl.constexpr,
        DUAL: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_rb = tl.program_id(0)
        pid_e = tl.program_id(1)
        r_offs = pid_rb * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r_offs < R
        rank = tl.load(ptrs_ptr + pid_e * 14 + 13).to(tl.int32)
        u1_base = tl.load(ptrs_ptr + pid_e * 14 + u1_col).to(tl.pointer_type(tl.float16))
        if DUAL:
            u3_base = tl.load(ptrs_ptr + pid_e * 14 + u3_col).to(tl.pointer_type(tl.float16))
        acc1 = tl.zeros([BLOCK_R], dtype=tl.float32)
        acc3 = tl.zeros([BLOCK_R], dtype=tl.float32)
        for k in range(0, 16):
            if k < rank:
                vv1 = tl.load(v1_out_ptr + pid_e * 16 + k)
                u1_val = tl.load(u1_base + r_offs * rank + k, mask=r_mask, other=0.0).to(tl.float32)
                acc1 += vv1 * u1_val
                if DUAL:
                    vv3 = tl.load(v3_out_ptr + pid_e * 16 + k)
                    u3_val = tl.load(u3_base + r_offs * rank + k, mask=r_mask, other=0.0).to(tl.float32)
                    acc3 += vv3 * u3_val
        tl.store(out1_ptr + pid_e * R + r_offs, acc1, mask=r_mask)
        if DUAL:
            tl.store(out3_ptr + pid_e * R + r_offs, acc3, mask=r_mask)

    @triton.jit
    def _batched_svd_u_w2_weighted_kernel(
        v2_out_ptr,
        ptrs_ptr,
        w_tok_ptr,
        tok_ids_ptr,
        u_out_ptr,
        R: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_rb = tl.program_id(0)
        pid_e = tl.program_id(1)
        r_offs = pid_rb * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r_offs < R
        rank = tl.load(ptrs_ptr + pid_e * 14 + 13).to(tl.int32)
        u2_base = tl.load(ptrs_ptr + pid_e * 14 + 10).to(tl.pointer_type(tl.float16))
        w_tok = tl.load(w_tok_ptr + pid_e)
        tok_id = tl.load(tok_ids_ptr + pid_e)
        acc = tl.zeros([BLOCK_R], dtype=tl.float32)
        for k in range(0, 16):
            if k < rank:
                vv = tl.load(v2_out_ptr + pid_e * 16 + k)
                u_val = tl.load(u2_base + r_offs * rank + k, mask=r_mask, other=0.0).to(tl.float32)
                acc += vv * u_val
        tl.atomic_add(u_out_ptr + tok_id * R + r_offs, acc * w_tok, mask=r_mask)

    @triton.jit
    def _batched_sc_lut_w13_dotlut_splitk_kernel(
        ptrs_ptr,
        dotlut1_ptr,
        dotlut2_ptr,
        dotlut3_ptr,
        tok_ids_ptr,
        out1_ptr,
        out3_ptr,
        R: tl.constexpr,
        SHARED_X: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_e = tl.program_id(2)
        r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r_offs < R

        tier = tl.load(ptrs_ptr + pid_e * 14 + 12).to(tl.int32)
        w1_packed_ptr = tl.load(ptrs_ptr + pid_e * 14 + 0).to(tl.pointer_type(tl.uint8))
        w1_scales_ptr = tl.load(ptrs_ptr + pid_e * 14 + 1).to(tl.pointer_type(tl.float16))
        w3_packed_ptr = tl.load(ptrs_ptr + pid_e * 14 + 4).to(tl.pointer_type(tl.uint8))
        w3_scales_ptr = tl.load(ptrs_ptr + pid_e * 14 + 5).to(tl.pointer_type(tl.float16))

        acc1 = tl.zeros([BLOCK_R], dtype=tl.float32)
        acc3 = tl.zeros([BLOCK_R], dtype=tl.float32)
        dl_e = 0 if SHARED_X else tl.load(tok_ids_ptr + pid_e)

        if tier == 1:
            packed_stride = 224 * 5
            scales_stride = 224
            dl_base = dotlut1_ptr + dl_e * (896 * 1024)
            b_start = pid_s * 8
            for sb in tl.static_range(0, 8):
                b = b_start + sb
                d_b = dl_base + (b * 4) * 1024
                p1 = w1_packed_ptr + r_offs * packed_stride + b * 5
                b10 = tl.load(p1 + 0, mask=r_mask, other=0).to(tl.int32)
                b11 = tl.load(p1 + 1, mask=r_mask, other=0).to(tl.int32)
                b12 = tl.load(p1 + 2, mask=r_mask, other=0).to(tl.int32)
                b13 = tl.load(p1 + 3, mask=r_mask, other=0).to(tl.int32)
                b14 = tl.load(p1 + 4, mask=r_mask, other=0).to(tl.int32)
                v10 = tl.load(d_b + 0 * 1024 + (b10 | ((b11 & 0x03) << 8)), mask=r_mask, other=0.0)
                v11 = tl.load(d_b + 1 * 1024 + ((b11 >> 2) | ((b12 & 0x0F) << 6)), mask=r_mask, other=0.0)
                v12 = tl.load(d_b + 2 * 1024 + ((b12 >> 4) | ((b13 & 0x3F) << 4)), mask=r_mask, other=0.0)
                v13 = tl.load(d_b + 3 * 1024 + ((b13 >> 6) | (b14 << 2)), mask=r_mask, other=0.0)
                sc1 = tl.load(w1_scales_ptr + r_offs * scales_stride + b, mask=r_mask, other=0.0).to(tl.float32)
                acc1 += (v10 + v11 + v12 + v13) * sc1

                p3 = w3_packed_ptr + r_offs * packed_stride + b * 5
                b30 = tl.load(p3 + 0, mask=r_mask, other=0).to(tl.int32)
                b31 = tl.load(p3 + 1, mask=r_mask, other=0).to(tl.int32)
                b32 = tl.load(p3 + 2, mask=r_mask, other=0).to(tl.int32)
                b33 = tl.load(p3 + 3, mask=r_mask, other=0).to(tl.int32)
                b34 = tl.load(p3 + 4, mask=r_mask, other=0).to(tl.int32)
                v30 = tl.load(d_b + 0 * 1024 + (b30 | ((b31 & 0x03) << 8)), mask=r_mask, other=0.0)
                v31 = tl.load(d_b + 1 * 1024 + ((b31 >> 2) | ((b32 & 0x0F) << 6)), mask=r_mask, other=0.0)
                v32 = tl.load(d_b + 2 * 1024 + ((b32 >> 4) | ((b33 & 0x3F) << 4)), mask=r_mask, other=0.0)
                v33 = tl.load(d_b + 3 * 1024 + ((b33 >> 6) | (b34 << 2)), mask=r_mask, other=0.0)
                sc3 = tl.load(w3_scales_ptr + r_offs * scales_stride + b, mask=r_mask, other=0.0).to(tl.float32)
                acc3 += (v30 + v31 + v32 + v33) * sc3
        else:
            n_blocks = 112 if tier == 2 else 56
            bps = 4 if tier == 2 else 2
            bpsc = 2 if tier == 2 else 1
            packed_stride = n_blocks * 9
            scales_stride = 56
            dl_base = (dotlut2_ptr + dl_e * (896 * 512)) if tier == 2 else (dotlut3_ptr + dl_e * (448 * 512))
            b_start = pid_s * bps
            for sb in range(0, bps):
                b = b_start + sb
                d_b = dl_base + (b * 8) * 512
                sc_idx = b // bpsc
                p1 = w1_packed_ptr + r_offs * packed_stride + b * 9
                b10 = tl.load(p1 + 0, mask=r_mask, other=0).to(tl.uint64)
                b11 = tl.load(p1 + 1, mask=r_mask, other=0).to(tl.uint64)
                b12 = tl.load(p1 + 2, mask=r_mask, other=0).to(tl.uint64)
                b13 = tl.load(p1 + 3, mask=r_mask, other=0).to(tl.uint64)
                b14 = tl.load(p1 + 4, mask=r_mask, other=0).to(tl.uint64)
                b15 = tl.load(p1 + 5, mask=r_mask, other=0).to(tl.uint64)
                b16 = tl.load(p1 + 6, mask=r_mask, other=0).to(tl.uint64)
                b17 = tl.load(p1 + 7, mask=r_mask, other=0).to(tl.uint64)
                b18 = tl.load(p1 + 8, mask=r_mask, other=0).to(tl.int32)
                lo1 = b10 | (b11 << 8) | (b12 << 16) | (b13 << 24) | (b14 << 32) | (b15 << 40) | (b16 << 48) | (b17 << 56)
                v10 = tl.load(d_b + 0 * 512 + (lo1 & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v11 = tl.load(d_b + 1 * 512 + ((lo1 >> 9) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v12 = tl.load(d_b + 2 * 512 + ((lo1 >> 18) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v13 = tl.load(d_b + 3 * 512 + ((lo1 >> 27) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v14 = tl.load(d_b + 4 * 512 + ((lo1 >> 36) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v15 = tl.load(d_b + 5 * 512 + ((lo1 >> 45) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v16 = tl.load(d_b + 6 * 512 + ((lo1 >> 54) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v17 = tl.load(d_b + 7 * 512 + (((lo1 >> 63).to(tl.int32) & 0x01) | ((b18 & 0xFF) << 1)), mask=r_mask, other=0.0)
                sc1 = tl.load(w1_scales_ptr + r_offs * scales_stride + sc_idx, mask=r_mask, other=0.0).to(tl.float32)
                acc1 += (v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17) * sc1

                p3 = w3_packed_ptr + r_offs * packed_stride + b * 9
                b30 = tl.load(p3 + 0, mask=r_mask, other=0).to(tl.uint64)
                b31 = tl.load(p3 + 1, mask=r_mask, other=0).to(tl.uint64)
                b32 = tl.load(p3 + 2, mask=r_mask, other=0).to(tl.uint64)
                b33 = tl.load(p3 + 3, mask=r_mask, other=0).to(tl.uint64)
                b34 = tl.load(p3 + 4, mask=r_mask, other=0).to(tl.uint64)
                b35 = tl.load(p3 + 5, mask=r_mask, other=0).to(tl.uint64)
                b36 = tl.load(p3 + 6, mask=r_mask, other=0).to(tl.uint64)
                b37 = tl.load(p3 + 7, mask=r_mask, other=0).to(tl.uint64)
                b38 = tl.load(p3 + 8, mask=r_mask, other=0).to(tl.int32)
                lo3 = b30 | (b31 << 8) | (b32 << 16) | (b33 << 24) | (b34 << 32) | (b35 << 40) | (b36 << 48) | (b37 << 56)
                v30 = tl.load(d_b + 0 * 512 + (lo3 & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v31 = tl.load(d_b + 1 * 512 + ((lo3 >> 9) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v32 = tl.load(d_b + 2 * 512 + ((lo3 >> 18) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v33 = tl.load(d_b + 3 * 512 + ((lo3 >> 27) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v34 = tl.load(d_b + 4 * 512 + ((lo3 >> 36) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v35 = tl.load(d_b + 5 * 512 + ((lo3 >> 45) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v36 = tl.load(d_b + 6 * 512 + ((lo3 >> 54) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v37 = tl.load(d_b + 7 * 512 + (((lo3 >> 63).to(tl.int32) & 0x01) | ((b38 & 0xFF) << 1)), mask=r_mask, other=0.0)
                sc3 = tl.load(w3_scales_ptr + r_offs * scales_stride + sc_idx, mask=r_mask, other=0.0).to(tl.float32)
                acc3 += (v30 + v31 + v32 + v33 + v34 + v35 + v36 + v37) * sc3

        tl.atomic_add(out1_ptr + pid_e * R + r_offs, acc1, mask=r_mask)
        tl.atomic_add(out3_ptr + pid_e * R + r_offs, acc3, mask=r_mask)

    @triton.jit
    def _batched_sc_lut_w2_dotlut_splitk_kernel(
        ptrs_ptr,
        dotlut1_ptr,
        dotlut2_ptr,
        dotlut3_ptr,
        w_tok_ptr,
        tok_ids_ptr,
        u_out_ptr,
        R: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_e = tl.program_id(2)
        r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r_offs < R

        tier = tl.load(ptrs_ptr + pid_e * 14 + 12).to(tl.int32)
        w2_packed_ptr = tl.load(ptrs_ptr + pid_e * 14 + 8).to(tl.pointer_type(tl.uint8))
        w2_scales_ptr = tl.load(ptrs_ptr + pid_e * 14 + 9).to(tl.pointer_type(tl.float16))
        w_tok = tl.load(w_tok_ptr + pid_e)
        tok_id = tl.load(tok_ids_ptr + pid_e)

        acc = tl.zeros([BLOCK_R], dtype=tl.float32)

        if tier == 1:
            packed_stride = 192 * 5
            scales_stride = 192
            dl_base = dotlut1_ptr + pid_e * (768 * 1024)
            b_start = pid_s * 8
            for sb in tl.static_range(0, 8):
                b = b_start + sb
                d_b = dl_base + (b * 4) * 1024
                p2 = w2_packed_ptr + r_offs * packed_stride + b * 5
                b0 = tl.load(p2 + 0, mask=r_mask, other=0).to(tl.int32)
                b1 = tl.load(p2 + 1, mask=r_mask, other=0).to(tl.int32)
                b2 = tl.load(p2 + 2, mask=r_mask, other=0).to(tl.int32)
                b3 = tl.load(p2 + 3, mask=r_mask, other=0).to(tl.int32)
                b4 = tl.load(p2 + 4, mask=r_mask, other=0).to(tl.int32)
                v0 = tl.load(d_b + 0 * 1024 + (b0 | ((b1 & 0x03) << 8)), mask=r_mask, other=0.0)
                v1 = tl.load(d_b + 1 * 1024 + ((b1 >> 2) | ((b2 & 0x0F) << 6)), mask=r_mask, other=0.0)
                v2 = tl.load(d_b + 2 * 1024 + ((b2 >> 4) | ((b3 & 0x3F) << 4)), mask=r_mask, other=0.0)
                v3 = tl.load(d_b + 3 * 1024 + ((b3 >> 6) | (b4 << 2)), mask=r_mask, other=0.0)
                sc = tl.load(w2_scales_ptr + r_offs * scales_stride + b, mask=r_mask, other=0.0).to(tl.float32)
                acc += (v0 + v1 + v2 + v3) * sc
        else:
            n_blocks = 96 if tier == 2 else 48
            bps = 4 if tier == 2 else 2
            bpsc = 2 if tier == 2 else 1
            packed_stride = n_blocks * 9
            scales_stride = 48
            dl_base = (dotlut2_ptr + pid_e * (768 * 512)) if tier == 2 else (dotlut3_ptr + pid_e * (384 * 512))
            b_start = pid_s * bps
            for sb in range(0, bps):
                b = b_start + sb
                d_b = dl_base + (b * 8) * 512
                sc_idx = b // bpsc
                p2 = w2_packed_ptr + r_offs * packed_stride + b * 9
                b0 = tl.load(p2 + 0, mask=r_mask, other=0).to(tl.uint64)
                b1 = tl.load(p2 + 1, mask=r_mask, other=0).to(tl.uint64)
                b2 = tl.load(p2 + 2, mask=r_mask, other=0).to(tl.uint64)
                b3 = tl.load(p2 + 3, mask=r_mask, other=0).to(tl.uint64)
                b4 = tl.load(p2 + 4, mask=r_mask, other=0).to(tl.uint64)
                b5 = tl.load(p2 + 5, mask=r_mask, other=0).to(tl.uint64)
                b6 = tl.load(p2 + 6, mask=r_mask, other=0).to(tl.uint64)
                b7 = tl.load(p2 + 7, mask=r_mask, other=0).to(tl.uint64)
                b8 = tl.load(p2 + 8, mask=r_mask, other=0).to(tl.int32)
                lo = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
                v0 = tl.load(d_b + 0 * 512 + (lo & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v1 = tl.load(d_b + 1 * 512 + ((lo >> 9) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v2 = tl.load(d_b + 2 * 512 + ((lo >> 18) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v3 = tl.load(d_b + 3 * 512 + ((lo >> 27) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v4 = tl.load(d_b + 4 * 512 + ((lo >> 36) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v5 = tl.load(d_b + 5 * 512 + ((lo >> 45) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v6 = tl.load(d_b + 6 * 512 + ((lo >> 54) & 0x1FF).to(tl.int32), mask=r_mask, other=0.0)
                v7 = tl.load(d_b + 7 * 512 + (((lo >> 63).to(tl.int32) & 0x01) | ((b8 & 0xFF) << 1)), mask=r_mask, other=0.0)
                sc = tl.load(w2_scales_ptr + r_offs * scales_stride + sc_idx, mask=r_mask, other=0.0).to(tl.float32)
                acc += (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7) * sc

        tl.atomic_add(u_out_ptr + tok_id * R + r_offs, acc * w_tok, mask=r_mask)
else:
    _fused_situ_kernel = None


@torch.no_grad()
def sc_lut_batched_experts_mlp(
    z: torch.Tensor,
    ptrs: torch.Tensor,
    flat_w: torch.Tensor,
    flat_t: torch.Tensor,
    luts_T_f32: Dict[int, torch.Tensor],
    qb_T_f32: Dict[int, torch.Tensor],
    qb_T_stack: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluates all T * K routed experts in parallel using pointer-table batched
    Split-K DotLUT + Batched SVD + Fused SiTU Triton kernels (~12 kernel launches/layer).
    """
    T = z.shape[0]
    E = ptrs.shape[0]
    device = z.device
    zf = z.float().contiguous()
    z_blocks = zf.view(T, 56, 64)
    dl1 = ((z_blocks @ qb_T_f32[1]).view(T, 896, 4) @ luts_T_f32[1]).contiguous()
    dl2 = ((z_blocks @ qb_T_f32[2]).view(T, 896, 4) @ luts_T_f32[2]).contiguous()
    dl3 = ((z_blocks @ qb_T_f32[3]).view(T, 448, 8) @ luts_T_f32[3]).contiguous()

    u = torch.zeros((T, 3584), device=device, dtype=torch.float32)
    max_e_chunk = 128
    for e0 in range(0, E, max_e_chunk):
        e1 = min(e0 + max_e_chunk, E)
        ec = e1 - e0
        c_ptrs = ptrs[e0:e1]
        c_w = flat_w[e0:e1]
        c_t = flat_t[e0:e1]
        z_in = zf if T == 1 else zf.index_select(0, c_t).contiguous()
        tiers_e = c_ptrs[:, 12]

        v1_tmp = torch.empty((ec, 16), device=device, dtype=torch.float32)
        v3_tmp = torch.empty((ec, 16), device=device, dtype=torch.float32)
        g1 = torch.empty((ec, 3072), device=device, dtype=torch.float32)
        u3 = torch.empty((ec, 3072), device=device, dtype=torch.float32)
        _batched_svd_v_kernel[(16, ec)](
            z_in, c_ptrs, 3, 7, v1_tmp, v3_tmp, C=3584, SHARED_X=(T == 1), DUAL=True, BLOCK_C=512, num_warps=4
        )
        _batched_svd_u_kernel[(48, ec)](
            v1_tmp, v3_tmp, c_ptrs, 2, 6, g1, u3, R=3072, DUAL=True, BLOCK_R=64, num_warps=4
        )
        _batched_sc_lut_w13_dotlut_splitk_kernel[(48, 28, ec)](
            c_ptrs, dl1, dl2, dl3, c_t, g1, u3, R=3072, SHARED_X=(T == 1), BLOCK_R=64, num_warps=4
        )

        h = torch.empty((ec, 3072), device=device, dtype=torch.float32)
        _fused_situ_kernel[(triton.cdiv(ec * 3072, 256),)](
            g1, u3, h, N=ec * 3072, BLOCK=256, num_warps=4
        )
        h_rot = torch.bmm(h.view(ec, 48, 64), qb_T_stack.index_select(0, tiers_e))
        dl_w2_1 = (h_rot.view(ec, 768, 4) @ luts_T_f32[1]).contiguous()
        dl_w2_2 = (h_rot.view(ec, 768, 4) @ luts_T_f32[2]).contiguous()
        dl_w2_3 = (h_rot.view(ec, 384, 8) @ luts_T_f32[3]).contiguous()

        v2_tmp = torch.empty((ec, 16), device=device, dtype=torch.float32)
        _batched_svd_v_kernel[(16, ec)](
            h, c_ptrs, 11, 11, v2_tmp, v2_tmp, C=3072, SHARED_X=False, DUAL=False, BLOCK_C=512, num_warps=4
        )
        _batched_svd_u_w2_weighted_kernel[(56, ec)](
            v2_tmp, c_ptrs, c_w, c_t, u, R=3584, BLOCK_R=64, num_warps=4
        )
        _batched_sc_lut_w2_dotlut_splitk_kernel[(56, 24, ec)](
            c_ptrs, dl_w2_1, dl_w2_2, dl_w2_3, c_w, c_t, u, R=3584, BLOCK_R=64, num_warps=4
        )
    return u
