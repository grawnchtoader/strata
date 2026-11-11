import os
import json
import logging
import shutil
import socket
import struct
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import torch
import torch.nn.functional as F
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from .decoder import (
    _HAS_TRITON,
    _fused_situ_kernel,
    build_dotlut,
    decode_sc_lut_matrix,
    resolve_optimal_dtype,
    sc_lut_batched_experts_mlp,
    sc_lut_expert_mlp,
)
from .tiered_moe_loader import (
    TieredMoECacheManager,
    detect_system_memory_gb,
    load_strata_coder_multi_gpu,
)


def _run_cleanup(actions, primary=None):
    """Try every cleanup; preserve an exception already in flight."""
    first_error = None
    try:
        for action in actions:
            try:
                action()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                if primary is not None:
                    logging.getLogger(__name__).error(
                        "Cleanup failed while handling another exception: %s: %s",
                        type(exc).__name__, str(exc),
                    )
        if first_error is not None and primary is None:
            raise first_error
    finally:
        # An exception's traceback points back to this frame. Avoid retaining
        # tensors until cyclic GC runs after a failed forward or cleanup.
        first_error = primary = None

if _HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _kda_decode_step_kernel(
        q_raw_ptr, k_raw_ptr, v_raw_ptr,
        q_hist_ptr, k_hist_ptr, v_hist_ptr,
        wc_q_ptr, wc_k_ptr, wc_v_ptr,
        z_ptr, A_ptr, dt_ptr,
        beta_in_ptr, gate_in_ptr,
        o_norm_w_ptr,
        S_ptr,
        out_ptr,
        num_warps=4,
    ):
        h = tl.program_id(0)
        d = tl.arange(0, 128)
        off = h * 128 + d

        q0 = tl.load(q_hist_ptr + 0 * 12288 + off).to(tl.float32)
        q1 = tl.load(q_hist_ptr + 1 * 12288 + off).to(tl.float32)
        q2 = tl.load(q_hist_ptr + 2 * 12288 + off).to(tl.float32)
        q3 = tl.load(q_raw_ptr + off).to(tl.float32)
        tl.store(q_hist_ptr + 0 * 12288 + off, q1)
        tl.store(q_hist_ptr + 1 * 12288 + off, q2)
        tl.store(q_hist_ptr + 2 * 12288 + off, q3)

        k0 = tl.load(k_hist_ptr + 0 * 12288 + off).to(tl.float32)
        k1 = tl.load(k_hist_ptr + 1 * 12288 + off).to(tl.float32)
        k2 = tl.load(k_hist_ptr + 2 * 12288 + off).to(tl.float32)
        k3 = tl.load(k_raw_ptr + off).to(tl.float32)
        tl.store(k_hist_ptr + 0 * 12288 + off, k1)
        tl.store(k_hist_ptr + 1 * 12288 + off, k2)
        tl.store(k_hist_ptr + 2 * 12288 + off, k3)

        v0 = tl.load(v_hist_ptr + 0 * 12288 + off).to(tl.float32)
        v1 = tl.load(v_hist_ptr + 1 * 12288 + off).to(tl.float32)
        v2 = tl.load(v_hist_ptr + 2 * 12288 + off).to(tl.float32)
        v3 = tl.load(v_raw_ptr + off).to(tl.float32)
        tl.store(v_hist_ptr + 0 * 12288 + off, v1)
        tl.store(v_hist_ptr + 1 * 12288 + off, v2)
        tl.store(v_hist_ptr + 2 * 12288 + off, v3)

        wq0 = tl.load(wc_q_ptr + off * 4 + 0).to(tl.float32)
        wq1 = tl.load(wc_q_ptr + off * 4 + 1).to(tl.float32)
        wq2 = tl.load(wc_q_ptr + off * 4 + 2).to(tl.float32)
        wq3 = tl.load(wc_q_ptr + off * 4 + 3).to(tl.float32)
        qc = q0 * wq0 + q1 * wq1 + q2 * wq2 + q3 * wq3
        qs = qc * tl.sigmoid(qc)

        wk0 = tl.load(wc_k_ptr + off * 4 + 0).to(tl.float32)
        wk1 = tl.load(wc_k_ptr + off * 4 + 1).to(tl.float32)
        wk2 = tl.load(wc_k_ptr + off * 4 + 2).to(tl.float32)
        wk3 = tl.load(wc_k_ptr + off * 4 + 3).to(tl.float32)
        kc = k0 * wk0 + k1 * wk1 + k2 * wk2 + k3 * wk3
        ks = kc * tl.sigmoid(kc)

        wv0 = tl.load(wc_v_ptr + off * 4 + 0).to(tl.float32)
        wv1 = tl.load(wc_v_ptr + off * 4 + 1).to(tl.float32)
        wv2 = tl.load(wc_v_ptr + off * 4 + 2).to(tl.float32)
        wv3 = tl.load(wc_v_ptr + off * 4 + 3).to(tl.float32)
        vc = v0 * wv0 + v1 * wv1 + v2 * wv2 + v3 * wv3
        vs = vc * tl.sigmoid(vc)

        q_norm = tl.sqrt(tl.maximum(tl.sum(qs * qs, axis=0), 1e-12))
        qs = (qs / q_norm) * 0.08838834764831845
        k_norm = tl.sqrt(tl.maximum(tl.sum(ks * ks, axis=0), 1e-12))
        ks = ks / k_norm

        A_h = tl.exp(tl.load(A_ptr + h).to(tl.float32))
        z_d = tl.load(z_ptr + off).to(tl.float32) + tl.load(dt_ptr + off).to(tl.float32)
        alp = tl.exp(-5.0 * tl.sigmoid(A_h * z_d))
        beta = tl.sigmoid(tl.load(beta_in_ptr + h).to(tl.float32))

        e = tl.arange(0, 128)
        s_off = h * 16384 + d[:, None] * 128 + e[None, :]
        S = tl.load(S_ptr + s_off).to(tl.float32)
        S = S * alp[:, None]
        proj = tl.sum(ks[:, None] * S, axis=0)
        delta = beta * (vs - proj)
        S = S + ks[:, None] * delta[None, :]
        tl.store(S_ptr + s_off, S)

        o = tl.sum(qs[:, None] * S, axis=0)
        o_rms = tl.rsqrt(tl.sum(o * o, axis=0) * (1.0 / 128.0) + 1e-5)
        nw = tl.load(o_norm_w_ptr + e).to(tl.float32)
        g_in = tl.load(gate_in_ptr + h * 128 + e).to(tl.float32)
        out = (o * o_rms * nw) * tl.sigmoid(g_in)
        tl.store(out_ptr + h * 128 + e, out)

FULL_ATTN_LAYERS = {
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
    51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92,
}
TIER_BITS = {1: 10, 2: 9, 3: 9}


def gpu_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()


def gpu_situ(g: torch.Tensor, u: torch.Tensor, b1: float = 4.0, b2: float = 25.0) -> torch.Tensor:
    if _HAS_TRITON and g.is_cuda and u.is_cuda:
        out = torch.empty_like(g, dtype=torch.float32)
        n = g.numel()
        _fused_situ_kernel[((n + 255) // 256,)](g.contiguous(), u.contiguous(), out, n, BLOCK=256)
        return out
    gf, uf = g.float(), u.float()
    return (b1 * torch.tanh(gf / b1) * torch.sigmoid(gf)) * (b2 * torch.tanh(uf / b2))


def gpu_res_mix(
    cur: torch.Tensor,
    bres: List[torch.Tensor],
    res_norm_w: torch.Tensor,
    res_proj_w: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if not bres:
        return cur
    V = torch.stack(bres + [cur], dim=1).float()
    K = (V * torch.rsqrt(V.pow(2).mean(-1, keepdim=True) + eps)) * res_norm_w.float()
    logits = torch.einsum("bsd,d->bs", K, res_proj_w.float().squeeze())
    probs = F.softmax(logits, dim=-1)
    return torch.einsum("bs,bsd->bd", probs, V)


def _fetch_weight(
    sf,
    cache_mgr: Optional[TieredMoECacheManager],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    if cache_mgr is not None:
        return cache_mgr.get_base_tensor(key, device)
    if isinstance(sf, dict):
        t = sf[key]
        return t if t.device == device else t.to(device, non_blocking=True)
    return sf.get_tensor(key).to(device, non_blocking=True)


def _mm(x: torch.Tensor, w) -> torch.Tensor:
    if isinstance(w, tuple):
        w_fp8, sc = w
        return (x.bfloat16() @ w_fp8.bfloat16().T).float() * sc
    if w.dtype in (torch.bfloat16, torch.float16) and x.is_cuda:
        return (x.to(w.dtype) @ w.T).float()
    return x.float() @ w.float().T


def gpu_kda_forward(
    sf,
    li: int,
    nrm: torch.Tensor,
    device: torch.device,
    cache_mgr: Optional[TieredMoECacheManager] = None,
    layer_state: Optional[dict] = None,
) -> torch.Tensor:
    p = f"language_model.model.layers.{li}.self_attn"
    H, D, K, lb = 96, 128, 4, -5.0
    L = nrm.shape[0]
    q_raw = _mm(nrm, _fetch_weight(sf, cache_mgr, f"{p}.q_proj.weight", device))
    k_raw = _mm(nrm, _fetch_weight(sf, cache_mgr, f"{p}.k_proj.weight", device))
    v_raw = _mm(nrm, _fetch_weight(sf, cache_mgr, f"{p}.v_proj.weight", device))
    wc_q = _fetch_weight(sf, cache_mgr, f"{p}.q_conv1d.weight", device).float().view(H * D, 1, K)
    wc_k = _fetch_weight(sf, cache_mgr, f"{p}.k_conv1d.weight", device).float().view(H * D, 1, K)
    wc_v = _fetch_weight(sf, cache_mgr, f"{p}.v_conv1d.weight", device).float().view(H * D, 1, K)

    f_a = _fetch_weight(sf, cache_mgr, f"{p}.f_a_proj.weight", device)
    f_b = _fetch_weight(sf, cache_mgr, f"{p}.f_b_proj.weight", device)
    A = torch.exp(_fetch_weight(sf, cache_mgr, f"{p}.A_log", device).float()[:H]).view(1, H, 1)
    dt = _fetch_weight(sf, cache_mgr, f"{p}.dt_bias", device).float().view(1, H, D)
    b_w = _fetch_weight(sf, cache_mgr, f"{p}.b_proj.weight", device)
    g_w = _fetch_weight(sf, cache_mgr, f"{p}.g_proj.weight", device)
    o_norm_w = _fetch_weight(sf, cache_mgr, f"{p}.o_norm.weight", device)
    o_w = _fetch_weight(sf, cache_mgr, f"{p}.o_proj.weight", device)

    if L == 1 and _HAS_TRITON and nrm.is_cuda and layer_state is not None and "S" in layer_state:
        z_raw = _mm(_mm(nrm, f_a), f_b)
        beta_in = _mm(nrm, b_w)
        gate_in = _mm(nrm, g_w)
        A_log = _fetch_weight(sf, cache_mgr, f"{p}.A_log", device)
        dt_w = _fetch_weight(sf, cache_mgr, f"{p}.dt_bias", device)
        q_hist = layer_state["q_hist"].to(device, dtype=torch.float32).contiguous()
        k_hist = layer_state["k_hist"].to(device, dtype=torch.float32).contiguous()
        v_hist = layer_state["v_hist"].to(device, dtype=torch.float32).contiguous()
        S = layer_state["S"].to(device, dtype=torch.float32).contiguous()
        layer_state["q_hist"] = q_hist
        layer_state["k_hist"] = k_hist
        layer_state["v_hist"] = v_hist
        layer_state["S"] = S
        gated_out = torch.empty((1, H * D), device=device, dtype=torch.float32)
        _kda_decode_step_kernel[(H,)](
            q_raw, k_raw, v_raw,
            q_hist, k_hist, v_hist,
            wc_q, wc_k, wc_v,
            z_raw, A_log, dt_w,
            beta_in, gate_in,
            o_norm_w,
            S,
            gated_out,
        )
        return _mm(gated_out, o_w)

    z = _mm(_mm(nrm, f_a), f_b).view(L, H, D) + dt
    alp_s = torch.exp(lb * torch.sigmoid(A * z))
    beta_s = torch.sigmoid(_mm(nrm, b_w))

    if layer_state is not None and "q_hist" in layer_state:
        q_cat = torch.cat([layer_state["q_hist"].to(device), q_raw], dim=0)
        k_cat = torch.cat([layer_state["k_hist"].to(device), k_raw], dim=0)
        v_cat = torch.cat([layer_state["v_hist"].to(device), v_raw], dim=0)
        S = layer_state["S"].to(device)
    else:
        pad = torch.zeros((K - 1, H * D), device=device, dtype=q_raw.dtype)
        q_cat = torch.cat([pad, q_raw], dim=0)
        k_cat = torch.cat([pad, k_raw], dim=0)
        v_cat = torch.cat([pad, v_raw], dim=0)
        S = torch.zeros(H, D, D, device=device, dtype=torch.float32)

    if layer_state is not None:
        layer_state["q_hist"] = q_cat[-(K - 1):].detach().contiguous()
        layer_state["k_hist"] = k_cat[-(K - 1):].detach().contiguous()
        layer_state["v_hist"] = v_cat[-(K - 1):].detach().contiguous()

    q_s = F.silu(F.conv1d(q_cat.T.unsqueeze(0), wc_q, groups=H * D)[0].T).view(L, H, D)
    k_s = F.silu(F.conv1d(k_cat.T.unsqueeze(0), wc_k, groups=H * D)[0].T).view(L, H, D)
    v_s = F.silu(F.conv1d(v_cat.T.unsqueeze(0), wc_v, groups=H * D)[0].T).view(L, H, D)
    q_s = F.normalize(q_s, p=2, dim=-1, eps=1e-6) * (D ** -0.5)
    k_s = F.normalize(k_s, p=2, dim=-1, eps=1e-6)

    seq_o = torch.empty(L, H, D, device=device, dtype=torch.float32)
    for t in range(L):
        S = S * alp_s[t].unsqueeze(-1)
        kt = k_s[t]
        proj = torch.einsum("hd,hde->he", kt, S)
        S = S + torch.einsum("hd,he->hde", kt, beta_s[t].unsqueeze(-1) * (v_s[t] - proj))
        seq_o[t] = torch.einsum("hd,hde->he", q_s[t], S)

    if layer_state is not None:
        layer_state["S"] = S.detach().contiguous()

    o_nrm = gpu_rmsnorm(seq_o, o_norm_w, 1e-5).view(L, H * D)
    gate = torch.sigmoid(_mm(nrm, g_w))
    return _mm(o_nrm * gate, o_w)


def gpu_mla_forward(
    sf,
    li: int,
    nrm: torch.Tensor,
    device: torch.device,
    cache_mgr: Optional[TieredMoECacheManager] = None,
    layer_state: Optional[dict] = None,
) -> torch.Tensor:
    p = f"language_model.model.layers.{li}.self_attn"
    H, nope, rope, vh, kvl = 96, 128, 64, 128, 512
    qh = nope + rope
    L = nrm.shape[0]
    qa = gpu_rmsnorm(
        _mm(nrm, _fetch_weight(sf, cache_mgr, f"{p}.q_a_proj.weight", device)),
        _fetch_weight(sf, cache_mgr, f"{p}.q_a_layernorm.weight", device),
        1e-5,
    )
    q = _mm(qa, _fetch_weight(sf, cache_mgr, f"{p}.q_b_proj.weight", device)).view(L, H, qh)
    ckv = _mm(nrm, _fetch_weight(sf, cache_mgr, f"{p}.kv_a_proj_with_mqa.weight", device))
    c = gpu_rmsnorm(ckv[:, :kvl], _fetch_weight(sf, cache_mgr, f"{p}.kv_a_layernorm.weight", device), 1e-5)
    k_rot = ckv[:, kvl:]
    kv = _mm(c, _fetch_weight(sf, cache_mgr, f"{p}.kv_b_proj.weight", device)).view(L, H, nope + vh)
    k_pass, v = kv[..., :nope], kv[..., nope:]

    if layer_state is not None:
        if "k_pass" in layer_state:
            k_pass = torch.cat([layer_state["k_pass"].to(device), k_pass], dim=0)
            k_rot = torch.cat([layer_state["k_rot"].to(device), k_rot], dim=0)
            v = torch.cat([layer_state["v"].to(device), v], dim=0)
        layer_state["k_pass"] = k_pass.detach()
        layer_state["k_rot"] = k_rot.detach()
        layer_state["v"] = v.detach()

    S_len = k_pass.shape[0]
    scale = qh ** -0.5

    sc = (
        torch.einsum("thd,shd->hts", q[..., :nope], k_pass)
        + torch.einsum("thd,sd->hts", q[..., nope:], k_rot)
    ) * scale
    if L > 1:
        mask = torch.triu(torch.ones(L, S_len, device=device, dtype=torch.bool), diagonal=S_len - L + 1)
        sc.masked_fill_(mask.unsqueeze(0), float("-inf"))
    w = F.softmax(sc, dim=-1)
    out = torch.einsum("hts,shd->thd", w, v).reshape(L, H * vh)
    g = torch.sigmoid(_mm(nrm, _fetch_weight(sf, cache_mgr, f"{p}.g_proj.weight", device)))
    return _mm(out * g, _fetch_weight(sf, cache_mgr, f"{p}.o_proj.weight", device))


def gpu_dense(sf, li: int, nrm: torch.Tensor, device: torch.device) -> torch.Tensor:
    p = f"language_model.model.layers.{li}.mlp"
    g = _mm(nrm, _fetch_weight(sf, None, f"{p}.gate_proj.weight", device))
    u = _mm(nrm, _fetch_weight(sf, None, f"{p}.up_proj.weight", device))
    return _mm(gpu_situ(g, u), _fetch_weight(sf, None, f"{p}.down_proj.weight", device))


def _gpu_moe_forward_chunk(
    sf,
    li: int,
    nrm: torch.Tensor,
    device: torch.device,
    cache_mgr: Optional[TieredMoECacheManager] = None,
    next_cache_mgr: Optional[TieredMoECacheManager] = None,
    force_arch_sm: int = 0,
    next2_cache_mgr: Optional[TieredMoECacheManager] = None,
    res_base: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    p = f"language_model.model.layers.{li}.block_sparse_moe"
    T = nrm.shape[0]
    K = 16
    sh_g = _mm(nrm, _fetch_weight(sf, cache_mgr, f"{p}.shared_experts.gate_proj.weight", device))
    sh_u = _mm(nrm, _fetch_weight(sf, cache_mgr, f"{p}.shared_experts.up_proj.weight", device))
    sh_out = _mm(gpu_situ(sh_g, sh_u), _fetch_weight(sf, cache_mgr, f"{p}.shared_experts.down_proj.weight", device))

    if cache_mgr is not None and cache_mgr.lookahead_w:
        base_vec = (res_base + sh_out) if res_base is not None else (nrm + sh_out)
        bf = base_vec.float()
        x0_normed = bf * torch.rsqrt(bf.pow(2).mean(-1, keepdim=True) + 1e-5)
        if next_cache_mgr is not None and 1 in cache_mgr.lookahead_w:
            cache_mgr.predict_and_prefetch_lookahead(x0_normed, next_cache_mgr, lead=1, top_k=K)
        if next2_cache_mgr is not None and 2 in cache_mgr.lookahead_w:
            cache_mgr.predict_and_prefetch_lookahead(x0_normed, next2_cache_mgr, lead=2, top_k=K)

    gw = _fetch_weight(sf, cache_mgr, f"{p}.gate.weight", device).float()
    bias = _fetch_weight(sf, cache_mgr, f"{p}.gate.e_score_correction_bias", device).float()
    lat_dn = _fetch_weight(sf, cache_mgr, f"{p}.routed_expert_down_proj.weight", device)
    lat_up = _fetch_weight(sf, cache_mgr, f"{p}.routed_expert_up_proj.weight", device)
    lat_nw = _fetch_weight(sf, cache_mgr, f"{p}.routed_expert_norm.weight", device)

    scores = torch.sigmoid(nrm @ gw.T)
    topk_idx = torch.topk(scores + bias.unsqueeze(0), k=K, dim=-1).indices
    topk_w = torch.gather(scores, 1, topk_idx)
    topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)

    z = _mm(nrm, lat_dn)
    flat_e = topk_idx.view(-1)

    if cache_mgr is not None and _HAS_TRITON and nrm.is_cuda:
        error = None
        try:
            ptrs = cache_mgr.get_experts_batch_ptrs_gpu(flat_e)
            if T == 1:
                flat_t_i64 = torch.zeros(K, device=device, dtype=torch.int64)
            else:
                flat_t_i64 = torch.arange(T, device=device, dtype=torch.int64).unsqueeze(1).expand(T, K).reshape(-1)
            u = sc_lut_batched_experts_mlp(
                z=z,
                ptrs=ptrs,
                flat_w=topk_w.view(-1).contiguous(),
                flat_t=flat_t_i64,
                luts_T_f32=cache_mgr.luts_T_f32,
                qb_T_f32=cache_mgr.qb_T_f32,
                qb_T_stack=cache_mgr.qb_T_stack,
            )
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                _run_cleanup([cache_mgr.release_experts_batch], error)
            finally:
                error = None
        y = _mm(gpu_rmsnorm(u, lat_nw, 1e-5), lat_up)
        return y + sh_out

    u = torch.zeros_like(z)
    flat_t = torch.arange(T, device=device).unsqueeze(1).expand(T, K).reshape(-1)
    flat_w = topk_w.view(-1, 1)
    order = torch.argsort(flat_e)
    sorted_e = flat_e[order]
    sorted_t = flat_t[order]
    sorted_w = flat_w[order]
    uniq_e, counts = torch.unique_consecutive(sorted_e, return_counts=True)

    uniq_e_list = uniq_e.tolist()
    counts_list = counts.tolist()

    if cache_mgr is not None:
        cache_mgr.warm_experts_parallel(uniq_e_list)
        dotluts = {
            t: build_dotlut(z, cache_mgr.luts[t], cache_mgr.q_bases[t])
            for t in cache_mgr.luts
        }
        if uniq_e_list:
            cache_mgr.stage_expert_to_gpu_async(uniq_e_list[0])
    else:
        expert_tiers = sf.get_tensor(f"{p}.expert_tiers").tolist()
        luts = {t: sf.get_tensor(f"{p}.sc_lut_tier{t}").to(device).float() for t in (1, 2, 3)}
        qbases = {t: sf.get_tensor(f"{p}.sc_qbasis_tier{t}").to(device).float() for t in (1, 2, 3)}
        dotluts = {t: build_dotlut(z, luts[t], qbases[t]) for t in (1, 2, 3)}

    offset = 0
    for idx_e, (e_val, cnt) in enumerate(zip(uniq_e_list, counts_list)):
        tok_ids = sorted_t[offset : offset + cnt]
        w_tok = sorted_w[offset : offset + cnt]
        offset += cnt
        z_e = z[tok_ids]
        if cache_mgr is not None:
            if idx_e + 1 < len(uniq_e_list):
                cache_mgr.stage_expert_to_gpu_async(uniq_e_list[idx_e + 1])
            raw_gpu, tier, lut, q_basis, bits = cache_mgr.get_expert_raw_gpu(e_val)
            dotlut_e = dotluts[tier].index_select(0, tok_ids)
            eo = sc_lut_expert_mlp(z_e, raw_gpu, lut, q_basis, bits, dotlut_z=dotlut_e)
        else:
            ep = f"{p}.experts.{e_val}"
            raw_t = int(expert_tiers[e_val])
            tier = raw_t + 1 if 0 in expert_tiers else raw_t
            bits = TIER_BITS[tier]
            raw_gpu = {}
            for proj in ("w1", "w2", "w3"):
                for suffix in ("idx_packed", "scales", "u_svd", "v_svd"):
                    raw_gpu[f"{proj}.{suffix}"] = sf.get_tensor(f"{ep}.{proj}.{suffix}").to(device, non_blocking=True)
            dotlut_e = dotluts[tier].index_select(0, tok_ids)
            eo = sc_lut_expert_mlp(z_e, raw_gpu, luts[tier], qbases[tier], bits, dotlut_z=dotlut_e)
        u.index_add_(0, tok_ids, eo * w_tok)

    y = _mm(gpu_rmsnorm(u, lat_nw, 1e-5), lat_up)
    return y + sh_out


def gpu_moe_forward(
    sf, li, nrm, device, cache_mgr=None, next_cache_mgr=None,
    force_arch_sm=0, next2_cache_mgr=None, res_base=None,
):
    # DotLUT workspace scales with tokens * 16 routed experts. Keep its
    # peak bounded while preserving token order and independent routing.
    chunk_tokens = 4
    outputs = []
    for start in range(0, nrm.shape[0], chunk_tokens):
        stop = min(start + chunk_tokens, nrm.shape[0])
        outputs.append(_gpu_moe_forward_chunk(
            sf, li, nrm[start:stop], device, cache_mgr, next_cache_mgr,
            force_arch_sm, next2_cache_mgr,
            res_base[start:stop] if res_base is not None else None,
        ))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


class KimiCoderEngine:
    """
    End-to-End 93-Layer Inference Engine for Grawnch/Kimi-K3.1-Pro-Coder.
    Supports:
      - Full local shard directories with 3-tier multi-GPU + RAM caching (`TieredMoECacheManager`)
      - On-demand shard caching/streaming from Hugging Face Hub (`Grawnch/Kimi-K3.1-Pro-Coder`)
      - Automatic V100 (`sm_70`) FP16 fallback and dynamic VRAM/RAM budgeting
    """

    def __init__(
        self,
        model_id_or_dir: str = "Grawnch/Kimi-K3.1-Pro-Coder",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        max_layers: int = 93,
        total_vram_gb: Optional[float] = None,
        total_ram_gb: Optional[float] = None,
        force_arch_sm: int = 0,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        peer_addr: Optional[str] = None,
    ):
        self.model_id_or_dir = model_id_or_dir
        self._forward_lock = threading.RLock()
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        if cache_dir is None:
            cache_dir = "/dev/shm/kimi_cache" if Path("/dev/shm").exists() else str(Path.home() / ".cache" / "strata_kimi")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_layers = min(max(max_layers, 1), 93)
        self.start_layer = max(0, min(start_layer, self.max_layers - 1))
        self.end_layer = min(max(end_layer if end_layer is not None else self.max_layers, self.start_layer + 1), self.max_layers)
        self.peer_addr = peer_addr
        self._peer_sock: Optional[socket.socket] = None
        self.peer_stats: Dict[str, float] = {}
        self.force_arch_sm = force_arch_sm

        self.local_dir = None
        if Path(model_id_or_dir).is_dir():
            cand = Path(model_id_or_dir)
            if list(cand.glob("model-*-of-000096.safetensors")):
                self.local_dir = cand
            else:
                hits = list(cand.glob("**/model-*-of-000096.safetensors"))
                self.local_dir = hits[0].parent if hits else cand
        tok_src = str(self.local_dir) if (self.local_dir is not None and (self.local_dir / "tiktoken.model").exists()) else "Grawnch/Kimi-K3.1-Pro-Coder"
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

        det_vram, det_ram = detect_system_memory_gb()
        self.total_vram_gb = total_vram_gb if total_vram_gb is not None else det_vram
        self.total_ram_gb = total_ram_gb if total_ram_gb is not None else det_ram
        self.out_dtype = resolve_optimal_dtype(self.device, force_arch_sm=force_arch_sm)

        # Cache Shard 1 (Layer 0 dense) if start_layer == 0, and Shard 94 (embed_tokens / norm / lm_head)
        self.layer0_cache: Dict[str, torch.Tensor] = {}
        if self.start_layer == 0:
            self.shard1_path, _ = self._resolve_shard(1, keep_cached=True)
            self._sf1 = safe_open(self.shard1_path, framework="pt", device="cpu")
            l0_dev = self.device if self.total_vram_gb >= 12.0 else torch.device("cpu")
            self.layer0_cache = {k: self._sf1.get_tensor(k).to(l0_dev) for k in self._sf1.keys()}

        self.shard94_path, _ = self._resolve_shard(94, keep_cached=True)
        self._sf94 = safe_open(self.shard94_path, framework="pt", device="cpu")
        s94_keys = set(self._sf94.keys())
        out_rn_k = "language_model.model.output_attn_res_norm.weight"
        out_rp_k = "language_model.model.output_attn_res_proj.weight"
        if self.peer_addr is None:
            self.out_attn_res_norm = self._sf94.get_tensor(out_rn_k).to(self.device) if out_rn_k in s94_keys else None
            self.out_attn_res_proj = self._sf94.get_tensor(out_rp_k).to(self.device).squeeze() if out_rp_k in s94_keys else None
            self.final_norm_w = self._sf94.get_tensor("language_model.model.norm.weight").to(self.device)
            head_dev = self.device if self.total_vram_gb >= 12.0 else torch.device("cpu")
            self.lm_head_w = self._sf94.get_tensor("language_model.lm_head.weight").to(device=head_dev, dtype=self.out_dtype)
        else:
            self.out_attn_res_norm = None
            self.out_attn_res_proj = None
            self.final_norm_w = None
            self.lm_head_w = None

        # Initialize persistent multi-GPU / RAM TieredMoECacheManagers for assigned MoE layer range
        self.layer_managers: Dict[int, TieredMoECacheManager] = {}
        shard_paths: Dict[int, str] = {}
        for li in range(max(1, self.start_layer), self.end_layer):
            fname = f"model-{li + 1:05d}-of-000096.safetensors"
            if self.local_dir is not None and (self.local_dir / fname).exists():
                shard_paths[li] = str(self.local_dir / fname)
            elif (self.cache_dir / fname).exists() and (self.cache_dir / fname).stat().st_size > 100_000_000:
                shard_paths[li] = str(self.cache_dir / fname)
            elif self.max_layers <= 8:
                sp, _ = self._resolve_shard(li + 1, keep_cached=True)
                shard_paths[li] = sp
        if shard_paths:
            profile_path = Path(__file__).resolve().parent / "kimi_tier_profile.json"
            if self.local_dir is not None and (self.local_dir / "kimi_tier_profile.json").exists():
                profile_path = self.local_dir / "kimi_tier_profile.json"
            layer_vram_weights: Optional[Dict[int, float]] = None
            layer_priority_overrides: Optional[Dict[int, List[int]]] = None
            lookahead_alphas: Dict[int, float] = {}
            if profile_path.exists():
                try:
                    prof = json.loads(profile_path.read_text(encoding="utf-8"))
                    if "layer_vram_weights" in prof:
                        layer_vram_weights = {int(k): float(v) for k, v in prof["layer_vram_weights"].items()}
                    if "vram_priority_override" in prof:
                        layer_priority_overrides = {int(k): [int(x) for x in v] for k, v in prof["vram_priority_override"].items()}
                    if "lookahead_alpha_lead1" in prof:
                        lookahead_alphas = {int(k): float(v) for k, v in prof["lookahead_alpha_lead1"].items()}
                except Exception:
                    pass
            self.layer_managers = load_strata_coder_multi_gpu(
                shard_paths=shard_paths,
                total_vram_gb=self.total_vram_gb,
                total_ram_gb=self.total_ram_gb,
                force_arch_sm=force_arch_sm,
                layer_vram_weights=layer_vram_weights,
                layer_priority_overrides=layer_priority_overrides,
            )
            self._init_folded_lookahead(lookahead_alphas)

    def _init_folded_lookahead(self, lookahead_alphas: Dict[int, float]):
        """Pre-folds consecutive MoE layer router + RMSNorm weights for lead=1 and lead=2 async prefetch."""
        router_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for li, tmgr in self.layer_managers.items():
            try:
                gw_k = f"{tmgr.pfx}gate.weight"
                bias_k = f"{tmgr.pfx}gate.e_score_correction_bias"
                gnorm_k = f"language_model.model.layers.{li}.post_attention_layernorm.weight"
                gw_t = tmgr.base_cache.get(gw_k) if gw_k in tmgr.base_cache else tmgr.get_tensor(gw_k)
                bias_t = tmgr.base_cache.get(bias_k) if bias_k in tmgr.base_cache else tmgr.get_tensor(bias_k)
                gnorm_t = tmgr.base_cache.get(gnorm_k) if gnorm_k in tmgr.base_cache else tmgr.get_tensor(gnorm_k)
                w_fold = gw_t.float().cpu() * gnorm_t.float().cpu().unsqueeze(0)
                router_cache[li] = (w_fold, bias_t.float().cpu())
            except Exception:
                continue
        for li, mgr in self.layer_managers.items():
            for lead in (1, 2):
                t_li = li + lead
                if t_li in router_cache:
                    w_fold, b_fold = router_cache[t_li]
                    alpha = lookahead_alphas.get(li, 1.0)
                    if lead == 2:
                        alpha = alpha * lookahead_alphas.get(li + 1, 1.0)
                    mgr.register_folded_lookahead(lead=lead, w_prime=w_fold * alpha, bias=b_fold)

    def _resolve_shard(self, shard_idx: int, keep_cached: Optional[bool] = None) -> Tuple[str, Optional[str]]:
        fname = f"model-{shard_idx:05d}-of-000096.safetensors"
        if self.local_dir is not None and (self.local_dir / fname).exists():
            return str(self.local_dir / fname), None
        cached = self.cache_dir / fname
        if cached.exists() and cached.stat().st_size > 100_000_000:
            return str(cached), None
        if keep_cached is None:
            try:
                free_bytes = shutil.disk_usage(self.cache_dir).free
                keep_cached = free_bytes > 15 * (1024 ** 3)
            except Exception:
                keep_cached = False
        if keep_cached:
            path = hf_hub_download(repo_id=self.model_id_or_dir, filename=fname, local_dir=str(self.cache_dir))
            return path, None
        tmp_dir = self.cache_dir / f"tmp_s{shard_idx:03d}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(repo_id=self.model_id_or_dir, filename=fname, local_dir=str(tmp_dir))
        return path, str(tmp_dir)

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        pos = 0
        while pos < n:
            r = sock.recv_into(view[pos:], n - pos)
            if r == 0:
                raise ConnectionError("Peer RPC socket closed")
            pos += r
        return bytes(buf)

    def _get_peer_socket(self) -> socket.socket:
        if self._peer_sock is None:
            host, port_s = self.peer_addr.rsplit(":", 1)
            s = socket.create_connection((host, int(port_s)), timeout=300)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            except OSError:
                pass
            self._peer_sock = s
        return self._peer_sock

    def _rpc_forward(self, cur: torch.Tensor, bres: List[torch.Tensor], reset_cache: bool) -> torch.Tensor:
        sock = self._get_peer_socket()
        seq_len, dim = cur.shape
        nb = len(bres)
        packed = torch.stack([cur, *bres], dim=0).to(dtype=torch.float16, device="cpu").contiguous()
        raw_bytes = bytes(packed.untyped_storage())
        hdr = struct.pack("<IIII", 1 if reset_cache else 0, seq_len, dim, nb)
        sock.sendall(hdr + raw_bytes)

        resp_hdr = self._recv_exact(sock, 32)
        vocab_sz, topk, v_used, v_tot, v_exp, r_used, r_tot, r_exp = struct.unpack("<IIffffff", resp_hdr)
        self.peer_stats = {
            "vram_used_gb": v_used, "vram_total_gb": v_tot, "vram_experts": int(v_exp),
            "ram_used_gb": r_used, "ram_total_gb": r_tot, "ram_experts": int(r_exp),
        }
        body = self._recv_exact(sock, topk * 6)
        vals = torch.frombuffer(bytearray(body[: topk * 2]), dtype=torch.float16).float()
        idxs = torch.frombuffer(bytearray(body[topk * 2 :]), dtype=torch.int32).long()
        logits = torch.full((vocab_sz,), -1e9, dtype=torch.float32)
        logits[idxs] = vals
        return logits

    @torch.no_grad()
    def _run_layer_range(
        self,
        cur: torch.Tensor,
        bres: List[torch.Tensor],
        layer_out_cache: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        try:
            return self._run_layer_range_impl(cur, bres, layer_out_cache)
        except BaseException as exc:
            # Future layers can already have speculative copies. A failed
            # forward must not leave them or partial attention state behind.
            _run_cleanup((mgr.release_transient_gpu_buffers for mgr in self.layer_managers.values()), exc)
            if layer_out_cache is not None:
                layer_out_cache.clear()
            raise

    @torch.no_grad()
    def _run_layer_range_impl(
        self,
        cur: torch.Tensor,
        bres: List[torch.Tensor],
        layer_out_cache: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        num_local = max(self.end_layer - max(1, self.start_layer), 1)
        for li in range(self.start_layer, self.end_layer):
            shard_idx = li + 1
            mgr = self.layer_managers.get(li)
            next_mgr = self.layer_managers.get(li + 1)
            next2_mgr = self.layer_managers.get(li + 2)
            if next_mgr is not None:
                next_mgr.prefetch_base_weights_async()
            layer_dev = mgr.device if mgr is not None else self.device
            cur = cur.to(layer_dev)
            bres = [b.to(layer_dev) for b in bres]

            tmp_dir = None
            sf_ctx = None
            if li == 0:
                sf = self.layer0_cache
            elif mgr is not None:
                sf = mgr._sf
            else:
                spath, tmp_dir = self._resolve_shard(shard_idx)
                if tmp_dir is None:
                    mgr = TieredMoECacheManager(
                        shard_path=spath,
                        layer_idx=li,
                        device=str(self.device),
                        vram_hot_budget_gb=(self.total_vram_gb * 0.84) / num_local,
                        ram_warm_budget_gb=(self.total_ram_gb * 0.80) / num_local,
                        promote_cold_to_ram=True,
                        force_arch_sm=self.force_arch_sm,
                        eager_preload=False,
                    )
                    self.layer_managers[li] = mgr
                    sf = mgr._sf
                else:
                    sf_ctx = safe_open(spath, framework="pt", device="cpu")
                    sf = sf_ctx.__enter__()
            with torch.cuda.device(layer_dev) if layer_dev.type == "cuda" else nullcontext():
                error = None
                try:
                    lp = f"language_model.model.layers.{li}"
                    if bres:
                        rn_a = _fetch_weight(sf, mgr, f"{lp}.self_attention_res_norm.weight", layer_dev)
                        rp_a = _fetch_weight(sf, mgr, f"{lp}.self_attention_res_proj.weight", layer_dev).squeeze()
                        h_in = gpu_res_mix(cur, bres, rn_a, rp_a)
                    else:
                        h_in = cur
                    if li % 12 == 0:
                        bres.append(cur)
                        have_prefix = False
                    else:
                        have_prefix = True
                    nrm_a = gpu_rmsnorm(h_in, _fetch_weight(sf, mgr, f"{lp}.input_layernorm.weight", layer_dev))
                    l_state = layer_out_cache.setdefault(li, {}) if layer_out_cache is not None else None
                    if li in FULL_ATTN_LAYERS:
                        ao = gpu_mla_forward(sf, li, nrm_a, layer_dev, mgr, layer_state=l_state)
                    else:
                        ao = gpu_kda_forward(sf, li, nrm_a, layer_dev, mgr, layer_state=l_state)
                    cur = (cur + ao) if have_prefix else ao

                    rn_m = _fetch_weight(sf, mgr, f"{lp}.mlp_res_norm.weight", layer_dev)
                    rp_m = _fetch_weight(sf, mgr, f"{lp}.mlp_res_proj.weight", layer_dev).squeeze()
                    h_mlp = gpu_res_mix(cur, bres, rn_m, rp_m)
                    nrm_m = gpu_rmsnorm(h_mlp, _fetch_weight(sf, mgr, f"{lp}.post_attention_layernorm.weight", layer_dev))
                    if li == 0:
                        mo = gpu_dense(sf, li, nrm_m, layer_dev)
                        cur = cur + mo
                    elif li == self.max_layers - 1 and cur.shape[0] > 1:
                        mo_last = gpu_moe_forward(sf, li, nrm_m[-1:], layer_dev, mgr, next_mgr, self.force_arch_sm, next2_mgr, cur[-1:])
                        cur = torch.cat([cur[:-1], cur[-1:] + mo_last], dim=0)
                    else:
                        mo = gpu_moe_forward(sf, li, nrm_m, layer_dev, mgr, next_mgr, self.force_arch_sm, next2_mgr, cur)
                        cur = cur + mo
                    # Do not retain one layer's intermediates across the next
                    # layer's allocation peak (especially across devices).
                    rn_a = rp_a = h_in = nrm_a = ao = None
                    rn_m = rp_m = h_mlp = nrm_m = mo = mo_last = None
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    actions = []
                    if mgr is not None:
                        actions.append(mgr.release_transient_gpu_buffers)
                    if sf_ctx is not None:
                        actions.append(lambda: sf_ctx.__exit__(None, None, None))
                    if tmp_dir is not None:
                        actions.append(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
                    try:
                        _run_cleanup(actions, error)
                    finally:
                        error = None
        return cur, bres

    @torch.no_grad()
    def _compute_final_logits(self, cur: torch.Tensor, bres: List[torch.Tensor]) -> torch.Tensor:
        cur = cur.to(self.device)
        bres = [b.to(self.device) for b in bres]
        if bres and self.out_attn_res_norm is not None and self.out_attn_res_proj is not None:
            h_final = gpu_res_mix(cur[-1:], [b[-1:] for b in bres], self.out_attn_res_norm, self.out_attn_res_proj)
        else:
            h_final = cur[-1:]
        h_last = gpu_rmsnorm(h_final, self.final_norm_w)
        logits = (h_last.to(device=self.lm_head_w.device, dtype=self.lm_head_w.dtype) @ self.lm_head_w.T).float().squeeze(0)
        return logits

    @torch.no_grad()
    def forward_last_logits(
        self,
        input_ids: List[int],
        layer_out_cache: Optional[Dict] = None,
    ) -> torch.Tensor:
        # The managers' pointer tables and transient staging buffers are shared
        # by HTTP requests; their lifetimes must not overlap between forwards.
        with self._forward_lock:
            error = None
            try:
                return self._forward_last_logits_impl(input_ids, layer_out_cache)
            except BaseException as exc:
                error = exc
                if layer_out_cache is not None:
                    layer_out_cache.clear()
                raise
            finally:
                try:
                    _run_cleanup((mgr.release_transient_gpu_buffers for mgr in self.layer_managers.values()), error)
                except BaseException:
                    if layer_out_cache is not None:
                        layer_out_cache.clear()
                    raise
                finally:
                    error = None

    @torch.no_grad()
    def _forward_last_logits_impl(
        self,
        input_ids: List[int],
        layer_out_cache: Optional[Dict] = None,
    ) -> torch.Tensor:
        cached_len = layer_out_cache.get("_seq_len", 0) if layer_out_cache is not None else 0
        reset_cache = not (0 < cached_len < len(input_ids))
        if not reset_cache:
            active_ids = input_ids[cached_len:]
        else:
            if layer_out_cache is not None:
                layer_out_cache.clear()
            active_ids = input_ids
        ids_cpu = torch.tensor(active_ids, dtype=torch.long, device="cpu")
        emb_slice = self._sf94.get_slice("language_model.model.embed_tokens.weight")
        h = emb_slice[ids_cpu].to(self.device).float()

        cur, bres = self._run_layer_range(h, [], layer_out_cache=layer_out_cache)
        if layer_out_cache is not None:
            layer_out_cache["_seq_len"] = len(input_ids)

        if self.peer_addr is not None:
            return self._rpc_forward(cur, bres, reset_cache=reset_cache)
        return self._compute_final_logits(cur, bres)

    @torch.no_grad()
    def serve_rpc_worker(self, host: str = "0.0.0.0", port: int = 9100) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(4)
        print(
            f"[Strata-Coder 100G DAC Worker] Listening on {host}:{port} "
            f"(Layers {self.start_layer}..{self.end_layer - 1}, VRAM={self.total_vram_gb:.1f}GB, RAM={self.total_ram_gb:.1f}GB)",
            flush=True,
        )
        worker_cache: Dict[int, Dict] = {}
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            except OSError:
                pass
            print(f"[Strata-Coder 100G DAC Worker] Connected to master {addr[0]}:{addr[1]}", flush=True)
            try:
                while True:
                    hdr = self._recv_exact(conn, 16)
                    reset_flag, seq_len, dim, nb = struct.unpack("<IIII", hdr)
                    if reset_flag:
                        worker_cache.clear()
                    nbytes = (1 + nb) * seq_len * dim * 2
                    payload = self._recv_exact(conn, nbytes)
                    stacked = torch.frombuffer(bytearray(payload), dtype=torch.float16).view(1 + nb, seq_len, dim).to(self.device).float()
                    cur_in = stacked[0]
                    bres_in = [stacked[i + 1] for i in range(nb)]
                    cur_out, bres_out = self._run_layer_range(cur_in, bres_in, layer_out_cache=worker_cache)
                    if self.peer_addr is not None:
                        logits = self._rpc_forward(cur_out, bres_out, reset_cache=bool(reset_flag))
                    else:
                        logits = self._compute_final_logits(cur_out, bres_out)
                    topk = min(256, logits.numel())
                    vals, idxs = torch.topk(logits, k=topk)
                    vals_b = bytes(vals.to(dtype=torch.float16, device="cpu").contiguous().untyped_storage())
                    idxs_b = bytes(idxs.to(dtype=torch.int32, device="cpu").contiguous().untyped_storage())
                    v_exp = sum(len(m.vram_cache) for m in self.layer_managers.values())
                    r_exp = sum(len(m.ram_cache) for m in self.layer_managers.values())
                    v_used = sum(m.vram_used_bytes for m in self.layer_managers.values()) / (1024 ** 3)
                    r_used = sum(m.ram_used_bytes for m in self.layer_managers.values()) / (1024 ** 3)
                    resp_hdr = struct.pack(
                        "<IIffffff",
                        int(logits.numel()), topk,
                        float(v_used), float(self.total_vram_gb), float(v_exp),
                        float(r_used), float(self.total_ram_gb), float(r_exp),
                    )
                    conn.sendall(resp_hdr + vals_b + idxs_b)
            except (ConnectionError, OSError) as exc:
                print(f"[Strata-Coder 100G DAC Worker] Connection closed ({exc}); awaiting reconnect...", flush=True)
            finally:
                conn.close()

    @torch.no_grad()
    def generate_stream(
        self,
        prompt_or_messages,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        reasoning_effort: str = "none",
    ) -> Generator[str, None, None]:
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = prompt_or_messages
        use_thinking = str(reasoning_effort).lower() not in ("none", "off", "false", "")
        effort_map = {"low": "low", "medium": "low", "high": "high", "max": "max"}
        t_effort = effort_map.get(str(reasoning_effort).lower(), "low")
        try:
            if use_thinking:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    thinking=True, thinking_effort=t_effort,
                )
            else:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    thinking=False,
                )
        except Exception:
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant: "

        in_think = False
        if prompt_text.endswith("<|open|>think<|sep|>"):
            if not use_thinking:
                prompt_text += "<|close|>think<|sep|><|open|>response<|sep|>"
            else:
                in_think = True
                yield "<think>"

        ids = self.tokenizer.encode(prompt_text)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        stop_ids = {163585, 163586}
        if eos_id is not None:
            stop_ids.add(eos_id)

        layer_out_cache: Dict[int, torch.Tensor] = {}
        for _ in range(max_new_tokens):
            logits = self.forward_last_logits(ids, layer_out_cache=layer_out_cache)
            if temperature <= 1e-5:
                next_id = int(torch.argmax(logits).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)
            if next_id in stop_ids:
                break
            if next_id == 163588:  # <|close|>
                if in_think:
                    ids.extend(self.tokenizer.encode("think<|sep|><|open|>response<|sep|>"))
                    in_think = False
                    yield "</think>"
                    continue
                else:
                    break
            if next_id in (163587, 163589):  # <|open|>, <|sep|>
                continue
            piece = self.tokenizer.decode([next_id], skip_special_tokens=True)
            if piece:
                yield piece
        if in_think:
            yield "</think>"
