from concurrent.futures import Future, ThreadPoolExecutor
import ctypes
import json
import os
import struct
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from safetensors import safe_open
from .decoder import decode_sc_lut_matrix, resolve_optimal_dtype

_SAFETENSORS_DTYPE_MAP = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
}


def _pread_exact(fd: int, length: int, offset: int) -> bytearray:
    buf = bytearray(length)
    mv = memoryview(buf)
    pos = 0
    while pos < length:
        chunk = os.pread(fd, length - pos, offset + pos)
        if not chunk:
            raise EOFError(f"Unexpected EOF in pread at offset {offset + pos}")
        k = len(chunk)
        mv[pos : pos + k] = chunk
        pos += k
    return buf


_SHARED_IO_POOL = ThreadPoolExecutor(max_workers=8)



def detect_system_memory_gb() -> Tuple[float, float]:
    """Detects total (VRAM_GB across all CUDA GPUs, Host_System_RAM_GB) dynamically at runtime."""
    vram_gb = 0.0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            vram_gb += torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
    ram_gb = 64.0
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            ram_gb = stat.ullTotalPhys / (1024 ** 3)
        elif hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
    except Exception:
        pass
    return vram_gb, ram_gb


class TieredMoECacheManager:
    """
    3-Tier Hot/Warm/Cold Mixed-Bit MoE Loader for Strata-Coder
    (Kimi-K3.1-Pro-Coder [896 experts/layer, 2.81 / 2.56 / 1.44 bpw]):
      - Tier 1 (Hot Coding/Math Experts, top 10%): Pinned in GPU VRAM (`device`)
      - Tier 2 (Warm General Experts, next 25%): Pinned in System RAM (`cpu`)
      - Tier 3 (Cold Tail Experts, remaining 65%): Kept on NVMe Disk (`safe_open` mmap)
        with background RouterLookahead prefetching, or promoted to System RAM on
        high-RAM servers (e.g. 512 GB - 768 GB DDR4/DDR5) via `promote_cold_to_ram=True`.
      - Automatically detects Volta / Turing GPUs (e.g. Tesla V100 SXM2 `sm_70`) and
        pre-casts stored BF16 LUT / basis / SVD tensors to native FP16 on load.
    """

    def __init__(
        self,
        shard_path: str,
        layer_idx: int,
        device: str = "cuda",
        vram_hot_budget_gb: float = 0.8,
        ram_warm_budget_gb: float = 2.5,
        promote_cold_to_ram: Optional[bool] = None,
        out_dtype: torch.dtype = torch.bfloat16,
        force_arch_sm: int = 0,
        eager_preload: bool = True,
        vram_priority_override: Optional[List[int]] = None,
        preload_ram: bool = True,
    ):
        self.shard_path = str(shard_path)
        self.layer_idx = layer_idx
        self.device = torch.device(device)
        self.force_arch_sm = force_arch_sm
        self.eager_preload = eager_preload
        self.preload_ram = preload_ram
        self.out_dtype = resolve_optimal_dtype(self.device, out_dtype, force_arch_sm=force_arch_sm)
        self.vram_hot_budget_bytes = int(vram_hot_budget_gb * (1024 ** 3))
        self.vram_base_budget_bytes = self.vram_hot_budget_bytes
        self.ram_warm_budget_bytes = int(ram_warm_budget_gb * (1024 ** 3))
        self.promote_cold_to_ram = (
            (ram_warm_budget_gb >= 5.0) if promote_cold_to_ram is None else promote_cold_to_ram
        )
        self.vram_used_bytes = 0
        self.ram_used_bytes = 0

        self._fd: Optional[int] = None
        self._hdr: Dict[str, dict] = {}
        self._data_base: int = 0
        if hasattr(os, "pread"):
            self._fd = os.open(self.shard_path, os.O_RDONLY)
            hdr_len = struct.unpack("<Q", _pread_exact(self._fd, 8, 0))[0]
            self._hdr = json.loads(_pread_exact(self._fd, hdr_len, 8))
            self._hdr.pop("__metadata__", None)
            self._data_base = 8 + hdr_len
            self._sf = self
            all_keys = set(self._hdr.keys())
        else:
            self._sf = safe_open(self.shard_path, framework="pt", device="cpu")
            all_keys = set(self._sf.keys())

        bsm_pfx = f"language_model.model.layers.{layer_idx}.block_sparse_moe."
        mlp_pfx = f"model.layers.{layer_idx}.mlp."
        self.pfx = bsm_pfx if f"{bsm_pfx}expert_tiers" in all_keys else mlp_pfx

        # Load per-layer embedded lookup tables & basis transforms into VRAM (pre-cast to FP16 on V100 sm_70)
        self.luts: Dict[int, torch.Tensor] = {}
        self.q_bases: Dict[int, torch.Tensor] = {}
        self.luts_T_f32: Dict[int, torch.Tensor] = {}
        self.qb_T_f32: Dict[int, torch.Tensor] = {}
        self.qb_T_stack = torch.zeros((4, 64, 64), dtype=torch.float32, device=self.device)
        for tier_id in (1, 2, 3):
            lut_k = f"{self.pfx}sc_lut_tier{tier_id}"
            qb_k = f"{self.pfx}sc_qbasis_tier{tier_id}"
            if lut_k in all_keys:
                self.luts[tier_id] = self.get_tensor(lut_k).to(device=self.device, dtype=self.out_dtype)
                self.q_bases[tier_id] = self.get_tensor(qb_k).to(device=self.device, dtype=self.out_dtype)
                self.luts_T_f32[tier_id] = self.luts[tier_id].float().T.contiguous()
                self.qb_T_f32[tier_id] = self.q_bases[tier_id].float().T.contiguous()
                self.qb_T_stack[tier_id] = self.qb_T_f32[tier_id]

        raw_tiers = [int(t) for t in self.get_tensor(f"{self.pfx}expert_tiers").tolist()]
        offset = 1 if 0 in raw_tiers else 0
        self.expert_tiers = [t + offset for t in raw_tiers]
        self.vram_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.ram_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.base_cache: Dict[str, torch.Tensor] = {}
        self.prefetch_buffer: Dict[int, Dict[str, torch.Tensor]] = {}
        self.expert_ptr_table = torch.zeros((len(self.expert_tiers), 14), dtype=torch.int64, device=self.device)
        self._vram_ptr_ready = set()
        self._last_batch_keepalive: List[Dict[str, torch.Tensor]] = []
        self._gpu_stage_cache: Dict[int, Tuple[Dict[str, torch.Tensor], Optional[torch.cuda.Event]]] = {}
        self._transfer_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device)
            if (self.device.type == "cuda" and torch.cuda.is_available())
            else None
        )
        self._pool = _SHARED_IO_POOL
        self._base_lock = threading.Lock()
        self._base_loaded = False
        self._base_fut: Optional[Future] = None
        self.priority_order: List[int] = list(vram_priority_override) if vram_priority_override is not None else []
        self.lookahead_w: Dict[int, torch.Tensor] = {}
        self.lookahead_bias: Dict[int, torch.Tensor] = {}

        if eager_preload:
            self.ensure_base_weights_loaded()
            if not self.priority_order:
                self.priority_order = self._compute_expert_priority_order()
            self._populate_tiers()

    def _pin_dict(self, entry: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return entry

    def keys(self):
        return self._hdr.keys() if self._fd is not None else self._sf.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        if self._fd is None:
            return self._sf.get_tensor(key)
        meta = self._hdr[key]
        s, e = meta["data_offsets"]
        dt = _SAFETENSORS_DTYPE_MAP[meta["dtype"]]
        buf = _pread_exact(self._fd, e - s, self._data_base + s)
        return torch.frombuffer(buf, dtype=dt).view(meta["shape"])

    def batch_pread_tensors(
        self,
        keys: List[str],
        max_gap: int = 131072,
        max_chunk: int = 8 * 1024 * 1024,
    ) -> Dict[str, torch.Tensor]:
        if not keys:
            return {}
        if self._fd is None:
            return {k: self._sf.get_tensor(k).clone() for k in keys}
        items = []
        for k in keys:
            m = self._hdr[k]
            s, e = m["data_offsets"]
            items.append((s, e, k, m))
        items.sort(key=lambda x: x[0])
        chunks = []
        cur_s, cur_e, cur_items = items[0][0], items[0][1], [items[0]]
        for it in items[1:]:
            s, e = it[0], it[1]
            if s - cur_e <= max_gap and (e - cur_s) <= max_chunk:
                cur_e = max(cur_e, e)
                cur_items.append(it)
            else:
                chunks.append((cur_s, cur_e, cur_items))
                cur_s, cur_e, cur_items = s, e, [it]
        chunks.append((cur_s, cur_e, cur_items))

        def _read_chunk(ch):
            cs, ce, citems = ch
            buf = _pread_exact(self._fd, ce - cs, self._data_base + cs)
            if len(citems) == 1 and (ce - cs) == (citems[0][1] - citems[0][0]):
                _, _, k, m = citems[0]
                dt = _SAFETENSORS_DTYPE_MAP[m["dtype"]]
                return {k: torch.frombuffer(buf, dtype=dt).view(m["shape"])}
            mv = memoryview(buf)
            out = {}
            for s, e, k, m in citems:
                sub = bytearray(mv[s - cs : e - cs])
                dt = _SAFETENSORS_DTYPE_MAP[m["dtype"]]
                out[k] = torch.frombuffer(sub, dtype=dt).view(m["shape"])
            return out

        res: Dict[str, torch.Tensor] = {}
        for d in self._pool.map(_read_chunk, chunks):
            res.update(d)
        return res

    def ensure_base_weights_loaded(self):
        if self._base_loaded:
            return
        with self._base_lock:
            if self._base_loaded:
                return
            base_keys = [
                k for k in self.keys()
                if ".experts." not in k
                and not k.endswith(("expert_tiers", "sc_lut_tier1", "sc_lut_tier2", "sc_lut_tier3", "sc_qbasis_tier1", "sc_qbasis_tier2", "sc_qbasis_tier3"))
                and k not in self.base_cache
            ]
            loaded = self.batch_pread_tensors(base_keys)
            items_sorted = sorted(loaded.items(), key=lambda kv: kv[1].numel() * kv[1].element_size())
            rem_bytes = sum(t.numel() * t.element_size() for _, t in items_sorted)
            can_fp8 = (
                self.device.type == "cuda"
                and torch.cuda.is_available()
                and hasattr(torch, "float8_e4m3fn")
                and torch.cuda.get_device_capability(self.device)[0] >= 8
            )
            for k, t in items_sorted:
                if t.dtype == torch.bfloat16 and self.out_dtype == torch.float16:
                    t = t.float().clamp(-65504.0, 65504.0).to(torch.float16)
                t_bytes = t.numel() * t.element_size()
                need_fp8 = (self.vram_used_bytes + rem_bytes) > self.vram_base_budget_bytes
                rem_bytes -= t_bytes
                if not need_fp8 and (self.vram_used_bytes + t_bytes <= self.vram_base_budget_bytes):
                    self.base_cache[k] = t.to(self.device, non_blocking=True)
                    self.vram_used_bytes += t_bytes
                elif (
                    can_fp8
                    and t.ndim == 2
                    and t.element_size() == 2
                    and t.numel() >= 8_000_000
                    and (self.vram_used_bytes + (t_bytes // 2) + t.shape[0] * 4 <= self.vram_base_budget_bytes)
                ):
                    tf = t.to(self.device, non_blocking=True).float()
                    sc = tf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
                    w_fp8 = (tf / sc).to(torch.float8_e4m3fn)
                    del tf
                    self.base_cache[k] = (w_fp8, sc.squeeze(1))
                    self.vram_used_bytes += (t_bytes // 2) + t.shape[0] * 4
                elif self.vram_used_bytes + t_bytes <= self.vram_base_budget_bytes:
                    self.base_cache[k] = t.to(self.device, non_blocking=True)
                    self.vram_used_bytes += t_bytes
                else:
                    self.base_cache[k] = t
                    self.ram_used_bytes += t_bytes
            self._base_loaded = True

    def prefetch_base_weights_async(self):
        if not self._base_loaded and self._base_fut is None:
            self._base_fut = self._pool.submit(self.ensure_base_weights_loaded)
        elif self._base_loaded and self._transfer_stream is not None:
            self.stage_base_weights_to_gpu_async()

    @torch.no_grad()
    def stage_base_weights_to_gpu_async(self):
        """Asynchronously DMA-copies any CPU-resident base_cache tensors to GPU on _transfer_stream."""
        if self._transfer_stream is None or not self._base_loaded:
            return
        if not hasattr(self, "_base_gpu_stage"):
            self._base_gpu_stage: Dict[str, Tuple[torch.Tensor, Optional[torch.cuda.Event]]] = {}
        cpu_items = [
            (k, v) for k, v in self.base_cache.items()
            if not isinstance(v, tuple) and not v.is_cuda and k not in self._base_gpu_stage
        ]
        if not cpu_items:
            return
        with torch.cuda.stream(self._transfer_stream):
            ev = torch.cuda.Event()
            staged = {k: v.to(self.device, non_blocking=True) for k, v in cpu_items}
            ev.record(self._transfer_stream)
        for k, gt in staged.items():
            self._base_gpu_stage[k] = (gt, ev)

    def get_base_tensor(self, key: str, target_dev: torch.device):
        if not self._base_loaded:
            self.ensure_base_weights_loaded()
        if hasattr(self, "_base_gpu_stage") and key in self._base_gpu_stage:
            gt, ev = self._base_gpu_stage.pop(key)
            cur_stream = torch.cuda.current_stream(target_dev)
            if ev is not None:
                cur_stream.wait_event(ev)
            gt.record_stream(cur_stream)
            return gt
        t = self.base_cache.get(key)
        if isinstance(t, tuple):
            return t
        if t is None:
            t = self.get_tensor(key)
            if t.dtype == torch.bfloat16 and self.out_dtype == torch.float16:
                t = t.float().clamp(-65504.0, 65504.0).to(torch.float16)
            t_bytes = t.numel() * t.element_size()
            self.base_cache[key] = t
            self.ram_used_bytes += t_bytes
        if t.device == target_dev:
            return t
        return t.to(target_dev, non_blocking=True)

    def _read_expert_raw(self, e: int, target_dev: torch.device) -> Dict[str, torch.Tensor]:
        epfx = f"{self.pfx}experts.{e}."
        out = {}
        for proj in ("w1", "w2", "w3"):
            for suffix in ("idx_packed", "scales", "u_svd", "v_svd"):
                k = f"{epfx}{proj}.{suffix}"
                t = self.get_tensor(k)
                if suffix in ("u_svd", "v_svd") and t.dtype != torch.float16:
                    t = t.float().clamp(-65504.0, 65504.0).to(torch.float16)
                out[f"{proj}.{suffix}"] = t.to(target_dev, non_blocking=True)
        return out

    def _make_ptr_row(self, e: int, raw: Dict[str, torch.Tensor]) -> List[int]:
        tier = self.expert_tiers[e]
        rank = raw["w1.u_svd"].shape[1]
        return [
            raw["w1.idx_packed"].data_ptr(), raw["w1.scales"].data_ptr(), raw["w1.u_svd"].data_ptr(), raw["w1.v_svd"].data_ptr(),
            raw["w3.idx_packed"].data_ptr(), raw["w3.scales"].data_ptr(), raw["w3.u_svd"].data_ptr(), raw["w3.v_svd"].data_ptr(),
            raw["w2.idx_packed"].data_ptr(), raw["w2.scales"].data_ptr(), raw["w2.u_svd"].data_ptr(), raw["w2.v_svd"].data_ptr(),
            tier, rank,
        ]

    def _compute_expert_priority_order(self) -> List[int]:
        """
        Orders experts strictly by:
          1. Quantization Tier (Tier 1 Hot -> Tier 2 Warm -> Tier 3 Cold)
          2. Intra-tier router activation priority: ||gate.weight[e]||_2 + 10.0 * e_score_correction_bias[e]
        """
        n_exp = len(self.expert_tiers)
        scores = [0.0] * n_exp
        all_keys = set(self.keys())
        gw_k = f"{self.pfx}gate.weight"
        bias_k = f"{self.pfx}gate.e_score_correction_bias"
        if gw_k in all_keys:
            gw = self.get_tensor(gw_k).float()
            gnorm = gw.norm(dim=-1)
            if bias_k in all_keys:
                bias = self.get_tensor(bias_k).float()
                prio = (gnorm + 10.0 * bias).tolist()
            else:
                prio = gnorm.tolist()
            for e in range(min(n_exp, len(prio))):
                scores[e] = float(prio[e])
        return sorted(
            range(n_exp),
            key=lambda e: (self.expert_tiers[e], -scores[e], e),
        )

    def _populate_tiers(self):
        vram_used = self.vram_used_bytes
        ram_used = self.ram_used_bytes
        tier_bytes: Dict[int, int] = {}

        # Waterfall Bucket Filling:
        #   Bucket 1 (GPU VRAM): Fills Tier 1 first, then promotes highest-priority Tier 2/3 if VRAM budget remains
        #   Bucket 2 (System RAM): Fills remaining Tier 1/2, then promotes Tier 3 if RAM budget remains
        #   Bucket 3 (NVMe mmap): Any remaining tail experts stay on disk (0 B VRAM, 0 B RAM)
        for e in self.priority_order:
            tier = self.expert_tiers[e]
            can_ram = self.preload_ram and (tier < 3 or self.promote_cold_to_ram)
            if tier in tier_bytes:
                tb = tier_bytes[tier]
                vram_fits = (vram_used + tb) <= self.vram_hot_budget_bytes
                ram_fits = can_ram and ((ram_used + tb) <= self.ram_warm_budget_bytes)
                if not vram_fits and not ram_fits:
                    if not self.preload_ram:
                        break
                    continue
            try:
                entry = self._read_expert_raw(e, torch.device("cpu"))
            except MemoryError:
                break
            ebytes = sum(t.numel() * t.element_size() for t in entry.values())
            tier_bytes[tier] = ebytes
            if vram_used + ebytes <= self.vram_hot_budget_bytes:
                gpu_e = {k: v.to(self.device, non_blocking=True) for k, v in entry.items()}
                self.vram_cache[e] = gpu_e
                self._vram_ptr_ready.add(e)
                vram_used += ebytes
                del entry
            elif can_ram and (ram_used + ebytes <= self.ram_warm_budget_bytes):
                self.ram_cache[e] = self._pin_dict(entry)
                ram_used += ebytes
            elif not self.preload_ram:
                del entry
                break
        if self.vram_cache and self.device.type == "cuda":
            idxs = sorted(self.vram_cache.keys())
            rows = [self._make_ptr_row(e, self.vram_cache[e]) for e in idxs]
            self.expert_ptr_table[torch.tensor(idxs, dtype=torch.long, device=self.device)] = torch.tensor(
                rows, dtype=torch.int64, device=self.device
            )
        self.vram_used_bytes = vram_used
        self.ram_used_bytes = ram_used

    def register_folded_lookahead(self, lead: int, w_prime: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Registers precomputed lookahead projection weights for lead d in {1, 2}."""
        self.lookahead_w[lead] = w_prime.to(device=self.device, dtype=torch.float32)
        if bias is not None:
            self.lookahead_bias[lead] = bias.to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def predict_and_prefetch_lookahead(
        self,
        x0_normed: torch.Tensor,
        target_mgr: "TieredMoECacheManager",
        lead: int = 1,
        top_k: int = 16,
    ):
        """Predicts target layer top-k experts from normed shared-expert base x0_normed and prefetches non-VRAM experts."""
        w_p = self.lookahead_w.get(lead)
        if w_p is None:
            return
        logits = x0_normed.float() @ w_p.T
        scores = torch.sigmoid(logits)
        b_p = self.lookahead_bias.get(lead)
        if b_p is not None:
            scores = scores + b_p.unsqueeze(0)
        pred_idx = torch.topk(scores, k=min(top_k, scores.shape[-1]), dim=-1).indices.view(-1).tolist()
        target_mgr.prefetch_cold_experts_async(pred_idx)
        for e in pred_idx:
            if e in target_mgr.ram_cache:
                target_mgr.stage_expert_to_gpu_async(e)

    def prefetch_cold_experts_async(self, predicted_experts: List[int]):
        """RouterLookahead hook: asynchronously prefetches predicted Cold NVMe experts."""
        for e in predicted_experts:
            if e not in self.vram_cache and e not in self.ram_cache and e not in self.prefetch_buffer:
                self._pool.submit(self._prefetch_worker, e)

    def _prefetch_worker(self, e: int):
        self.prefetch_buffer[e] = self._read_expert_raw(e, torch.device("cpu"))

    def warm_experts_parallel(self, expert_ids: List[int]):
        missing = [
            e for e in expert_ids
            if e not in self.vram_cache and e not in self.ram_cache and e not in self.prefetch_buffer
        ]
        if not missing:
            return
        all_keys: List[str] = []
        for e in missing:
            epfx = f"{self.pfx}experts.{e}."
            for proj in ("w1", "w2", "w3"):
                for suffix in ("idx_packed", "scales", "u_svd", "v_svd"):
                    all_keys.append(f"{epfx}{proj}.{suffix}")
        loaded = self.batch_pread_tensors(all_keys)
        for e in missing:
            epfx = f"{self.pfx}experts.{e}."
            entry: Dict[str, torch.Tensor] = {}
            for proj in ("w1", "w2", "w3"):
                for suffix in ("idx_packed", "scales", "u_svd", "v_svd"):
                    t = loaded[f"{epfx}{proj}.{suffix}"]
                    if suffix in ("u_svd", "v_svd") and t.dtype != torch.float16:
                        t = t.float().clamp(-65504.0, 65504.0).to(torch.float16)
                    entry[f"{proj}.{suffix}"] = t
            self.prefetch_buffer[e] = entry

    @torch.no_grad()
    def get_experts_batch_ptrs_gpu(self, flat_e: torch.Tensor) -> torch.Tensor:
        """
        Ensures all routed experts in `flat_e` are staged in GPU memory and returns
        their [E, 14] int64 pointer table on GPU. Zero CPU round-trip when all experts are in VRAM.
        """
        self._last_batch_keepalive.clear()
        if len(self._vram_ptr_ready) == len(self.expert_tiers):
            return self.expert_ptr_table.index_select(0, flat_e).contiguous()
        e_list = flat_e.tolist()
        missing = [e for e in set(e_list) if e not in self._vram_ptr_ready]
        if missing:
            self.warm_experts_parallel(missing)
            upd_idxs = []
            upd_rows = []
            for e in missing:
                raw, _, _, _, _ = self.get_expert_raw_gpu(e)
                if e in self.vram_cache:
                    self._vram_ptr_ready.add(e)
                else:
                    self._last_batch_keepalive.append(raw)
                upd_idxs.append(e)
                upd_rows.append(self._make_ptr_row(e, raw))
            self.expert_ptr_table[torch.tensor(upd_idxs, dtype=torch.long, device=self.device)] = torch.tensor(
                upd_rows, dtype=torch.int64, device=self.device
            )
        return self.expert_ptr_table.index_select(0, flat_e).contiguous()

    @torch.no_grad()
    def stage_expert_to_gpu_async(self, e: int):
        """Asynchronously stages a RAM/prefetched expert to GPU on the dedicated transfer stream."""
        if self._transfer_stream is None or e in self.vram_cache or e in self._gpu_stage_cache:
            return
        cpu_raw = self.ram_cache.get(e)
        if cpu_raw is None:
            return
        with torch.cuda.stream(self._transfer_stream):
            gpu_raw = {k: v.to(self.device, non_blocking=True) for k, v in cpu_raw.items()}
            ev = torch.cuda.Event()
            ev.record(self._transfer_stream)
        self._gpu_stage_cache[e] = (gpu_raw, ev)

    @torch.no_grad()
    def get_expert_raw_gpu(self, e: int) -> Tuple[Dict[str, torch.Tensor], int, torch.Tensor, torch.Tensor, int]:
        """Returns (raw_gpu_dict, tier, lut, q_basis, bits) on GPU without materializing dense weights."""
        tier = self.expert_tiers[e]
        lut = self.luts[tier]
        q_basis = self.q_bases[tier]
        bits = lut.shape[0].bit_length() - 1

        if e in self.vram_cache:
            raw = self.vram_cache[e]
        elif e in self._gpu_stage_cache:
            raw, ev = self._gpu_stage_cache.pop(e)
            cur_stream = torch.cuda.current_stream(self.device)
            if ev is not None:
                cur_stream.wait_event(ev)
            for t in raw.values():
                t.record_stream(cur_stream)
        elif e in self.ram_cache:
            raw = {k: v.to(self.device, non_blocking=True) for k, v in self.ram_cache[e].items()}
        else:
            if e in self.prefetch_buffer:
                cpu_raw = self.prefetch_buffer.pop(e)
            else:
                cpu_raw = self._read_expert_raw(e, torch.device("cpu"))
            ebytes = sum(t.numel() * t.element_size() for t in cpu_raw.values())
            if (tier == 1 or self.eager_preload) and (self.vram_used_bytes + ebytes <= self.vram_hot_budget_bytes):
                raw = {k: v.to(self.device, non_blocking=True) for k, v in cpu_raw.items()}
                self.vram_cache[e] = raw
                self.vram_used_bytes += ebytes
            elif (tier < 3 or self.promote_cold_to_ram) and (self.ram_used_bytes + ebytes <= self.ram_warm_budget_bytes):
                pinned_raw = self._pin_dict(cpu_raw)
                self.ram_cache[e] = pinned_raw
                self.ram_used_bytes += ebytes
                raw = {k: v.to(self.device, non_blocking=True) for k, v in pinned_raw.items()}
            else:
                raw = {k: v.to(self.device, non_blocking=True) for k, v in cpu_raw.items()}
                del cpu_raw
        return raw, tier, lut, q_basis, bits

    @torch.no_grad()
    def get_decoded_expert(self, e: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns decoded (w1, w3, w2) in BF16 (or FP16 on V100 sm_70) on GPU."""
        raw, _, lut, q_basis, bits = self.get_expert_raw_gpu(e)
        mats = []
        for proj in ("w1", "w3", "w2"):
            w = decode_sc_lut_matrix(
                raw[f"{proj}.idx_packed"],
                raw[f"{proj}.scales"],
                raw[f"{proj}.u_svd"],
                raw[f"{proj}.v_svd"],
                lut,
                q_basis,
                bits=bits,
                out_dtype=self.out_dtype,
                force_arch_sm=self.force_arch_sm,
            )
            mats.append(w)
        return mats[0], mats[1], mats[2]


def load_strata_coder_layer(
    shard_path: str,
    layer_idx: int,
    device: str = "cuda",
    vram_hot_budget_gb: float = 0.8,
    ram_warm_budget_gb: float = 2.5,
    promote_cold_to_ram: Optional[bool] = None,
    force_arch_sm: int = 0,
    eager_preload: bool = True,
) -> TieredMoECacheManager:
    return TieredMoECacheManager(
        shard_path=shard_path,
        layer_idx=layer_idx,
        device=device,
        vram_hot_budget_gb=vram_hot_budget_gb,
        ram_warm_budget_gb=ram_warm_budget_gb,
        promote_cold_to_ram=promote_cold_to_ram,
        force_arch_sm=force_arch_sm,
        eager_preload=eager_preload,
    )


def load_strata_coder_multi_gpu(
    shard_paths: Dict[int, str],
    num_gpus: Optional[int] = None,
    total_vram_gb: Optional[float] = None,
    total_ram_gb: Optional[float] = None,
    promote_cold_to_ram: Optional[bool] = None,
    force_arch_sm: int = 0,
    eager_preload: Optional[bool] = None,
    layer_vram_weights: Optional[Dict[int, float]] = None,
    layer_priority_overrides: Optional[Dict[int, List[int]]] = None,
) -> Dict[int, TieredMoECacheManager]:
    """
    Hardware-aware Multi-GPU + System RAM loader.
    Dynamically detects total GPU VRAM and Host RAM when not explicitly specified,
    filling Tier 1 -> Tier 2 -> Tier 3 in strict priority order up to the machine's actual memory.
    Supports optional per-layer VRAM budget weights (`layer_vram_weights`).
    """
    det_vram_gb, det_ram_gb = detect_system_memory_gb()
    if total_vram_gb is None:
        total_vram_gb = det_vram_gb if det_vram_gb > 0 else 24.0
    if total_ram_gb is None:
        total_ram_gb = det_ram_gb

    n_layers = max(len(shard_paths), 1)
    n_budget_layers = n_layers
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_gpus = max(num_gpus, 1)

    # Reserve ~16% of VRAM for lm_head + workspace, ~20% of RAM for OS
    total_usable_vram = total_vram_gb * 0.84
    usable_vram_per_layer = total_usable_vram / n_budget_layers
    usable_ram_per_layer = (total_ram_gb * 0.80) / n_budget_layers
    if promote_cold_to_ram is None:
        promote_cold_to_ram = True
    if eager_preload is None:
        eager_preload = True
    preload_ram = len(shard_paths) <= 8 or os.environ.get("STRATA_PRELOAD_RAM") == "1"

    weight_sum = sum(layer_vram_weights.get(li, 1.0) for li in shard_paths) if layer_vram_weights else float(n_budget_layers)
    managers: Dict[int, TieredMoECacheManager] = {}
    for idx, (layer_idx, spath) in enumerate(sorted(shard_paths.items())):
        gpu_id = (idx * num_gpus) // n_layers
        dev = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        l_vram_gb = (
            total_usable_vram * (layer_vram_weights.get(layer_idx, 1.0) / max(weight_sum, 1e-6))
            if layer_vram_weights
            else usable_vram_per_layer
        )
        p_override = layer_priority_overrides.get(layer_idx) if layer_priority_overrides else None
        managers[layer_idx] = TieredMoECacheManager(
            shard_path=spath,
            layer_idx=layer_idx,
            device=dev,
            vram_hot_budget_gb=l_vram_gb,
            ram_warm_budget_gb=usable_ram_per_layer,
            promote_cold_to_ram=promote_cold_to_ram,
            force_arch_sm=force_arch_sm,
            eager_preload=eager_preload,
            vram_priority_override=p_override,
            preload_ram=preload_ram,
        )
    return managers
