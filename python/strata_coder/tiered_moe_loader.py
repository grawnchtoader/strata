from concurrent.futures import Future, ThreadPoolExecutor
import contextlib
import ctypes
import json
import os
import struct
import threading
import time
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
            dev_ctx = (
                torch.cuda.device(self.device)
                if (self.device.type == "cuda" and torch.cuda.is_available())
                else contextlib.nullcontext()
            )
            with dev_ctx:
                for k, t in items_sorted:
                    if t.dtype == torch.bfloat16 and self.out_dtype == torch.float16:
                        t = t.float().clamp(-65504.0, 65504.0).to(torch.float16)
                    t_bytes = t.numel() * t.element_size()
                    need_fp8 = (self.vram_used_bytes + rem_bytes) > self.vram_base_budget_bytes
                    rem_bytes -= t_bytes
                    try:
                        if (
                            not need_fp8
                            and (self.vram_used_bytes + t_bytes <= self.vram_base_budget_bytes)
                            and self._has_gpu_headroom(t_bytes)
                        ):
                            self.base_cache[k] = t.to(self.device, non_blocking=True)
                            self.vram_used_bytes += t_bytes
                        elif (
                            can_fp8
                            and t.ndim == 2
                            and t.element_size() == 2
                            and t.numel() >= 8_000_000
                            and (self.vram_used_bytes + (t_bytes // 2) + t.shape[0] * 4 <= self.vram_base_budget_bytes)
                            and self._has_gpu_headroom((t_bytes // 2) + t.shape[0] * 4)
                        ):
                            tf = t.to(self.device, non_blocking=True).float()
                            sc = tf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
                            w_fp8 = (tf / sc).to(torch.float8_e4m3fn)
                            del tf
                            self.base_cache[k] = (w_fp8, sc.squeeze(1))
                            self.vram_used_bytes += (t_bytes // 2) + t.shape[0] * 4
                        elif (
                            (self.vram_used_bytes + t_bytes <= self.vram_base_budget_bytes)
                            and self._has_gpu_headroom(t_bytes)
                        ):
                            self.base_cache[k] = t.to(self.device, non_blocking=True)
                            self.vram_used_bytes += t_bytes
                        else:
                            self.base_cache[k] = t
                            self.ram_used_bytes += t_bytes
                    except torch.OutOfMemoryError:
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
        total_b = sum(v.numel() * v.element_size() for _, v in cpu_items)
        if not self._has_gpu_headroom(total_b):
            return
        try:
            with torch.cuda.device(self.device):
                with torch.cuda.stream(self._transfer_stream):
                    ev = torch.cuda.Event()
                    staged = {k: v.to(self.device, non_blocking=True) for k, v in cpu_items}
                    ev.record(self._transfer_stream)
            for k, gt in staged.items():
                self._base_gpu_stage[k] = (gt, ev)
        except torch.OutOfMemoryError:
            self._base_gpu_stage.clear()

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
                if suffix in ("scales", "u_svd", "v_svd") and t.dtype != torch.float16:
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

    def _has_gpu_headroom(self, extra_bytes: int, min_free_bytes: int = 3_221_225_472) -> bool:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return True
        try:
            free_b, _ = torch.cuda.mem_get_info(self.device)
            return (free_b - extra_bytes) >= min_free_bytes
        except Exception:
            return True

    def _estimate_expert_bytes(self, e: int) -> int:
        if not self._hdr:
            return 0
        epfx = f"{self.pfx}experts.{e}."
        total = 0
        for proj in ("w1", "w2", "w3"):
            for suffix in ("idx_packed", "scales", "u_svd", "v_svd"):
                m = self._hdr.get(f"{epfx}{proj}.{suffix}")
                if m is None:
                    return 0
                numel = 1
                for dim in m["shape"]:
                    numel *= dim
                elem_sz = 2 if suffix != "idx_packed" else _SAFETENSORS_DTYPE_MAP[m["dtype"]].itemsize
                total += numel * elem_sz
        return total

    def _populate_tiers(self):
        vram_used = self.vram_used_bytes
        ram_used = self.ram_used_bytes
        sim_vram = vram_used
        sim_ram = ram_used
        tier_bytes: Dict[int, int] = {}
        dev_ctx = (
            torch.cuda.device(self.device)
            if (self.device.type == "cuda" and torch.cuda.is_available())
            else contextlib.nullcontext()
        )

        def _flush_batch(batch_exps: List[int]) -> bool:
            nonlocal vram_used, ram_used, sim_vram, sim_ram
            if not batch_exps:
                return False
            b_keys: List[str] = []
            for e_id in batch_exps:
                epfx = f"{self.pfx}experts.{e_id}."
                for proj in ("w1", "w2", "w3"):
                    for suffix in ("idx_packed", "scales", "u_svd", "v_svd"):
                        b_keys.append(f"{epfx}{proj}.{suffix}")
            try:
                loaded = self.batch_pread_tensors(b_keys, max_gap=262144, max_chunk=16 * 1024 * 1024)
            except MemoryError:
                return True
            stop_early = False
            with dev_ctx:
                for e_id in batch_exps:
                    epfx = f"{self.pfx}experts.{e_id}."
                    entry: Dict[str, torch.Tensor] = {}
                    for proj in ("w1", "w2", "w3"):
                        for suffix in ("idx_packed", "scales", "u_svd", "v_svd"):
                            t = loaded.pop(f"{epfx}{proj}.{suffix}")
                            if suffix in ("scales", "u_svd", "v_svd") and t.dtype != torch.float16:
                                t = t.float().clamp(-65504.0, 65504.0).to(torch.float16)
                            entry[f"{proj}.{suffix}"] = t
                    ebytes = sum(t.numel() * t.element_size() for t in entry.values())
                    tier = self.expert_tiers[e_id]
                    tier_bytes[tier] = ebytes
                    can_ram = self.preload_ram and (tier < 3 or self.promote_cold_to_ram)
                    placed_vram = False
                    if (vram_used + ebytes <= self.vram_hot_budget_bytes) and self._has_gpu_headroom(ebytes):
                        try:
                            gpu_e = {k: v.to(self.device, non_blocking=True) for k, v in entry.items()}
                            self.vram_cache[e_id] = gpu_e
                            self._vram_ptr_ready.add(e_id)
                            vram_used += ebytes
                            placed_vram = True
                            del entry
                        except torch.OutOfMemoryError:
                            placed_vram = False
                    if not placed_vram:
                        if can_ram and (ram_used + ebytes <= self.ram_warm_budget_bytes):
                            self.ram_cache[e_id] = self._pin_dict(entry)
                            ram_used += ebytes
                        else:
                            del entry
                            if not self.preload_ram:
                                stop_early = True
                                break
            loaded.clear()
            sim_vram = vram_used
            sim_ram = ram_used
            return stop_early

        batch: List[int] = []
        for e in self.priority_order:
            tier = self.expert_tiers[e]
            can_ram = self.preload_ram and (tier < 3 or self.promote_cold_to_ram)
            tb = self._estimate_expert_bytes(e) or tier_bytes.get(tier, 0)
            if tb > 0:
                vram_fits = ((sim_vram + tb) <= self.vram_hot_budget_bytes) and self._has_gpu_headroom(
                    (sim_vram - vram_used) + tb
                )
                ram_fits = can_ram and ((sim_ram + tb) <= self.ram_warm_budget_bytes)
                if not vram_fits and not ram_fits:
                    if not self.preload_ram:
                        break
                    continue
                if vram_fits:
                    sim_vram += tb
                else:
                    sim_ram += tb
            batch.append(e)
            if len(batch) >= 64 or (tb == 0 and not self.preload_ram and len(batch) >= 8):
                if _flush_batch(batch):
                    batch.clear()
                    break
                batch.clear()
        if batch:
            _flush_batch(batch)

        if self.vram_cache and self.device.type == "cuda":
            with dev_ctx:
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
        raw_idx = torch.topk(scores, k=min(top_k, scores.shape[-1]), dim=-1).indices.view(-1).tolist()
        pred_idx = list(dict.fromkeys(raw_idx))
        target_mgr.prefetch_cold_experts_async(pred_idx)
        for e in pred_idx:
            if len(target_mgr._gpu_stage_cache) >= 24:
                break
            if e in target_mgr.ram_cache:
                target_mgr.stage_expert_to_gpu_async(e)

    def prefetch_cold_experts_async(self, predicted_experts: List[int]):
        """RouterLookahead hook: asynchronously prefetches predicted Cold NVMe experts."""
        for e in predicted_experts:
            if len(self.prefetch_buffer) >= 64:
                break
            if e not in self.vram_cache and e not in self.ram_cache and e not in self.prefetch_buffer:
                self._pool.submit(self._prefetch_worker, e)

    def _prefetch_worker(self, e: int):
        if len(self.prefetch_buffer) < 64:
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
                    if suffix in ("scales", "u_svd", "v_svd") and t.dtype != torch.float16:
                        t = t.float().clamp(-65504.0, 65504.0).to(torch.float16)
                    entry[f"{proj}.{suffix}"] = t
            self.prefetch_buffer[e] = entry

    def _clear_batch_keepalive(self):
        if not self._last_batch_keepalive:
            return
        if self.device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(self.device):
                cur_stream = torch.cuda.current_stream(self.device)
                for raw in self._last_batch_keepalive:
                    for t in raw.values():
                        if t.is_cuda:
                            t.record_stream(cur_stream)
        self._last_batch_keepalive.clear()

    def release_transient_gpu_cache(self):
        """Releases per-layer transient GPU buffers and unconsumed lookahead stages cleanly on their streams."""
        self._clear_batch_keepalive()
        if self._gpu_stage_cache:
            if self._transfer_stream is not None:
                for raw, _ in self._gpu_stage_cache.values():
                    for t in raw.values():
                        if t.is_cuda:
                            t.record_stream(self._transfer_stream)
            self._gpu_stage_cache.clear()
        if hasattr(self, "_base_gpu_stage") and self._base_gpu_stage:
            if self._transfer_stream is not None:
                for gt, _ in self._base_gpu_stage.values():
                    if gt.is_cuda:
                        gt.record_stream(self._transfer_stream)
            self._base_gpu_stage.clear()

    @torch.no_grad()
    def get_experts_batch_ptrs_gpu(self, flat_e: torch.Tensor) -> torch.Tensor:
        """
        Ensures all routed experts in `flat_e` are staged in GPU memory and returns
        their [E, 14] int64 pointer table on GPU. Zero CPU round-trip when all experts are in VRAM.
        """
        self._clear_batch_keepalive()
        if len(self._vram_ptr_ready) == len(self.expert_tiers):
            return self.expert_ptr_table.index_select(0, flat_e).contiguous()
        e_list = flat_e.tolist()
        missing = [e for e in dict.fromkeys(e_list) if e not in self._vram_ptr_ready]
        if not missing:
            return self.expert_ptr_table.index_select(0, flat_e).contiguous()
        self.warm_experts_parallel(missing)
        upd_vram_idxs = []
        upd_vram_rows = []
        temp_rows: Dict[int, List[int]] = {}
        for e in missing:
            raw, _, _, _, _ = self.get_expert_raw_gpu(e)
            row = self._make_ptr_row(e, raw)
            if e in self.vram_cache:
                self._vram_ptr_ready.add(e)
                upd_vram_idxs.append(e)
                upd_vram_rows.append(row)
            else:
                self._last_batch_keepalive.append(raw)
                temp_rows[e] = row
        if upd_vram_idxs:
            self.expert_ptr_table[torch.tensor(upd_vram_idxs, dtype=torch.long, device=self.device)] = torch.tensor(
                upd_vram_rows, dtype=torch.int64, device=self.device
            )
        if not temp_rows:
            return self.expert_ptr_table.index_select(0, flat_e).contiguous()
        out = self.expert_ptr_table.index_select(0, flat_e).contiguous()
        t_idxs = [i for i, e in enumerate(e_list) if e in temp_rows]
        t_vals = [temp_rows[e_list[i]] for i in t_idxs]
        out[torch.tensor(t_idxs, dtype=torch.long, device=self.device)] = torch.tensor(
            t_vals, dtype=torch.int64, device=self.device
        )
        return out

    @torch.no_grad()
    def stage_expert_to_gpu_async(self, e: int):
        """Asynchronously stages a RAM/prefetched expert to GPU on the dedicated transfer stream."""
        if (
            self._transfer_stream is None
            or e in self.vram_cache
            or e in self._gpu_stage_cache
            or len(self._gpu_stage_cache) >= 24
        ):
            return
        cpu_raw = self.ram_cache.get(e)
        if cpu_raw is None:
            return
        ebytes = sum(v.numel() * v.element_size() for v in cpu_raw.values())
        if not self._has_gpu_headroom(ebytes):
            return
        try:
            with torch.cuda.device(self.device):
                with torch.cuda.stream(self._transfer_stream):
                    gpu_raw = {k: v.to(self.device, non_blocking=True) for k, v in cpu_raw.items()}
                    ev = torch.cuda.Event()
                    ev.record(self._transfer_stream)
            self._gpu_stage_cache[e] = (gpu_raw, ev)
        except torch.OutOfMemoryError:
            self.release_transient_gpu_cache()

    def _copy_dict_to_gpu(self, src: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dev_ctx = (
            torch.cuda.device(self.device)
            if (self.device.type == "cuda" and torch.cuda.is_available())
            else contextlib.nullcontext()
        )
        try:
            with dev_ctx:
                return {k: v.to(self.device, non_blocking=True) for k, v in src.items()}
        except torch.OutOfMemoryError:
            self.release_transient_gpu_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with dev_ctx:
                return {k: v.to(self.device, non_blocking=True) for k, v in src.items()}

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
            with torch.cuda.device(self.device):
                cur_stream = torch.cuda.current_stream(self.device)
                if ev is not None:
                    cur_stream.wait_event(ev)
                for t in raw.values():
                    t.record_stream(cur_stream)
        elif e in self.ram_cache:
            raw = self._copy_dict_to_gpu(self.ram_cache[e])
        else:
            if e in self.prefetch_buffer:
                cpu_raw = self.prefetch_buffer.pop(e)
            else:
                cpu_raw = self._read_expert_raw(e, torch.device("cpu"))
            ebytes = sum(t.numel() * t.element_size() for t in cpu_raw.values())
            if (
                (tier == 1 or self.eager_preload)
                and (self.vram_used_bytes + ebytes <= self.vram_hot_budget_bytes)
                and self._has_gpu_headroom(ebytes)
            ):
                raw = self._copy_dict_to_gpu(cpu_raw)
                self.vram_cache[e] = raw
                self.vram_used_bytes += ebytes
            elif (tier < 3 or self.promote_cold_to_ram) and (self.ram_used_bytes + ebytes <= self.ram_warm_budget_bytes):
                pinned_raw = self._pin_dict(cpu_raw)
                self.ram_cache[e] = pinned_raw
                self.ram_used_bytes += ebytes
                raw = self._copy_dict_to_gpu(pinned_raw)
            else:
                raw = self._copy_dict_to_gpu(cpu_raw)
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

    sorted_items = sorted(shard_paths.items())
    gpu_layer_idxs: Dict[int, List[int]] = {g: [] for g in range(num_gpus)}
    for idx, (layer_idx, _) in enumerate(sorted_items):
        g = (idx * num_gpus) // n_layers
        gpu_layer_idxs[g].append(layer_idx)

    per_gpu_usable = total_usable_vram / num_gpus
    managers: Dict[int, TieredMoECacheManager] = {}
    t_load0 = time.time()
    orig_threads = torch.get_num_threads()
    if orig_threads > 4:
        torch.set_num_threads(4)
    try:
        for idx, (layer_idx, spath) in enumerate(sorted_items):
            gpu_id = (idx * num_gpus) // n_layers
            dev = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            g_layers = gpu_layer_idxs[gpu_id]
            g_budget = per_gpu_usable
            if num_gpus > 1 and n_layers > 8:
                if gpu_id == 0:
                    g_budget = max(g_budget - 2.6, g_budget * 0.85)
                elif gpu_id == num_gpus - 1:
                    g_budget = max(g_budget - 2.2, g_budget * 0.88)
            if layer_vram_weights:
                g_wsum = sum(layer_vram_weights.get(li, 1.0) for li in g_layers)
                l_vram_gb = g_budget * (layer_vram_weights.get(layer_idx, 1.0) / max(g_wsum, 1e-6))
            else:
                l_vram_gb = g_budget / max(len(g_layers), 1)
            p_override = layer_priority_overrides.get(layer_idx) if layer_priority_overrides else None
            dev_ctx = torch.cuda.device(dev) if dev.startswith("cuda") else contextlib.nullcontext()
            with dev_ctx:
                mgr = TieredMoECacheManager(
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
            managers[layer_idx] = mgr
            if eager_preload and len(sorted_items) > 1:
                tot_vram_gb = sum(m.vram_used_bytes for m in managers.values()) / (1024 ** 3)
                tot_ram_gb = sum(m.ram_used_bytes for m in managers.values()) / (1024 ** 3)
                print(
                    f"[Strata-Coder] Loaded MoE layer {idx + 1}/{len(sorted_items)} "
                    f"(layer {layer_idx}, {dev}) | VRAM: {tot_vram_gb:.1f} GB ({len(mgr.vram_cache)} hot) | "
                    f"RAM: {tot_ram_gb:.1f} GB ({len(mgr.ram_cache)} warm) | "
                    f"elapsed {time.time() - t_load0:.1f}s",
                    flush=True,
                )
    finally:
        if torch.get_num_threads() != orig_threads:
            torch.set_num_threads(orig_threads)
    return managers
