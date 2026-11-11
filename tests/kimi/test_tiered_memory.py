"""CPU regressions for ownership and budgeting; no model or CUDA allocations."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import gc
import importlib.util
from pathlib import Path
import sys
import threading
import types
import weakref

import pytest
import torch


@pytest.fixture
def loader(monkeypatch):
    package = "_tiered_memory_test"
    monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    decoder = types.ModuleType(f"{package}.decoder")
    decoder.decode_sc_lut_matrix = None
    decoder.resolve_optimal_dtype = lambda device, dtype, **kwargs: dtype
    monkeypatch.setitem(sys.modules, decoder.__name__, decoder)
    path = Path(__file__).resolve().parents[2] / "python/strata_coder/tiered_moe_loader.py"
    spec = importlib.util.spec_from_file_location(f"{package}.tiered_moe_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    module._SHARED_IO_POOL.shutdown(wait=True)


class Tensor:
    def __init__(self, nbytes, device="cpu", records=None):
        self.nbytes = nbytes
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.records = records if records is not None else []

    def numel(self):
        return self.nbytes // 4

    def element_size(self):
        return 4

    def to(self, device=None, **kwargs):
        return Tensor(self.nbytes, device, self.records)

    def record_stream(self, stream):
        self.records.append(stream)


@pytest.fixture
def cuda(monkeypatch):
    streams = []
    stream = object()
    monkeypatch.setattr(torch.cuda, "device", lambda dev: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda dev: stream)

    class Event:
        def record(self, s):
            streams.append(s)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    return stream, streams


def manager(loader):
    mgr = loader.TieredMoECacheManager.__new__(loader.TieredMoECacheManager)
    mgr.device = torch.device("cuda:1")
    mgr._transfer_stream = object()
    mgr._last_batch_keepalive = []
    mgr._gpu_stage_cache = {}
    mgr._base_gpu_stage = {}
    mgr.vram_used_bytes = 100
    mgr.vram_hot_budget_bytes = 200
    mgr.vram_cache = {}
    mgr.ram_cache = {}
    mgr.base_cache = {}
    mgr._base_loaded = True
    mgr._base_fut = None
    mgr._prefetch_lock = threading.Lock()
    mgr._prefetch_pending = set()
    mgr._prefetch_epoch = 0
    mgr.prefetch_buffer = {}
    mgr.ram_used_bytes = 0
    mgr.ram_warm_budget_bytes = 200
    mgr.lookahead_w = {}
    mgr.lookahead_bias = {}
    return mgr


def test_false_predictions_are_bounded_and_released_each_visit(loader, cuda):
    mgr = manager(loader)
    mgr.ram_cache = {e: {"weight": Tensor(40)} for e in range(8)}
    for start in (0, 2, 4, 6):
        for e in range(start, start + 2):
            mgr.stage_expert_to_gpu_async(e)
        assert len(mgr._gpu_stage_cache) == 2
        mgr.stage_expert_to_gpu_async((start + 2) % 8)
        assert len(mgr._gpu_stage_cache) == 2
        refs = [weakref.ref(raw["weight"]) for raw, _ in mgr._gpu_stage_cache.values()]
        mgr.release_transient_gpu_buffers()
        assert not mgr._gpu_stage_cache
        assert all(ref() is None for ref in refs)
    assert mgr.vram_used_bytes == 100


def test_base_and_expert_prefetch_share_remaining_budget(loader, cuda):
    mgr = manager(loader)
    mgr.base_cache = {"used": Tensor(60), "unused": Tensor(80)}
    mgr.ram_cache = {0: {"weight": Tensor(40)}, 1: {"weight": Tensor(40)}}
    mgr.stage_base_weights_to_gpu_async()
    assert set(mgr._base_gpu_stage) == {"used"}
    mgr.stage_expert_to_gpu_async(0)
    mgr.stage_expert_to_gpu_async(1)
    assert set(mgr._gpu_stage_cache) == {0}
    mgr.release_transient_gpu_buffers()
    assert not mgr._base_gpu_stage and not mgr._gpu_stage_cache
    assert mgr._available_staging_bytes() == 100


def test_pointer_batch_owners_record_compute_stream_before_release(loader, cuda):
    mgr = manager(loader)
    records = []
    tensor = Tensor(40, "cuda:1", records)
    ref = weakref.ref(tensor)
    mgr._last_batch_keepalive = [{"weight": tensor}]
    del tensor
    mgr.release_experts_batch()
    assert records == [cuda[0]]
    assert ref() is None
    mgr.release_transient_gpu_buffers()  # Cleanup is safe after partial failure.


def test_folded_lookahead_counts_replacements_and_respects_budget(loader, cuda):
    mgr = manager(loader)
    mgr.register_folded_lookahead(1, Tensor(40), Tensor(8))
    assert mgr.vram_used_bytes == 148
    mgr.register_folded_lookahead(1, Tensor(20))
    assert mgr.vram_used_bytes == 120
    assert 1 not in mgr.lookahead_bias
    mgr.register_folded_lookahead(2, Tensor(100))
    assert 2 not in mgr.lookahead_w
    assert mgr.vram_used_bytes == 120


def test_failed_base_prefetch_releases_traceback_but_surfaces_error(loader, cuda):
    mgr = manager(loader)
    ref = None

    def failed_load():
        nonlocal ref
        temporary = Tensor(100, "cuda:1")
        ref = weakref.ref(temporary)
        raise RuntimeError("loader allocation failed")

    mgr._base_fut = Future()
    try:
        failed_load()
    except RuntimeError as error:
        mgr._base_fut.set_exception(error)
    assert ref() is not None
    mgr.release_transient_gpu_buffers()
    assert ref() is None
    with pytest.raises(RuntimeError, match="loader allocation failed"):
        mgr.ensure_base_weights_loaded()
    assert mgr._base_fut is None


def test_cold_prefetch_deduplicates_bounds_and_discards_late_completions(loader, cuda):
    mgr = manager(loader)
    tasks = []
    mgr._pool = types.SimpleNamespace(submit=lambda fn, *args: tasks.append((fn, args)))
    mgr._read_expert_raw = lambda e, device: {"weight": Tensor(40)}
    mgr.prefetch_cold_experts_async(list(range(64)))
    mgr.prefetch_cold_experts_async(list(range(64)))
    assert len(tasks) == 8
    assert len(mgr._prefetch_pending) == 8
    mgr.release_transient_gpu_buffers()
    for fn, args in tasks:
        fn(*args)
    assert not mgr.prefetch_buffer  # A finished old visit cannot resurrect copies.
    assert not mgr._prefetch_pending
    tasks.clear()
    mgr.prefetch_cold_experts_async(list(range(64)))
    for fn, args in tasks:
        fn(*args)
    assert len(mgr.prefetch_buffer) == 5  # 200-byte remaining RAM budget.


@pytest.mark.parametrize("consumer", ["ensure_base_weights_loaded", "prefetch_base_weights_async"])
def test_failed_future_consumer_does_not_need_cyclic_gc(loader, cuda, consumer):
    mgr = manager(loader)
    refs = []

    def failed_load():
        temporary = Tensor(100, "cuda:1")
        refs.append(weakref.ref(temporary))
        raise RuntimeError("loader allocation failed")

    # Use a real Future: its error owns the worker traceback, including the
    # temporary. A consumer must detach both its field and its local Future.
    with ThreadPoolExecutor(max_workers=1) as pool:
        mgr._base_fut = pool.submit(failed_load)
        mgr._base_fut.exception()  # Wait for the worker without consuming its error.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        try:
            getattr(mgr, consumer)()
        except RuntimeError as error:
            assert str(error) == "loader allocation failed"
        else:
            pytest.fail("failed prefetch was not propagated")
        assert mgr._base_fut is None
        assert refs[0]() is None
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("explicit_cause", [False, True])
def test_cleanup_detaches_worker_tracebacks_through_exception_chain(loader, cuda, explicit_cause):
    mgr = manager(loader)
    refs = []

    def failed_load(owner):
        temporary = Tensor(100, "cuda:1")
        refs.append(weakref.ref(temporary))
        try:
            raise ValueError("underlying read failed")
        except ValueError as cause:
            if explicit_cause:
                raise RuntimeError("loader allocation failed") from cause
            raise RuntimeError("loader allocation failed")

    with ThreadPoolExecutor(max_workers=1) as pool:
        mgr._base_fut = pool.submit(failed_load, mgr)
        mgr._base_fut.exception()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        mgr.release_transient_gpu_buffers()
        assert refs[0]() is None
        # Preserve the exception objects and their messages for the consumer.
        try:
            mgr.ensure_base_weights_loaded()
        except RuntimeError as error:
            assert str(error) == "loader allocation failed"
            assert isinstance(error.__context__, ValueError)
            assert str(error.__context__) == "underlying read failed"
            if explicit_cause:
                assert error.__cause__ is error.__context__
        else:
            pytest.fail("failed prefetch was not propagated")
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("requested", [36.0, 64.0])
def test_multi_gpu_budget_reserves_existing_weights_per_device(loader, monkeypatch, requested):
    gib = 1024 ** 3
    allocations = {}
    monkeypatch.setattr(loader, "detect_system_memory_gb", lambda: (64.0, 600.0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: ((26 if dev.index == 0 else 32) * gib, 32 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev: (6 if dev.index == 0 else 0) * gib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev: (5 if dev.index == 0 else 0) * gib)

    def capture(**kwargs):
        allocations[kwargs["layer_idx"]] = kwargs
        return kwargs

    monkeypatch.setattr(loader, "TieredMoECacheManager", capture)
    loader.load_strata_coder_multi_gpu(
        {i: "unused" for i in range(4)}, total_vram_gb=requested,
        layer_vram_weights={0: 10, 1: 1, 2: 1, 3: 1},
    )
    gpu0 = sum(v["vram_hot_budget_gb"] for v in allocations.values() if v["device"] == "cuda:0")
    gpu1 = sum(v["vram_hot_budget_gb"] for v in allocations.values() if v["device"] == "cuda:1")
    assert gpu0 == pytest.approx(min(requested * .84 / 2, 27 - 32 * .16))
    assert gpu1 == pytest.approx(min(requested * .84 / 2, 32 * .84))
    assert gpu0 + gpu1 <= requested * .84
    assert allocations[0]["vram_hot_budget_gb"] / allocations[1]["vram_hot_budget_gb"] == pytest.approx(10)
