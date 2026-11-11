import gc
import threading
import unittest
import weakref
from unittest import mock

import torch

from strata_coder import engine


class EngineMemoryTests(unittest.TestCase):
    def test_chunked_moe_matches_unchunked_cpu(self):
        torch.manual_seed(42)
        prefix = 'language_model.model.layers.1.block_sparse_moe'
        weights = {
            'shared_experts.gate_proj.weight': torch.randn(3, 3),
            'shared_experts.up_proj.weight': torch.randn(3, 3),
            'shared_experts.down_proj.weight': torch.randn(3, 3),
            'gate.weight': torch.randn(16, 3),
            'gate.e_score_correction_bias': torch.randn(16),
            'routed_expert_down_proj.weight': torch.randn(3, 3),
            'routed_expert_up_proj.weight': torch.randn(3, 3),
            'routed_expert_norm.weight': torch.ones(3),
            'expert_tiers': torch.ones(16, dtype=torch.long),
        }
        for tier in (1, 2, 3):
            weights[f'sc_lut_tier{tier}'] = torch.ones(1)
            weights[f'sc_qbasis_tier{tier}'] = torch.ones(1)
        for expert in range(16):
            for proj in ('w1', 'w2', 'w3'):
                for suffix in ('idx_packed', 'scales', 'u_svd', 'v_svd'):
                    weights[f'experts.{expert}.{proj}.{suffix}'] = torch.tensor(float(expert + 1))

        class Weights:
            def get_tensor(self, key):
                return weights[key[len(prefix) + 1:]]

        x = torch.randn(11, 3)
        def expert(z, raw, *args, **kwargs):
            return z * raw['w1.scales']
        with mock.patch.object(engine, 'build_dotlut', lambda z, *a: z), \
             mock.patch.object(engine, 'sc_lut_expert_mlp', expert):
            expected = engine._gpu_moe_forward_chunk(Weights(), 1, x, torch.device('cpu'))
            actual = engine.gpu_moe_forward(Weights(), 1, x, torch.device('cpu'))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_failure_releases_all_managers_and_partial_cache(self):
        obj = engine.KimiCoderEngine.__new__(engine.KimiCoderEngine)
        obj._forward_lock = threading.RLock()
        managers = [mock.Mock(), mock.Mock()]
        obj.layer_managers = dict(enumerate(managers))
        cache = {1: {'partial': torch.ones(1)}}
        obj._forward_last_logits_impl = mock.Mock(side_effect=RuntimeError('failed'))
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            obj.forward_last_logits([1], cache)
        self.assertEqual(cache, {})
        for manager in managers:
            manager.release_transient_gpu_buffers.assert_called_once()

    def test_cleanup_failure_does_not_mask_forward_failure_or_skip_manager(self):
        obj = engine.KimiCoderEngine.__new__(engine.KimiCoderEngine)
        obj._forward_lock = threading.RLock()
        first, second = mock.Mock(), mock.Mock()
        first.release_transient_gpu_buffers.side_effect = RuntimeError('cleanup failed')
        obj.layer_managers = {1: first, 2: second}
        obj._forward_last_logits_impl = mock.Mock(side_effect=RuntimeError('forward failed'))
        cache = {1: {'partial': torch.ones(1)}}
        with self.assertLogs(engine.__name__, level='ERROR'):
            with self.assertRaisesRegex(RuntimeError, 'forward failed'):
                obj.forward_last_logits([1], cache)
        self.assertEqual(cache, {})
        second.release_transient_gpu_buffers.assert_called_once()

    def test_cleanup_failure_propagates_after_success_and_cleans_rest(self):
        obj = engine.KimiCoderEngine.__new__(engine.KimiCoderEngine)
        obj._forward_lock = threading.RLock()
        first, second = mock.Mock(), mock.Mock()
        first.release_transient_gpu_buffers.side_effect = RuntimeError('cleanup failed')
        obj.layer_managers = {1: first, 2: second}
        obj._forward_last_logits_impl = mock.Mock(return_value=torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            obj.forward_last_logits([1], {})
        second.release_transient_gpu_buffers.assert_called_once()

    def test_cleanup_actions_continue_after_error(self):
        close, remove = mock.Mock(), mock.Mock()
        def fail():
            raise RuntimeError('release failed')
        with self.assertRaisesRegex(RuntimeError, 'release failed'):
            engine._run_cleanup([fail, close, remove])
        close.assert_called_once()
        remove.assert_called_once()

    def test_failed_forward_and_cleanup_release_tensors_without_cyclic_gc(self):
        refs = []
        obj = engine.KimiCoderEngine.__new__(engine.KimiCoderEngine)
        obj._forward_lock = threading.RLock()

        class Manager:
            def release_transient_gpu_buffers(self):
                tensor = torch.ones(4)
                refs.append(weakref.ref(tensor))
                raise RuntimeError('cleanup failed')

        obj.layer_managers = {1: Manager()}

        def forward(ids, cache):
            tensor = torch.ones(4)
            refs.append(weakref.ref(tensor))
            if ids[0] == 1:
                raise RuntimeError('forward failed')
            return torch.zeros(1)

        obj._forward_last_logits_impl = forward
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with mock.patch.object(engine.logging, 'getLogger'):
                for code in (1, 1, 0, 0):
                    try:
                        obj.forward_last_logits([code], {})
                    except RuntimeError:
                        pass
                    else:
                        self.fail('expected failure')
                    self.assertTrue(all(ref() is None for ref in refs))
        finally:
            if was_enabled:
                gc.enable()

    def test_failed_layer_releases_tensors_without_cyclic_gc(self):
        refs = []
        obj = engine.KimiCoderEngine.__new__(engine.KimiCoderEngine)
        obj._forward_lock = threading.RLock()
        obj.start_layer, obj.end_layer, obj.max_layers = 1, 2, 3
        obj.device = torch.device('cpu')

        class Manager:
            device = torch.device('cpu')
            _sf = None

            def release_transient_gpu_buffers(self):
                tensor = torch.ones(4)
                refs.append(weakref.ref(tensor))
                raise RuntimeError('cleanup failed')

        obj.layer_managers = {1: Manager()}

        def fail_fetch(*args):
            tensor = torch.ones(4)
            refs.append(weakref.ref(tensor))
            raise RuntimeError('layer failed')

        def forward(ids, cache):
            tensor = torch.ones(1, 3)
            refs.append(weakref.ref(tensor))
            return obj._run_layer_range(tensor, [], cache)

        obj._forward_last_logits_impl = forward
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with mock.patch.object(engine.logging, 'getLogger'), \
                 mock.patch.object(engine, '_fetch_weight', fail_fetch):
                for _ in range(3):
                    try:
                        obj.forward_last_logits([1], {})
                    except RuntimeError as exc:
                        self.assertEqual(str(exc), 'layer failed')
                    else:
                        self.fail('expected failure')
                    self.assertTrue(all(ref() is None for ref in refs))
        finally:
            if was_enabled:
                gc.enable()

    def test_layer_failure_clears_future_prefetch_and_attention_state(self):
        obj = engine.KimiCoderEngine.__new__(engine.KimiCoderEngine)
        obj.start_layer, obj.end_layer, obj.max_layers = 1, 2, 3
        obj.device = torch.device('cpu')
        current, future = mock.Mock(), mock.Mock()
        current.device = future.device = obj.device
        obj.layer_managers = {1: current, 2: future}
        cache = {1: {'partial': torch.ones(1)}}
        with mock.patch.object(engine, '_fetch_weight', side_effect=RuntimeError('layer failed')):
            with self.assertRaisesRegex(RuntimeError, 'layer failed'):
                obj._run_layer_range(torch.ones(1, 3), [], cache)
        self.assertEqual(cache, {})
        self.assertGreaterEqual(current.release_transient_gpu_buffers.call_count, 1)
        future.release_transient_gpu_buffers.assert_called_once()

    def test_concurrent_forwards_are_serialized(self):
        obj = engine.KimiCoderEngine.__new__(engine.KimiCoderEngine)
        obj._forward_lock = threading.RLock()
        obj.layer_managers = {}
        entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()
        results = []
        def forward(ids, cache):
            if ids[0] == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('timed out')
            else:
                second_entered.set()
            return ids[0]
        obj._forward_last_logits_impl = forward
        a = threading.Thread(target=lambda: results.append(obj.forward_last_logits([1], {})))
        b = threading.Thread(target=lambda: results.append(obj.forward_last_logits([2], {})))
        a.start()
        self.assertTrue(entered.wait(5))
        b.start()
        try:
            self.assertFalse(second_entered.wait(0.1))
        finally:
            release.set()
            a.join(5)
            b.join(5)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertEqual(results, [1, 2])


if __name__ == '__main__':
    unittest.main()
