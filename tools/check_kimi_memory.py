"""Opt-in CUDA checks for a local Strata-Coder checkpoint.

Run from a source checkout with the runtime dependencies installed:
    python tools/check_kimi_memory.py --model /path/to/Kimi-K3.1-Pro-Coder
The checkpoint is not downloaded by this check.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch
from strata_coder.decoder import decode_sc_lut_matrix, sc_lut_batched_experts_mlp
from strata_coder.engine import KimiCoderEngine


def assert_clean(engine):
    counts = {
        "expert_stages": sum(len(m._gpu_stage_cache) for m in engine.layer_managers.values()),
        "base_stages": sum(len(m._base_gpu_stage) for m in engine.layer_managers.values()),
        "batch_owners": sum(len(m._last_batch_keepalive) for m in engine.layer_managers.values()),
    }
    assert not any(counts.values()), counts
    return counts


@torch.no_grad()
def check_experts(engine):
    """Compare fused kernels with reconstructed FP32 matrices on each used GPU.

    The selected weights are temporary, so this check does not seed the resident
    expert cache or change which base weights fit in its budget.
    """
    representatives = {}
    for manager in engine.layer_managers.values():
        representatives.setdefault(str(manager.device), manager)
    results = []
    for device, manager in representatives.items():
        with torch.cuda.device(manager.device):
            for tier in (1, 2, 3):
                expert = manager.expert_tiers.index(tier)
                raw = manager._read_expert_raw(expert, manager.device)
                lut, basis = manager.luts[tier], manager.q_bases[tier]
                bits = lut.shape[0].bit_length() - 1
                weights = {
                    projection: decode_sc_lut_matrix(
                        raw[projection + ".idx_packed"], raw[projection + ".scales"],
                        raw[projection + ".u_svd"], raw[projection + ".v_svd"],
                        lut, basis, bits=bits, out_dtype=torch.float32,
                    )
                    for projection in ("w1", "w3", "w2")
                }
                row = manager._make_ptr_row(expert, raw)
                for tokens in (1, 4, 7):
                    columns = torch.arange(3584, device=manager.device).float()
                    rows = torch.arange(tokens, device=manager.device).float()
                    z = torch.sin(columns[None, :] * .017 + rows[:, None] * .3) * .1
                    pointers = torch.tensor([row] * tokens, device=manager.device, dtype=torch.int64)
                    actual = sc_lut_batched_experts_mlp(
                        z, pointers, torch.ones(tokens, device=manager.device),
                        torch.arange(tokens, device=manager.device, dtype=torch.int64),
                        manager.luts_T_f32, manager.qb_T_f32, manager.qb_T_stack,
                    )
                    gate, up = z @ weights["w1"].T, z @ weights["w3"].T
                    hidden = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(up / 25))
                    reference = hidden @ weights["w2"].T
                    difference = actual - reference
                    relative = float(torch.linalg.vector_norm(difference) /
                                     torch.linalg.vector_norm(reference).clamp_min(1e-12))
                    result = {"device": device, "layer": manager.layer_idx,
                              "tier": tier, "expert": expert, "tokens": tokens,
                              "relative_l2": relative,
                              "max_abs_error": float(difference.abs().max()),
                              "finite": bool(torch.isfinite(actual).all())}
                    results.append(result)
                    assert result["finite"] and relative < 1e-3, result
                del raw, weights, actual, reference, difference, gate, up, hidden, pointers, z
            torch.cuda.synchronize(manager.device)
    assert_clean(engine)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ram-gb", type=float)
    parser.add_argument("--vram-gb", type=float)
    args = parser.parse_args()
    model = Path(args.model)
    if not model.is_dir():
        parser.error("--model must be an existing local checkpoint directory")
    required = [f"model-{i:05d}-of-000096.safetensors" for i in range(1, 97)]
    required += ["tiktoken.model", "tokenizer_config.json"]
    if any(not (model / name).is_file() for name in required):
        parser.error("the local checkpoint is incomplete")
    if not torch.cuda.is_available():
        parser.error("this opt-in check requires CUDA")
    engine = KimiCoderEngine(args.model, total_ram_gb=args.ram_gb, total_vram_gb=args.vram_gb)
    print(json.dumps({"expert_checks": check_experts(engine)}), flush=True)
    cases = [
        ("What is 2+2? Reply with only the number.", "4"),
        ("Как называется столица Франции? Ответь одним словом.", "Париж"),
        ("What is 2+2? Reply with only the number.", "4"),
    ]
    for prompt, expected in cases:
        answer = "".join(engine.generate_stream(prompt, max_new_tokens=8, temperature=0))
        assert expected.casefold() in answer.casefold(), (prompt, answer)
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        print(json.dumps({"prompt": prompt, "answer": answer,
                          "transients": assert_clean(engine),
                          "allocated": [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
