"""bench/memory.py: tensor placement, the LUT storage inventory (with a small fake LUT op placed on
the CPU and, when there is one, on the GPU), and the peak-memory breakdown."""

import functools

import pytest
import torch

from baseline import host as base
from bench import memory
from LUT.stages import LUTArch


def _fake_lut_softmax(scores, dim=-1, *, arch, tables=None):
    """Stands in for a LUT op: looks up exp() in tables["exp"] (here only so the op has a table)."""
    return torch.softmax(scores, dim)


def _ops_with(slot, n_layers=2):
    """A ModelOps with every stage a plain function, except softmax in layer 1 = slot."""
    plain = lambda *a, **k: None  # noqa: E731
    layers = [{"norm": plain, "softmax": plain} for _ in range(n_layers)]
    layers[1]["softmax"] = slot
    return base.ModelOps(embedding=plain, layers=layers, final_norm=plain, lm_head=plain)


ARCH = LUTArch("exp_table", inputs_per_table=1, bits_per_input=8, out_bits=16)


def test_all_base_ops_say_none_explicitly():
    cfg = base.TransformerConfig(vocab_size=50, context_length=16, d_model=8, n_heads=2, n_layers=2, d_ff=16)
    inv = memory.lut_inventory(base.base_ops(cfg, "inference"))
    assert inv["rows"] == [] and inv["lut_stages"] == 0 and inv["label"] == "none (all base ops)"
    assert inv["totals"]["lut_bytes_gpu_vram"] == 0 and inv["totals"]["lut_bytes_host_dram"] == 0


def test_a_lut_op_without_tables_is_listed_as_not_built():
    inv = memory.lut_inventory(_ops_with(functools.partial(_fake_lut_softmax, arch=ARCH)))
    (row,) = inv["rows"]
    assert row["stage"] == "softmax" and row["layer"] == 1 and row["place"].startswith("not materialized")
    assert row["planned_table_bits"] == 256 * 16 and inv["label"] == "1 LUT ops, tables not built yet"


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device"))])
def test_a_fake_lut_table_is_found_with_its_place_and_size(device):
    table = torch.zeros(256, dtype=torch.int16, device=device)              # 256 entries x 16 bits = 512 bytes
    slot = functools.partial(_fake_lut_softmax, arch=ARCH, tables={"exp": table})
    inv = memory.lut_inventory(_ops_with(slot))
    (row,) = inv["rows"]
    assert row["name"] == "tables['exp']" and row["shape"] == [256] and row["dtype"] == "int16" and row["bytes"] == 512
    place = "GPU VRAM" if device == "cuda" else "host DRAM"
    assert row["place"] == place and row["device"].startswith(device)
    key = "lut_bytes_gpu_vram" if device == "cuda" else "lut_bytes_host_dram"
    assert inv["totals"][key] == 512 and inv["totals"]["lut_tables"] == 1
    assert inv["label"] == f"{place}, 0.000512 MB"


def test_tables_held_in_a_closure_or_train_pair_are_found_too():
    table = torch.ones(16, 4)

    def lut_fn(x, *, arch):                      # a table captured by a closure, not passed as a keyword
        return table[x]
    slot = (functools.partial(lut_fn, arch=ARCH), lambda *a: None)          # train mode: (forward, backward)
    inv = memory.lut_inventory(_ops_with(slot))
    assert inv["totals"]["lut_bytes_host_dram"] == 16 * 4 * 4 and "closure table" in inv["rows"][0]["name"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_pinned_host_tables_are_marked():
    t = torch.zeros(64).pin_memory()
    assert memory.place_of(t) == {"device": "cpu", "place": "host DRAM", "pinned": True, "mmap_file": None}


def test_memory_mapped_tables_are_on_disk(tmp_path):
    torch.save({"t": torch.arange(1024, dtype=torch.int32)}, tmp_path / "t.pt")
    t = torch.load(tmp_path / "t.pt", mmap=True, weights_only=True)["t"]
    place = memory.place_of(t)
    assert place["place"] == "disk (memory-mapped)" and place["mmap_file"].endswith("t.pt")


def test_breakdown_splits_the_peak():
    phase = {"torch_peak_allocated_bytes": 100, "torch_peak_reserved_bytes": 120, "process_gpu_peak_bytes": 400}
    b = memory.breakdown(phase, weights_gpu=60, lut_gpu=5, kv_cache_gpu=10)
    assert b["activations_workspace_bytes"] == 25 and b["allocator_cached_bytes"] == 20
    assert b["context_and_allocator_bytes"] == 280


def test_phase_memory_on_the_cpu_records_host_memory():
    with memory.PhaseMemory("cpu") as m:
        x = torch.zeros(10**6)
        del x
    assert m.result["host_rss_peak_bytes"] > 0 and "torch_peak_allocated_bytes" not in m.result


def test_the_rope_table_is_listed_as_a_precomputed_table():
    from baseline import inference

    inference.clear_rope_tables()
    inference.rope_tables(1, 48, 10000.0, 7)
    rows = memory.precomputed_tables()
    assert [r["name"].split(":")[0] for r in rows] == ["rope cos", "rope sin"]
    assert all(r["kind"] == "precomputed table (not a LUT op)" and r["shape"] == [256, 24] and r["bytes"] == 256 * 24 * 4
               for r in rows)
    cfg = base.TransformerConfig(vocab_size=50, context_length=16, d_model=8, n_heads=2, n_layers=2, d_ff=16)
    inv = memory.with_precomputed(memory.lut_inventory(base.base_ops(cfg, "inference")), rows)
    assert inv["label"] == "no LUT ops; rope cos/sin table: host DRAM, 0.0492 MB"
    assert inv["totals"]["precomputed_bytes_host_dram"] == 2 * 256 * 24 * 4 and inv["totals"]["lut_bytes_host_dram"] == 0
    inference.clear_rope_tables()
