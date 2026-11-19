# LUT-Transformer

A low-resource, LUT-based transformer attention mechanism combined with input quantization, aiming for high accuracy with lower power and latency. The work starts with a from-scratch PyTorch transformer whose every function is written out as its mathematical base operations, then swaps in quantized and LUT-based versions stage by stage, and later ports the result to Vitis HLS for an Alveo U50.

Status: setup. The PyTorch functions are documented stubs; the baseline inference ops come first, then quantization, the LUT variants and the HLS port.

## Layout

| Path | What's there |
| --- | --- |
| `python/baseline/host.py` | The transformer definition (no LUTs) and the train, evaluate and generate entry points. Start reading here. |
| `python/baseline/train.py`, `inference.py` | Non-LUT functions for training (forward and hand-written backward) and inference, from base math |
| `python/baseline/reference_torch.py` | Ground-truth model from `torch.nn` for correctness and speed comparison |
| `python/LUT/host.py` | The LUT model: the `TOGGLES` block says, per stage, whether to use a LUT, which architecture, the table shape, which layers and the input quantization |
| `python/LUT/inference_lut.py`, `train_lut.py` | LUT versions with the same signatures as the baseline, plus table-cost accounting |
| `python/common/` | Fixed-point and integer quantization (`quant.py`), datasets (`data.py`) |
| `python/configs/` | Model shapes (`model/`) and device per task (`hw/`) |
| `python/bench/` | Timing, power telemetry (CPU RAPL, NVIDIA, XRT) and run logging |
| `python/tests/` | Each from-scratch function against PyTorch, and the LUT toggles |
| `results/` | One directory per run plus `summary.csv` |

## Quick start

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 5080 needs CUDA 12.8+
pip install -r requirements.txt
pytest                                   # from the repository root
cd python
python -m baseline.host info
python -m LUT.host info
python -m bench.run_benchmark --workload matmul --device cuda --size 4096 --power
```
