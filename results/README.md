# Results

Every benchmark, training or evaluation run writes one directory under `results/runs/`, named `<UTC time>_<run name>`:

| File | Contents |
| --- | --- |
| `meta.json` | Run name, git commit (and whether the tree was dirty), command line, configs, which device did which task, host, CPU, GPU, driver, Python, PyTorch and CUDA versions |
| `metrics.jsonl` | One JSON object per logged step or measurement |
| `power.json` | Power and energy per source (CPU package RAPL, NVIDIA GPU), never summed across sources |
| `summary.json` | Final numbers for the run |

`summary.csv` in this folder gets one row per finished run, so runs can be compared in a spreadsheet. A number without a run id is not a result.

A benchmark sweep (`python -m bench.sweep`, from `python/`) writes one run directory per CSV row, as above (`power_<phase>_T<context>.json` for each power window), and one directory per sweep under `sweeps/<UTC time>_<csv name>/`:

| File | Contents |
| --- | --- |
| `models.csv` | The sweep CSV as it was run |
| `plan.csv` | Each row's estimated time and where the estimate came from (timing history or a calibration run), and why a row didn't run (disabled, or it needs training first) |
| `summary.csv` | One line per row, benchmark and context length (speed) or dataset (perplexity): whether the model was trained before (`weights`), prefill and per-token decode latency (median, p10, p90), tokens per second, power and energy per token per phase, idle power; or perplexity with the tokens, window and stride it used; plus the run id, estimated and actual time |
| `status.json` | Rows done, elapsed and remaining time, updated after every row, for checking on a sweep from elsewhere |

Raw power samples (`power_samples*.json`) are gitignored; everything else here can be committed. The timing history the estimates come from is outside the repository, in `../LUT-Transformer-local/sweep-history.jsonl`.
