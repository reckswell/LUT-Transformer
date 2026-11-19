# Results

Every benchmark, training or evaluation run writes one directory under `results/runs/`, named `<UTC time>_<run name>`:

| File | Contents |
| --- | --- |
| `meta.json` | Run name, git commit (and whether the tree was dirty), command line, configs, which device did which task, host, CPU, GPU, driver, Python, PyTorch and CUDA versions |
| `metrics.jsonl` | One JSON object per logged step or measurement |
| `power.json` | Power and energy per source (CPU package RAPL, NVIDIA GPU), never summed across sources |
| `summary.json` | Final numbers for the run |

`summary.csv` in this folder gets one row per finished run, so runs can be compared in a spreadsheet. A number without a run id is not a result.
