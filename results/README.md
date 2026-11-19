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
| `results_long.csv` | The same numbers in long format for a spreadsheet: one line per row, dataset, benchmark, metric and context length, with `value` and `unit`, the repeats, device and dtype, the LUT preset and a compact string of its stage settings, the git commit, the run id and time, and the dataset file's sha256 |
| `status.json` | Rows done, elapsed and remaining time, updated after every row, for checking on a sweep from elsewhere |

`all_results.csv` in this folder collects `results_long.csv` from every sweep, appended after every row; a run id that is already in it is never added again. All of these are plain UTF-8 CSV with a header row and open directly in Excel (a pivot table on `row` or `model` against `metric` gives the comparison table). An `.xlsx` writer would need the openpyxl package, which is not installed.

Perplexity is measured on the same text for every model: the dataset's evaluation split with its document separators replaced by blank lines, cut to a fixed number of characters (not tokens) at a space. Token perplexity depends on the tokenizer, so models with different tokenizers are compared with `bits_per_byte` (total negative log-likelihood / (ln 2 x UTF-8 bytes)) and `word_perplexity` (exp(total NLL / words)); each result records the text's sha256, characters, bytes and words, and the window and stride.

Raw power samples (`power_samples*.json`) are gitignored; everything else here can be committed. The timing history the estimates come from is outside the repository, in `../LUT-Transformer-local/sweep-history.jsonl`.
