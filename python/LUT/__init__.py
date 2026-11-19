"""LUT transformer: the baseline model with LUT versions of chosen stages.

Files:
    host.py           the LUT toggles (what each stage does with LUTs) and entry points (start here)
    inference_lut.py  LUT versions of baseline/inference.py functions, table-size and cost accounting
    train_lut.py      LUT versions of baseline/train.py functions (only for LUT-aware training)
    ops_registry.py   turns the toggles into the functions the model calls

Run from python/:  python -m LUT.host info
"""
