"""LUT transformer: its own wiring of the transformer, with LUT versions of chosen stages.

Files:
    host.py           the PARAMETERS block (what each stage does: LUTs, precision, KV cache), the
                      LUT model's own wiring and its entry points (start here)
    stages.py         what a toggle can say: LUTArch, Stage (one slot per technique), BASE,
                      KVStore, KV_BASE
    techniques.py     the precision techniques the slots call, one function each (decision D4,
                      Alpha's; pending until written)
    nonlinear.py      replaceable nonlinear functions inside a stage: softmax exp, norm rsqrt,
                      activation
    observer.py       calibration: per-site, per-layer, per-channel ranges, saved and loaded
    presets.py        FP16 / FP8 / INT8 / INT4 settings and the advisor's sweep as named presets
    inference_lut.py  LUT versions of baseline/inference.py functions, table-size and cost accounting
    train_lut.py      LUT versions of baseline/train.py functions (only for LUT-aware training)
    ops_registry.py   turns the toggles into the functions the model calls

Run from python/:  python -m LUT.host info
"""
