"""LUT transformer: its own wiring of the transformer, with LUT versions of chosen stages.

Files:
    host.py           the PARAMETERS block (what each stage does with LUTs), the LUT model's own
                      wiring and its entry points (start here)
    stages.py         what a toggle can say: LUTArch, Stage, BASE
    inference_lut.py  LUT versions of baseline/inference.py functions, table-size and cost accounting
    train_lut.py      LUT versions of baseline/train.py functions (only for LUT-aware training)
    ops_registry.py   turns the toggles into the functions the model calls

Run from python/:  python -m LUT.host info
"""
