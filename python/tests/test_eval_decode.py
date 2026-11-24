"""bench/eval_decode.py: the decode-step protocol walks the same windows, steps and scored tokens as
the HLS inference host and bench/sweep.py measure_perplexity (no GPU, no model)."""

import json

import torch

from bench import check_hls_inference, eval_decode, sweep


def test_windows_steps_and_scored_tokens_match_the_hls_records():
    # the slice (50,212 token ids) and the full span (321,527): HLS host.windows / host.decode_steps
    for n, windows, steps in ((50_212, 392, 99_868), (321_527, 2_511, 640_296)):
        spans = list(eval_decode.windows(n, 256, 128))
        assert len(spans) == windows == sweep.count_windows(n, 256, 128)
        assert sum(e - b - 1 for b, e, _ in spans) == steps
        assert sum(e - f for _, e, f in spans) == n - 1                      # every token after the first, once
        scored = torch.cat([torch.arange(f, e) for _, e, f in spans])
        assert torch.equal(scored, torch.arange(1, n))


class _Echo:
    """A fake decoder whose logits at each step put all mass on (the fed token + 1) mod V, and that
    counts its cache, so the test can see which logits row predicts which token."""

    def __init__(self, vocab):
        self.vocab, self.length, self.fed = vocab, 0, []

    def reset(self):
        self.length = 0

    def step(self, token):
        self.length += 1
        self.fed.append(int(token))
        z = torch.full((1, self.vocab), -10.0)
        z[0, (int(token) + 1) % self.vocab] = 10.0
        return z


def test_decode_window_keeps_the_logits_of_the_scored_positions_only():
    vocab, n = 50, 600
    ids = torch.arange(n) % vocab                       # token t + 1 = token t + 1 (mod V): the echo is always right
    buf = torch.empty(256, vocab)
    dec = _Echo(vocab)
    for b, e, f in eval_decode.windows(n, 256, 128):
        dec.fed = []
        k = eval_decode.decode_window(dec, ids, b, e, f, buf)
        assert dec.fed == ids[b:e - 1].tolist()         # positions 0 .. L-2 from an empty cache
        assert k == e - f                               # one row per scored token
        assert torch.equal(buf[:k].argmax(-1), ids[f:e])


def test_the_hls_check_finds_the_reference_named_by_the_record_manifest(tmp_path):
    ref = tmp_path / "llama2.c-stories15M_tinystories_chars1200000_fp32_baseline.pt"
    ref.write_bytes(b"x")
    data = tmp_path / "vectors" / "tinystories_full_w256s128"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text(json.dumps({"source_reference": str(ref)}))
    summary = {"dataset": "tinystories_full_w256s128", "dataset_dir": str(data)}
    assert check_hls_inference.reference_of(summary) == ref
