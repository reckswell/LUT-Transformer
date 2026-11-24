"""Modeled cost of each stage of one inference step, from the config's shapes alone (no run needed).

Estimated, not measured (agent.md 4): first-principles counts per stage, so they sit next to the
measured stage times. On a GPU at batch 1 the measured time of a small op is mostly kernel-launch
overhead, so time alone says little about what a stage would cost in hardware; these counts do.

    from bench import opcount
    costs = opcount.stage_costs(cfg, phase="decode", context=256)   # {stage: StageCost}, one layer
    costs["qkv_proj"].mul_adds                                        # for the whole step
    opcount.FORWARD_ORDER                                             # embedding ... lm_head

Hardware form (2026-11-24, the advisor's reference): what a datapath has to compute, not the
PyTorch code's literal operations. Where they differ:
    RMSNorm   one rsqrt per row, then 3 multiplies per element (square, scale by the rsqrt, gamma);
              the code divides every element by sqrt (an emulation pays D divides per row)
    softmax   one exp per score, one reciprocal per row, one multiply per score (normalize); the
              code divides every score
    SiLU      one evaluation per element (a table or a function unit); the code does exp + divide
    scores    the 1/sqrt(Dh) scale is folded into W_q (or is a shift), so it costs nothing here
A step processes n new tokens: n = T for a prefill of a T-token prompt, n = 1 for a decode step at
context T, where the new token is the T-th: it attends to T keys (T - 1 cached plus itself). keys =
T in both phases; prefill counts the dense T x T score matrix the code computes (a causal-aware
kernel needs T(T+1)/2). D = d_model, H = heads, Dh = D / H, F = d_ff, V = vocabulary.

Per stage and step (layer stages: one layer; the model has n_layers identical ones):
    mul_adds   multiplies (each fused with an accumulate where there is one)
    nonlinear  exp, rsqrt, reciprocal and activation-function evaluations (what a LUT or a special
               unit would serve)
    elements   values read and written as the code materializes them: weights, input and output
               activations, and the K/V reads (nothing reused across stages). bytes = elements x
               the width: 4 at FP32 (bytes), 1 at INT8 everywhere (bytes_int8)

    norm (2 calls)   per call: 3nD mul, n rsqrt; elements 2nD + D (x in, y out, gamma)
    qkv_proj         3nD^2 mul; elements 3(D^2 + 2nD)
    rope (2 calls)   q and k: 4 mul per pair, D/2 pairs each: 4nD mul; elements 2(2nD + n*Dh) (cos/sin)
    attn_scores      H*n*keys*Dh mul; elements nD + keys*D (K read) + H*n*keys (scores out)
    softmax          H*n*keys exp + H*n reciprocal, H*n*keys mul; elements 2*H*n*keys
    attn_values      H*n*keys*Dh mul; elements H*n*keys + keys*D (V read) + nD
    out_proj         nD^2 mul; elements D^2 + 2nD
    ffn_up           W1 and W3 (glu) or W1 (mlp): u*nDF mul; elements u(DF + nD + nF)
    ffn_act          SiLU or GELU: nF evaluations (ReLU: 0); elements 2nF
    ffn_gate         glu only: nF mul; elements 3nF
    ffn_down         nFD mul; elements FD + nF + nD
    embedding        n row reads, no arithmetic; elements 2nD
    final_norm       the positions that reach the head: 1 (prefill keeps the last one): 3D mul, 1 rsqrt
    lm_head          1 position: DV mul; elements VD + D + V
Not in any stage (the "unattributed" part of a timed step): split/merge heads, the KV-cache append,
residual adds, the rotary table (built once per step), the causal mask and the argmax.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

BYTES = 4      # float32
# Forward-pass order of the stages (embedding first, lm_head last): the order of every table.
FORWARD_ORDER = ("embedding", "norm", "qkv_proj", "rope", "attn_scores", "softmax", "attn_values", "out_proj",
                 "ffn_up", "ffn_act", "ffn_gate", "ffn_down", "final_norm", "lm_head")
MODEL_LEVEL = ("embedding", "final_norm", "lm_head")


@dataclass
class StageCost:
    stage: str
    calls: int          # calls per step (per layer for layer stages)
    tokens: int         # new tokens in the step (n)
    mul_adds: int
    nonlinear: int
    elements: int       # values read and written (weights, activations, K/V)

    @property
    def bytes(self) -> int:
        """FP32 bytes."""
        return BYTES * self.elements

    @property
    def bytes_int8(self) -> int:
        return self.elements

    def per_token(self) -> dict[str, float]:
        return {"mul_adds_per_token": self.mul_adds / self.tokens, "nonlinear_per_token": self.nonlinear / self.tokens,
                "bytes_per_token": self.bytes / self.tokens, "bytes_int8_per_token": self.bytes_int8 / self.tokens}

    def scaled(self, k: int) -> "StageCost":
        """The cost of k identical copies (k layers)."""
        return StageCost(self.stage, self.calls * k, self.tokens, self.mul_adds * k, self.nonlinear * k,
                         self.elements * k)

    def to_dict(self) -> dict[str, int | str]:
        return {**asdict(self), "bytes": self.bytes, "bytes_int8": self.bytes_int8}


def step_shape(phase: str, context: int) -> tuple[int, int]:
    """(n new tokens, keys per query): a prefill of `context` tokens, or the decode step that makes
    the context-th token (it attends to `context` keys)."""
    if phase == "prefill":
        return context, context
    if phase == "decode":
        return 1, context
    raise ValueError(f"phase must be prefill or decode, got {phase!r}")


def stage_costs(cfg, phase: str, context: int) -> dict[str, StageCost]:
    """{stage: StageCost} for one step; layer stages are for one layer. Stages the model doesn't run
    (rope without rotary positions, ffn_gate without a gated feed-forward) are left out."""
    n, keys = step_shape(phase, context)
    d, h, dh, f, v = cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.d_ff, cfg.vocab_size
    glu = cfg.ffn == "glu"
    up = 2 if glu else 1
    act_evals = 0 if cfg.activation == "relu" else n * f
    if cfg.norm == "rmsnorm":
        norm_mul, norm_nl, norm_el = 3 * n * d, n, 2 * n * d + d
    else:                            # LayerNorm, roughly: mean, variance, scale, gamma, beta
        norm_mul, norm_nl, norm_el = 4 * n * d, n, 2 * n * d + 2 * d
    scores = h * n * keys
    costs = {
        "embedding": StageCost("embedding", 1, n, 0, 0, 2 * n * d),
        "norm": StageCost("norm", 2, n, 2 * norm_mul, 2 * norm_nl, 2 * norm_el),
        "qkv_proj": StageCost("qkv_proj", 3, n, 3 * n * d * d, 0, 3 * (d * d + 2 * n * d)),
        "rope": StageCost("rope", 2, n, 4 * n * d, 0, 2 * (2 * n * d + n * dh)),
        "attn_scores": StageCost("attn_scores", 1, n, scores * dh, 0, n * d + keys * d + scores),
        "softmax": StageCost("softmax", 1, n, scores, scores + h * n, 2 * scores),
        "attn_values": StageCost("attn_values", 1, n, scores * dh, 0, scores + keys * d + n * d),
        "out_proj": StageCost("out_proj", 1, n, n * d * d, 0, d * d + 2 * n * d),
        "ffn_up": StageCost("ffn_up", up, n, up * n * d * f, 0, up * (d * f + n * d + n * f)),
        "ffn_act": StageCost("ffn_act", 1, n, 0, act_evals, 2 * n * f),
        "ffn_gate": StageCost("ffn_gate", 1, n, n * f, 0, 3 * n * f),
        "ffn_down": StageCost("ffn_down", 1, n, n * f * d, 0, f * d + n * f + n * d),
        "final_norm": StageCost("final_norm", 1, n, norm_mul // n, norm_nl // n, norm_el - (2 * n * d) + 2 * d),
        "lm_head": StageCost("lm_head", 1, n, d * v, 0, v * d + d + v),
    }
    if cfg.positional_encoding != "rope":
        del costs["rope"]
    if not glu:
        del costs["ffn_gate"]
    return costs


def model_costs(cfg, phase: str, context: int) -> dict[str, StageCost]:
    """{stage: StageCost} for the whole model in forward order: layer stages summed over the layers."""
    costs = stage_costs(cfg, phase, context)
    return {s: costs[s] if s in MODEL_LEVEL else costs[s].scaled(cfg.n_layers) for s in FORWARD_ORDER if s in costs}


def step_total(cfg, phase: str, context: int) -> StageCost:
    """The whole step: layer stages times n_layers, plus the model-level stages."""
    costs = model_costs(cfg, phase, context)
    n = next(iter(costs.values())).tokens
    total = StageCost("step", 0, n, 0, 0, 0)
    for c in costs.values():
        total.calls += c.calls
        total.mul_adds += c.mul_adds
        total.nonlinear += c.nonlinear
        total.elements += c.elements
    return total
