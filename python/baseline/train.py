"""Training functions, non-LUT ("base") versions, written from mathematical base operations.

Training needs a backward pass, so each operation comes as a pair:
    <op>_forward(...)  -> (output, cache)   cache = whatever backward needs, saved explicitly
    <op>_backward(d_output, cache) -> gradients with respect to the inputs and parameters
Backpropagation is written by hand from the chain rule; autograd is not used here (tests use
it only to check these functions). Some forward math repeats inference.py on purpose: Alpha
asked for training and inference to live in separate files even where they overlap.

The LUT/quantized training versions (if training ever sees LUTs, for example
quantization-aware training) live in python/LUT/train_lut.py with the same signatures.

Shape names: B batch, T sequence length, D model width, H heads, Dh = D / H, F feed-forward
width, V vocabulary. N = B * T, the number of predicted tokens.

Status: the op bodies are written in window 1, each tested against autograd in
float64. training_loop() at the bottom already works and is used by baseline/host.py.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

Cache = dict[str, Any]


def _todo(name: str) -> NotImplementedError:
    return NotImplementedError(f"train.{name} is written in window 1")


# ----------------------------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------------------------

def embedding_forward(token_ids: Tensor, table: Tensor) -> tuple[Tensor, Cache]:
    """x[b, t] = table[token_ids[b, t]]. Saves token_ids and the table shape. Returns (B, T, D)."""
    raise _todo("embedding_forward")


def embedding_backward(dx: Tensor, cache: Cache) -> Tensor:
    """d_table[v] = sum of dx[b, t] over every (b, t) with token_ids[b, t] == v (scatter-add).

    Returns d_table (V, D). Tokens that never appear get zero gradient.
    """
    raise _todo("embedding_backward")


# ----------------------------------------------------------------------------------------------
# Linear layers and activations
# ----------------------------------------------------------------------------------------------

def linear_forward(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> tuple[Tensor, Cache]:
    """y = x @ W^T + b. x: (..., D_in), W: (D_out, D_in), b: (D_out,). Saves x and W."""
    raise _todo("linear_forward")


def linear_backward(dy: Tensor, cache: Cache) -> tuple[Tensor, Tensor, Tensor | None]:
    """Given dL/dy (..., D_out), return (dx, dW, db):

    dx = dy @ W                       (..., D_in)
    dW = sum over leading axes of dy^T @ x     (D_out, D_in)
    db = sum over leading axes of dy           (D_out,), or None if there was no bias
    """
    raise _todo("linear_backward")


def relu_forward(x: Tensor) -> tuple[Tensor, Cache]:
    """y = max(x, 0). Saves the mask x > 0."""
    raise _todo("relu_forward")


def relu_backward(dy: Tensor, cache: Cache) -> Tensor:
    """dx = dy where x > 0, else 0."""
    raise _todo("relu_backward")


def gelu_forward(x: Tensor) -> tuple[Tensor, Cache]:
    """GELU (tanh approximation), see inference.gelu. Saves x."""
    raise _todo("gelu_forward")


def gelu_backward(dy: Tensor, cache: Cache) -> Tensor:
    """dx = dy * d/dx GELU(x), differentiating the tanh approximation exactly."""
    raise _todo("gelu_backward")


# ----------------------------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------------------------

def layernorm_forward(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> tuple[Tensor, Cache]:
    """y = gamma * x_hat + beta with x_hat = (x - mean) / sqrt(var + eps) over the last axis.
    Saves x_hat, 1 / sqrt(var + eps) and gamma."""
    raise _todo("layernorm_forward")


def layernorm_backward(dy: Tensor, cache: Cache) -> tuple[Tensor, Tensor, Tensor]:
    """Return (dx, d_gamma, d_beta).

    d_gamma = sum(dy * x_hat), d_beta = sum(dy) over leading axes;
    dx = (1/sqrt(var+eps)) * (g - mean(g) - x_hat * mean(g * x_hat)) with g = dy * gamma,
    means taken over the last axis.
    """
    raise _todo("layernorm_backward")


def rmsnorm_forward(x: Tensor, gamma: Tensor, eps: float = 1e-6) -> tuple[Tensor, Cache]:
    """y = gamma * x / rms, rms = sqrt(mean(x^2) + eps). Saves x, rms and gamma."""
    raise _todo("rmsnorm_forward")


def rmsnorm_backward(dy: Tensor, cache: Cache) -> tuple[Tensor, Tensor]:
    """Return (dx, d_gamma). With g = dy * gamma: dx = g / rms - x * mean(g * x) / rms^3."""
    raise _todo("rmsnorm_backward")


# ----------------------------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------------------------

def attention_scores_forward(q: Tensor, k: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Cache]:
    """s = (q @ k^T) / sqrt(Dh) with masked entries set to -inf. q: (B, H, T, Dh), k: (B, H, S, Dh).
    Saves q, k and the mask. Returns (B, H, T, S)."""
    raise _todo("attention_scores_forward")


def attention_scores_backward(ds: Tensor, cache: Cache) -> tuple[Tensor, Tensor]:
    """Return (dq, dk): dq = ds @ k / sqrt(Dh), dk = ds^T @ q / sqrt(Dh), masked entries of ds zeroed."""
    raise _todo("attention_scores_backward")


def softmax_forward(scores: Tensor, dim: int = -1) -> tuple[Tensor, Cache]:
    """Stable softmax along `dim` (see inference.softmax). Saves the output probabilities p."""
    raise _todo("softmax_forward")


def softmax_backward(dp: Tensor, cache: Cache) -> Tensor:
    """ds = p * (dp - sum(dp * p, along dim, keepdim)). This is the softmax Jacobian applied to dp."""
    raise _todo("softmax_backward")


def attention_values_forward(probs: Tensor, v: Tensor) -> tuple[Tensor, Cache]:
    """out = probs @ v. probs: (B, H, T, S), v: (B, H, S, Dh). Saves probs and v."""
    raise _todo("attention_values_forward")


def attention_values_backward(d_out: Tensor, cache: Cache) -> tuple[Tensor, Tensor]:
    """Return (d_probs, dv): d_probs = d_out @ v^T, dv = probs^T @ d_out."""
    raise _todo("attention_values_backward")


def dropout_forward(x: Tensor, p: float, generator: torch.Generator | None = None) -> tuple[Tensor, Cache]:
    """Inverted dropout: zero each element with probability p and scale the rest by 1 / (1 - p).
    p = 0 returns x unchanged. Saves the keep mask. DECISION (window 1): use dropout at all?"""
    raise _todo("dropout_forward")


def dropout_backward(dy: Tensor, cache: Cache) -> Tensor:
    """dx = dy * keep_mask / (1 - p)."""
    raise _todo("dropout_backward")


# ----------------------------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------------------------

def cross_entropy_forward(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> tuple[Tensor, Cache]:
    """Mean next-token cross-entropy.

    logits: (B, T, V), targets: (B, T) integer ids. loss = mean over counted tokens of
    logsumexp(logits) - logits[target], computed with the max subtracted for stability.
    Tokens equal to ignore_index are not counted. Saves the softmax probabilities and targets.
    """
    raise _todo("cross_entropy_forward")


def cross_entropy_backward(cache: Cache) -> Tensor:
    """d_logits = (softmax(logits) - one_hot(targets)) / N_counted, zero for ignored tokens."""
    raise _todo("cross_entropy_backward")


# ----------------------------------------------------------------------------------------------
# Optimizer and schedule
# ----------------------------------------------------------------------------------------------

def clip_grad_norm(grads: dict[str, Tensor], max_norm: float) -> float:
    """Scale all gradients in place so their global L2 norm is at most max_norm. Returns the norm
    before clipping (log it: spikes show instability)."""
    raise _todo("clip_grad_norm")


def adamw_step(params: dict[str, Tensor], grads: dict[str, Tensor], state: dict[str, Any], *,
               lr: float, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
               weight_decay: float = 0.1) -> None:
    """One AdamW update, in place, for every parameter name in grads.

    m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2;  m_hat = m / (1 - b1^t);  v_hat = v / (1 - b2^t)
    p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)   (decoupled weight decay)
    state holds m, v per parameter and the step count t. DECISION (window 1): which parameters get
    weight decay (usually not norms, biases or embeddings).
    """
    raise _todo("adamw_step")


def lr_at(step: int, *, max_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0) -> float:
    """Linear warm-up to max_lr over warmup_steps, then cosine decay to min_lr at total_steps."""
    raise _todo("lr_at")


# ----------------------------------------------------------------------------------------------
# Training loop (works now)
# ----------------------------------------------------------------------------------------------

def training_loop(
    step_fn: Callable[[int, Any], dict[str, Any]],
    batches: Iterable[Any],
    *,
    num_steps: int,
    run: Any | None = None,
    log_every: int = 10,
    eval_fn: Callable[[int], dict[str, Any]] | None = None,
    eval_every: int | None = None,
    device: str = "cpu",
    desc: str = "train",
) -> dict[str, Any]:
    """Run num_steps training steps with a tqdm progress bar (rate and time remaining) and log them.

    step_fn(step, batch) does one forward, backward and optimizer update and returns a dict with
    at least "loss" and optionally "tokens" (tokens in the batch) and anything else to log.
    eval_fn(step) returns validation metrics; it runs every eval_every steps.
    run is a bench.runlog.Run (or None to skip logging). Returns a summary including the total
    wall-clock training time.
    """
    from tqdm import tqdm

    from bench.timing import Stopwatch

    batch_iter = iter(batches)
    tokens_seen = 0
    last_loss = float("nan")
    with Stopwatch(device) as total:
        progress = tqdm(range(num_steps), desc=desc, unit="step", dynamic_ncols=True)
        for step in progress:
            batch = next(batch_iter)
            start = time.perf_counter()
            metrics = {key: (float(value) if isinstance(value, (int, float, Tensor)) else value)
                       for key, value in step_fn(step, batch).items()}
            step_time = time.perf_counter() - start
            last_loss = metrics["loss"]
            tokens_seen += int(metrics.get("tokens", 0))
            progress.set_postfix(loss=f"{last_loss:.4f}")
            if run is not None and (step % log_every == 0 or step == num_steps - 1):
                run.log(kind="train", step=step, step_time_s=step_time, **metrics)
            if eval_fn is not None and eval_every and (step + 1) % eval_every == 0:
                eval_metrics = eval_fn(step)
                if run is not None:
                    run.log(kind="eval", step=step, **eval_metrics)
                progress.write(f"step {step + 1}: " + ", ".join(f"{k}={v:.4g}" for k, v in eval_metrics.items()))
    return {
        "train_time_s": total.elapsed_s,
        "steps": num_steps,
        "final_train_loss": last_loss,
        "tokens_seen": tokens_seen,
        "tokens_per_s": tokens_seen / total.elapsed_s if total.elapsed_s > 0 else None,
    }
