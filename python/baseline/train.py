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

Every op is tested against autograd in float64 (python/tests/test_backward_against_autograd.py).
The model-level forward_train / backward that chain them live in baseline/host.py.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

Cache = dict[str, Any]


# ----------------------------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------------------------

def embedding_forward(token_ids: Tensor, table: Tensor) -> tuple[Tensor, Cache]:
    """x[b, t] = table[token_ids[b, t]]. Saves token_ids and the table shape. Returns (B, T, D)."""
    return table[token_ids], {"token_ids": token_ids, "table_shape": table.shape}


def embedding_backward(dx: Tensor, cache: Cache) -> Tensor:
    """d_table[v] = sum of dx[b, t] over every (b, t) with token_ids[b, t] == v (scatter-add).

    Returns d_table (V, D). Tokens that never appear get zero gradient. A token that appears
    several times collects the sum of its rows, because each appearance used the same row.
    """
    v, d = cache["table_shape"]
    d_table = torch.zeros(v, d, device=dx.device, dtype=dx.dtype)
    # index_add_ adds row r of the source into row ids[r] of d_table, summing repeats.
    return d_table.index_add_(0, cache["token_ids"].reshape(-1), dx.reshape(-1, d))


# ----------------------------------------------------------------------------------------------
# Linear layers and activations
# ----------------------------------------------------------------------------------------------

def linear_forward(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> tuple[Tensor, Cache]:
    """y = x @ W^T + b. x: (..., D_in), W: (D_out, D_in), b: (D_out,). Saves x and W."""
    y = x @ weight.transpose(0, 1)
    if bias is not None:
        y = y + bias
    return y, {"x": x, "weight": weight, "has_bias": bias is not None}


def linear_backward(dy: Tensor, cache: Cache) -> tuple[Tensor, Tensor, Tensor | None]:
    """Given dL/dy (..., D_out), return (dx, dW, db):

    dx = dy @ W                       (..., D_in)
    dW = sum over leading axes of dy^T @ x     (D_out, D_in)
    db = sum over leading axes of dy           (D_out,), or None if there was no bias

    From y[n, o] = sum_i x[n, i] W[o, i] + b[o]: dy[n, o] reaches x[n, i] through W[o, i]
    (so dx = dy @ W), reaches W[o, i] through x[n, i] for every row n (so dW sums dy^T x over
    rows), and reaches b[o] directly for every row (so db sums dy over rows).
    """
    x, weight = cache["x"], cache["weight"]
    dx = dy @ weight
    dy_rows = dy.reshape(-1, dy.shape[-1])                    # (N, D_out), N = all leading axes
    x_rows = x.reshape(-1, x.shape[-1])                       # (N, D_in)
    dw = dy_rows.transpose(0, 1) @ x_rows                     # (D_out, D_in)
    db = dy_rows.sum(dim=0) if cache["has_bias"] else None
    return dx, dw, db


def relu_forward(x: Tensor) -> tuple[Tensor, Cache]:
    """y = max(x, 0). Saves the mask x > 0."""
    mask = x > 0
    return torch.where(mask, x, torch.zeros_like(x)), {"mask": mask}


def relu_backward(dy: Tensor, cache: Cache) -> Tensor:
    """dx = dy where x > 0, else 0. (The derivative at exactly 0 is taken as 0, as torch does.)"""
    return torch.where(cache["mask"], dy, torch.zeros_like(dy))


_GELU_C = math.sqrt(2.0 / math.pi)
_GELU_A = 0.044715


def gelu_forward(x: Tensor) -> tuple[Tensor, Cache]:
    """GELU (tanh approximation), see inference.gelu. Saves x and tanh(u)."""
    tanh_u = torch.tanh(_GELU_C * (x + _GELU_A * x * x * x))
    return 0.5 * x * (1.0 + tanh_u), {"x": x, "tanh_u": tanh_u}


def gelu_backward(dy: Tensor, cache: Cache) -> Tensor:
    """dx = dy * d/dx GELU(x), differentiating the tanh approximation exactly.

    With u = c (x + a x^3) and y = 0.5 x (1 + tanh u), the product rule gives
        dy/dx = 0.5 (1 + tanh u) + 0.5 x (1 - tanh^2 u) c (1 + 3 a x^2)
    since d tanh(u)/du = 1 - tanh^2 u and du/dx = c (1 + 3 a x^2).
    """
    x, t = cache["x"], cache["tanh_u"]
    du_dx = _GELU_C * (1.0 + 3.0 * _GELU_A * x * x)
    return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx)


def silu_forward(x: Tensor) -> tuple[Tensor, Cache]:
    """SiLU: y = x / (1 + exp(-x)) = x * sigmoid(x) (see inference.silu). Saves x and sigmoid(x)."""
    e = torch.exp(-x)
    return x / (1.0 + e), {"x": x, "sig": 1.0 / (1.0 + e)}


def silu_backward(dy: Tensor, cache: Cache) -> Tensor:
    """dx = dy * s (1 + x (1 - s)), with s = sigmoid(x).

    Product rule on y = x s(x), using s'(x) = s (1 - s):  dy/dx = s + x s (1 - s) = s (1 + x (1 - s)).
    """
    x, s = cache["x"], cache["sig"]
    return dy * (s * (1.0 + x * (1.0 - s)))


def glu_gate_forward(a: Tensor, b: Tensor) -> tuple[Tensor, Cache]:
    """y = a * b, elementwise (the GLU gate, see inference.glu_gate). Saves a and b."""
    return a * b, {"a": a, "b": b}


def glu_gate_backward(dy: Tensor, cache: Cache) -> tuple[Tensor, Tensor]:
    """Return (da, db) = (dy * b, dy * a): each operand's gradient is dy times the other operand."""
    return dy * cache["b"], dy * cache["a"]


# ----------------------------------------------------------------------------------------------
# Rotary positions
# ----------------------------------------------------------------------------------------------

def rope_forward(x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Cache]:
    """Rotate each column pair (i, i + Dh/2) of x (B, H, T, Dh) by its position's angle (see
    inference.rope): y1 = x1 cos - x2 sin, y2 = x2 cos + x1 sin. Saves cos and sin (T, Dh/2)."""
    mid = x.shape[-1] // 2                                     # Dh/2, where the second half starts
    x1, x2 = x[..., :mid], x[..., mid:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1), {"cos": cos, "sin": sin}


def rope_backward(dy: Tensor, cache: Cache) -> Tensor:
    """dx = the inverse rotation of dy (angle -p w_i), because a rotation is orthogonal:
        dx1 = dy1 cos + dy2 sin
        dx2 = dy2 cos - dy1 sin
    From y1 = x1 c - x2 s and y2 = x2 c + x1 s: x1 reaches y1 through c and y2 through s, so
    dL/dx1 = dy1 c + dy2 s; x2 reaches y1 through -s and y2 through c, so dL/dx2 = -dy1 s + dy2 c.
    cos and sin are constants (a table of positions), so they get no gradient.
    """
    cos, sin = cache["cos"], cache["sin"]
    mid = dy.shape[-1] // 2
    dy1, dy2 = dy[..., :mid], dy[..., mid:]
    return torch.cat((dy1 * cos + dy2 * sin, dy2 * cos - dy1 * sin), dim=-1)


# ----------------------------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------------------------

def _sum_leading(t: Tensor) -> Tensor:
    """Sum over every axis but the last: (..., D) -> (D,). Parameter gradients sum over all rows."""
    return t.reshape(-1, t.shape[-1]).sum(dim=0)


def layernorm_forward(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> tuple[Tensor, Cache]:
    """y = gamma * x_hat + beta with x_hat = (x - mean) / sqrt(var + eps) over the last axis.
    Saves x_hat, 1 / sqrt(var + eps) and gamma."""
    d = x.shape[-1]
    mu = x.sum(dim=-1, keepdim=True) / d
    centered = x - mu
    var = (centered * centered).sum(dim=-1, keepdim=True) / d
    rstd = 1.0 / torch.sqrt(var + eps)                        # (..., 1)
    x_hat = centered * rstd
    return gamma * x_hat + beta, {"x_hat": x_hat, "rstd": rstd, "gamma": gamma}


def layernorm_backward(dy: Tensor, cache: Cache) -> tuple[Tensor, Tensor, Tensor]:
    """Return (dx, d_gamma, d_beta).

    d_gamma = sum(dy * x_hat), d_beta = sum(dy) over leading axes;
    dx = (1/sqrt(var+eps)) * (g - mean(g) - x_hat * mean(g * x_hat)) with g = dy * gamma,
    means taken over the last axis.

    Why dx has three terms: x_hat depends on x directly, through the mean (every x_i shifts mu,
    hence the - mean(g) term), and through the variance (every x_i changes var, hence the
    - x_hat * mean(g * x_hat) term).
    """
    x_hat, rstd, gamma = cache["x_hat"], cache["rstd"], cache["gamma"]
    d = x_hat.shape[-1]
    g = dy * gamma
    mean_g = g.sum(dim=-1, keepdim=True) / d
    mean_gx = (g * x_hat).sum(dim=-1, keepdim=True) / d
    dx = rstd * (g - mean_g - x_hat * mean_gx)
    return dx, _sum_leading(dy * x_hat), _sum_leading(dy)


def rmsnorm_forward(x: Tensor, gamma: Tensor, eps: float = 1e-6) -> tuple[Tensor, Cache]:
    """y = gamma * x / rms, rms = sqrt(mean(x^2) + eps). Saves x, rms and gamma."""
    d = x.shape[-1]
    rms = torch.sqrt((x * x).sum(dim=-1, keepdim=True) / d + eps)   # (..., 1)
    return gamma * (x / rms), {"x": x, "rms": rms, "gamma": gamma}


def rmsnorm_backward(dy: Tensor, cache: Cache) -> tuple[Tensor, Tensor]:
    """Return (dx, d_gamma). With g = dy * gamma: dx = g / rms - x * mean(g * x) / rms^3.

    The second term is the path through rms: d rms / d x_i = x_i / (D * rms).
    """
    x, rms, gamma = cache["x"], cache["rms"], cache["gamma"]
    d = x.shape[-1]
    g = dy * gamma
    mean_gx = (g * x).sum(dim=-1, keepdim=True) / d
    dx = g / rms - x * mean_gx / (rms * rms * rms)
    return dx, _sum_leading(dy * (x / rms))


# ----------------------------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------------------------

def attention_scores_forward(q: Tensor, k: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Cache]:
    """s = (q @ k^T) / sqrt(Dh) with masked entries set to -inf. q: (B, H, T, Dh), k: (B, H, S, Dh).
    Saves q, k and the mask. Returns (B, H, T, S)."""
    dh = q.shape[-1]
    s = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
    if mask is not None:
        s = torch.where(mask, s, torch.full_like(s, float("-inf")))
    return s, {"q": q, "k": k, "mask": mask}


def attention_scores_backward(ds: Tensor, cache: Cache) -> tuple[Tensor, Tensor]:
    """Return (dq, dk): dq = ds @ k / sqrt(Dh), dk = ds^T @ q / sqrt(Dh), masked entries of ds zeroed.

    A masked score is a constant (-inf), so no gradient flows through it to q or k.
    """
    q, k, mask = cache["q"], cache["k"], cache["mask"]
    if mask is not None:
        ds = torch.where(mask, ds, torch.zeros_like(ds))
    scale = 1.0 / math.sqrt(q.shape[-1])
    dq = (ds @ k) * scale                                     # (B, H, T, S) @ (B, H, S, Dh)
    dk = (ds.transpose(-2, -1) @ q) * scale                   # (B, H, S, T) @ (B, H, T, Dh)
    return dq, dk


def softmax_forward(scores: Tensor, dim: int = -1) -> tuple[Tensor, Cache]:
    """Stable softmax along `dim` (see inference.softmax). Saves the output probabilities p."""
    e = torch.exp(scores - scores.amax(dim=dim, keepdim=True))
    p = e / e.sum(dim=dim, keepdim=True)
    return p, {"p": p, "dim": dim}


def softmax_backward(dp: Tensor, cache: Cache) -> Tensor:
    """ds = p * (dp - sum(dp * p, along dim, keepdim)). This is the softmax Jacobian applied to dp.

    The Jacobian is dp_i/ds_j = p_i (delta_ij - p_j), so ds_j = sum_i dp_i p_i (delta_ij - p_j)
    = p_j dp_j - p_j sum_i dp_i p_i, which is the formula above without building the matrix.
    """
    p, dim = cache["p"], cache["dim"]
    return p * (dp - (dp * p).sum(dim=dim, keepdim=True))


def attention_values_forward(probs: Tensor, v: Tensor) -> tuple[Tensor, Cache]:
    """out = probs @ v. probs: (B, H, T, S), v: (B, H, S, Dh). Saves probs and v."""
    return probs @ v, {"probs": probs, "v": v}


def attention_values_backward(d_out: Tensor, cache: Cache) -> tuple[Tensor, Tensor]:
    """Return (d_probs, dv): d_probs = d_out @ v^T, dv = probs^T @ d_out."""
    probs, v = cache["probs"], cache["v"]
    return d_out @ v.transpose(-2, -1), probs.transpose(-2, -1) @ d_out


def dropout_forward(x: Tensor, p: float, generator: torch.Generator | None = None) -> tuple[Tensor, Cache]:
    """Inverted dropout: zero each element with probability p and scale the rest by 1 / (1 - p).
    p = 0 returns x unchanged. Saves the keep mask. DECISION (window 1): use dropout at all?
    (Every config sets dropout 0.0 for now.)"""
    if p == 0.0:
        return x, {"keep": None, "p": p}
    keep = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) >= p
    return torch.where(keep, x / (1.0 - p), torch.zeros_like(x)), {"keep": keep, "p": p}


def dropout_backward(dy: Tensor, cache: Cache) -> Tensor:
    """dx = dy * keep_mask / (1 - p)."""
    if cache["keep"] is None:
        return dy
    return torch.where(cache["keep"], dy / (1.0 - cache["p"]), torch.zeros_like(dy))


# ----------------------------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------------------------

def cross_entropy_forward(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> tuple[Tensor, Cache]:
    """Mean next-token cross-entropy.

    logits: (B, T, V), targets: (B, T) integer ids. loss = mean over counted tokens of
    logsumexp(logits) - logits[target], computed with the max subtracted for stability.
    Tokens equal to ignore_index are not counted. Saves the softmax probabilities and targets.

    logsumexp(z) = m + log(sum(exp(z - m))) with m = max(z): the same identity as the stable
    softmax, so no exp() overflows.
    """
    v = logits.shape[-1]
    z = logits.reshape(-1, v)                                 # (N, V)
    t = targets.reshape(-1)                                   # (N,)
    counted = t != ignore_index                               # (N,) which tokens count
    safe_t = torch.where(counted, t, torch.zeros_like(t))     # any valid index for ignored rows
    m = z.amax(dim=-1, keepdim=True)
    lse = m + torch.log(torch.exp(z - m).sum(dim=-1, keepdim=True))   # (N, 1)
    target_logit = z.gather(1, safe_t[:, None])               # (N, 1) logits[n, t[n]]
    per_token = (lse - target_logit).squeeze(1)
    n_counted = counted.sum()
    loss = torch.where(counted, per_token, torch.zeros_like(per_token)).sum() / n_counted
    probs = torch.exp(z - lse)                                # softmax(z), reused by backward
    return loss, {"probs": probs, "targets": safe_t, "counted": counted, "n_counted": n_counted,
                  "shape": logits.shape}


def cross_entropy_backward(cache: Cache) -> Tensor:
    """d_logits = (softmax(logits) - one_hot(targets)) / N_counted, zero for ignored tokens.

    d/dz_v [logsumexp(z) - z_t] = softmax(z)_v - [v == t]; the mean divides by N_counted.
    """
    probs, t, counted = cache["probs"], cache["targets"], cache["counted"]
    d = probs.clone()
    d[torch.arange(d.shape[0], device=d.device), t] -= 1.0    # subtract the one-hot target
    d = d * (counted[:, None].to(d.dtype) / cache["n_counted"])
    return d.reshape(cache["shape"])


# ----------------------------------------------------------------------------------------------
# Optimizer and schedule
# ----------------------------------------------------------------------------------------------

def clip_grad_norm(grads: dict[str, Tensor], max_norm: float) -> float:
    """Scale all gradients in place so their global L2 norm is at most max_norm. Returns the norm
    before clipping (log it: spikes show instability).

    norm = sqrt(sum over every gradient entry of g^2). If norm > max_norm, every gradient is
    multiplied by max_norm / (norm + 1e-6), as torch.nn.utils.clip_grad_norm_ does.
    """
    total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    coef = max_norm / (total + 1e-6)
    if coef < 1.0:
        for g in grads.values():
            g.mul_(coef)
    return total


def default_decay(name: str, param: Tensor) -> bool:
    """Which parameters get weight decay: matrices and embeddings (2 or more axes), not biases or
    norm gains (1 axis). DECISION: this is the usual choice (GPT-2, nanoGPT); confirm with Alpha."""
    return param.dim() >= 2


def adamw_step(params: dict[str, Tensor], grads: dict[str, Tensor], state: dict[str, Any], *,
               lr: float, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
               weight_decay: float = 0.1, decay: Callable[[str, Tensor], bool] = default_decay) -> None:
    """One AdamW update, in place, for every parameter name in grads.

    m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2;  m_hat = m / (1 - b1^t);  v_hat = v / (1 - b2^t)
    p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)   (decoupled weight decay)
    state holds m, v per parameter and the step count t. decay(name, p) says which parameters
    get weight decay (default_decay: matrices and embeddings only).

    m is a running mean of the gradient and v of its square; dividing by (1 - b^t) undoes their
    bias toward the zero they start at. "Decoupled" means the decay shrinks p directly instead of
    being added to the gradient, so it is not rescaled by 1 / sqrt(v_hat).
    """
    b1, b2 = betas
    state["t"] = state.get("t", 0) + 1
    t = state["t"]
    m_all, v_all = state.setdefault("m", {}), state.setdefault("v", {})
    for name, g in grads.items():
        p = params[name]
        m = m_all.setdefault(name, torch.zeros_like(p))
        v = v_all.setdefault(name, torch.zeros_like(p))
        m.mul_(b1).add_((1 - b1) * g)
        v.mul_(b2).add_((1 - b2) * g * g)
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        update = m_hat / (torch.sqrt(v_hat) + eps)
        if weight_decay and decay(name, p):
            update = update + weight_decay * p
        p.sub_(lr * update)


def lr_at(step: int, *, max_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0) -> float:
    """Linear warm-up to max_lr over warmup_steps, then cosine decay to min_lr at total_steps.

    warm-up (step < warmup_steps):  lr = max_lr * (step + 1) / warmup_steps
    decay:  progress = (step - warmup_steps) / (total_steps - warmup_steps), capped at 1;
            lr = min_lr + 0.5 (max_lr - min_lr) (1 + cos(pi * progress))
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


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
    progress: bool | None = None,
) -> dict[str, Any]:
    """Run num_steps training steps with a tqdm progress bar (rate, elapsed and time remaining) and
    log them.

    step_fn(step, batch) does one forward, backward and optimizer update and returns a dict with
    at least "loss" and optionally "tokens" (tokens in the batch) and anything else to log.
    eval_fn(step) returns validation metrics; it runs every eval_every steps.
    run is a bench.runlog.Run (or None to skip logging). Returns a summary including the total
    wall-clock training time.

    progress: the bar (bench/progress.py, the same style as the benchmark sweep and the eval
    harness). None shows it in a terminal and hides it in pytest and logs; True or False forces it.
    The bar is updated after each step's stop timestamp, so it is not in step_time_s.
    """
    from bench.progress import progress_bar
    from bench.timing import Stopwatch

    batch_iter = iter(batches)
    tokens_seen = 0
    last_loss = float("nan")
    with Stopwatch(device) as total:
        progress_iter = progress_bar(range(num_steps), desc=desc, unit="step", show=progress)
        for step in progress_iter:
            batch = next(batch_iter)
            start = time.perf_counter()
            metrics = {key: (float(value) if isinstance(value, (int, float, Tensor)) else value)
                       for key, value in step_fn(step, batch).items()}
            step_time = time.perf_counter() - start
            last_loss = metrics["loss"]
            tokens_seen += int(metrics.get("tokens", 0))
            postfix = {"loss": f"{last_loss:.4f}"}
            if metrics.get("tokens") and step_time > 0:
                postfix["tok/s"] = f"{metrics['tokens'] / step_time:,.0f}"
            progress_iter.set_postfix(postfix, refresh=False)
            if run is not None and (step % log_every == 0 or step == num_steps - 1):
                run.log(kind="train", step=step, step_time_s=step_time, **metrics)
            if eval_fn is not None and eval_every and (step + 1) % eval_every == 0:
                eval_metrics = eval_fn(step)
                if run is not None:
                    run.log(kind="eval", step=step, **eval_metrics)
                progress_iter.write(f"step {step + 1}: " + ", ".join(f"{k}={v:.4g}" for k, v in eval_metrics.items()))
    return {
        "train_time_s": total.elapsed_s,
        "steps": num_steps,
        "final_train_loss": last_loss,
        "tokens_seen": tokens_seen,
        "tokens_per_s": tokens_seen / total.elapsed_s if total.elapsed_s > 0 else None,
    }
