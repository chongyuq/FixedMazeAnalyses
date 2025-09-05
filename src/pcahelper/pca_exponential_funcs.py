from __future__ import annotations

from typing import Optional, Tuple, Literal

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


def compute_ema(
    x: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Compute an exponential moving average (EMA) over the first dimension.

    If `t` is None:
        ema_0 = alpha * x_0
        ema_i = (1 - alpha) * ema_{i-1} + alpha * x_i

    If `t` is provided (must be non-decreasing):
        Let dt_i = t_i - t_{i-1} for i >= 1
        ema_0 = x_0
        ema_i = exp(-alpha * dt_i) * ema_{i-1} + x_i

    Note: `alpha` is ignored when `t` is provided (preserves original behavior).

    Args:
        x: Tensor of shape (n, d).
        t: Optional times of shape (n,) on the same device as `x`.
        alpha: Smoothing factor when `t` is None.

    Returns:
        Tensor of shape (n, d) with EMA per row.
    """
    if x.ndim != 2:
        raise ValueError(f"`x` must be 2D (n, d). Got shape {tuple(x.shape)}")

    n, _ = x.shape
    out = torch.empty_like(x)

    if t is None:
        # Discrete-time EMA with alpha
        out[0] = x[0]
        for i in range(1, n):
            out[i] = (1.0 - alpha) * out[i - 1] + x[i]
    else:
        if t.shape[0] != n:
            raise ValueError("`t` must have the same length as `x` (n).")
        if torch.any((t[1:] - t[:-1]) < 0):
            raise ValueError("`t` must be non-decreasing.")

        out[0] = x[0]
        dt = t[1:] - t[:-1]
        for i in range(1, n):
            decay = torch.exp(-alpha * dt[i - 1])
            out[i] = decay * out[i - 1] + x[i]
    return out


def make_decay_features(
    idx: torch.Tensor,
    start_idx: torch.Tensor,
    tot_obs: int,
    alpha: float = 0.1,
    history: bool = True,
    future: bool = True,
    t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build exponentially-decayed one-hot features per session, optionally forward (history)
    and/or backward (future), concatenated along the sample axis (history first, then future).

    Args:
        idx: (n,) integer indices of observations.
        start_idx: (s,) starting indices for sequences to be treated separately (0-based, ascending).
        tot_obs: total number of possible unique observations (one-hot dimension).
        alpha: EMA alpha when `t` is None.
        history: include forward-in-time EMA.
        future: include backward-in-time EMA (time-reversed EMA flipped back).
        t: optional (n,) times for each observation.

    Returns:
        Tensor of shape:
            (n, tot_obs) if exactly one of {history, future} is True,
            (2n, tot_obs) if both are True.
        Order is all-history rows, then all-future rows.
    """
    if idx.ndim != 1:
        raise ValueError("`idx` must be 1D.")
    if start_idx.ndim != 1:
        raise ValueError("`start_idx` must be 1D.")
    if t is not None and t.ndim != 1:
        raise ValueError("`t` must be 1D when provided.")

    device = idx.device
    n = idx.shape[0]

    # Build session boundaries (end-exclusive)
    ends = torch.cat([start_idx.to(device), torch.tensor([n], device=device)])

    # One-hot encode to float for EMA math
    state_one_hot = F.one_hot(idx.to(torch.int64), num_classes=tot_obs).to(torch.float32)

    history_chunks = []
    future_chunks = []

    for i in range(ends.shape[0] - 1):
        i_start, i_end = ends[i].item(), ends[i + 1].item()

        seg = state_one_hot[i_start:i_end]

        if history:
            if t is not None:
                seg_t = t[i_start:i_end]
                history_chunks.append(compute_ema(seg, t=seg_t, alpha=alpha))
            else:
                history_chunks.append(compute_ema(seg, alpha=alpha))

        if future:
            if t is not None:
                # backward time: last time - t, reversed
                seg_t_back = torch.flip(t[i_start:i_end], dims=(0,))
                seg_t_back = seg_t_back[0] - seg_t_back  # make non-decreasing from the end
                ema_rev = compute_ema(torch.flip(seg, dims=(0,)), t=seg_t_back, alpha=alpha)
            else:
                ema_rev = compute_ema(torch.flip(seg, dims=(0,)), alpha=alpha)
            future_chunks.append(torch.flip(ema_rev, dims=(0,)))

    outs = []
    if history:
        outs.append(torch.cat(history_chunks, dim=0))
    if future:
        outs.append(torch.cat(future_chunks, dim=0))

    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def _fit_pca_rows(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit PCA on rows of x using scikit-learn and return (components, explained_variance_ratio).

    Returns:
        v: (d, k) principal axes as columns (same as sklearn components_.T)
        e: (k,) explained variance ratio
    """
    device = x.device
    x_np = x.detach().cpu().numpy()

    model = PCA()  # default: all components up to min(n_samples, n_features)
    model.fit(x_np)

    v = torch.from_numpy(model.components_.T).to(device=device, dtype=torch.float32)
    e = torch.from_numpy(model.explained_variance_ratio_).to(device=device, dtype=torch.float32)
    return v, e


def fit_decay_pca_basis(
    idx: torch.Tensor,
    start_idx: torch.Tensor,
    tot_obs: int = 196,
    t: Optional[torch.Tensor] = None,
    alpha: float = 0.1,
    normalize: bool = True,
    history: bool = True,
    future: bool = True,
    combine: Literal["stack", "sum"] = "stack"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (history+future) exponentially-decayed one-hot features and run PCA.

    Note: When both history and future are included (default), there are 2n samples.

    Returns:
        v: (tot_obs, k) principal axes as columns
        e: (k,) explained variance ratio
    """
    assert combine in ("stack", "sum"), "`combine` must be 'stack' or 'sum'."
    if combine == "sum" and not (history and future):
        raise ValueError("`combine='sum'` requires both `history` and `future` to be True.")
    feats = make_decay_features(
        idx=idx,
        start_idx=start_idx,
        tot_obs=tot_obs,
        t=t,
        alpha=alpha,
        history=history,
        future=future,
    )
    if combine == "sum":
        n = idx.shape[0]
        feats = feats.reshape(2, n, tot_obs).sum(dim=0)
        # as both features contain the present, we need to zero one of them to avoid doubling the present
        feats[0, idx] -= 1.0
    if normalize:
        feats = F.normalize(feats, p=2, dim=-1)  # L2 normalize rows, ensuring each row has magnitude 1
    return _fit_pca_rows(feats)


