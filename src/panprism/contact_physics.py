"""
Physical 3D Chromatin Contact Operator (`P_phys`) & Borzoi/Flashzoi Unsquashing Utilities.

Computes diagonal-normalized physical contact probabilities from predicted
distance-normalized contact residual matrices:

    C_norm(i, j) = C(i, j) - median_{|u - v| = |i - j|} C(u, v)
    P_phys(i, j) = ((d_ij + 5000) / 5000)^(-0.87) * exp(clip(C_norm(i, j), -2.5, +2.5))

where `d_ij = |i - j| * bin_bp` (default `bin_bp = 2048 bp` for AlphaGenome 2D contact maps).
Negative `log(O/E)` residuals (`C_norm < 0`) are NEVER zero-clamped: they smoothly attenuate
physical contact probability below the `-0.87` polymer distance-decay prior while preserving
strict positivity `P_phys(i, j) > 0`.

Also provides exact, invertible `undo_squashed_scale` / `apply_squashed_scale` transforms
for `Borzoi-524kb` and `Flashzoi-524kb` (`sum_sqrt` coverage squashing) supporting both
NumPy arrays and PyTorch tensors without requiring PyTorch as a hard dependency.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - CPU/lightweight env without torch
    torch = None  # type: ignore

DEFAULT_POLYMER_DIST_OFFSET_BP: float = 5000.0
DEFAULT_POLYMER_EXPONENT: float = -0.87
DEFAULT_CNORM_CLIP: float = 2.5
DEFAULT_CONTACT_BIN_BP: float = 2048.0


def _is_torch_tensor(x: Any) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _symmetrize_and_rc_average(
    contact_map: Any,
    contact_map_rc: Optional[Any] = None,
) -> Any:
    """Average optional FWD + RC contact maps and enforce exact spatial transpose symmetry."""
    if _is_torch_tensor(contact_map):
        c = contact_map.float()
        if contact_map_rc is not None:
            c_rc = contact_map_rc.to(device=c.device, dtype=torch.float32)
            c = 0.5 * (c + torch.flip(c_rc, dims=(0, 1)))
        return 0.5 * (c + c.transpose(0, 1))

    c = np.asarray(contact_map, dtype=np.float32)
    if contact_map_rc is not None:
        c_rc = np.asarray(contact_map_rc, dtype=np.float32)
        c = 0.5 * (c + np.flip(c_rc, axis=(0, 1)))
    if c.ndim == 2:
        return 0.5 * (c + c.T)
    if c.ndim == 3:
        return 0.5 * (c + np.transpose(c, (1, 0, 2)))
    raise ValueError(f"Expected 2D (N, N) or 3D (N, N, C) contact_map, got shape {c.shape}")


def compute_diagonal_baseline(
    contact_map: Any,
    min_diag_samples: int = 1,
    min_std: float = 0.05,
) -> Tuple[Any, Any]:
    """
    Compute per-diagonal median and standard deviation of a symmetric contact map.

    Supports 2D `(N, N)` or 3D `(N, N, C)` NumPy arrays or PyTorch tensors.
    Returns `(diag_med, diag_std)` of shape `(N,)` for 2D inputs or `(N, C)` for 3D inputs.
    """
    if _is_torch_tensor(contact_map):
        c = _symmetrize_and_rc_average(contact_map)
        if c.ndim not in (2, 3) or c.shape[0] != c.shape[1]:
            raise ValueError(f"Expected square (N, N) or (N, N, C) tensor, got shape {tuple(c.shape)}")
        n = c.shape[0]
        out_shape = (n,) if c.ndim == 2 else (n, c.shape[2])
        med = torch.zeros(out_shape, dtype=torch.float32, device=c.device)
        std = torch.full(out_shape, float(min_std), dtype=torch.float32, device=c.device)
        for k in range(n):
            vals = torch.diagonal(c, offset=k, dim1=0, dim2=1)  # (N-k,) or (C, N-k)
            diag_len = vals.shape[-1]
            if k == 0 or diag_len >= min_diag_samples:
                sv = torch.sort(vals, dim=-1).values
                mid = diag_len // 2
                med[k] = sv[..., mid] if (diag_len % 2 == 1) else 0.5 * (sv[..., mid - 1] + sv[..., mid])
                s = torch.std(vals, dim=-1, correction=0) if diag_len > 1 else torch.zeros_like(med[k])
                std[k] = torch.clamp(s, min=float(min_std))
            else:
                med[k] = med[k - 1]
                std[k] = std[k - 1]
        return med, std

    c = _symmetrize_and_rc_average(contact_map)
    if c.ndim not in (2, 3) or c.shape[0] != c.shape[1]:
        raise ValueError(f"Expected square (N, N) or (N, N, C) array, got shape {c.shape}")
    n = c.shape[0]
    out_shape = (n,) if c.ndim == 2 else (n, c.shape[2])
    med = np.zeros(out_shape, dtype=np.float32)
    std = np.full(out_shape, float(min_std), dtype=np.float32)
    for k in range(n):
        vals = np.diagonal(c, offset=k, axis1=0, axis2=1)  # (N-k,) or (C, N-k)
        diag_len = vals.shape[-1]
        if k == 0 or diag_len >= min_diag_samples:
            med[k] = np.median(vals, axis=-1)
            s = np.std(vals, axis=-1) if diag_len > 1 else np.zeros_like(med[k])
            std[k] = np.maximum(float(min_std), s)
        else:
            med[k] = med[k - 1]
            std[k] = std[k - 1]
    return med, std


def compute_physical_3d_contact(
    contact_map: Any,
    bin_bp: float = DEFAULT_CONTACT_BIN_BP,
    contact_map_rc: Optional[Any] = None,
    diag_medians: Optional[Any] = None,
    diag_stds: Optional[Any] = None,
    dist_offset_bp: float = DEFAULT_POLYMER_DIST_OFFSET_BP,
    polymer_exponent: float = DEFAULT_POLYMER_EXPONENT,
    clip_c_norm: float = DEFAULT_CNORM_CLIP,
    min_diag_samples: int = 1,
    min_std: float = 0.05,
    return_z_score: bool = False,
) -> Tuple[Any, ...]:
    """
    Transform raw signed `log(O/E)` contact map(s) into diagonal-normalized physical 3D
    contact probabilities:

        P_phys(i, j) = ((|i - j| * bin_bp + dist_offset_bp) / dist_offset_bp)^polymer_exponent
                       * exp(clip(C(i, j) - diag_median(|i - j|), -clip_c_norm, +clip_c_norm))

    Works on both 2D `(N, N)` and 3D `(N, N, C)` NumPy arrays and PyTorch tensors.

    Returns:
      - `(p_phys, c_norm, diag_medians)` when `return_z_score=False` (default)
      - `(p_phys, c_norm, z_3d, diag_medians, diag_stds)` when `return_z_score=True`
    """
    c = _symmetrize_and_rc_average(contact_map, contact_map_rc)
    if c.ndim not in (2, 3) or c.shape[0] != c.shape[1]:
        raise ValueError(f"Expected square (N, N) or (N, N, C) contact_map, got shape {tuple(c.shape)}")
    n = c.shape[0]

    if diag_medians is None or (return_z_score and diag_stds is None):
        comp_med, comp_std = compute_diagonal_baseline(
            c, min_diag_samples=min_diag_samples, min_std=min_std
        )
        if diag_medians is None:
            diag_medians = comp_med
        if diag_stds is None:
            diag_stds = comp_std

    if _is_torch_tensor(c):
        idx = torch.arange(n, device=c.device)
        dist_bins = torch.abs(idx[:, None] - idx[None, :])
        dm = diag_medians.to(device=c.device, dtype=torch.float32) if _is_torch_tensor(diag_medians) else torch.as_tensor(diag_medians, dtype=torch.float32, device=c.device)
        c_norm = c - dm[dist_bins]
        d_bp = dist_bins.float() * float(bin_bp)
        p_decay = torch.pow((d_bp + float(dist_offset_bp)) / float(dist_offset_bp), float(polymer_exponent))
        if c.ndim == 3:
            p_decay = p_decay.unsqueeze(-1)
        p_phys = p_decay * torch.exp(torch.clamp(c_norm, -float(clip_c_norm), float(clip_c_norm)))
        if return_z_score:
            ds = diag_stds.to(device=c.device, dtype=torch.float32) if _is_torch_tensor(diag_stds) else torch.as_tensor(diag_stds, dtype=torch.float32, device=c.device)
            z_3d = c_norm / torch.clamp(ds[dist_bins], min=float(min_std))
            return p_phys, c_norm, z_3d, dm, ds
        return p_phys, c_norm, dm

    idx = np.arange(n, dtype=np.int64)
    dist_bins = np.abs(idx[:, None] - idx[None, :])
    dm_np = np.asarray(diag_medians, dtype=np.float32)
    c_norm = c - dm_np[dist_bins]
    d_bp = dist_bins.astype(np.float32) * float(bin_bp)
    p_decay = np.power((d_bp + float(dist_offset_bp)) / float(dist_offset_bp), float(polymer_exponent)).astype(np.float32)
    if c.ndim == 3:
        p_decay = p_decay[:, :, None]
    p_phys = (p_decay * np.exp(np.clip(c_norm, -float(clip_c_norm), float(clip_c_norm)))).astype(np.float32)
    if return_z_score:
        ds_np = np.asarray(diag_stds, dtype=np.float32)
        z_3d = (c_norm / np.maximum(float(min_std), ds_np[dist_bins])).astype(np.float32)
        return p_phys, c_norm, z_3d, dm_np, ds_np
    return p_phys, c_norm, dm_np


def physical_contact_from_raw(
    c_raw: float,
    bin_sep: int,
    dist_bp: float,
    diag_med: np.ndarray,
    diag_std: Optional[np.ndarray] = None,
    c_idx: Optional[int] = None,
    dist_offset_bp: float = DEFAULT_POLYMER_DIST_OFFSET_BP,
    polymer_exponent: float = DEFAULT_POLYMER_EXPONENT,
    clip_c_norm: float = DEFAULT_CNORM_CLIP,
    min_std: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Compute scalar `(c_norm, z_3d, p_phys)` for a single bin pair `(i, j)` with genomic
    separation `dist_bp` and diagonal baseline `(diag_med, diag_std)`.
    """
    b = min(len(diag_med) - 1, max(0, int(bin_sep)))
    med_val = float(diag_med[b] if c_idx is None else diag_med[b, c_idx])
    if diag_std is not None:
        std_val = float(diag_std[b] if c_idx is None else diag_std[b, c_idx])
    else:
        std_val = 1.0
    c_norm = float(c_raw - med_val)
    z_3d = float(c_norm / max(float(min_std), std_val))
    p_decay = ((max(0.0, float(dist_bp)) + float(dist_offset_bp)) / float(dist_offset_bp)) ** float(polymer_exponent)
    p_phys = float(p_decay * math.exp(max(-float(clip_c_norm), min(float(clip_c_norm), c_norm))))
    return c_norm, z_3d, p_phys


def apply_squashed_scale(
    y: Union[float, np.ndarray, Any],
    scale: float = 0.30,
    clip_soft: Optional[float] = 384.0,
    power: float = 0.75,
    old_transform: bool = True,
) -> Union[float, np.ndarray, Any]:
    """
    Apply Borzoi / Flashzoi forward `sum_sqrt` squashing transform (exact inverse of
    `undo_squashed_scale` for non-negative coverage `y >= 0`).

    - `old_transform=True` (default, standard Borzoi/Flashzoi human RNA-seq `targets_human.txt`):
        1. `u = max(y, 0)^power`
        2. `u = where(u > clip_soft, clip_soft + sqrt(u - clip_soft), u)`
        3. `x = u * scale`
    - `old_transform=False` (Baskerville new transform):
        1. `u = (max(y, 0) * scale + 1)^power - 1`
        2. `x = where(u > clip_soft, clip_soft - 1 + sqrt(u - clip_soft + 1), u)`
    """
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    if power <= 0.0:
        raise ValueError(f"power must be positive, got {power}")

    if _is_torch_tensor(y):
        v = torch.clamp(y.float(), min=0.0)
        if old_transform:
            u = torch.pow(v, float(power))
            if clip_soft is not None:
                cs = float(clip_soft)
                u = torch.where(u > cs, cs + torch.sqrt(torch.clamp(u - cs, min=0.0)), u)
            return u * float(scale)
        u = torch.pow(v * float(scale) + 1.0, float(power)) - 1.0
        if clip_soft is not None:
            cs = float(clip_soft)
            u = torch.where(u > cs, cs - 1.0 + torch.sqrt(torch.clamp(u - cs + 1.0, min=0.0)), u)
        return u

    is_scalar = np.isscalar(y)
    arr = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    if old_transform:
        u = np.power(arr, float(power))
        if clip_soft is not None:
            cs = float(clip_soft)
            u = np.where(u > cs, cs + np.sqrt(np.maximum(u - cs, 0.0)), u)
        out = u * float(scale)
    else:
        u = np.power(arr * float(scale) + 1.0, float(power)) - 1.0
        if clip_soft is not None:
            cs = float(clip_soft)
            u = np.where(u > cs, cs - 1.0 + np.sqrt(np.maximum(u - cs + 1.0, 0.0)), u)
        out = u
    if is_scalar:
        return float(out)
    return out.astype(np.asarray(y).dtype if np.issubdtype(np.asarray(y).dtype, np.floating) else np.float32)


def undo_squashed_scale(
    x: Union[float, np.ndarray, Any],
    scale: float = 0.30,
    clip_soft: Optional[float] = 384.0,
    power: float = 0.75,
    old_transform: bool = True,
) -> Union[float, np.ndarray, Any]:
    """
    Invert Borzoi / Flashzoi `sum_sqrt` squashing AFTER averaging FWD + RC predictions
    and BEFORE summing across exon or promoter bins.

    Exact inverse of `apply_squashed_scale` for all non-negative inputs (both below and
    above `clip_soft`). Supports Python floats, NumPy arrays, and PyTorch tensors.
    """
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    if power <= 0.0:
        raise ValueError(f"power must be positive, got {power}")

    inv_p = 1.0 / float(power)
    if _is_torch_tensor(x):
        if old_transform:
            u = x.float() / float(scale)
            if clip_soft is not None:
                cs = float(clip_soft)
                u = torch.where(u > cs, (u - cs) ** 2 + cs, u)
            return torch.pow(torch.clamp(u, min=0.0), inv_p)
        u = x.float()
        if clip_soft is not None:
            cs = float(clip_soft)
            u = torch.where(u > cs, (u - cs + 1.0) ** 2 + cs - 1.0, u)
        return (torch.pow(torch.clamp(u + 1.0, min=0.0), inv_p) - 1.0) / float(scale)

    is_scalar = np.isscalar(x)
    arr = np.asarray(x, dtype=np.float64)
    if old_transform:
        u = arr / float(scale)
        if clip_soft is not None:
            cs = float(clip_soft)
            u = np.where(u > cs, (u - cs) ** 2 + cs, u)
        out = np.power(np.maximum(u, 0.0), inv_p)
    else:
        u = arr.copy()
        if clip_soft is not None:
            cs = float(clip_soft)
            u = np.where(u > cs, (u - cs + 1.0) ** 2 + cs - 1.0, u)
        out = (np.power(np.maximum(u + 1.0, 0.0), inv_p) - 1.0) / float(scale)
    if is_scalar:
        return float(out)
    return out.astype(np.asarray(x).dtype if np.issubdtype(np.asarray(x).dtype, np.floating) else np.float32)
