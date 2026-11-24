"""
Antisymmetric diploid cis-regulatory coupler and chromosome-grouped
cross-validation evaluators.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.optimize import minimize

PANPRISM_NODE_NAMES = [
    "exon_sense_rna_sum",          # 0
    "exon_sense_rna_density",      # 1
    "exon_survival_fraction",      # 2
    "prom_sense_cage",             # 3
    "prom_atac",                   # 4
    "prom_h3k27ac_h3k4me3",        # 5
    "canonical_splice_prob",       # 6
    "canonical_splice_usage",      # 7
    "exon_h3k36me3",               # 8
    "ep_3d_contact_enhancer",      # 9
    "ep_3d_loop_strength",         # 10
    "ep_1d_decay_enhancer",        # 11
    "ins_denovo_enhancer_3d",      # 12
    "ins_denovo_sense_cage",       # 13
    "ins_antisense_interference",  # 14
    "ins_cryptic_splice_trap",     # 15
    "ins_cryptic_usage_trap",      # 16
    "ins_len_log1p",               # 17
    "del_len_log1p",               # 18
    "del_bkp_chromatin",           # 19
    "inv_len_log1p",               # 20
    "prom_antisense_cage",         # 21
]
PB_NODE_NAMES = PANPRISM_NODE_NAMES

BZ_NODE_NAMES = [
    "exon_sense_rna_sum",          # 0
    "exon_sense_rna_density",      # 1
    "exon_survival_fraction",      # 2
    "prom_sense_cage",             # 3
    "prom_dnase",                  # 4
    "prom_h3k27ac_h3k4me3",        # 5
    "ep_rewired_enhancer",         # 6
    "ins_denovo_enhancer",         # 7
    "ins_antisense_interference",  # 8
]

# Biophysical monotonicity masks: canonical transcriptional/chromatin/enhancer/exon-survival channels
# are non-negative (w_j >= 0), whereas polymorphic insertion roadblock/cryptic-splice channels have free sign.
PANPRISM_CHANNEL_NONNEG_MASK = np.array([True] * 10 + [False, False], dtype=bool)
PB_CHANNEL_NONNEG_MASK = PANPRISM_CHANNEL_NONNEG_MASK
KS_CHANNEL_NONNEG_MASK = np.array([True, True, True, True], dtype=bool)
BZ_CHANNEL_NONNEG_MASK = np.array([True] * 7 + [False, False], dtype=bool)


def zero_shot_korsakova_score(delta_ks: np.ndarray) -> np.ndarray:
    """Zero-shot mechanistic expression score from 4-dim Korsakova stitched or Center-Crop delta (`0` fitted params)."""
    return (
        0.45 * delta_ks[..., 0]
        + 0.25 * delta_ks[..., 1]
        + 0.20 * delta_ks[..., 2]
        + 0.10 * delta_ks[..., 3]
    ).astype(np.float32)


def zero_shot_borzoi_korsakova_score(delta_ks: np.ndarray) -> np.ndarray:
    """Zero-shot mechanistic expression score for Borzoi / Flashzoi 4-dim Korsakova stitched delta."""
    return zero_shot_korsakova_score(delta_ks)


def borzoi_9channel_features(delta_bz: np.ndarray, mode: str = "full") -> np.ndarray:
    """9-channel biophysical representation for Borzoi-524kb and Flashzoi-524kb Breakpoint-Graph nodes."""
    out = np.asarray(delta_bz, dtype=np.float32).copy()
    if mode == "no_ins_denovo":
        out[..., 7] = 0.0
        out[..., 8] = 0.0
    elif mode == "no_exon_survival":
        out[..., 2] = 0.0
        out[..., 7] = 0.0
        out[..., 8] = 0.0
    return out


def panprism_12channel_features(delta_pb: np.ndarray, mode: str = "full") -> np.ndarray:
    """
    Map 22-dim AlphaGenome PanPrism node deltas into 12 scale-separated orthogonal biophysical channels.
    Supports cumulative ablation modes:
      - `'full'`              : all 12 biophysical channels active
      - `'no_3d_contact'`     : replace 4DN 3D Hi-C loop channels `[8,9]` with 1D distance-decay enhancer
      - `'no_ins_denovo'`     : additionally zero *de novo* insertion channels `[10,11]`
      - `'no_splice_inv'`     : additionally zero canonical splice & H3K36me3 elongation `[6,7]`
      - `'no_exon_survival'`  : additionally zero exon survival fraction `[1]` & bidirectional promoter `[5]`
    """
    c_exon_rna = 0.5 * (delta_pb[..., 0] + delta_pb[..., 1])
    c_exon_surv = np.zeros_like(c_exon_rna) if mode == "no_exon_survival" else delta_pb[..., 2]
    c_prom_cage = delta_pb[..., 3]
    c_prom_atac = delta_pb[..., 4]
    c_prom_hist = delta_pb[..., 5]
    c_prom_bidir = np.zeros_like(c_exon_rna) if mode == "no_exon_survival" else delta_pb[..., 21]

    if mode in ("no_splice_inv", "no_exon_survival"):
        c_elong = np.zeros_like(c_exon_rna)
        c_splice = np.zeros_like(c_exon_rna)
    else:
        c_elong = delta_pb[..., 8]
        c_splice = 0.5 * (delta_pb[..., 6] + delta_pb[..., 7])

    if mode in ("no_3d_contact", "no_ins_denovo", "no_splice_inv", "no_exon_survival"):
        c_ep1 = delta_pb[..., 11]
        c_ep2 = np.zeros_like(c_exon_rna)
    else:
        c_ep1 = delta_pb[..., 9]
        c_ep2 = delta_pb[..., 10]

    if mode in ("no_ins_denovo", "no_splice_inv", "no_exon_survival"):
        c_ins_act = np.zeros_like(c_exon_rna)
        c_ins_sp = np.zeros_like(c_exon_rna)
    else:
        # Strictly separate sense activating insertion nodes (3D enhancer [12] + sense CAGE [13])
        # from repressive roadblock insertion nodes (antisense interference [14] + cryptic splice traps [15,16]).
        c_ins_act = 0.5 * (delta_pb[..., 12] + delta_pb[..., 13])
        c_ins_sp = (delta_pb[..., 14] + delta_pb[..., 15] + delta_pb[..., 16]) / 3.0

    return np.stack(
        [
            c_exon_rna,
            c_exon_surv,
            c_prom_cage,
            c_prom_atac,
            c_prom_hist,
            c_prom_bidir,
            c_elong,
            c_splice,
            c_ep1,
            c_ep2,
            c_ins_act,
            c_ins_sp,
        ],
        axis=-1,
    ).astype(np.float32)


def zero_shot_panprism_score(delta_pb: np.ndarray, mode: str = "full") -> np.ndarray:
    """Unit-consistent Zero-Shot mechanistic expression score (`0` cohort parameters fitted)."""
    ex_surv = 0.0 if mode == "no_exon_survival" else (0.35 * delta_pb[..., 2])
    base = (
        0.28 * delta_pb[..., 0]
        + 0.24 * delta_pb[..., 1]
        + ex_surv
        + 0.25 * delta_pb[..., 3]
        + 0.08 * delta_pb[..., 4]
        + 0.12 * delta_pb[..., 5]
    )
    if mode in ("no_splice_inv", "no_exon_survival"):
        splice_elong = 0.0
    else:
        splice_elong = 0.15 * delta_pb[..., 6] + 0.15 * delta_pb[..., 7] + 0.45 * delta_pb[..., 8]

    if mode in ("no_3d_contact", "no_ins_denovo", "no_splice_inv", "no_exon_survival"):
        ep_term = 0.04 * delta_pb[..., 11]
    else:
        ep_term = 0.03 * delta_pb[..., 9] + 0.02 * delta_pb[..., 10]

    if mode in ("no_ins_denovo", "no_splice_inv", "no_exon_survival"):
        ins_term = 0.0
    else:
        ins_term = (
            +0.015 * delta_pb[..., 12]
            + 0.010 * delta_pb[..., 13]
            - 0.018 * delta_pb[..., 14]
            - 0.045 * delta_pb[..., 15]
            - 0.040 * delta_pb[..., 16]
        )

    return np.asarray(base + splice_elong + ep_term + ins_term, dtype=np.float32)


def zero_shot_borzoi_panprism_score(delta_bz_pb: np.ndarray) -> np.ndarray:
    """Unit-consistent Zero-Shot mechanistic expression score for Borzoi-524kb and Flashzoi-524kb (`0` fitted params)."""
    return (
        0.35 * delta_bz_pb[..., 0]
        + 0.25 * delta_bz_pb[..., 1]
        + 0.25 * delta_bz_pb[..., 2]
        + 0.20 * delta_bz_pb[..., 3]
        + 0.08 * delta_bz_pb[..., 4]
        + 0.10 * delta_bz_pb[..., 5]
        + 0.02 * delta_bz_pb[..., 6]
        + 0.02 * delta_bz_pb[..., 7]
        - 0.015 * delta_bz_pb[..., 8]
    ).astype(np.float32)


def fit_ridge_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Tuple[float, ...] = (1.0, 10.0, 50.0, 200.0, 1000.0),
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Standardized Ridge regression with internal GCV strictly inside `(X, y)`."""
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    Xs = (X - x_mean) / x_std
    y_mean = float(y.mean())
    yc = y - y_mean

    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    n = float(Xs.shape[0])
    best_gcv = np.inf
    best_w = np.zeros(Xs.shape[1], dtype=np.float64)

    for alpha in alphas:
        d = (s**2) / (s**2 + alpha)
        y_hat = U @ (d * (U.T @ yc))
        eff_df = float(d.sum())
        denom = max(1.0, n - eff_df) ** 2
        gcv = float(np.sum((yc - y_hat) ** 2)) / denom
        if gcv < best_gcv:
            best_gcv = gcv
            best_w = Vt.T @ ((s / (s**2 + alpha)) * (U.T @ yc))

    return best_w.astype(np.float32), y_mean, x_mean.astype(np.float32), x_std.astype(np.float32)


def predict_ridge(X: np.ndarray, w: np.ndarray, b: float, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    Xs = (X - x_mean) / x_std
    return (Xs @ w + b).astype(np.float32)


def fit_antisymmetric_ridge_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Tuple[float, ...] = (5.0, 25.0, 100.0, 500.0, 2000.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline 8 (`Unconstrained 12-Ch Ridge`): H1<->H2 parity-invariant (zero-intercept, RMS-scaled)
    unconstrained Ridge regression with internal GCV strictly inside `(X, y)`.
    Guarantees exact antisymmetry `f(-X) = -f(X)` (`|f(H1,H2) + f(H2,H1)| == 0.0`).
    """
    x_rms = np.sqrt(np.mean(X.astype(np.float64) ** 2, axis=0))
    x_rms = np.where(x_rms < 1e-8, 1.0, x_rms)
    Xs = X.astype(np.float64) / x_rms
    yv = y.astype(np.float64)

    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    n = float(Xs.shape[0])
    best_gcv = np.inf
    best_w = np.zeros(Xs.shape[1], dtype=np.float64)

    for alpha in alphas:
        d = (s**2) / (s**2 + alpha)
        y_hat = U @ (d * (U.T @ yv))
        eff_df = float(d.sum())
        denom = max(1.0, n - eff_df) ** 2
        gcv = float(np.sum((yv - y_hat) ** 2)) / denom
        if gcv < best_gcv:
            best_gcv = gcv
            best_w = Vt.T @ ((s / (s**2 + alpha)) * (U.T @ yv))

    return best_w.astype(np.float32), x_rms.astype(np.float32)


def predict_antisymmetric_ridge(X: np.ndarray, w: np.ndarray, x_rms: np.ndarray) -> np.ndarray:
    """Exact zero-intercept prediction in float64 before float32 cast so `f(-X) + f(X) == 0.0`."""
    pred64 = (np.asarray(X, dtype=np.float64) / np.asarray(x_rms, dtype=np.float64)) @ np.asarray(w, dtype=np.float64)
    return pred64.astype(np.float32)


def fit_biophysical_antisymmetric_ridge(
    X: np.ndarray,
    y: np.ndarray,
    nonneg_mask: np.ndarray,
    alphas: Tuple[float, ...] = (5.0, 25.0, 100.0, 500.0, 2000.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    H1<->H2 parity-invariant (zero-intercept, training-RMS scaled) Ridge regression with biophysical
    sign constraints (`w_j >= 0` where `nonneg_mask[j] == True`; unconstrained otherwise).
    Selects regularization `alpha` via deterministic 4-fold internal CV strictly inside `(X, y)`.
    """
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    # Symmetrize `(X, y)` with `(-X, -y)` so the L-BFGS-B trajectory is 100% invariant to training-row H1<->H2 flips!
    X_sym = np.vstack([X64, -X64])
    y_sym = np.concatenate([y64, -y64])

    x_rms = np.sqrt(np.mean(X_sym**2, axis=0))
    x_rms = np.where(x_rms < 1e-8, 1.0, x_rms)
    Xs = X64 / x_rms
    d = Xs.shape[1]
    bounds = [(0.0, None) if bool(nonneg_mask[j]) else (None, None) for j in range(d)]

    def _solve(Xa: np.ndarray, ya: np.ndarray, alpha: float) -> np.ndarray:
        XtX = Xa.T @ Xa + alpha * np.eye(d)
        Xty = Xa.T @ ya
        res = minimize(
            lambda w: 0.5 * float(w @ (XtX @ w)) - float(Xty @ w),
            np.zeros(d, dtype=np.float64),
            jac=lambda w: XtX @ w - Xty,
            method="L-BFGS-B",
            bounds=bounds,
        )
        return res.x

    idx = np.arange(len(y64)) % 4
    best_loss = np.inf
    best_alpha = alphas[len(alphas) // 2]
    for alpha in alphas:
        val_err = 0.0
        for k in range(4):
            mtr, mva = idx != k, idx == k
            if np.sum(mtr) < 4 or np.sum(mva) < 1:
                continue
            wk = _solve(Xs[mtr], y64[mtr], alpha)
            val_err += float(np.sum((y64[mva] - Xs[mva] @ wk) ** 2))
        if val_err < best_loss:
            best_loss = val_err
            best_alpha = alpha

    w_full = _solve(Xs, y64, best_alpha)
    return w_full.astype(np.float32), x_rms.astype(np.float32)


def run_permuted_null_control(
    X: np.ndarray,
    y: np.ndarray,
    chroms: np.ndarray,
    nonneg_mask: np.ndarray,
    n_seeds: int = 20,
    strong_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Baseline 9 (`20-Seed Permuted-Haplotype Sign-Flip Null`):
    Permutes haplotype rows AND randomly flips `(H1, H2)` signs across `n_seeds` independent seeds,
    evaluating through the exact same 22-Chr LOCO `fit_biophysical_antisymmetric_ridge` pipeline.
    """
    unique_chroms = sorted(set(chroms))
    all_accs: List[float] = []
    all_rs: List[float] = []
    strong_accs: List[float] = []
    strong_rs: List[float] = []

    nonzero_y = np.abs(y) > 1e-8
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        perm_idx = rng.permutation(len(y))
        sign_flip = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(len(y), 1))
        X_null = X[perm_idx] * sign_flip

        preds = np.zeros(len(y), dtype=np.float32)
        for ch in unique_chroms:
            te = chroms == ch
            tr = ~te
            if np.sum(tr) < 8 or np.sum(te) < 1:
                continue
            w, x_rms = fit_biophysical_antisymmetric_ridge(X_null[tr], y[tr], nonneg_mask)
            preds[te] = predict_antisymmetric_ridge(X_null[te], w, x_rms)

        acc_all = float(np.mean(np.sign(preds[nonzero_y]) == np.sign(y[nonzero_y]))) if np.any(nonzero_y) else 0.5
        r_all, _ = safe_corr(y[nonzero_y], preds[nonzero_y])
        all_accs.append(acc_all)
        all_rs.append(r_all)

        if strong_mask is not None and np.any(strong_mask & nonzero_y):
            sm = strong_mask & nonzero_y
            strong_accs.append(float(np.mean(np.sign(preds[sm]) == np.sign(y[sm]))))
            sr, _ = safe_corr(y[sm], preds[sm])
            strong_rs.append(sr)

    res = {
        "n_seeds": int(n_seeds),
        "all_dir_acc_mean": round(float(np.mean(all_accs)), 4),
        "all_dir_acc_std": round(float(np.std(all_accs)), 4),
        "all_pearson_r_mean": round(float(np.mean(all_rs)), 4),
        "all_pearson_r_std": round(float(np.std(all_rs)), 4),
    }
    if strong_accs:
        res["strong_dir_acc_mean"] = round(float(np.mean(strong_accs)), 4)
        res["strong_dir_acc_std"] = round(float(np.std(strong_accs)), 4)
        res["strong_pearson_r_mean"] = round(float(np.mean(strong_rs)), 4)
        res["strong_pearson_r_std"] = round(float(np.std(strong_rs)), 4)
    return res


def compute_loco_trans_pcs(
    expr_matrix_genes_by_samples: np.ndarray,
    gene_chroms: np.ndarray,
    excluded_chroms: set,
    fit_sample_indices: np.ndarray,
    n_pcs: int = 12,
) -> np.ndarray:
    """
    Strictly leak-free Leave-One-Chromosome-Out (LOCO) Trans-Regulatory Principal Components:
      1) Excludes ALL genes on `excluded_chroms` (zero cis/syntenic leakage).
      2) Fits gene mean/std and SVD loadings EXCLUSIVELY on `fit_sample_indices`.
      3) Projects all samples onto the training-fitted trans subspace -> `(N_samples, n_pcs)`.
    """
    keep_gene_mask = np.array([c not in excluded_chroms for c in gene_chroms], dtype=bool)
    sub_expr = expr_matrix_genes_by_samples[keep_gene_mask, :]

    train_slice = sub_expr[:, fit_sample_indices]
    g_mean = train_slice.mean(axis=1, keepdims=True)
    g_std = train_slice.std(axis=1, keepdims=True)
    g_std = np.where(g_std < 1e-6, 1.0, g_std)

    X_train = ((train_slice - g_mean) / g_std).T
    X_all = ((sub_expr - g_mean) / g_std).T

    gram = (X_train @ X_train.T) / float(X_train.shape[1])
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:n_pcs]
    top_vals = np.maximum(eigvals[order], 1e-8)
    top_u = eigvecs[:, order]
    V = (X_train.T @ top_u) / np.sqrt(float(X_train.shape[1]) * top_vals)[None, :]
    Z_all = (X_all @ V) / np.sqrt(float(X_train.shape[1]))
    return Z_all.astype(np.float32)


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return `(Pearson r, Spearman rho)`, returning `(0.0, 0.0)` on constant arrays."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) < 4 or np.std(yt) < 1e-7 or np.std(yp) < 1e-7 or len(np.unique(yt)) < 2 or len(np.unique(yp)) < 2:
        return 0.0, 0.0
    pr = float(stats.pearsonr(yt, yp)[0])
    sr = float(stats.spearmanr(yt, yp)[0])
    return (0.0 if np.isnan(pr) else pr), (0.0 if np.isnan(sr) else sr)


def exact_mcnemar_pvalue(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, float]:
    """Exact binomial paired McNemar test comparing directional correctness of Method A vs Method B."""
    ca = np.asarray(correct_a, dtype=bool)
    cb = np.asarray(correct_b, dtype=bool)
    n_a_only = int(np.sum(ca & (~cb)))
    n_b_only = int(np.sum((~ca) & cb))
    n_disc = n_a_only + n_b_only
    if n_disc == 0:
        p_one, p_two = 1.0, 1.0
    else:
        p_one = float(stats.binomtest(n_a_only, n_disc, 0.5, alternative="greater").pvalue)
        p_two = float(stats.binomtest(n_a_only, n_disc, 0.5, alternative="two-sided").pvalue)
    return {
        "n_a_only_wins": n_a_only,
        "n_b_only_wins": n_b_only,
        "n_discordant": n_disc,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
    }


def paired_bootstrap_pvalue(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float], float]:
    """Compute mean delta `(A - B)`, 95% bootstrap CI, and two-sided Wilcoxon signed-rank p-value."""
    diff = np.asarray(metric_a, dtype=np.float64) - np.asarray(metric_b, dtype=np.float64)
    mean_d = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[idx].mean(axis=1)
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    nonzero = diff[np.abs(diff) > 1e-12]
    pval = float(stats.wilcoxon(nonzero, alternative="two-sided").pvalue) if len(nonzero) >= 5 else 1.0
    return mean_d, (ci_lo, ci_hi), pval
