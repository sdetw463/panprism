"""
DNA-only linkage-disequilibrium (LD) haplotype scaffold phase registration.

Independent chromosome-scale statistical phasing callsets (e.g., 1KG_ONT_VIENNA SHAPEIT5,
HGSVC3 PanGenie+SHAPEIT5, or HPRC v2 assembly haplotypes `(H1, H2)`) exhibit ~50% random
maternal/paternal scaffold polarity inversions per `(chromosome, individual)` relative to
pedigree-phased reference SNV scaffolds (e.g., NYGC 1kGP 30x GRCh38).

Without phase registration, combining SVs and cis-SNVs phased independently per
autosome randomizes haplotype polarity across roughly half of all scaffolds.

This module aligns `(H1_SV, H2_SV)` onto `(H1_SNV, H2_SNV)` using population diploid genotype LD covariance `cov(G_SV, G_SNV)` alone,
without accessing RNA-seq or allele-specific expression counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class PhaseRegistrationSummary:
    """Summary and per-chromosome diagnostics of RNA-free diploid LD scaffold registration."""
    n_samples: int
    n_chromosomes: int
    n_het_sv_blocks_total: int
    n_het_sv_blocks_flipped: int
    flipped_fraction: float
    pre_reg_high_ld_concordance_mean: float
    pre_reg_high_ld_concordance_median: float
    post_reg_high_ld_concordance_mean: float
    post_reg_high_ld_concordance_median: float
    n_high_ld_anchor_pairs: int
    per_chrom_flip_rates: Dict[str, float]


def evaluate_ld_phase_concordance(
    sv_pos: np.ndarray,
    sv_h1: np.ndarray,
    sv_h2: np.ndarray,
    snv_pos: np.ndarray,
    snv_h1: np.ndarray,
    snv_h2: np.ndarray,
    max_dist_bp: int = 25000,
    min_abs_r: float = 0.50,
) -> Tuple[float, float, int]:
    """
    Evaluate exact haplotype phase concordance across high-LD `(SV, SNV)` pairs (`|r_diploid| >= min_abs_r`)
    within `max_dist_bp` on a single chromosome or locus set.

    Returns `(mean_concordance, median_concordance, n_pairs)`.
    """
    if len(sv_pos) == 0 or len(snv_pos) == 0:
        return 0.5, 0.5, 0

    u_dip = (snv_h1 + snv_h2).astype(np.float64)
    u_dif = (snv_h1 - snv_h2).astype(np.float64)
    u_c = u_dip - u_dip.mean(axis=1, keepdims=True)
    u_ss = np.sum(u_c**2, axis=1)

    concordances: List[float] = []
    for i in range(len(sv_pos)):
        near = np.abs(snv_pos - int(sv_pos[i])) <= max_dist_bp
        if not np.any(near):
            continue
        vh1 = sv_h1[i].astype(np.float64)
        vh2 = sv_h2[i].astype(np.float64)
        vdip = vh1 + vh2
        vdif = vh1 - vh2
        vc = vdip - vdip.mean()
        vss = float(np.sum(vc**2))
        if vss < 1e-6:
            continue
        sub_c = u_c[near]
        sub_ss = u_ss[near]
        r_vec = (sub_c @ vc) / (np.sqrt(sub_ss * vss) + 1e-8)
        best_j = int(np.argmax(np.abs(r_vec)))
        best_r = float(r_vec[best_j])
        if abs(best_r) < min_abs_r:
            continue
        udif_best = u_dif[near][best_j]
        both_het = (vdif != 0) & (udif_best != 0)
        if int(np.sum(both_het)) < 4:
            continue
        expected_sign = 1.0 if best_r > 0 else -1.0
        agree = (vdif[both_het] * udif_best[both_het] * expected_sign) > 0
        concordances.append(float(np.mean(agree)))

    if not concordances:
        return 0.5, 0.5, 0
    arr = np.array(concordances, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), len(concordances)


def register_sv_phase_scaffolds(
    sv_chroms: List[str],
    sv_gene_ids: List[str],
    sv_pos: np.ndarray,
    sv_h1: np.ndarray,
    sv_h2: np.ndarray,
    snv_chroms: List[str],
    snv_gene_ids: List[str],
    snv_pos: np.ndarray,
    snv_h1: np.ndarray,
    snv_h2: np.ndarray,
    max_dist_bp: int = 25000,
    min_ld_r: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PhaseRegistrationSummary]:
    """
    Register SV haplotype matrices `(sv_h1, sv_h2)` of shape `(M_sv, N_samples)` onto
    the reference SNV scaffold `(snv_h1, snv_h2)` of shape `(M_snv, N_samples)` using
    100% genotype-only diploid LD covariance.

    Returns `(reg_h1, reg_h2, chrom_donor_flip_matrix, summary)`.
    """
    m_sv, n_samp = sv_h1.shape
    reg_h1 = sv_h1.copy()
    reg_h2 = sv_h2.copy()

    # Group indices by gene_id and by chrom
    snv_by_gene: Dict[str, List[int]] = {}
    for idx, gid in enumerate(snv_gene_ids):
        snv_by_gene.setdefault(gid, []).append(idx)

    sv_by_gene: Dict[str, List[int]] = {}
    gene_to_chrom: Dict[str, str] = {}
    for idx, (gid, ch) in enumerate(zip(sv_gene_ids, sv_chroms)):
        sv_by_gene.setdefault(gid, []).append(idx)
        gene_to_chrom[gid] = ch

    unique_chroms = sorted(set(sv_chroms), key=lambda c: int(c[3:]) if c.startswith("chr") and c[3:].isdigit() else 99)
    chrom_to_idx = {c: i for i, c in enumerate(unique_chroms)}

    gene_sample_score: Dict[str, np.ndarray] = {}
    chrom_sample_score: Dict[str, np.ndarray] = {c: np.zeros(n_samp, dtype=np.float64) for c in unique_chroms}

    for gid, s_indices in sv_by_gene.items():
        score_s = np.zeros(n_samp, dtype=np.float64)
        u_indices = snv_by_gene.get(gid, [])
        if u_indices:
            u_idx_arr = np.array(u_indices, dtype=np.int64)
            u_p = snv_pos[u_idx_arr]
            u_1 = snv_h1[u_idx_arr].astype(np.float64)
            u_2 = snv_h2[u_idx_arr].astype(np.float64)
            u_dip = u_1 + u_2
            u_dif = u_1 - u_2
            u_c = u_dip - u_dip.mean(axis=1, keepdims=True)
            u_ss = np.sum(u_c**2, axis=1)

            for si in s_indices:
                near = np.abs(u_p - int(sv_pos[si])) <= max_dist_bp
                if not np.any(near):
                    continue
                vh1 = sv_h1[si].astype(np.float64)
                vh2 = sv_h2[si].astype(np.float64)
                vdip = vh1 + vh2
                vdif = vh1 - vh2
                vc = vdip - vdip.mean()
                vss = float(np.sum(vc**2))
                if vss < 1e-6:
                    continue
                sub_c = u_c[near]
                sub_ss = u_ss[near]
                r_vec = (sub_c @ vc) / (np.sqrt(sub_ss * vss) + 1e-8)
                w_vec = np.where(np.abs(r_vec) >= min_ld_r, r_vec * (np.abs(r_vec) ** 2), 0.0)
                if np.sum(np.abs(w_vec)) > 1e-6:
                    expected_sign = w_vec @ u_dif[near]
                    score_s += vdif * expected_sign

        gene_sample_score[gid] = score_s
        chrom_sample_score[gene_to_chrom[gid]] += score_s

    # Pre-registration high-LD concordance per chromosome
    pre_conc_list: List[float] = []
    pre_med_list: List[float] = []
    n_anchor_total = 0
    sv_by_chrom: Dict[str, List[int]] = {}
    snv_by_chrom: Dict[str, List[int]] = {}
    for i, c in enumerate(sv_chroms):
        sv_by_chrom.setdefault(c, []).append(i)
    for i, c in enumerate(snv_chroms):
        snv_by_chrom.setdefault(c, []).append(i)

    for c in unique_chroms:
        sv_i = np.array(sv_by_chrom.get(c, []), dtype=np.int64)
        snv_i = np.array(snv_by_chrom.get(c, []), dtype=np.int64)
        if len(sv_i) and len(snv_i):
            cm, cmed, npairs = evaluate_ld_phase_concordance(
                sv_pos[sv_i], sv_h1[sv_i], sv_h2[sv_i],
                snv_pos[snv_i], snv_h1[snv_i], snv_h2[snv_i],
                max_dist_bp=max_dist_bp,
            )
            if npairs > 0:
                pre_conc_list.extend([cm] * npairs)
                pre_med_list.append(cmed)
                n_anchor_total += npairs

    # Apply registration per SV row (preferring local gene LD score when informative, falling back to chromosome scaffold score)
    n_het_total = 0
    n_het_flipped = 0
    per_chr_het: Dict[str, int] = {c: 0 for c in unique_chroms}
    per_chr_flip: Dict[str, int] = {c: 0 for c in unique_chroms}
    chrom_donor_flips = np.zeros((len(unique_chroms), n_samp), dtype=bool)
    for c in unique_chroms:
        chrom_donor_flips[chrom_to_idx[c]] = chrom_sample_score[c] < 0.0

    for i in range(m_sv):
        gid = sv_gene_ids[i]
        ch = sv_chroms[i]
        sc_loc = gene_sample_score[gid]
        sc_chr = chrom_sample_score[ch]
        eff_sc = np.where(np.abs(sc_loc) > 1e-6, sc_loc, sc_chr)
        flip_m = eff_sc < 0.0
        het_m = reg_h1[i] != reg_h2[i]
        nh = int(np.sum(het_m))
        nf = int(np.sum(het_m & flip_m))
        n_het_total += nh
        n_het_flipped += nf
        per_chr_het[ch] += nh
        per_chr_flip[ch] += nf
        tmp = reg_h1[i, flip_m].copy()
        reg_h1[i, flip_m] = reg_h2[i, flip_m]
        reg_h2[i, flip_m] = tmp

    post_conc_list: List[float] = []
    post_med_list: List[float] = []
    for c in unique_chroms:
        sv_i = np.array(sv_by_chrom.get(c, []), dtype=np.int64)
        snv_i = np.array(snv_by_chrom.get(c, []), dtype=np.int64)
        if len(sv_i) and len(snv_i):
            cm, cmed, npairs = evaluate_ld_phase_concordance(
                sv_pos[sv_i], reg_h1[sv_i], reg_h2[sv_i],
                snv_pos[snv_i], snv_h1[snv_i], snv_h2[snv_i],
                max_dist_bp=max_dist_bp,
            )
            if npairs > 0:
                post_conc_list.extend([cm] * npairs)
                post_med_list.append(cmed)

    per_chr_rates = {
        c: float(per_chr_flip[c]) / max(1, per_chr_het[c]) for c in unique_chroms
    }

    summary = PhaseRegistrationSummary(
        n_samples=n_samp,
        n_chromosomes=len(unique_chroms),
        n_het_sv_blocks_total=n_het_total,
        n_het_sv_blocks_flipped=n_het_flipped,
        flipped_fraction=float(n_het_flipped) / max(1, n_het_total),
        pre_reg_high_ld_concordance_mean=float(np.mean(pre_conc_list)) if pre_conc_list else 0.5,
        pre_reg_high_ld_concordance_median=float(np.median(pre_med_list)) if pre_med_list else 0.5,
        post_reg_high_ld_concordance_mean=float(np.mean(post_conc_list)) if post_conc_list else 0.5,
        post_reg_high_ld_concordance_median=float(np.median(post_med_list)) if post_med_list else 0.5,
        n_high_ld_anchor_pairs=n_anchor_total,
        per_chrom_flip_rates=per_chr_rates,
    )
    return reg_h1, reg_h2, chrom_donor_flips, summary
