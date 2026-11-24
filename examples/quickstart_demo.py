"""
Minimal CPU quickstart demo for panprism.

Runs in ~1-2 seconds on a standard CPU without downloading external genomes or
model weights. Demonstrates:
  1. Two-tier cross-assembly syntenic grafting (V_15 + DASG / V_graft)
  2. DNA-only LD haplotype scaffold phase registration
  3. Breakpoint-graph coordinate bijection over an unclipped insertion bubble
  4. Diagonal-normalized physical 3D contact decay (P_phys)
  5. Open-source mode feature masking and antisymmetric Ridge regression
"""

from __future__ import annotations

import time
import numpy as np

from panprism import (
    PANPRISM_V2_FEATURE_NAMES,
    VariantAllele,
    build_haplotype_graph_sequence,
    compute_physical_3d_contact,
    fit_grouped_antisymmetric_ridge,
    mask_features_for_mode,
    undo_squashed_scale,
    verify_and_graft_sv_two_tier,
)
from panprism.phase_registration import register_sv_phase_scaffolds


class _ToyShiftChain:
    """Minimal LiftOver stub mapping hs1 coordinates to GRCh38 (+100 bp shift)."""

    def __init__(self, shift: int = 100):
        self.shift = shift

    def convert_coordinate(self, chrom: str, pos: int, strand: str = "+"):
        return [(chrom, pos + self.shift, strand, 1.0)]


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(7)
    bases = np.array(list("ACGT"))

    # 1. Cross-assembly syntenic grafting (Tier-2 DASG rescue across a 1-bp REF mismatch)
    hs1_seq = "".join(rng.choice(bases, size=400))
    grch38_prefix = "".join(rng.choice(bases, size=100))
    grch38_body = list(hs1_seq)
    # Introduce a 1-bp reference divergence inside the SV span so naive V_15 fails
    sv_pos_hs1 = 150
    sv_ref_hs1 = hs1_seq[sv_pos_hs1 : sv_pos_hs1 + 60]
    sv_alt_hs1 = hs1_seq[sv_pos_hs1]  # 59-bp deletion
    grch38_body[sv_pos_hs1 + 10] = "A" if grch38_body[sv_pos_hs1 + 10] != "A" else "C"
    grch38_seq = grch38_prefix + "".join(grch38_body)

    res = verify_and_graft_sv_two_tier(
        chrom="chr1",
        pos0_source=sv_pos_hs1,
        ref_allele=sv_ref_hs1,
        alt_allele=sv_alt_hs1,
        grch38_chrom_seq=grch38_seq,
        hs1_chrom_seq=hs1_seq,
        liftover_chain=_ToyShiftChain(shift=100),
    )
    print(
        f"[1] Syntenic grafting: passed={res.passed}, tier={res.tier}, "
        f"grch38_span=[{res.pos0_grch38}, {res.end0_grch38}), "
        f"grafted_alt_len={len(res.alt_grch38)} bp"
    )

    # 2. DNA-only LD phase registration on 60 synthetic donors with 50% inverted scaffolds
    n_donors, n_sv, n_snv = 60, 6, 12
    true_sv_h1 = rng.binomial(1, 0.3, size=(n_sv, n_donors)).astype(np.int8)
    true_sv_h2 = rng.binomial(1, 0.3, size=(n_sv, n_donors)).astype(np.int8)
    # Create flanking SNVs in strong LD with the SVs
    snv_h1 = np.repeat(true_sv_h1, 2, axis=0).copy()
    snv_h2 = np.repeat(true_sv_h2, 2, axis=0).copy()
    noise = rng.random(snv_h1.shape) < 0.05
    snv_h1[noise] ^= 1

    # Randomly invert H1/H2 labels on half of the donors in the SV callset
    flipped_truth = rng.random(n_donors) < 0.5
    obs_sv_h1 = np.where(flipped_truth[None, :], true_sv_h2, true_sv_h1)
    obs_sv_h2 = np.where(flipped_truth[None, :], true_sv_h1, true_sv_h2)

    sv_pos = np.arange(1, n_sv + 1) * 5_000
    snv_pos = np.repeat(sv_pos, 2) + np.tile([500, 1200], n_sv)
    _, _, _, summary = register_sv_phase_scaffolds(
        sv_chroms=["chr1"] * n_sv,
        sv_gene_ids=["GENE1"] * n_sv,
        sv_pos=sv_pos,
        sv_h1=obs_sv_h1,
        sv_h2=obs_sv_h2,
        snv_chroms=["chr1"] * n_snv,
        snv_gene_ids=["GENE1"] * n_snv,
        snv_pos=snv_pos,
        snv_h1=snv_h1,
        snv_h2=snv_h2,
    )
    print(
        f"[2] LD phase registration: pre-reg concordance={summary.pre_reg_high_ld_concordance_mean:.3f} -> "
        f"post-reg concordance={summary.post_reg_high_ld_concordance_mean:.3f}"
    )

    # 3. Breakpoint-graph projection with an unclipped 200-bp insertion bubble
    ref_window = "".join(rng.choice(bases, size=20_000))
    ins_seq = "".join(rng.choice(bases, size=200))
    ins_var = VariantAllele(
        pos0=11_500,
        ref=ref_window[11_500],
        alt=ref_window[11_500] + ins_seq,
        var_id="toy_ins_200bp",
        is_sv=True,
        sv_class="INS",
        sv_len=200,
    )
    mapping = build_haplotype_graph_sequence(
        ref_seq=ref_window,
        ref_seq_start0=0,
        tss_pos0=10_000,
        strand="+",
        variants=[ins_var],
        window_len=8_192,
    )
    print(
        f"[3] Breakpoint graph: seq_len={len(mapping.sequence)} bp, "
        f"insertion_bubbles={len(mapping.insertion_bubbles)}, "
        f"bubble_len={mapping.insertion_bubbles[0].ins_len} bp"
    )

    # 4. Physical 3D contact normalization and Flashzoi coverage unsquashing
    raw_c = rng.normal(loc=-0.4, scale=0.6, size=(32, 32)).astype(np.float32)
    raw_c = 0.5 * (raw_c + raw_c.T)
    p_phys = compute_physical_3d_contact(raw_c, bin_bp=2048.0)[0]
    unsquashed = undo_squashed_scale(np.array([0.5, 2.0, 10.0], dtype=np.float32))
    print(
        f"[4] 3D contact physics: min(P_phys)={float(p_phys.min()):.3e} (>0 across negative log O/E), "
        f"unsquashed_sample={[round(float(v), 3) for v in unsquashed]}"
    )

    # 5. Open-source feature masking and antisymmetric grouped Ridge regression
    x_full = rng.normal(size=(80, len(PANPRISM_V2_FEATURE_NAMES))).astype(np.float32)
    x_open, kept_mask = mask_features_for_mode(x_full, PANPRISM_V2_FEATURE_NAMES, mode="mit")
    groups = np.repeat(np.arange(8), 10)
    w_true = np.zeros(x_open.shape[1])
    active_idx = np.where(kept_mask)[0][:3]
    w_true[active_idx] = [0.8, 0.5, 0.4]
    y = x_open @ w_true + rng.normal(scale=0.1, size=80)
    fit = fit_grouped_antisymmetric_ridge(x_open, y, groups=groups)
    pred_fwd = fit.predict(x_open)
    pred_rev = fit.predict(-x_open)
    antisym_err = float(np.max(np.abs(pred_fwd + pred_rev)))
    zero_err = float(np.max(np.abs(fit.predict(np.zeros_like(x_open)))))
    print(
        f"[5] Open-source mode & antisymmetric Ridge: active_open_channels={int(kept_mask.sum())}/{len(kept_mask)}, "
        f"max|f(H1,H2)+f(H2,H1)|={antisym_err:.1e}, max|f(H,H)|={zero_err:.1e}"
    )
    print(f"Demo finished in {time.perf_counter() - t0:.2f} s.")


if __name__ == "__main__":
    main()
