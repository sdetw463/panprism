"""
Comprehensive Unit & Mathematical Invariance Test Suite for `panprism` v1.0.0 (26 tests).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from panprism import (
    KS_CHANNEL_NONNEG_MASK,
    PANPRISM_CHANNEL_NONNEG_MASK,
    VariantAllele,
    borzoi_9channel_features,
    build_center_crop_naive_sequence,
    build_haplotype_graph_sequence,
    cadd_sv_heuristic_score,
    classify_sv_allele,
    compute_compression_telemetry,
    compute_loco_trans_pcs,
    compute_within_ancestry_peer_zscores,
    decompose_multiallelic_vcf_record,
    exact_mcnemar_pvalue,
    finemap_outlier_windows,
    fit_antisymmetric_ridge_closed_form,
    fit_biophysical_antisymmetric_ridge,
    haplotype_sha1,
    panprism_12channel_features,
    parse_gfa_bubbles,
    pool_alphagenome_on_gpu,
    pool_borzoi_on_gpu,
    predict_antisymmetric_ridge,
    register_sv_phase_scaffolds,
    reverse_complement,
    run_permuted_null_control,
    verify_and_liftover_sv,
    watershed_annotation_prior_score,
    zero_shot_borzoi_panprism_score,
    zero_shot_korsakova_score,
    zero_shot_panprism_score,
)
from panprism.cli import main as cli_main
from panprism.s2f_runners import measure_fp32_vs_bf16_parity


def _make_synthetic_chrom(length: int = 300000, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"))
    return "".join(alphabet[rng.integers(0, 4, size=length)])


def _make_mock_ag_preds(L: int = 131072, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "rna_1bp": rng.uniform(0.0, 2.0, size=(L, 3)).astype(np.float32),
        "cage_1bp": rng.uniform(0.0, 1.0, size=(L, 2)).astype(np.float32),
        "atac_1bp": rng.uniform(0.0, 1.5, size=(L,)).astype(np.float32),
        "dnase_1bp": rng.uniform(0.0, 1.5, size=(L,)).astype(np.float32),
        "histone_128bp": rng.uniform(0.0, 2.0, size=(L // 128, 4)).astype(np.float32),
        "splice_probs_1bp": rng.uniform(0.0, 0.2, size=(L, 4)).astype(np.float32),
        "splice_usage_1bp": rng.uniform(0.0, 0.2, size=(L, 2)).astype(np.float32),
        "contact_gm12878": rng.uniform(0.0, 1.0, size=(L // 2048, L // 2048)).astype(np.float32),
    }


# 1. Zero-SV reference identity bijection & exact zero delta
def test_zero_sv_reference_identity_bijection():
    ref = _make_synthetic_chrom(260000, seed=1)
    tss = 130000
    exons = [(130000, 130300), (132500, 132900), (136000, 136500)]
    mp = build_haplotype_graph_sequence(ref, 0, tss, "+", [], window_len=131072)
    assert mp.sequence == ref[tss - 65536 : tss + 65536]
    for es, ee in exons:
        mapped = mp.map_ref_interval_to_hap(es, ee)
        assert len(mapped) == 1
        hs, he, inv = mapped[0]
        assert not inv
        assert mp.sequence[hs:he] == ref[es:ee]
    delta_zero = np.zeros((1, 22), dtype=np.float32)
    assert float(zero_shot_panprism_score(delta_zero)[0]) == 0.0


# 2. Polymorphic MEI_Alu insertion bubble shifts downstream exons by exact +312 bp
def test_mei_alu_insertion_bubble_downstream_shift():
    ref = _make_synthetic_chrom(260000, seed=2)
    tss = 130000
    exons = [(130000, 130200), (133000, 133400)]
    ins_pos = 131500
    ins_seq = "G" * 312
    alu = VariantAllele(pos0=ins_pos, ref=ref[ins_pos], alt=ref[ins_pos] + ins_seq, var_id="ALU1", is_sv=True, sv_class="MEI_Alu", sv_len=312)
    mp_ref = build_haplotype_graph_sequence(ref, 0, tss, "+", [], window_len=131072)
    mp_alt = build_haplotype_graph_sequence(ref, 0, tss, "+", [alu], window_len=131072)

    assert len(mp_alt.insertion_bubbles) == 1
    bub = mp_alt.insertion_bubbles[0]
    assert bub.hap_end - bub.hap_start == 312
    assert mp_alt.sequence[bub.hap_start : bub.hap_end] == ins_seq

    hs_r, he_r, _ = mp_ref.map_ref_interval_to_hap(*exons[1])[0]
    hs_a, he_a, _ = mp_alt.map_ref_interval_to_hap(*exons[1])[0]
    assert hs_a - hs_r == 312
    assert he_a - he_r == 312
    assert mp_alt.sequence[hs_a:he_a] == ref[exons[1][0] : exons[1][1]]


# 3. Deletion pulls downstream exons upstream & computes partial exon survival fraction
def test_large_deletion_upstream_pull_and_exon_survival():
    ref = _make_synthetic_chrom(260000, seed=3)
    tss = 130000
    exons = [(130000, 130400), (134000, 134600)]  # total 1000 bp
    # Delete 2000 bp [132200, 134200) anchored at 132199, including first 200 bp of exon 2
    del_pos = 132199
    del_ref = ref[del_pos : del_pos + 2001]
    del_v = VariantAllele(pos0=del_pos, ref=del_ref, alt=del_ref[0], var_id="DEL2K", is_sv=True, sv_class="DEL", sv_len=2000)
    mp = build_haplotype_graph_sequence(ref, 0, tss, "+", [del_v], window_len=131072)
    mapped_ex2 = mp.map_ref_interval_to_hap(134000, 134600)
    assert len(mapped_ex2) == 1
    hs, he, _ = mapped_ex2[0]
    assert (he - hs) == 400  # 200 bp deleted, 400 bp surviving
    assert mp.sequence[hs:he] == ref[134200:134600]

    preds = _make_mock_ag_preds(131072, seed=4)
    pooled = pool_alphagenome_on_gpu(preds, mp, exons)
    assert np.isclose(pooled["panprism_nodes"][2], 0.80, atol=1e-5)  # (400 + 400) / 1000


# 4. Inversion flips internal exon strand orientation & reverse-complements DNA
def test_inversion_reverse_complement_strand_flip():
    ref = _make_synthetic_chrom(260000, seed=5)
    tss = 130000
    inv_s, inv_e = 132000, 132600
    inv_v = VariantAllele(pos0=inv_s, ref=ref[inv_s:inv_e], alt="<INV>", var_id="INV600", is_sv=True, sv_class="INV_COMPLEX", sv_len=600)
    mp = build_haplotype_graph_sequence(ref, 0, tss, "+", [inv_v], window_len=131072)
    mapped = mp.map_ref_interval_to_hap(132100, 132350)
    assert len(mapped) == 1
    hs, he, inv_flag = mapped[0]
    assert inv_flag is True
    assert mp.sequence[hs:he] == reverse_complement(ref[132100:132350])


# 5. Multi-SV topological composition (SNV + DEL + MEI_Alu + INV)
def test_multi_sv_topological_composition_del_ins_inv_snv():
    ref = _make_synthetic_chrom(280000, seed=6)
    tss = 140000
    v_snv = VariantAllele(138000, ref[138000], "A" if ref[138000] != "A" else "G", "SNV1", False, "SNV", 1)
    v_del = VariantAllele(141000, ref[141000:141851], ref[141000], "DEL850", True, "DEL", 850)
    v_ins = VariantAllele(143000, ref[143000], ref[143000] + ("T" * 308), "ALU308", True, "MEI_Alu", 308)
    v_inv = VariantAllele(145000, ref[145000:145500], "<INV>", "INV500", True, "INV_COMPLEX", 500)

    mp_ref = build_haplotype_graph_sequence(ref, 0, tss, "+", [], window_len=131072)
    mp_comp = build_haplotype_graph_sequence(ref, 0, tss, "+", [v_inv, v_ins, v_snv, v_del], window_len=131072)

    # Exon downstream of all 4 variants at [147000, 147350): net shift = -850 + 308 = -542 bp!
    hs_r, _, _ = mp_ref.map_ref_interval_to_hap(147000, 147350)[0]
    hs_c, he_c, inv_c = mp_comp.map_ref_interval_to_hap(147000, 147350)[0]
    assert not inv_c
    assert hs_c - hs_r == -542
    assert mp_comp.sequence[hs_c:he_c] == ref[147000:147350]


# 6. Overlapping SV + SNV deduplication prioritizes SV cleanly
def test_overlapping_variant_deduplication_sv_priority():
    ref = _make_synthetic_chrom(260000, seed=8)
    tss = 130000
    v_del = VariantAllele(131000, ref[131000:131201], ref[131000], "DEL200", True, "DEL", 200)
    v_snv_inside = VariantAllele(131050, ref[131050], "C", "SNV_IN_DEL", False, "SNV", 1)
    mp = build_haplotype_graph_sequence(ref, 0, tss, "+", [v_snv_inside, v_del], window_len=131072)
    assert mp.map_ref_interval_to_hap(131010, 131190) == []
    hs, he, _ = mp.map_ref_interval_to_hap(132000, 132200)[0]
    assert mp.sequence[hs:he] == ref[132000:132200]


# 7. +-5 kb window-shift stability preserves exact mapped exon sequences
@pytest.mark.parametrize("shift_bp", [-5000, -2500, -3, 0, 3, 2500, 5000])
def test_window_shift_jitter_stability(shift_bp: int):
    ref = _make_synthetic_chrom(280000, seed=9)
    tss = 140000
    alu = VariantAllele(141200, ref[141200], ref[141200] + ("A" * 290), "ALU290", True, "MEI_Alu", 290)
    mp = build_haplotype_graph_sequence(ref, 0, tss, "+", [alu], window_len=131072, center_shift_bp=shift_bp)
    hs, he, _ = mp.map_ref_interval_to_hap(143000, 143450)[0]
    assert mp.sequence[hs:he] == ref[143000:143450]


# 8. Haplotype SHA-1 lossless deduplication
def test_haplotype_sha1_lossless_deduplication():
    ref = _make_synthetic_chrom(260000, seed=10)
    v1 = VariantAllele(131000, ref[131000], "A" if ref[131000] != "A" else "T", "S1", False, "SNV", 1)
    m1 = build_haplotype_graph_sequence(ref, 0, 130000, "+", [v1], window_len=131072)
    m2 = build_haplotype_graph_sequence(ref, 0, 130000, "+", [v1], window_len=131072)
    m3 = build_haplotype_graph_sequence(ref, 0, 130000, "+", [], window_len=131072)
    assert haplotype_sha1(m1, "ENSG1") == haplotype_sha1(m2, "ENSG1")
    assert haplotype_sha1(m1, "ENSG1") != haplotype_sha1(m3, "ENSG1")
    assert haplotype_sha1(m1, "ENSG1") != haplotype_sha1(m1, "ENSG2")


# 9. Naive Center-Crop-Personal-FASTA coordinate drift vs Breakpoint-Graph accuracy
def test_center_crop_naive_drift_vs_breakpoint_graph():
    ref = _make_synthetic_chrom(260000, seed=12)
    tss = 130000
    # Upstream 3,000 bp insertion shifts TSS and exons by +3,000 bp if unprojected!
    v_up = VariantAllele(126000, ref[126000], ref[126000] + ("C" * 3000), "INS3K", True, "INS_other", 3000)
    exons = [(132000, 132500)]
    mp = build_haplotype_graph_sequence(ref, 0, tss, "+", [v_up], window_len=131072)
    preds = {k: np.zeros_like(v) for k, v in _make_mock_ag_preds(131072, seed=13).items()}
    hs, he, _ = mp.map_ref_interval_to_hap(132000, 132500)[0]
    preds["rna_1bp"][hs:he, :] = 10.0  # True exon transcription spike on haplotype!

    out = pool_alphagenome_on_gpu(preds, mp, exons)
    assert out["panprism_nodes"][0] > 8.0      # Breakpoint-Graph captures 100% of exon spike
    assert out["center_crop_naive"][0] == 0.0   # Unprojected naive offset misses shifted exon completely!
    naive_seq = build_center_crop_naive_sequence(ref, 0, tss, [v_up], window_len=131072)
    assert len(naive_seq) == 131072


# 10. Exact diploid parity |f(H1,H2) + f(H2,H1)| == 0.0 for all Zero-Shot scorers
def test_exact_diploid_parity_zero_shot_all_backbones():
    rng = np.random.default_rng(14)
    h1_pb = rng.normal(size=(50, 22)).astype(np.float32)
    h2_pb = rng.normal(size=(50, 22)).astype(np.float32)
    s12 = zero_shot_panprism_score(h1_pb - h2_pb)
    s21 = zero_shot_panprism_score(h2_pb - h1_pb)
    assert float(np.max(np.abs(s12 + s21))) == 0.0

    h1_ks = rng.normal(size=(50, 4)).astype(np.float32)
    h2_ks = rng.normal(size=(50, 4)).astype(np.float32)
    assert float(np.max(np.abs(zero_shot_korsakova_score(h1_ks - h2_ks) + zero_shot_korsakova_score(h2_ks - h1_ks)))) == 0.0

    h1_bz = rng.normal(size=(50, 9)).astype(np.float32)
    h2_bz = rng.normal(size=(50, 9)).astype(np.float32)
    assert float(np.max(np.abs(zero_shot_borzoi_panprism_score(h1_bz - h2_bz) + zero_shot_borzoi_panprism_score(h2_bz - h1_bz)))) == 0.0


# 11. Exact diploid parity for Unconstrained 12-Ch Antisymmetric Ridge
def test_exact_diploid_parity_unconstrained_ridge():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(80, 12)).astype(np.float32)
    y = (X[:, 0] * 0.8 - X[:, 10] * 0.5 + rng.normal(scale=0.2, size=80)).astype(np.float32)
    w1, rms1 = fit_antisymmetric_ridge_closed_form(X, y)
    p_pos = predict_antisymmetric_ridge(X, w1, rms1)
    p_neg = predict_antisymmetric_ridge(-X, w1, rms1)
    assert float(np.max(np.abs(p_pos + p_neg))) == 0.0

    # Co-flipping arbitrary training rows (X_i, y_i) -> (-X_i, -y_i) leaves weights identical
    flips = rng.choice([-1.0, 1.0], size=(80, 1)).astype(np.float32)
    w2, rms2 = fit_antisymmetric_ridge_closed_form(X * flips, y * flips[:, 0])
    assert np.allclose(w1, w2, atol=1e-6)
    assert np.allclose(rms1, rms2, atol=1e-6)


# 12. Exact diploid parity for Biophysical Constrained Antisymmetric Ridge
def test_exact_diploid_parity_biophysical_constrained_ridge():
    rng = np.random.default_rng(16)
    X = rng.normal(size=(96, 12)).astype(np.float32)
    y = (0.7 * X[:, 0] + 0.4 * X[:, 2] - 0.6 * X[:, 11] + rng.normal(scale=0.3, size=96)).astype(np.float32)
    w, rms = fit_biophysical_antisymmetric_ridge(X, y, PANPRISM_CHANNEL_NONNEG_MASK)
    assert np.all(w[:10] >= -1e-9)
    p12 = predict_antisymmetric_ridge(X, w, rms)
    p21 = predict_antisymmetric_ridge(-X, w, rms)
    assert float(np.max(np.abs(p12 + p21))) == 0.0


# 13. Assembly validator catches hs1 coordinate offset and fixes via LiftOver + flank check
def test_assembly_validator_catches_hs1_offset_and_fixes_via_liftover():
    hg38_seq = _make_synthetic_chrom(400000, seed=17)
    true_hg38_pos = 250000
    hs1_offset = 120000  # hs1 coordinate is 130000 (shifted by +120 kb on GRCh38)
    pos_hs1 = true_hg38_pos - hs1_offset

    class MockChain:
        def convert_coordinate(self, chrom: str, pos0: int):
            return [(chrom, pos0 + hs1_offset, "+", 1000)]

    ref_del = hg38_seq[true_hg38_pos : true_hg38_pos + 120]
    res = verify_and_liftover_sv("chr1", pos_hs1, ref_del, ref_del[0], hg38_seq, liftover_chain=MockChain(), flank_bp=15)
    assert res.passed is True
    assert res.naive_grch38_match is False
    assert res.pos0_grch38 == true_hg38_pos
    assert res.coord_offset_bp == hs1_offset


# 14. Assembly validator rejects gapped pericentromeric flank on short insertion
def test_assembly_validator_rejects_gapped_pericentromeric_flank():
    hg38_seq = _make_synthetic_chrom(100000, seed=18)

    class GappedChain:
        def convert_coordinate(self, chrom: str, pos0: int):
            # Introduce a 4-bp indel distortion on right flank
            shift = 10000 if pos0 <= 40000 else 10004
            return [(chrom, pos0 + shift, "+", 1000)]

    pos_hs1 = 40000
    anchor = hg38_seq[50000]
    res = verify_and_liftover_sv("chr1", pos_hs1, anchor, anchor + ("A" * 300), hg38_seq, liftover_chain=GappedChain(), flank_bp=15)
    assert res.passed is False
    assert res.failure_reason == "gapped_or_noncollinear_flank"


# 15. Assembly validator rejects 1-bp GRCh38 nucleotide mismatch
def test_assembly_validator_rejects_ref_nucleotide_mismatch():
    hg38_seq = _make_synthetic_chrom(50000, seed=19)
    true_ref = hg38_seq[20000:20080]
    corrupted = true_ref[:40] + ("A" if true_ref[40] != "A" else "G") + true_ref[41:]
    res = verify_and_liftover_sv("chr2", 20000, corrupted, corrupted[0], hg38_seq, liftover_chain=None)
    assert res.passed is False
    assert res.failure_reason == "grch38_ref_nucleotide_mismatch"


# 16. SV mechanism classifier covers MEI_Alu, MEI_L1, MEI_SVA, VNTR_DUP, DEL, INV_COMPLEX
def test_classify_sv_allele_mei_vntr_del_inv():
    assert classify_sv_allele("v1", "A", "A" + "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTT" + "T" * 270)[0] == "MEI_Alu"
    assert classify_sv_allele("v2", "A", "A" + "G" * 5200 + "A" * 20)[0] == "MEI_L1"
    assert classify_sv_allele("v3", "A", "A" + ("CCCTCT" * 8) + "G" * 1300)[0] == "MEI_SVA"
    assert classify_sv_allele("v4", "A", "A" + ("ACGTACGTACGTAA" * 8))[0] == "VNTR_DUP"
    assert classify_sv_allele("v5", "A" * 250, "A")[0] == "DEL"
    assert classify_sv_allele("v6", "A" * 400, "<INV>")[0] == "INV_COMPLEX"


# 17. RNA-free diploid LD covariance scaffold polarity registration recovers 50% chromosome flips
def test_rna_free_diploid_ld_scaffold_registration_recovers_flips():
    rng = np.random.default_rng(20)
    n_samp = 140
    chroms = ["chr1", "chr2", "chr3"]
    sv_chroms, sv_gids, sv_pos = [], [], []
    snv_chroms, snv_gids, snv_pos = [], [], []
    true_sv_h1, true_sv_h2, corrupted_sv_h1, corrupted_sv_h2 = [], [], [], []
    snv_h1_list, snv_h2_list = [], []

    for c_idx, ch in enumerate(chroms):
        # Random 50% chromosome-level maternal/paternal scaffold flip per donor!
        donor_flip = rng.random(n_samp) < 0.50
        for g_idx in range(4):
            gid = f"G_{ch}_{g_idx}"
            base_pos = 100000 * (g_idx + 1)
            h1 = (rng.random(n_samp) < 0.35).astype(np.int8)
            h2 = (rng.random(n_samp) < 0.35).astype(np.int8)
            sv_chroms.append(ch)
            sv_gids.append(gid)
            sv_pos.append(base_pos)
            true_sv_h1.append(h1)
            true_sv_h2.append(h2)

            ch1 = np.where(donor_flip, h2, h1)
            ch2 = np.where(donor_flip, h1, h2)
            corrupted_sv_h1.append(ch1)
            corrupted_sv_h2.append(ch2)

            # 3 flanking SNVs in strong LD (95% allele coupling) on true NYGC scaffold
            for k in range(3):
                noise1 = rng.random(n_samp) < 0.03
                noise2 = rng.random(n_samp) < 0.03
                u1 = np.where(noise1, 1 - h1, h1).astype(np.int8)
                u2 = np.where(noise2, 1 - h2, h2).astype(np.int8)
                snv_chroms.append(ch)
                snv_gids.append(gid)
                snv_pos.append(base_pos + (k + 1) * 1500)
                snv_h1_list.append(u1)
                snv_h2_list.append(u2)

    reg_h1, reg_h2, _flips, summary = register_sv_phase_scaffolds(
        sv_chroms, sv_gids, np.array(sv_pos),
        np.vstack(corrupted_sv_h1), np.vstack(corrupted_sv_h2),
        snv_chroms, snv_gids, np.array(snv_pos),
        np.vstack(snv_h1_list), np.vstack(snv_h2_list),
    )
    assert 0.35 <= summary.pre_reg_high_ld_concordance_mean <= 0.65
    assert summary.post_reg_high_ld_concordance_mean > 0.94
    het_mask = np.vstack(true_sv_h1) != np.vstack(true_sv_h2)
    exact_recovery = np.mean(reg_h1[het_mask] == np.vstack(true_sv_h1)[het_mask])
    assert exact_recovery > 0.98


# 18. GFA v1.1 walk bubble parser extracts INS, DEL, and INV VariantAlleles
def test_gfa_v11_walk_bubble_parser_ins_del_inv():
    s1 = "ACGT" * 25   # 100 bp anchor
    s2 = "GCTA" * 30   # 120 bp deleted in HG002#1
    s3 = "TTAA" * 25   # 100 bp anchor
    s_ins = "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTT" + ("A" * 270)  # 310 bp Alu insertion in HG002#2
    s4 = "CCGGAT" * 15 # 90 bp inverted in HG005#1
    s5 = "ATGC" * 25   # 100 bp end anchor

    gfa = [
        f"S\ts1\t{s1}",
        f"S\ts2\t{s2}",
        f"S\ts3\t{s3}",
        f"S\ts_ins\t{s_ins}",
        f"S\ts4\t{s4}",
        f"S\ts5\t{s5}",
        "W\tGRCh38\t0\tchr1\t1000\t1510\t>s1>s2>s3>s4>s5",
        "W\tHG002\t1\tchr1\t0\t390\t>s1>s3>s4>s5",
        "W\tHG002\t2\tchr1\t0\t820\t>s1>s2>s_ins>s3>s4>s5",
        "W\tHG005\t1\tchr1\t0\t510\t>s1>s2>s3<s4>s5",
    ]
    parsed = parse_gfa_bubbles(gfa, ref_sample_prefix="GRCh38", min_sv_len=50)
    assert parsed["HG002#1"][0].sv_class == "DEL"
    assert parsed["HG002#1"][0].sv_len == 120
    assert parsed["HG002#2"][0].sv_class == "MEI_Alu"
    assert parsed["HG002#2"][0].sv_len == 310
    assert parsed["HG005#1"][0].sv_class == "INV_COMPLEX"
    assert parsed["HG005#1"][0].sv_len == 90


# 19. Multi-allelic HPRC v2 wave VCF record decomposition & flank trimming
def test_multiallelic_hprc_vcf_decomposition():
    motif = "ACGTACGTACGTAA"
    ref = "GGG" + motif + "TTT"
    alt1 = "GGG" + (motif * 6) + "TTT"   # +70 bp VNTR expansion
    alt2 = "GGG" + (motif * 12) + "TTT"  # +154 bp VNTR expansion
    alt_snv = "GGGA" + motif[1:] + "TTT" # 1 bp SNV (filtered out by min_sv_len=50)
    decomposed = decompose_multiallelic_vcf_record("chr7", 50000, "HPRC_BUB1", ref, [alt1, alt_snv, alt2], min_sv_len=50)
    assert [idx for idx, _ in decomposed] == [1, 3]
    assert decomposed[0][1].sv_class == "VNTR_DUP"
    assert decomposed[0][1].sv_len == 70
    assert decomposed[1][1].sv_len == 154


# 20. AlphaGenome Breakpoint-Graph pooling captures cryptic TE roadblocks missed by Korsakova 1D stitch
def test_pool_alphagenome_in_memory_and_cryptic_te_nodes():
    ref = _make_synthetic_chrom(260000, seed=21)
    tss = 130000
    exons = [(130000, 130300), (135000, 135400)]
    alu = VariantAllele(132500, ref[132500], ref[132500] + ("T" * 300), "ALU_TRAP", True, "MEI_Alu", 300)

    mp_ref = build_haplotype_graph_sequence(ref, 0, tss, "+", [], window_len=131072)
    mp_alt = build_haplotype_graph_sequence(ref, 0, tss, "+", [alu], window_len=131072)

    preds_ref = {k: np.ones_like(v) * 0.5 for k, v in _make_mock_ag_preds(131072, seed=22).items()}
    preds_alt = {k: v.copy() for k, v in preds_ref.items()}
    bub = mp_alt.insertion_bubbles[0]
    # Implant strong cryptic splice site & antisense transcription inside the intronic Alu bubble!
    preds_alt["splice_probs_1bp"][bub.hap_start + 120, 0] = 0.98
    preds_alt["cage_1bp"][bub.hap_start : bub.hap_end, 1] = 8.0

    out_ref = pool_alphagenome_on_gpu(preds_ref, mp_ref, exons)
    out_alt = pool_alphagenome_on_gpu(preds_alt, mp_alt, exons, return_locus_tracks=True)

    # Korsakova 1D stitch excises the intronic insertion bubble -> identical 4-dim vector!
    assert np.allclose(out_alt["korsakova_stitch"], out_ref["korsakova_stitch"], atol=1e-6)
    # PanPrism captures both antisense interference [14] and cryptic splice trap [15]!
    assert out_alt["panprism_nodes"][14] > 5.0
    assert out_alt["panprism_nodes"][15] > 0.95
    assert "locus_tracks_32bp" in out_alt


# 21. Borzoi / Flashzoi 524kb @ 32bp Breakpoint-Graph pooling
def test_pool_borzoi_flashzoi_32bp_pooling():
    ref = _make_synthetic_chrom(600000, seed=23)
    tss = 300000
    exons = [(300000, 300640), (304000, 304960)]
    alu = VariantAllele(302000, ref[302000], ref[302000] + ("A" * 640), "ALU640", True, "MEI_Alu", 640)
    mp = build_haplotype_graph_sequence(ref, 0, tss, "+", [alu], window_len=524288)
    rng = np.random.default_rng(24)
    bz_preds = {
        k: rng.uniform(0.1, 1.5, size=(6144,)).astype(np.float32)
        for k in ["rna_plus", "rna_minus", "rna_unstr", "cage_plus", "cage_minus", "dnase", "h3k27ac", "h3k4me3"]
    }
    res = pool_borzoi_on_gpu(bz_preds, mp, exons)
    assert res["panprism_nodes"].shape == (9,)
    assert res["korsakova_stitch"].shape == (4,)
    assert res["center_crop_naive"].shape == (4,)
    assert borzoi_9channel_features(res["panprism_nodes"], mode="no_ins_denovo")[7] == 0.0


# 22. Cumulative 7-step ablation channel masking & compression telemetry
def test_ablation_modes_and_compression_telemetry():
    rng = np.random.default_rng(25)
    dpb = rng.normal(size=(10, 22)).astype(np.float32)
    f_full = panprism_12channel_features(dpb, "full")
    f_no3d = panprism_12channel_features(dpb, "no_3d_contact")
    f_noins = panprism_12channel_features(dpb, "no_ins_denovo")
    f_nosp = panprism_12channel_features(dpb, "no_splice_inv")
    f_noex = panprism_12channel_features(dpb, "no_exon_survival")

    assert f_full.shape == (10, 12)
    assert np.all(f_no3d[:, 9] == 0.0)
    assert np.all(f_noins[:, 10:] == 0.0)
    assert np.all(f_nosp[:, 6:8] == 0.0)
    assert np.all(f_noex[:, 1] == 0.0)

    tel = compute_compression_telemetry(n_alphagenome_seqs=12000, n_borzoi_seqs=4000, n_flashzoi_seqs=4000)
    assert tel["in_memory_compression_ratio"] > 150000.0
    parity = measure_fp32_vs_bf16_parity(f_full, f_full + 1e-4)
    assert parity["pearson_r"] > 0.9999


# 23. 20-Seed Permuted-Haplotype Sign-Flip Null Control stays unbiased around 50%
def test_permuted_null_control_unbiased_50_percent():
    rng = np.random.default_rng(26)
    X = rng.normal(size=(160, 12)).astype(np.float32)
    y = (0.9 * X[:, 0] - 0.5 * X[:, 10] + rng.normal(scale=0.5, size=160)).astype(np.float32)
    chroms = np.array([f"chr{(i % 8) + 1}" for i in range(160)])
    null_res = run_permuted_null_control(X, y, chroms, PANPRISM_CHANNEL_NONNEG_MASK, n_seeds=20, strong_mask=np.abs(y) > 0.5)
    assert 0.43 <= null_res["all_dir_acc_mean"] <= 0.57
    assert abs(null_res["all_pearson_r_mean"]) < 0.12


# 24. Leak-free LOCO Trans-PCs strictly exclude target chromosome and test samples
def test_loco_trans_pcs_strictly_excludes_target_chromosome():
    rng = np.random.default_rng(27)
    expr = rng.normal(size=(200, 60)).astype(np.float32)
    g_chroms = np.array(["chr1"] * 40 + ["chr2"] * 80 + ["chr3"] * 80)
    fit_idx = np.arange(0, 30)  # e.g. EUR-only training donors

    z1 = compute_loco_trans_pcs(expr, g_chroms, excluded_chroms={"chr1"}, fit_sample_indices=fit_idx, n_pcs=5)
    expr_corrupted = expr.copy()
    expr_corrupted[:40, :] = 1e6  # Wildly corrupt chr1 expression!
    z2 = compute_loco_trans_pcs(expr_corrupted, g_chroms, excluded_chroms={"chr1"}, fit_sample_indices=fit_idx, n_pcs=5)
    assert np.allclose(z1, z2, atol=1e-6)


# 25. Within-ancestry PEER Z-scores & Rare/Singleton SV Outlier Fine-Mapping
def test_outlier_finemapping_and_within_ancestry_peer_zscores():
    rng = np.random.default_rng(28)
    G, N = 120, 100
    anc = np.array(["EUR"] * 50 + ["AFR"] * 50)
    # Implant strong latent batch factor + population shift
    latent = rng.normal(size=(3, N))
    loadings = rng.normal(size=(G, 3))
    expr = loadings @ latent + rng.normal(scale=0.4, size=(G, N))
    expr[:, 50:] += 3.0  # Population offset on AFR that should NOT trigger false outliers
    expr[5, 72] -= 4.5   # True singleton L1 underexpression outlier in AFR donor #72!

    Z = compute_within_ancestry_peer_zscores(expr, anc, n_peer_pcs=5)
    assert Z[5, 72] < -3.5
    assert abs(float(Z[:, 50:].mean())) < 0.05

    # Fine-map candidate window with 1 causal singleton SV + 4 bystander rare SNVs
    wids = np.array(["W1"] * 5 + ["W2"] * 4)
    z_out = np.array([-4.2] * 5 + [3.8] * 4, dtype=np.float32)
    s2f_delta = np.array([-2.8, 0.05, -0.1, 0.02, 0.0, 2.5, -0.2, 0.1, 0.0], dtype=np.float32)
    _evid, pips = finemap_outlier_windows(wids, z_out, s2f_delta, temperature=0.8)
    assert pips[0] > 0.75
    assert pips[5] > 0.75

    cadd = cadd_sv_heuristic_score([1, 0], [320, 1], [500, 15000], [0, 0], [1, 0], ["MEI_Alu", "SNV"])
    assert cadd[0] > cadd[1]
    ws = watershed_annotation_prior_score(rng.normal(size=(6, 4)), rng.normal(size=(40, 4)), rng.integers(0, 2, size=40))
    assert ws.shape == (6,) and np.all((ws >= 0.0) & (ws <= 1.0))
    mcn = exact_mcnemar_pvalue(np.array([1, 1, 1, 0]), np.array([0, 0, 1, 0]))
    assert mcn["n_a_only_wins"] == 2 and mcn["n_b_only_wins"] == 0


# 26. End-to-end CLI smoke tests across subcommands
def test_cli_end_to_end_subcommands(tmp_path: Path):
    # 1) verify-assembly
    fa_p = tmp_path / "mini.fa"
    seq = _make_synthetic_chrom(150000, seed=30)
    fa_p.write_text(f">chr1\n{seq}\n")
    vcf_p = tmp_path / "mini.vcf"
    del_ref = seq[70000:70080]
    vcf_p.write_text(f"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t70001\tDEL80\t{del_ref}\t{del_ref[0]}\t.\tPASS\tSVTYPE=DEL\n")
    ver_out = tmp_path / "ver.json"
    assert cli_main(["verify-assembly", "--input", str(vcf_p), "--fasta", str(fa_p), "--output", str(ver_out)]) == 0
    assert json.loads(ver_out.read_text())["summary"]["n_passed"] == 1

    # 2) register-phase
    reg_in = tmp_path / "reg_in.npz"
    reg_out = tmp_path / "reg_out.npz"
    h1 = np.array([[1, 0, 1, 0, 1, 0]], dtype=np.int8)
    h2 = np.array([[0, 1, 0, 0, 0, 1]], dtype=np.int8)
    np.savez(reg_in, sv_chroms=["chr1"], sv_gene_ids=["G1"], sv_pos=[5000], sv_h1=h2, sv_h2=h1,
             snv_chroms=["chr1"], snv_gene_ids=["G1"], snv_pos=[5500], snv_h1=h1, snv_h2=h2)
    assert cli_main(["register-phase", "--input-npz", str(reg_in), "--output-npz", str(reg_out)]) == 0
    assert reg_out.exists()

    # 3) infer --dry-run
    man_p = tmp_path / "man.json"
    inf_out = tmp_path / "inf_dry.json"
    man_p.write_text(json.dumps({"loci": [{"gene_id": "G1", "ref_seq": seq, "ref_seq_start0": 0, "tss_pos0": 75000, "strand": "+", "exons0": [[75000, 75300]], "variants": []}]}))
    assert cli_main(["infer", "--manifest-json", str(man_p), "--dry-run", "--output", str(inf_out)]) == 0
    assert json.loads(inf_out.read_text())["n_unique_haplotypes"] == 1

    # 4) couple
    cpl_in = tmp_path / "cpl.npz"
    cpl_out = tmp_path / "cpl.json"
    rng = np.random.default_rng(31)
    dpb = rng.normal(size=(40, 22)).astype(np.float32)
    y_ase = (0.8 * (dpb[:, 0] + dpb[:, 1]) + rng.normal(scale=0.15, size=40)).astype(np.float32)
    np.savez(cpl_in, delta_panprism=dpb, y_ase=y_ase, chroms=[f"chr{(i%4)+1}" for i in range(40)])
    assert cli_main(["couple", "--input-npz", str(cpl_in), "--null-seeds", "5", "--output", str(cpl_out)]) == 0
    assert json.loads(cpl_out.read_text())["loco_biophysical_dir_acc"] > 0.70

    # 5) finemap
    fm_in = tmp_path / "fm.tsv"
    fm_out = tmp_path / "fm_out.tsv"
    pd.DataFrame({"window_id": ["W1", "W1"], "z_outlier": [-3.5, -3.5], "panprism_delta": [-2.2, 0.05]}).to_csv(fm_in, sep="\t", index=False)
    assert cli_main(["finemap", "--input-parquet", str(fm_in), "--output", str(fm_out)]) == 0
    assert fm_out.exists()


# 27. FP32 vs BF16 parity metric & PyTorch native SDPA FlashAttention GQA+RoPE shim
def test_fp32_bf16_parity_and_flashzoi_sdpa_shim():
    from panprism.s2f_runners import _install_pytorch_sdpa_flash_mha_shim, measure_fp32_vs_bf16_parity

    rng = np.random.default_rng(99)
    fp32 = rng.normal(size=(16, 22)).astype(np.float32)
    bf16 = fp32 + rng.normal(scale=1e-3, size=(16, 22)).astype(np.float32)
    par = measure_fp32_vs_bf16_parity(fp32, bf16)
    assert par["pearson_r"] > 0.9999
    assert par["max_abs_diff"] < 0.02

    torch = pytest.importorskip("torch")
    _install_pytorch_sdpa_flash_mha_shim()
    from flash_attn.modules.mha import MHA

    mha = MHA(embed_dim=1536, num_heads=8, num_heads_kv=4, rotary_emb_dim=128, rotary_emb_base=20000.0)
    assert mha.Wqkv.weight.shape == (3072, 1536)
    x = torch.randn(2, 32, 1536)
    y = mha(x)
    assert y.shape == (2, 32, 1536)


# 28. CLI `--help` and all 5 subcommand `--help` invocations exit 0 without argparse %-format crashes
def test_cli_help_and_all_subcommand_helps_exit_zero():
    with pytest.raises(SystemExit) as exc_top:
        cli_main(["--help"])
    assert exc_top.value.code == 0

    for subcmd in ("verify-assembly", "register-phase", "infer", "couple", "finemap"):
        with pytest.raises(SystemExit) as exc_sub:
            cli_main([subcmd, "--help"])
        assert exc_sub.value.code == 0
