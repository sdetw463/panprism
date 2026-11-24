"""
Unit and algebraic invariance tests for panprism grafting, contact physics, masking, and GAT modules:
  1. Divergence-Adaptive Syntenic Grafting (`DASG / V_graft`) two-tier validation
  2. Diagonal-Normalized Physical 3D Contact (`P_phys`) & Borzoi/Flashzoi `undo_squashed_scale`
  3. Pure-PyTorch `SiameseBreakpointGAT`, `SiameseSpliceCompetitionGAT`, `SiameseMagnitudeGAT`
     exact machine-zero (`0.0`) antisymmetry, homozygous identity, and swap invariance
  4. Dual-Mode (`mit` default vs `full`) runner config & `0%` AlphaGenome MIT channel/coordinate masking
  5. CLI subcommand flags and API guarantees
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

import panprism
from panprism import (
    ALPHAGENOME_SUPPORTED_WINDOWS,
    HAS_TORCH,
    MIT_ALLOWED_PANPRISM_V2_INDICES,
    PANPRISM_V2_FEATURE_NAMES,
    apply_squashed_scale,
    build_haplotype_graph_sequence,
    build_v2_runner_spec,
    compute_diagonal_baseline,
    compute_physical_3d_contact,
    get_mode_config,
    graft_sv_dasg,
    mask_features_for_mode,
    mask_graph_for_mode,
    physical_contact_from_raw,
    pool_alphagenome_on_gpu,
    resolve_backbones_for_mode,
    undo_squashed_scale,
    verify_and_graft_sv_two_tier,
    verify_and_liftover_sv,
)
from panprism.cli import main as cli_main


def _make_dna(length: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    return "".join(bases[rng.integers(0, 4, size=length)])


def _flip_base(b: str) -> str:
    return {"A": "G", "G": "A", "C": "T", "T": "C"}[b.upper()]


# ===========================================================================
# 1. Version & Package Integrity
# ===========================================================================


def test_package_version_is_2_0_0(capsys):
    assert panprism.__version__ == "2.0.0"
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "panprism 2.0.0" in captured.out


# ===========================================================================
# 2. DASG (V_graft) Two-Tier Syntenic Grafting Tests
# ===========================================================================


def test_dasg_tier1_tier2_parity_when_v15_passes():
    hg38_seq = _make_dna(10000, seed=101)
    offset = 1200
    # hs1 is identical to hg38 shifted by -offset
    hs1_seq = hg38_seq[offset:]

    class LinearChain:
        def convert_coordinate(self, chrom: str, pos0: int):
            return [(chrom, pos0 + offset, "+", 1000)]

    pos_hs1 = 2000
    ref_del = hs1_seq[pos_hs1 : pos_hs1 + 80]
    alt_del = ref_del[0]

    t1 = verify_and_liftover_sv("chr1", pos_hs1, ref_del, alt_del, hg38_seq, liftover_chain=LinearChain(), flank_bp=15)
    two = verify_and_graft_sv_two_tier(
        "chr1",
        pos_hs1,
        ref_del,
        alt_del,
        grch38_chrom_seq=hg38_seq,
        hs1_chrom_seq=hs1_seq,
        liftover_chain=LinearChain(),
        flank_bp=15,
    )
    assert t1.passed is True
    assert two.passed is True
    assert two.tier == "tier1_v15"
    assert two.dasg_result is None
    assert two.pos0_grch38 == t1.pos0_grch38 == pos_hs1 + offset
    assert two.end0_grch38 == t1.end0_grch38 == pos_hs1 + offset + 80
    assert two.ref_grch38 == t1.verified_ref_seq == ref_del
    assert two.alt_grch38 == alt_del


def test_dasg_rescues_local_ref_divergence_with_zero_anchor_mismatch():
    hs1_seq = _make_dna(6000, seed=102)
    pos_hs1 = 2500
    rlen = 120
    # Create hg38 identical to hs1 shifted by +500 bp EXCEPT a 1-bp divergence inside [pos_hs1, pos_hs1 + rlen)
    hg38_list = list(hs1_seq)
    hg38_list[pos_hs1 + 40] = _flip_base(hg38_list[pos_hs1 + 40])
    prefix = _make_dna(500, seed=103)
    hg38_seq = prefix + "".join(hg38_list)
    offset = 500

    class ShiftChain:
        def convert_coordinate(self, chrom: str, pos0: int):
            return [(chrom, pos0 + offset, "+", 1000)]

    ref_hs1 = hs1_seq[pos_hs1 : pos_hs1 + rlen]
    alt_hs1 = ref_hs1[0]

    res = verify_and_graft_sv_two_tier(
        "chr7",
        pos_hs1,
        ref_hs1,
        alt_hs1,
        grch38_chrom_seq=hg38_seq,
        hs1_chrom_seq=hs1_seq,
        liftover_chain=ShiftChain(),
        flank_bp=15,
        max_search_bp=250,
    )
    assert res.tier1_result.passed is False
    assert res.tier1_result.failure_reason == "grch38_ref_nucleotide_mismatch"
    assert res.passed is True
    assert res.tier == "tier2_vgraft"
    assert res.dasg_result is not None
    assert res.dasg_result.rescued is True
    assert res.dasg_result.left_anchor_mismatch_bp == 0
    assert res.dasg_result.right_anchor_mismatch_bp == 0
    assert res.dasg_result.anchor_mismatch_total == 0
    # Exact 15-bp flanking anchor identity between hs1 and hg38
    g = res.dasg_result
    assert hs1_seq[g.l_hs1 - 15 : g.l_hs1] == hg38_seq[g.l_hg38 - 15 : g.l_hg38]
    assert hs1_seq[g.r_hs1 : g.r_hs1 + 15] == hg38_seq[g.r_hg38 : g.r_hg38 + 15]
    assert res.ref_grch38 == hg38_seq[g.l_hg38 : g.r_hg38]
    assert res.alt_grch38 == hs1_seq[g.l_hs1 : pos_hs1] + alt_hs1 + hs1_seq[pos_hs1 + rlen : g.r_hs1]


def test_dasg_rescues_flank_indel_gap_by_walking_outward():
    hs1_seq = _make_dna(6000, seed=104)
    pos_hs1 = 2200
    # Insert 4 extra bases in hg38 6 bp downstream of pos_hs1 (inside Tier-1's +-15bp flank!)
    ins_at = pos_hs1 + 6
    extra4 = "CGTA"
    hg38_seq = hs1_seq[:ins_at] + extra4 + hs1_seq[ins_at:]

    class IndelChain:
        def convert_coordinate(self, chrom: str, pos0: int):
            if pos0 < ins_at:
                return [(chrom, pos0, "+", 1000)]
            return [(chrom, pos0 + 4, "+", 1000)]

    ref_ins = hs1_seq[pos_hs1]
    alt_ins = ref_ins + ("A" * 280)

    res = verify_and_graft_sv_two_tier(
        "chr2",
        pos_hs1,
        ref_ins,
        alt_ins,
        grch38_chrom_seq=hg38_seq,
        hs1_chrom_seq=hs1_seq,
        liftover_chain=IndelChain(),
        flank_bp=15,
        max_search_bp=250,
    )
    assert res.tier1_result.passed is False
    assert res.tier1_result.failure_reason == "gapped_or_noncollinear_flank"
    assert res.passed is True
    assert res.tier == "tier2_vgraft"
    g = res.dasg_result
    assert g is not None and g.rescued is True
    assert g.d_right_bp >= 5  # walked past the 4-bp indel at pos_hs1 + 6
    assert g.span_diff_bp == 4
    assert g.anchor_mismatch_total == 0
    assert hs1_seq[g.r_hs1 : g.r_hs1 + 15] == hg38_seq[g.r_hg38 : g.r_hg38 + 15]


def test_dasg_rejects_mismatched_anchors_hs1_mismatch_and_span_distortion():
    hs1_seq = _make_dna(6000, seed=105)
    pos_hs1 = 2500

    class IdentChain:
        def convert_coordinate(self, chrom: str, pos0: int):
            return [(chrom, pos0, "+", 1000)]

    # 1) Wrong REF on hs1 -> rejected immediately
    bad_ref = _flip_base(hs1_seq[pos_hs1]) + hs1_seq[pos_hs1 + 1 : pos_hs1 + 60]
    r_hs1_mm = graft_sv_dasg("chr1", pos_hs1, bad_ref, bad_ref[0], hs1_seq, hs1_seq, IdentChain())
    assert r_hs1_mm.rescued is False
    assert r_hs1_mm.fail_reason == "hs1_ref_mismatch"

    # 2) Heavily diverged right flank (every 10th base mutated across +300bp) -> no exact 15-bp right anchor
    hg38_div = list(hs1_seq)
    for k in range(pos_hs1 + 60, pos_hs1 + 400, 10):
        hg38_div[k] = _flip_base(hg38_div[k])
    r_no_anc = graft_sv_dasg(
        "chr1", pos_hs1, hs1_seq[pos_hs1 : pos_hs1 + 60], hs1_seq[pos_hs1], hs1_seq, "".join(hg38_div), IdentChain(), max_search_bp=200
    )
    assert r_no_anc.rescued is False
    assert "no_right_anchor" in r_no_anc.fail_reason

    # 3) Excessive span distortion (+120 bp gap > max(50, 0.10*60)) -> rejected
    hg38_big_gap = hs1_seq[: pos_hs1 + 30] + ("T" * 120) + hs1_seq[pos_hs1 + 30 :]

    class BigGapChain:
        def convert_coordinate(self, chrom: str, pos0: int):
            return [(chrom, pos0 if pos0 < pos_hs1 + 30 else pos0 + 120, "+", 1000)]

    r_dist = graft_sv_dasg(
        "chr1", pos_hs1, hs1_seq[pos_hs1 : pos_hs1 + 60], hs1_seq[pos_hs1], hs1_seq, hg38_big_gap, BigGapChain()
    )
    assert r_dist.rescued is False
    assert r_dist.fail_reason == "span_distortion_exceeded"

    # 4) Unmapped start is NOT eligible for Tier-2 rescue under verify_and_graft_sv_two_tier
    class UnmappedStartChain:
        def convert_coordinate(self, chrom: str, pos0: int):
            if pos0 == pos_hs1:
                return []
            return [(chrom, pos0, "+", 1000)]

    r_unmap = verify_and_graft_sv_two_tier(
        "chr1", pos_hs1, hs1_seq[pos_hs1 : pos_hs1 + 60], hs1_seq[pos_hs1], hs1_seq, hs1_seq, UnmappedStartChain()
    )
    assert r_unmap.passed is False
    assert r_unmap.tier == "failed"
    assert r_unmap.failure_reason == "unmapped_or_noncollinear_start"


# ===========================================================================
# 3. Physical 3D Contact (`P_phys`) & `undo_squashed_scale` Tests
# ===========================================================================


def test_physical_3d_contact_preserves_negative_log_oe_and_power_law_decay():
    n = 32
    rng = np.random.default_rng(202)
    # 1) Constant-per-diagonal map -> C_norm == 0 -> exact ((d + 5000) / 5000)^(-0.87) decay!
    idx = np.arange(n)
    dist_bins = np.abs(idx[:, None] - idx[None, :])
    cmap_flat = -0.75 + 0.02 * dist_bins.astype(np.float32)  # predominantly negative!
    p_flat, c_norm_flat, diag_meds = compute_physical_3d_contact(cmap_flat, bin_bp=2048.0)

    expected_decay = np.power((dist_bins * 2048.0 + 5000.0) / 5000.0, -0.87)
    assert np.allclose(c_norm_flat, 0.0, atol=1e-6)
    assert np.allclose(p_flat, expected_decay, rtol=1e-6, atol=1e-6)
    assert np.all(p_flat > 0.0)

    # 2) Signed negative-heavy log(O/E) map (~70% negative, mirroring AlphaGenome contact_maps)
    raw = rng.normal(loc=-0.45, scale=0.65, size=(n, n)).astype(np.float32)
    raw = 0.5 * (raw + raw.T)
    raw[2, 12] = -4.0  # extreme negative -> clipped at -2.5
    raw[12, 2] = -4.0
    raw[4, 14] = +4.0  # extreme positive -> clipped at +2.5
    raw[14, 4] = +4.0

    p_phys, c_norm, z_3d, d_med, d_std = compute_physical_3d_contact(raw, bin_bp=2048.0, return_z_score=True)
    assert p_phys.shape == (n, n)
    assert np.allclose(p_phys, p_phys.T, atol=1e-7)
    # Negative C_norm entries MUST remain strictly positive physical probabilities (NO zero-clamp!)
    neg_mask = c_norm < -0.1
    assert np.any(neg_mask)
    assert np.all(p_phys[neg_mask] > 0.0)
    assert np.all(p_phys[neg_mask] < expected_decay[neg_mask])
    # Saturation bounds at exp(-2.5) and exp(+2.5)
    assert np.isclose(p_phys[2, 12], expected_decay[2, 12] * np.exp(-2.5), rtol=1e-5)
    assert np.isclose(p_phys[4, 14], expected_decay[4, 14] * np.exp(+2.5), rtol=1e-5)

    # 3) Scalar helper matches matrix operator identically
    cn_s, z_s, pp_s = physical_contact_from_raw(
        float(raw[5, 11]), bin_sep=6, dist_bp=6 * 2048.0, diag_med=d_med, diag_std=d_std
    )
    assert np.isclose(cn_s, float(c_norm[5, 11]), atol=1e-6)
    assert np.isclose(z_s, float(z_3d[5, 11]), atol=1e-6)
    assert np.isclose(pp_s, float(p_phys[5, 11]), atol=1e-6)

    # 4) 3D multi-channel contact map (N, N, C) + FWD/RC averaging
    raw3 = np.stack([raw, raw * 0.5], axis=-1)
    p3, cn3, dm3 = compute_physical_3d_contact(raw3, contact_map_rc=np.flip(raw3, axis=(0, 1)), bin_bp=2048.0)
    assert p3.shape == (n, n, 2)
    assert dm3.shape == (n, 2)
    assert np.allclose(p3[:, :, 0], p_phys, atol=1e-6)
    assert np.all(p3 > 0.0)


def test_pool_alphagenome_physical_3d_vs_v1_zero_clamp():
    L = 32768
    ref = _make_dna(60000, seed=203)
    mp = build_haplotype_graph_sequence(ref, 0, 30000, "+", [], window_len=L)
    n_cbins = L // 2048
    preds = {
        "rna_1bp": np.ones((L, 3), dtype=np.float32),
        "cage_1bp": np.ones((L, 2), dtype=np.float32),
        "atac_1bp": np.ones((L,), dtype=np.float32),
        "dnase_1bp": np.ones((L,), dtype=np.float32),
        "histone_128bp": np.ones((L // 128, 4), dtype=np.float32),
        "splice_probs_1bp": np.zeros((L, 4), dtype=np.float32),
        "splice_usage_1bp": np.zeros((L, 2), dtype=np.float32),
        # Strictly negative log(O/E) map: v1 clamp(min=0) collapses to 0.0, v2 P_phys stays strictly > 0!
        "contact_gm12878": np.full((n_cbins, n_cbins), -0.6, dtype=np.float32),
    }
    out_v1 = pool_alphagenome_on_gpu(preds, mp, [(30000, 30400)], use_physical_3d=False)
    out_v2 = pool_alphagenome_on_gpu(preds, mp, [(30000, 30400)], use_physical_3d=True)
    assert out_v1["panprism_nodes"][9] == 0.0
    assert out_v1["panprism_nodes"][10] == 0.0
    assert out_v2["panprism_nodes"][9] > 0.0
    assert out_v2["panprism_nodes"][10] > 0.0


@pytest.mark.parametrize("old_transform", [True, False])
@pytest.mark.parametrize("scale", [0.30, 0.01])
def test_undo_squashed_scale_exact_invertibility(old_transform: bool, scale: float):
    # Cover zero, sub-clip, near-clip, and deeply clipped values (> 384^(4/3) ~ 2806)
    y_true = np.array([0.0, 1e-4, 0.25, 5.0, 120.0, 2800.0, 8500.0, 50000.0, 250000.0], dtype=np.float64)
    squashed = apply_squashed_scale(y_true, scale=scale, clip_soft=384.0, power=0.75, old_transform=old_transform)
    recovered = undo_squashed_scale(squashed, scale=scale, clip_soft=384.0, power=0.75, old_transform=old_transform)
    np.testing.assert_allclose(recovered, y_true, rtol=1e-6, atol=1e-6)

    # Scalar roundtrip
    s_in = 12345.67
    s_sq = apply_squashed_scale(s_in, scale=scale, clip_soft=384.0, power=0.75, old_transform=old_transform)
    s_rec = undo_squashed_scale(s_sq, scale=scale, clip_soft=384.0, power=0.75, old_transform=old_transform)
    assert np.isclose(s_rec, s_in, rtol=1e-6)

    if HAS_TORCH:
        import torch

        yt = torch.tensor(y_true, dtype=torch.float32)
        sq_t = apply_squashed_scale(yt, scale=scale, clip_soft=384.0, power=0.75, old_transform=old_transform)
        rec_t = undo_squashed_scale(sq_t, scale=scale, clip_soft=384.0, power=0.75, old_transform=old_transform)
        np.testing.assert_allclose(rec_t.numpy(), y_true.astype(np.float32), rtol=1e-4, atol=1e-3)


def test_physical_3d_contact_torch_numpy_parity():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(204)
    raw = rng.normal(loc=-0.3, scale=0.8, size=(16, 16)).astype(np.float32)
    p_np, cn_np, dm_np = compute_physical_3d_contact(raw, bin_bp=2048.0)
    p_th, cn_th, dm_th = compute_physical_3d_contact(torch.from_numpy(raw), bin_bp=2048.0)
    np.testing.assert_allclose(p_th.numpy(), p_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cn_th.numpy(), cn_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dm_th.numpy(), dm_np, rtol=1e-5, atol=1e-5)


# ===========================================================================
# 4. SiameseGAT Exact `0.0` Invariants (CPU <1s)
# ===========================================================================


def _make_synthetic_paired_graph_batch(batch_size: int = 6, n_nodes: int = 10, node_dim: int = 8, edge_dim: int = 6, readout_dim: int = 2, seed: int = 301):
    import torch
    from panprism.siamese_gat import PairedGraphBatch

    g = torch.Generator().manual_seed(seed)

    def _one_side(offset_seed: int):
        gg = torch.Generator().manual_seed(seed + offset_seed)
        nx = torch.randn((batch_size, n_nodes, node_dim), generator=gg, dtype=torch.float32)
        nt = torch.randint(0, 5, (batch_size, n_nodes), generator=gg, dtype=torch.long)
        nt[:, 0] = 0
        nt[:, 1] = 1
        na = torch.rand((batch_size, n_nodes, 1), generator=gg, dtype=torch.float32)
        na[:, 0, 0] = 1.0
        # Force slot -1 to be an inactive insertion bubble (node_act == 0.0)
        nt[:, -1] = 3
        na[:, -1, 0] = 0.0
        nm = torch.ones((batch_size, n_nodes), dtype=torch.bool)
        nm[:, -2] = False
        ex = torch.randn((batch_size, n_nodes, n_nodes, edge_dim), generator=gg, dtype=torch.float32)
        em = nm.unsqueeze(2) & nm.unsqueeze(1)
        rw = torch.zeros((batch_size, n_nodes, readout_dim), dtype=torch.float32)
        rw[:, 0, 0] = 1.0
        if readout_dim > 1:
            rw[:, 1:4, 1] = 1.0 / 3.0
        if readout_dim > 2:
            rw[:, 4:6, 2] = 0.5
        return nx, nt, na, nm, ex, em, rw

    s1 = _one_side(0)
    s2 = _one_side(77)
    return PairedGraphBatch(*s1, *s2)


def test_siamese_breakpoint_gat_exact_zero_invariants_and_inert_bubble_gate():
    torch = pytest.importorskip("torch")
    from panprism.siamese_gat import SiameseBreakpointGAT, count_parameters, verify_siamese_invariants

    torch.manual_seed(401)
    batch = _make_synthetic_paired_graph_batch(batch_size=6, n_nodes=10, node_dim=8, edge_dim=6, readout_dim=2)
    model = SiameseBreakpointGAT(node_dim=8, edge_dim=6, readout_dim=2, hidden_dim=16, edge_hidden_dim=8, num_heads=4, num_layers=2)
    model.eval()
    assert count_parameters(model) > 0

    inv = verify_siamese_invariants(model, batch)
    assert inv["max_antisym_err"] == 0.0
    assert inv["max_zero_sv_err"] == 0.0

    # Also verify exact 0.0 in float32 forward_pair
    with torch.no_grad():
        f12 = model.forward_pair(batch)
        f21 = model.forward_pair(batch.swap_haplotypes())
        f11 = model.forward_pair(batch.identical_pair())
    assert float(torch.max(torch.abs(f12 + f21)).item()) == 0.0
    assert float(torch.max(torch.abs(f11)).item()) == 0.0

    # Inert bubble (slot -1 has node_act == 0.0): mutating its node_x wildly changes output by exact 0.0!
    base_out = model.predict_antisymmetric(batch)
    batch_mut = _make_synthetic_paired_graph_batch(batch_size=6, n_nodes=10, node_dim=8, edge_dim=6, readout_dim=2)
    batch_mut.node_x1[:, -1, :] = 1234.5
    mut_out = model.predict_antisymmetric(batch_mut)
    assert float(np.max(np.abs(base_out - mut_out))) == 0.0


def test_siamese_splice_competition_gat_exact_zero_invariants():
    torch = pytest.importorskip("torch")
    from panprism.siamese_gat import SiameseSpliceCompetitionGAT, verify_siamese_invariants

    torch.manual_seed(402)
    batch = _make_synthetic_paired_graph_batch(batch_size=6, n_nodes=10, node_dim=9, edge_dim=10, readout_dim=3)
    w_prior = [1.2, -1.5, -1.5, -0.5, -0.5, -0.5, -1.5, -1.5, 0.8]
    model = SiameseSpliceCompetitionGAT(node_dim=9, edge_dim=10, w_prior_init=w_prior, hidden_dim=16, num_heads=4)
    model.eval()

    inv = verify_siamese_invariants(model, batch)
    assert inv["max_antisym_err"] == 0.0
    assert inv["max_zero_sv_err"] == 0.0

    with torch.no_grad():
        f12 = model(batch)
        f21 = model(batch.swap_haplotypes())
        f11 = model(batch.identical_pair())
    assert float(torch.max(torch.abs(f12 + f21)).item()) == 0.0
    assert float(torch.max(torch.abs(f11)).item()) == 0.0


def test_siamese_magnitude_gat_exact_zero_swap_invariance_and_identity():
    torch = pytest.importorskip("torch")
    from panprism.siamese_gat import SiameseMagnitudeGAT, verify_siamese_invariants

    torch.manual_seed(403)
    batch = _make_synthetic_paired_graph_batch(batch_size=6, n_nodes=10, node_dim=8, edge_dim=6, readout_dim=2)
    model = SiameseMagnitudeGAT(node_dim=8, edge_dim=6, readout_dim=2, hidden_dim=16, edge_hidden_dim=8, num_heads=4, num_layers=2)
    model.eval()

    inv = verify_siamese_invariants(model, batch)
    assert inv["max_swap_err"] == 0.0
    assert inv["max_zero_sv_err"] == 0.0

    g12 = model.predict_mag(batch)
    g21 = model.predict_mag(batch.swap_haplotypes())
    g11 = model.predict_mag(batch.identical_pair())
    assert np.all(g12 >= 0.0)
    assert float(np.max(np.abs(g12 - g21))) == 0.0
    assert float(np.max(np.abs(g11))) == 0.0

    with torch.no_grad():
        l12 = model.forward_logit(batch)
        l21 = model.forward_logit(batch.swap_haplotypes())
    assert float(torch.max(torch.abs(l12 - l21)).item()) == 0.0


def test_siamese_optional_torch_guard_behavior():
    # Package import succeeds regardless of torch availability; if torch is absent, instantiation raises ImportError
    if not HAS_TORCH:
        with pytest.raises(ImportError, match="PyTorch"):
            panprism.SiameseBreakpointGAT(node_dim=4, edge_dim=4)
    else:
        assert HAS_TORCH is True


# ===========================================================================
# 5. Dual-Mode (`mit` default vs `full`) & MIT 0% AlphaGenome Masking Tests
# ===========================================================================


def test_dual_mode_registry_and_mit_blocks_alphagenome():
    mit_cfg = get_mode_config("mit")
    full_cfg = get_mode_config("full")
    assert mit_cfg.uses_alphagenome is False
    assert mit_cfg.alphagenome_track_fraction == 0.0
    assert mit_cfg.alphagenome_coord_fraction == 0.0
    assert mit_cfg.backbones == ("flashzoi-4x", "pangolin-12x")

    assert full_cfg.uses_alphagenome is True
    assert full_cfg.backbones == ("alphagenome", "flashzoi-4x", "pangolin-12x")
    assert 524288 in ALPHAGENOME_SUPPORTED_WINDOWS
    assert 1048576 in ALPHAGENOME_SUPPORTED_WINDOWS

    assert resolve_backbones_for_mode("mit", "auto") == ["flashzoi-4x", "pangolin-12x"]
    assert resolve_backbones_for_mode("full", "auto") == ["alphagenome", "flashzoi-4x", "pangolin-12x"]
    with pytest.raises(ValueError, match="mode='mit'"):
        resolve_backbones_for_mode("mit", "alphagenome")

    spec_mit = build_v2_runner_spec(mode="mit")
    assert spec_mit["uses_alphagenome"] is False
    assert spec_mit["alphagenome_track_fraction"] == 0.0
    assert spec_mit["alphagenome_coord_fraction"] == 0.0
    assert spec_mit["window_len"] == 524288

    spec_full_1m = build_v2_runner_spec(mode="full", window_len=1048576)
    assert spec_full_1m["uses_alphagenome"] is True
    assert spec_full_1m["window_len"] == 1048576
    assert spec_full_1m["use_physical_3d"] is True


def test_mit_mode_feature_masking_zeros_all_alphagenome_tracks():
    rng = np.random.default_rng(501)
    X = rng.uniform(0.5, 2.0, size=(12, len(PANPRISM_V2_FEATURE_NAMES))).astype(np.float32)

    X_mit, keep_mit = mask_features_for_mode(X, PANPRISM_V2_FEATURE_NAMES, mode="mit")
    assert tuple(np.nonzero(keep_mit)[0].tolist()) == MIT_ALLOWED_PANPRISM_V2_INDICES
    # All AG 1D/3D channels (including index 23 `fz_ep_3d_gm_abc`!) are strictly 0.0
    assert np.all(X_mit[:, ~keep_mit] == 0.0)
    assert np.all(X_mit[:, 23] == 0.0)
    # MIT Flashzoi-4x 1D + geometric exon survival channels are preserved identically
    np.testing.assert_array_equal(X_mit[:, list(MIT_ALLOWED_PANPRISM_V2_INDICES)], X[:, list(MIT_ALLOWED_PANPRISM_V2_INDICES)])

    X_full, keep_full = mask_features_for_mode(X, PANPRISM_V2_FEATURE_NAMES, mode="full")
    assert np.all(keep_full)
    np.testing.assert_array_equal(X_full, X)


def test_mit_mode_graph_masking_zero_alphagenome_tracks_and_coordinates():
    rng = np.random.default_rng(502)

    # 1) ASE schema (32 nodes: slots 7..22 are AG peak-called enhancers)
    nodes_ase = rng.uniform(0.2, 1.5, size=(4, 32, 24)).astype(np.float32)
    mask_ase = np.ones((4, 32), dtype=bool)
    edges_ase = rng.uniform(0.2, 1.5, size=(4, 32, 32, 12)).astype(np.float32)
    res_ase = mask_graph_for_mode(nodes_ase, mask_ase, edges_ase, schema="ase", mode="mit")
    assert res_ase["ag_channel_count"] == 0
    assert res_ase["ag_coord_slot_count"] == 0
    assert np.all(res_ase["node_mask"][:, 7:23] == False)  # noqa: E712
    assert np.all(res_ase["nodes"][:, 7:23, :] == 0.0)
    assert np.all(res_ase["nodes"][:, :, 6:22] == 0.0)
    assert np.all(res_ase["edges"][:, :, :, [3, 4, 5, 6, 7, 8, 10, 11]] == 0.0)
    # TSS/Exon Flashzoi channel [23] preserved on canonical slots 0..6
    np.testing.assert_array_equal(res_ase["nodes"][:, :7, 23], nodes_ase[:, :7, 23])

    # 2) CRISPRi schema (32 nodes: slots 3..31 are AG peak-called background peaks)
    nodes_cr = rng.uniform(0.2, 1.5, size=(4, 32, 24)).astype(np.float32)
    mask_cr = np.ones((4, 32), dtype=bool)
    edges_cr = rng.uniform(0.2, 1.5, size=(4, 32, 32, 12)).astype(np.float32)
    res_cr = mask_graph_for_mode(nodes_cr, mask_cr, edges_cr, schema="crispri", mode="mit")
    assert res_cr["ag_channel_count"] == 0
    assert res_cr["ag_coord_slot_count"] == 0
    assert np.all(res_cr["node_mask"][:, 3:] == False)  # noqa: E712
    assert np.all(res_cr["nodes"][:, :, 6:20] == 0.0)
    assert np.all(res_cr["edges"][:, :, :, 4:] == 0.0)

    # 3) Splice schema (18 nodes, 18 node dims, 10 edge dims + slot-17 geometric act)
    nodes_sp = rng.uniform(0.2, 1.0, size=(4, 18, 18)).astype(np.float32)
    mask_sp = np.ones((4, 18), dtype=bool)
    edges_sp = rng.uniform(0.2, 1.0, size=(4, 18, 18, 10)).astype(np.float32)
    act_sp = rng.uniform(0.5, 1.0, size=(4, 18, 1)).astype(np.float32)
    res_sp = mask_graph_for_mode(nodes_sp, mask_sp, edges_sp, schema="splice", mode="mit", node_act=act_sp)
    assert res_sp["ag_channel_count"] == 0
    assert res_sp["ag_coord_slot_count"] == 0
    assert np.all(res_sp["nodes"][:, :, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17]] == 0.0)
    assert np.all(res_sp["edges"][:, :, :, [2, 9]] == 0.0)
    expected_slot17_act = 0.25 * (1.0 - nodes_sp[:, 0, 0]) * nodes_sp[:, 17, 0]
    np.testing.assert_allclose(res_sp["node_act"][:, 17, 0], expected_slot17_act, atol=1e-6)


# ===========================================================================
# 6. CLI dual-mode smoke tests
# ===========================================================================


def test_cli_v2_dual_mode_infer_dry_run(tmp_path: Path):
    seq = _make_dna(150000, seed=601)
    man_p = tmp_path / "manifest_v2.json"
    man_p.write_text(
        json.dumps(
            {
                "loci": [
                    {
                        "gene_id": "ENSG_TEST",
                        "ref_seq": seq,
                        "ref_seq_start0": 0,
                        "tss_pos0": 75000,
                        "strand": "+",
                        "exons0": [[75000, 75400]],
                        "variants": [],
                    }
                ]
            }
        )
    )

    # Default mode is open-source (no AlphaGenome tracks)
    out_mit = tmp_path / "dry_mit.json"
    assert cli_main(["infer", "--manifest-json", str(man_p), "--dry-run", "--output", str(out_mit)]) == 0
    rep_mit = json.loads(out_mit.read_text())
    assert rep_mit["version"] == "2.0.0"
    assert rep_mit["mode"] == "mit"
    assert rep_mit["uses_alphagenome"] is False
    assert rep_mit["alphagenome_track_fraction"] == 0.0
    assert rep_mit["alphagenome_coord_fraction"] == 0.0
    assert rep_mit["backbones"] == ["flashzoi-4x", "pangolin-12x"]
    assert rep_mit["window_len"] == 524288

    # Full mode supports 1 Mb AlphaGenome window
    out_full = tmp_path / "dry_full.json"
    assert (
        cli_main(
            [
                "infer",
                "--manifest-json",
                str(man_p),
                "--mode",
                "full",
                "--window-len",
                "1048576",
                "--dry-run",
                "--output",
                str(out_full),
            ]
        )
        == 0
    )
    rep_full = json.loads(out_full.read_text())
    assert rep_full["mode"] == "full"
    assert rep_full["uses_alphagenome"] is True
    assert rep_full["window_len"] == 1048576

    # Explicitly combining --mode mit with --backbone alphagenome raises ValueError
    with pytest.raises(ValueError, match="mode='mit'"):
        cli_main(
            [
                "infer",
                "--manifest-json",
                str(man_p),
                "--mode",
                "mit",
                "--backbone",
                "alphagenome",
                "--dry-run",
                "--output",
                str(tmp_path / "bad.json"),
            ]
        )
