"""
Divergence-Adaptive Syntenic Grafting (`DASG / V_graft`) Two-Tier Cross-Assembly Validator.

Upgrades Tier-1 `V_15` (`verify_and_liftover_sv`: 100% exact GRCh38 REF + ±15 bp gapless flank)
with Tier-2 `V_graft` (`DASG`): for variants failing `V_15` due to local reference divergence
(`grch38_ref_nucleotide_mismatch` or `gapped_or_noncollinear_flank`), searches outward up to
`±max_search_bp` (default `±250 bp`) on the source assembly (`T2T-CHM13v2.0 / hs1`) for the
nearest collinear `anchor_bp`-bp (default `15 bp`) left and right flanking anchors with:
  1. 100.0% exact nucleotide match (`0` mismatches) between `hs1` and `GRCh38` on `+` strand;
  2. Zero ambiguous `'N'` bases inside anchors or grafted spans;
  3. Bounded span distortion `|span_hg38 - span_hs1| <= max(50, 0.10 * span_hs1)`.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from panprism.assembly_validator import AssemblyCheckResult, verify_and_liftover_sv

DEFAULT_DASG_ELIGIBLE_FAILURES: Tuple[str, ...] = (
    "grch38_ref_nucleotide_mismatch",
    "gapped_or_noncollinear_flank",
)


@dataclass
class DASGResult:
    """Outcome of Tier-2 Dual-Anchor Syntenic Graph Grafting (`V_graft`)."""

    rescued: bool
    l_hs1: int = -1
    r_hs1: int = -1
    l_hg38: int = -1
    r_hg38: int = -1
    d_left_bp: int = -1
    d_right_bp: int = -1
    span_hs1: int = 0
    span_hg38: int = 0
    span_diff_bp: int = 0
    est_pos0_hg38: int = -1
    left_anchor_mismatch_bp: int = -1
    right_anchor_mismatch_bp: int = -1
    ref_graft: str = ""
    alt_graft: str = ""
    hg38_span_seq: str = ""
    fail_reason: str = ""

    @property
    def anchor_mismatch_total(self) -> int:
        if not self.rescued:
            return -1
        return int(self.left_anchor_mismatch_bp) + int(self.right_anchor_mismatch_bp)


@dataclass
class TwoTierAssemblyResult:
    """Unified output of Tier-1 `V_15` + Tier-2 `DASG (V_graft)` two-tier syntenic validation."""

    passed: bool
    tier: str  # "tier1_v15" | "tier2_vgraft" | "failed"
    chrom_grch38: str
    pos0_source: int
    pos0_grch38: int
    end0_grch38: int
    ref_grch38: str
    alt_grch38: str
    source_ref_graft: str
    tier1_result: AssemblyCheckResult
    dasg_result: Optional[DASGResult] = None
    failure_reason: str = ""


def _lift_left_anchor(
    chrom: str,
    l_hs1: int,
    hs1_seq: str,
    hg38_seq: str,
    liftover_chain: Any,
    anchor_bp: int = 15,
) -> Optional[Tuple[int, int]]:
    """
    Check whether the `anchor_bp`-bp left flanking anchor `[l_hs1 - anchor_bp, l_hs1)` lifts
    gaplessly on `+` strand to `[l_hg38 - anchor_bp, l_hg38)` with 100.0% exact base match
    and zero `'N'` bases. Returns `(l_hs1, l_hg38)` or `None`.
    """
    a0 = l_hs1 - anchor_bp
    if a0 < 0 or l_hs1 > len(hs1_seq):
        return None
    anc_hs1 = hs1_seq[a0:l_hs1].upper()
    if "N" in anc_hs1:
        return None
    h0 = liftover_chain.convert_coordinate(chrom, a0)
    if not h0 or h0[0][0] != chrom or h0[0][2] != "+":
        return None
    h1 = liftover_chain.convert_coordinate(chrom, l_hs1 - 1)
    if not h1 or h1[0][0] != chrom or h1[0][2] != "+":
        return None
    g0 = int(h0[0][1])
    g1 = int(h1[0][1])
    if g1 - g0 != anchor_bp - 1:
        return None
    l_hg38 = g0 + anchor_bp
    if g0 < 0 or l_hg38 > len(hg38_seq):
        return None
    if hg38_seq[g0:l_hg38].upper() != anc_hs1:
        return None
    return (l_hs1, l_hg38)


def _lift_right_anchor(
    chrom: str,
    r_hs1: int,
    hs1_seq: str,
    hg38_seq: str,
    liftover_chain: Any,
    anchor_bp: int = 15,
) -> Optional[Tuple[int, int]]:
    """
    Check whether the `anchor_bp`-bp right flanking anchor `[r_hs1, r_hs1 + anchor_bp)` lifts
    gaplessly on `+` strand to `[r_hg38, r_hg38 + anchor_bp)` with 100.0% exact base match
    and zero `'N'` bases. Returns `(r_hs1, r_hg38)` or `None`.
    """
    b1 = r_hs1 + anchor_bp
    if r_hs1 < 0 or b1 > len(hs1_seq):
        return None
    anc_hs1 = hs1_seq[r_hs1:b1].upper()
    if "N" in anc_hs1:
        return None
    h0 = liftover_chain.convert_coordinate(chrom, r_hs1)
    if not h0 or h0[0][0] != chrom or h0[0][2] != "+":
        return None
    h1 = liftover_chain.convert_coordinate(chrom, b1 - 1)
    if not h1 or h1[0][0] != chrom or h1[0][2] != "+":
        return None
    r_hg38 = int(h0[0][1])
    g1 = int(h1[0][1])
    if g1 - r_hg38 != anchor_bp - 1:
        return None
    if r_hg38 < 0 or r_hg38 + anchor_bp > len(hg38_seq):
        return None
    if hg38_seq[r_hg38 : r_hg38 + anchor_bp].upper() != anc_hs1:
        return None
    return (r_hs1, r_hg38)


def graft_sv_dasg(
    chrom: str,
    pos0_hs1: int,
    ref_raw: str,
    alt_raw: str,
    hs1_seq: str,
    hg38_seq: str,
    liftover_chain: Any,
    max_search_bp: int = 250,
    anchor_bp: int = 15,
    max_span_abs_tol: float = 50.0,
    max_span_rel_tol: float = 0.10,
    max_candidates_per_side: int = 6,
) -> DASGResult:
    """
    Attempt Tier-2 Divergence-Adaptive Syntenic Grafting (`V_graft`) for an SV on `hs1`.

    Searches outward up to `±max_search_bp` on `hs1` for the nearest collinear `anchor_bp`-bp
    exact-match anchors `(L_hs1 -> L_hg38, R_hs1 -> R_hg38)` with bounded span distortion and
    verifies `0` anchor mismatches before constructing the grafted alleles:
      - `hg38_span_seq = hg38[L_hg38 : R_hg38]` (GRCh38 interval replaced)
      - `ref_graft     = hs1[L_hs1 : R_hs1]`    (source reference haplotype span)
      - `alt_graft     = hs1[L_hs1 : pos0_hs1] + alt_raw + hs1[end0_hs1 : R_hs1]`
    """
    if anchor_bp <= 0:
        raise ValueError(f"anchor_bp must be positive, got {anchor_bp}")
    ref_u = ref_raw.upper()
    alt_u = alt_raw.upper()
    rlen = len(ref_u)
    end0_hs1 = pos0_hs1 + rlen

    if rlen == 0 or pos0_hs1 < anchor_bp or end0_hs1 + anchor_bp > len(hs1_seq):
        return DASGResult(rescued=False, fail_reason="hs1_out_of_bounds")
    if "N" in alt_u or "N" in ref_u:
        return DASGResult(rescued=False, fail_reason="allele_contains_N")
    if hs1_seq[pos0_hs1:end0_hs1].upper() != ref_u:
        return DASGResult(rescued=False, fail_reason="hs1_ref_mismatch")

    left_cands: List[Tuple[int, int, int]] = []   # (d_L, l_hs1, l_hg38)
    right_cands: List[Tuple[int, int, int]] = []  # (d_R, r_hs1, r_hg38)

    max_dl = min(max_search_bp + 1, pos0_hs1 - anchor_bp + 1)
    for d_l in range(0, max_dl):
        hit = _lift_left_anchor(chrom, pos0_hs1 - d_l, hs1_seq, hg38_seq, liftover_chain, anchor_bp=anchor_bp)
        if hit is not None:
            left_cands.append((d_l, hit[0], hit[1]))
            if len(left_cands) >= max_candidates_per_side:
                break

    if not left_cands:
        return DASGResult(rescued=False, fail_reason=f"no_left_anchor_within_{max_search_bp}bp")

    max_dr = min(max_search_bp + 1, len(hs1_seq) - anchor_bp - end0_hs1 + 1)
    for d_r in range(0, max_dr):
        hit = _lift_right_anchor(chrom, end0_hs1 + d_r, hs1_seq, hg38_seq, liftover_chain, anchor_bp=anchor_bp)
        if hit is not None:
            right_cands.append((d_r, hit[0], hit[1]))
            if len(right_cands) >= max_candidates_per_side:
                break

    if not right_cands:
        return DASGResult(rescued=False, fail_reason=f"no_right_anchor_within_{max_search_bp}bp")

    pairs = [
        (d_l + d_r, d_l, l_hs1, l_hg38, d_r, r_hs1, r_hg38)
        for (d_l, l_hs1, l_hg38) in left_cands
        for (d_r, r_hs1, r_hg38) in right_cands
    ]
    pairs.sort(key=lambda x: x[0])

    last_fail = "span_distortion_or_gap"
    for _tot_d, d_l, l_hs1, l_hg38, d_r, r_hs1, r_hg38 in pairs:
        span_hs1 = r_hs1 - l_hs1
        span_hg38 = r_hg38 - l_hg38
        if span_hs1 <= 0 or span_hg38 <= 0:
            last_fail = "nonpositive_span"
            continue
        if span_hg38 > max(200000, 4 * span_hs1):
            last_fail = "huge_unaligned_gap"
            continue
        span_diff = abs(span_hg38 - span_hs1)
        tol = max(float(max_span_abs_tol), float(max_span_rel_tol) * float(span_hs1))
        if span_diff > tol:
            last_fail = "span_distortion_exceeded"
            continue

        ref_graft = hs1_seq[l_hs1:r_hs1].upper()
        hg38_span = hg38_seq[l_hg38:r_hg38].upper()
        if "N" in ref_graft or "N" in hg38_span:
            last_fail = "span_contains_N"
            continue
        alt_graft = hs1_seq[l_hs1:pos0_hs1].upper() + alt_u + hs1_seq[end0_hs1:r_hs1].upper()

        l_anc_hs1 = hs1_seq[l_hs1 - anchor_bp : l_hs1].upper()
        l_anc_hg38 = hg38_seq[l_hg38 - anchor_bp : l_hg38].upper()
        r_anc_hs1 = hs1_seq[r_hs1 : r_hs1 + anchor_bp].upper()
        r_anc_hg38 = hg38_seq[r_hg38 : r_hg38 + anchor_bp].upper()
        l_mm = sum(1 for a, b in zip(l_anc_hs1, l_anc_hg38) if a != b)
        r_mm = sum(1 for a, b in zip(r_anc_hs1, r_anc_hg38) if a != b)
        if l_mm != 0 or r_mm != 0:
            last_fail = "anchor_junction_mismatch"
            continue

        h_pos = liftover_chain.convert_coordinate(chrom, pos0_hs1)
        if h_pos and h_pos[0][0] == chrom and h_pos[0][2] == "+" and l_hg38 <= int(h_pos[0][1]) <= r_hg38:
            est_pos0 = int(h_pos[0][1])
        else:
            est_pos0 = l_hg38 + int(round((pos0_hs1 - l_hs1) * float(span_hg38) / float(span_hs1)))

        return DASGResult(
            rescued=True,
            l_hs1=l_hs1,
            r_hs1=r_hs1,
            l_hg38=l_hg38,
            r_hg38=r_hg38,
            d_left_bp=d_l,
            d_right_bp=d_r,
            span_hs1=span_hs1,
            span_hg38=span_hg38,
            span_diff_bp=span_diff,
            est_pos0_hg38=est_pos0,
            left_anchor_mismatch_bp=l_mm,
            right_anchor_mismatch_bp=r_mm,
            ref_graft=ref_graft,
            alt_graft=alt_graft,
            hg38_span_seq=hg38_span,
            fail_reason="",
        )

    return DASGResult(rescued=False, fail_reason=last_fail)


def verify_and_graft_sv_two_tier(
    chrom: str,
    pos0_source: int,
    ref_allele: str,
    alt_allele: str,
    grch38_chrom_seq: str,
    hs1_chrom_seq: Optional[str] = None,
    liftover_chain: Optional[Any] = None,
    flank_bp: int = 15,
    max_search_bp: int = 250,
    is_symbolic_inv: bool = False,
    end0_source_override: Optional[int] = None,
    eligible_tier2_failures: Sequence[str] = DEFAULT_DASG_ELIGIBLE_FAILURES,
) -> TwoTierAssemblyResult:
    """
    Run two-tier syntenic verification (Tier-1 V_15 followed by Tier-2 DASG):
      - **Tier 1 (`V_15`)**: exact GRCh38 REF match + `±flank_bp` gapless synteny via
        `verify_and_liftover_sv`.
      - **Tier 2 (`V_graft / DASG`)**: invoked when Tier 1 fails on a local divergence or
        flank-indel failure (`eligible_tier2_failures`) and both `hs1_chrom_seq` and
        `liftover_chain` are supplied. Searches outward up to `±max_search_bp` for exact
        `flank_bp`-bp (15-bp) flanking anchors with `0%` anchor mismatch.
    """
    t1 = verify_and_liftover_sv(
        chrom=chrom,
        pos0_source=pos0_source,
        ref_allele=ref_allele,
        alt_allele=alt_allele,
        grch38_chrom_seq=grch38_chrom_seq,
        liftover_chain=liftover_chain,
        flank_bp=flank_bp,
        is_symbolic_inv=is_symbolic_inv,
        end0_source_override=end0_source_override,
    )
    if t1.passed:
        return TwoTierAssemblyResult(
            passed=True,
            tier="tier1_v15",
            chrom_grch38=t1.chrom_grch38,
            pos0_source=t1.pos0_source,
            pos0_grch38=t1.pos0_grch38,
            end0_grch38=t1.end0_grch38,
            ref_grch38=t1.verified_ref_seq,
            alt_grch38=alt_allele.upper(),
            source_ref_graft=t1.verified_ref_seq,
            tier1_result=t1,
            dasg_result=None,
            failure_reason="",
        )

    if (
        not is_symbolic_inv
        and hs1_chrom_seq is not None
        and liftover_chain is not None
        and t1.failure_reason in eligible_tier2_failures
    ):
        gres = graft_sv_dasg(
            chrom=chrom,
            pos0_hs1=pos0_source,
            ref_raw=ref_allele,
            alt_raw=alt_allele,
            hs1_seq=hs1_chrom_seq,
            hg38_seq=grch38_chrom_seq,
            liftover_chain=liftover_chain,
            max_search_bp=max_search_bp,
            anchor_bp=flank_bp,
        )
        if gres.rescued:
            return TwoTierAssemblyResult(
                passed=True,
                tier="tier2_vgraft",
                chrom_grch38=chrom,
                pos0_source=pos0_source,
                pos0_grch38=gres.l_hg38,
                end0_grch38=gres.r_hg38,
                ref_grch38=gres.hg38_span_seq,
                alt_grch38=gres.alt_graft,
                source_ref_graft=gres.ref_graft,
                tier1_result=t1,
                dasg_result=gres,
                failure_reason="",
            )
        return TwoTierAssemblyResult(
            passed=False,
            tier="failed",
            chrom_grch38=chrom,
            pos0_source=pos0_source,
            pos0_grch38=t1.pos0_grch38,
            end0_grch38=t1.end0_grch38,
            ref_grch38="",
            alt_grch38="",
            source_ref_graft="",
            tier1_result=t1,
            dasg_result=gres,
            failure_reason=f"{t1.failure_reason}|dasg:{gres.fail_reason}",
        )

    return TwoTierAssemblyResult(
        passed=False,
        tier="failed",
        chrom_grch38=t1.chrom_grch38,
        pos0_source=t1.pos0_source,
        pos0_grch38=t1.pos0_grch38,
        end0_grch38=t1.end0_grch38,
        ref_grch38="",
        alt_grch38="",
        source_ref_graft="",
        tier1_result=t1,
        dasg_result=None,
        failure_reason=t1.failure_reason,
    )
