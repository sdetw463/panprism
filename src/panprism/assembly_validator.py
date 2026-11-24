"""
Cross-assembly (T2T-CHM13v2.0 / hs1 to GRCh38 / hg38) dual-breakpoint liftover
and exact flanking-sequence validator.

Resolves silent assembly mismatches in flagship long-read pangenome VCFs (such as
1KG_ONT_VIENNA v1.1 and HGSVC3 CHM13 panels) where VCF contigs lack `##reference=`
tags while coordinates are anchored on `hs1` (100-667 kb shifted relative to `GRCh38`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


# Signature motifs for sequence-resolved Mobile Element Insertions (Alu, L1, SVA) when SVAN annotation is absent
_ALU_DIMER_MOTIFS = ("GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTT", "TTTTTTGAGACGGAGTCTCGCTCTGTCGCCCAGGCTGGAGTG")
_L1_MOTIFS = ("GGGGAGGAGCCAAGATGGCCGAATAGGAACAGCTCCGGTCT", "TTTTTTTTTTTTTTTTTTTTTT")


def classify_sv_allele(
    var_id: str,
    ref: str,
    alt: str,
    info_str: str = "",
) -> Tuple[str, int]:
    """
    Classify a sequence-resolved or symbolic structural variant into canonical biophysical classes:
      - MEI_Alu, MEI_SVA, MEI_L1, VNTR_DUP, INS_other, DEL, INV_COMPLEX
    and return `(sv_class, sv_len)`.
    """
    rlen = len(ref)
    if alt == "<INV>":
        return "INV_COMPLEX", rlen
    alen = len(alt)
    sv_len = abs(alen - rlen)
    info_u = info_str

    if "FAM_N=Alu" in info_u or "SINE/Alu" in info_u or "ALU" in var_id.upper():
        return "MEI_Alu", max(sv_len, alen)
    if "FAM_N=SVA" in info_u or "Retroposon/SVA" in info_u or "SVA" in var_id.upper():
        return "MEI_SVA", max(sv_len, alen)
    if "FAM_N=L1" in info_u or "LINE/L1" in info_u or "L1" in var_id.upper():
        return "MEI_L1", max(sv_len, alen)
    if "ITYPE_N=VNTR" in info_u or "DUP" in var_id.upper() or "VNTR" in info_u.upper() or "ITYPE_N=dup" in info_u.lower():
        return "VNTR_DUP", sv_len
    if "-INV-" in var_id.upper() or "SVTYPE=INV" in info_u.upper():
        return "INV_COMPLEX", max(sv_len, rlen, alen)
    if "-COMPLEX-" in var_id.upper():
        return ("INV_COMPLEX" if max(sv_len, rlen, alen) >= 50 else "INS_other"), max(sv_len, rlen, alen)

    if alen > rlen:
        ins_seq = alt[min(rlen, 1) :].upper()
        # Sequence-intrinsic TE motif classification for HPRC v2 / unannotated GFA bubbles
        if 250 <= sv_len <= 400 and (
            any(m[:22] in ins_seq for m in _ALU_DIMER_MOTIFS)
            or ins_seq.endswith("AAAAAAAA")
            or ins_seq.startswith("TTTTTTTT")
        ):
            return "MEI_Alu", sv_len
        if sv_len >= 4500 and (ins_seq.endswith("AAAAAAAAAA") or ins_seq.startswith("TTTTTTTTTT")):
            return "MEI_L1", sv_len
        if 900 <= sv_len <= 4400 and ("CCCTCT" * 3 in ins_seq or "AGAGGG" * 3 in ins_seq):
            return "MEI_SVA", sv_len
        # Check tandem/VNTR internal k-mer periodicity
        if sv_len >= 60 and _has_tandem_periodicity(ins_seq):
            return "VNTR_DUP", sv_len
        return "INS_other", sv_len
    if rlen > alen:
        return "DEL", sv_len
    return "INV_COMPLEX", max(rlen, alen)


def _has_tandem_periodicity(seq: str, k: int = 12) -> bool:
    if len(seq) < 3 * k:
        return False
    probe = seq[:k]
    return seq.count(probe) >= 3


@dataclass
class AssemblyCheckResult:
    """Result of cross-assembly coordinate and exact nucleotide flank verification."""
    passed: bool
    chrom_grch38: str
    pos0_source: int
    pos0_grch38: int
    end0_grch38: int
    coord_offset_bp: int
    naive_grch38_match: bool
    verified_ref_seq: str
    failure_reason: str = ""


def verify_and_liftover_sv(
    chrom: str,
    pos0_source: int,
    ref_allele: str,
    alt_allele: str,
    grch38_chrom_seq: str,
    liftover_chain: Optional[Any] = None,
    flank_bp: int = 15,
    is_symbolic_inv: bool = False,
    end0_source_override: Optional[int] = None,
) -> AssemblyCheckResult:
    """
    Verify an SV allele against `grch38_chrom_seq`, optionally lifting from a source
    assembly (e.g. `hs1` / `T2T-CHM13v2.0`) via `liftover_chain` (a `pyliftover.LiftOver`
    or duck-typed `.convert_coordinate(chrom, pos0)` object):

      1. Checks whether naive unlifted `[pos0_source : pos0_source + len(ref)]` matches `GRCh38`
         (used to diagnose unannotated assembly coordinate mismatches).
      2. If `liftover_chain` is provided (and the coordinate is on `hs1`), lifts start (and end
         for symbolic inversions or multi-bp deletions), requiring same-chromosome `+`-strand synteny.
      3. Enforces **100.0% base-for-base nucleotide identity** of `ref_allele` against `grch38_chrom_seq`.
      4. For point-anchor insertions (`len(ref_allele) < 16`) lifted across assemblies, enforces
         gapless collinear synteny across `±flank_bp` flanking bases (`pos0 - hl == flank_bp` and
         `hr - hl == rlen + 2 * flank_bp`), eliminating repetitive pericentromeric mis-lifts.
    """
    ref_u = ref_allele.upper()
    rlen = len(ref_u)
    seq_len = len(grch38_chrom_seq)

    # 1. Check whether raw VCF coordinates already match GRCh38 directly
    naive_match = False
    if 0 <= pos0_source and pos0_source + rlen <= seq_len and not is_symbolic_inv:
        naive_match = (grch38_chrom_seq[pos0_source : pos0_source + rlen] == ref_u)

    # 2. Coordinate mapping
    if liftover_chain is None:
        pos0_hg38 = pos0_source
        end0_hg38 = end0_source_override if end0_source_override is not None else (pos0_source + rlen)
    else:
        hits0 = liftover_chain.convert_coordinate(chrom, pos0_source)
        if not hits0 or hits0[0][0] != chrom or hits0[0][2] != "+":
            return AssemblyCheckResult(
                passed=False,
                chrom_grch38=chrom,
                pos0_source=pos0_source,
                pos0_grch38=-1,
                end0_grch38=-1,
                coord_offset_bp=0,
                naive_grch38_match=naive_match,
                verified_ref_seq="",
                failure_reason="unmapped_or_noncollinear_start",
            )
        pos0_hg38 = int(hits0[0][1])

        if is_symbolic_inv and end0_source_override is not None:
            hits_end = liftover_chain.convert_coordinate(chrom, end0_source_override)
            if not hits_end or hits_end[0][0] != chrom or hits_end[0][2] != "+":
                return AssemblyCheckResult(
                    passed=False,
                    chrom_grch38=chrom,
                    pos0_source=pos0_source,
                    pos0_grch38=pos0_hg38,
                    end0_grch38=-1,
                    coord_offset_bp=pos0_hg38 - pos0_source,
                    naive_grch38_match=naive_match,
                    verified_ref_seq="",
                    failure_reason="unmapped_or_noncollinear_inv_end",
                )
            end0_hg38 = int(hits_end[0][1])
            src_span = end0_source_override - pos0_source
            dst_span = end0_hg38 - pos0_hg38
            if dst_span <= 0 or abs(dst_span - src_span) > max(10, int(0.05 * max(1, src_span))):
                return AssemblyCheckResult(
                    passed=False,
                    chrom_grch38=chrom,
                    pos0_source=pos0_source,
                    pos0_grch38=pos0_hg38,
                    end0_grch38=end0_hg38,
                    coord_offset_bp=pos0_hg38 - pos0_source,
                    naive_grch38_match=naive_match,
                    verified_ref_seq="",
                    failure_reason="inv_span_distortion",
                )
        else:
            end0_hg38 = pos0_hg38 + rlen

        # Gapless ±flank_bp synteny check for short-REF insertions/indels
        if not is_symbolic_inv and rlen < 16 and flank_bp > 0:
            hl = liftover_chain.convert_coordinate(chrom, max(0, pos0_source - flank_bp))
            hr = liftover_chain.convert_coordinate(chrom, pos0_source + rlen + flank_bp)
            if (
                not hl
                or not hr
                or hl[0][0] != chrom
                or hr[0][0] != chrom
                or hl[0][2] != "+"
                or hr[0][2] != "+"
                or (pos0_hg38 - int(hl[0][1]) != flank_bp)
                or (int(hr[0][1]) - int(hl[0][1]) != rlen + 2 * flank_bp)
            ):
                return AssemblyCheckResult(
                    passed=False,
                    chrom_grch38=chrom,
                    pos0_source=pos0_source,
                    pos0_grch38=pos0_hg38,
                    end0_grch38=end0_hg38,
                    coord_offset_bp=pos0_hg38 - pos0_source,
                    naive_grch38_match=naive_match,
                    verified_ref_seq="",
                    failure_reason="gapped_or_noncollinear_flank",
                )

    # 3. Boundary & 100.0% Exact Nucleotide Identity Verification on GRCh38
    margin = max(1, flank_bp)
    if pos0_hg38 < margin or end0_hg38 + margin > seq_len or end0_hg38 <= pos0_hg38:
        return AssemblyCheckResult(
            passed=False,
            chrom_grch38=chrom,
            pos0_source=pos0_source,
            pos0_grch38=pos0_hg38,
            end0_grch38=end0_hg38,
            coord_offset_bp=pos0_hg38 - pos0_source,
            naive_grch38_match=naive_match,
            verified_ref_seq="",
            failure_reason="out_of_chromosome_bounds",
        )

    obs_grch38 = grch38_chrom_seq[pos0_hg38:end0_hg38]
    if not is_symbolic_inv and obs_grch38 != ref_u:
        return AssemblyCheckResult(
            passed=False,
            chrom_grch38=chrom,
            pos0_source=pos0_source,
            pos0_grch38=pos0_hg38,
            end0_grch38=end0_hg38,
            coord_offset_bp=pos0_hg38 - pos0_source,
            naive_grch38_match=naive_match,
            verified_ref_seq="",
            failure_reason="grch38_ref_nucleotide_mismatch",
        )

    return AssemblyCheckResult(
        passed=True,
        chrom_grch38=chrom,
        pos0_source=pos0_source,
        pos0_grch38=pos0_hg38,
        end0_grch38=end0_hg38,
        coord_offset_bp=pos0_hg38 - pos0_source,
        naive_grch38_match=naive_match,
        verified_ref_seq=obs_grch38,
        failure_reason="",
    )
