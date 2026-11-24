"""
GFA v1.0/v1.1 Pangenome Walk Bubble Parser & Multi-Allelic HPRC v2 VCF Decomposer.

Converts both:
  1. Graphical Fragment Assembly (`GFA` v1.0/v1.1 `S`/`P`/`W` lines, e.g., `hprc-v2.0-mc-grch38.sv.gfa`), and
  2. Multi-allelic pangenome wave VCF bubbles (`hprc-v2.0-mc-grch38.wave.vcf.gz`)
into clean, left-trimmed, sequence-resolved `VariantAllele` records in 0-based half-open GRCh38 coordinates.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from panprism.assembly_validator import classify_sv_allele
from panprism.breakpoint_graph import VariantAllele, reverse_complement


_WALK_STEP_RE = re.compile(r"([><])([^><\s]+)")


def trim_common_flanks(pos0: int, ref: str, alt: str) -> Tuple[int, str, str]:
    """Left- and right-trim identical flanking bases while keeping at least 1 anchor base."""
    r = ref.upper()
    a = alt.upper()
    # Right-trim first while both have > 1 base
    while len(r) > 1 and len(a) > 1 and r[-1] == a[-1]:
        r = r[:-1]
        a = a[:-1]
    # Left-trim while both have > 1 base
    shift = 0
    while len(r) > 1 and len(a) > 1 and r[0] == a[0]:
        r = r[1:]
        a = a[1:]
        shift += 1
    return pos0 + shift, r, a


def decompose_multiallelic_vcf_record(
    chrom: str,
    pos0: int,
    var_id: str,
    ref: str,
    alts: Iterable[str],
    info_str: str = "",
    min_sv_len: int = 50,
    max_sv_len: int = 50000,
) -> List[Tuple[int, VariantAllele]]:
    """
    Decompose a potentially multi-allelic HPRC v2 / Minigraph-Cactus VCF bubble into
    `(allele_index_1based, VariantAllele)` pairs for all structural alleles in `[min_sv_len, max_sv_len]`.
    """
    out: List[Tuple[int, VariantAllele]] = []
    for alt_idx_0, raw_alt in enumerate(alts):
        allele_num = alt_idx_0 + 1
        if not raw_alt or raw_alt == "*" or "<" in raw_alt and raw_alt != "<INV>":
            continue
        if raw_alt == "<INV>":
            sv_len = len(ref)
            if min_sv_len <= sv_len <= max_sv_len:
                out.append(
                    (
                        allele_num,
                        VariantAllele(
                            pos0=pos0,
                            ref=ref.upper(),
                            alt="<INV>",
                            var_id=f"{var_id}_a{allele_num}",
                            is_sv=True,
                            sv_class="INV_COMPLEX",
                            sv_len=sv_len,
                        ),
                    )
                )
            continue

        t_pos0, t_ref, t_alt = trim_common_flanks(pos0, ref, raw_alt)
        sv_cls, sv_len = classify_sv_allele(f"{var_id}_a{allele_num}", t_ref, t_alt, info_str=info_str)
        if min_sv_len <= sv_len <= max_sv_len:
            out.append(
                (
                    allele_num,
                    VariantAllele(
                        pos0=t_pos0,
                        ref=t_ref,
                        alt=t_alt,
                        var_id=f"{var_id}_a{allele_num}",
                        is_sv=True,
                        sv_class=sv_cls,
                        sv_len=sv_len,
                    ),
                )
            )
    return out


def parse_gfa_bubbles(
    gfa_lines: Iterable[str],
    ref_sample_prefix: str = "GRCh38",
    min_sv_len: int = 50,
    max_sv_len: int = 50000,
) -> Dict[str, List[VariantAllele]]:
    """
    Parse GFA v1.0/v1.1 `S` (segment) and `W` (walk) / `P` (path) records into per-sample
    `VariantAllele` lists anchored against the reference walk (`ref_sample_prefix`).

    Supports Minigraph-Cactus / PGGB GFA v1.1 walks (`W <sample> <hap> <seqid> <start> <end> >s1>s2...`).
    """
    segments: Dict[str, str] = {}
    ref_walks: List[Tuple[str, int, List[Tuple[str, str]]]] = []
    alt_walks: List[Tuple[str, str, List[Tuple[str, str]]]] = []

    for raw_line in gfa_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        rec_type = parts[0]
        if rec_type == "S" and len(parts) >= 3:
            segments[parts[1]] = parts[2].upper()
        elif rec_type == "W" and len(parts) >= 7:
            sample, hap_idx, seqid, start_s, _end_s, walk_str = parts[1:7]
            steps = [(m.group(1), m.group(2)) for m in _WALK_STEP_RE.finditer(walk_str)]
            start0 = int(start_s) if start_s.isdigit() else 0
            if sample.startswith(ref_sample_prefix) or seqid.startswith(ref_sample_prefix):
                chrom = seqid.split("#")[-1]
                ref_walks.append((chrom, start0, steps))
            else:
                sample_hap = f"{sample}#{hap_idx}"
                alt_walks.append((sample_hap, seqid, steps))
        elif rec_type == "P" and len(parts) >= 3:
            pname, path_str = parts[1], parts[2]
            steps = []
            for tok in path_str.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                orient = ">" if not tok.endswith("-") else "<"
                sname = tok[:-1] if tok[-1] in "+-" else tok
                steps.append((orient, sname))
            if pname.startswith(ref_sample_prefix):
                chrom = pname.split("#")[-1]
                ref_walks.append((chrom, 0, steps))
            else:
                alt_walks.append((pname, pname, steps))

    def _step_seq(orient: str, seg_id: str) -> str:
        s = segments.get(seg_id, "")
        return s if orient == ">" else reverse_complement(s)

    per_hap_alleles: Dict[str, List[VariantAllele]] = {}
    if not ref_walks:
        return per_hap_alleles

    # Index reference node positions along each reference walk
    for _chrom, ref_start0, ref_steps in ref_walks:
        ref_node_pos0: Dict[str, Tuple[int, int, str]] = {}
        cursor = ref_start0
        for idx_r, (orient, sid) in enumerate(ref_steps):
            slen = len(segments.get(sid, ""))
            ref_node_pos0[sid] = (idx_r, cursor, orient)
            cursor += slen

        for sample_hap, _seqid, a_steps in alt_walks:
            # Find shared anchor nodes along the walk to extract bubbles
            anchors: List[Tuple[int, int]] = []  # (alt_step_idx, ref_step_idx)
            for idx_a, (ao, asid) in enumerate(a_steps):
                if asid in ref_node_pos0 and ref_node_pos0[asid][2] == ao:
                    anchors.append((idx_a, ref_node_pos0[asid][0]))

            for k in range(len(anchors) - 1):
                ia0, ir0 = anchors[k]
                ia1, ir1 = anchors[k + 1]
                if ir1 <= ir0 or (ia1 - ia0 == 1 and ir1 - ir0 == 1):
                    continue
                # Extract reference path between anchors (including left anchor's last base)
                left_orient, left_sid = ref_steps[ir0]
                left_seq = _step_seq(left_orient, left_sid)
                anchor_base = left_seq[-1:] if left_seq else "N"
                anchor_pos0 = ref_node_pos0[left_sid][1] + max(0, len(left_seq) - 1)

                ref_mid = "".join(_step_seq(ro, rsid) for ro, rsid in ref_steps[ir0 + 1 : ir1])
                alt_mid = "".join(_step_seq(ao, asid) for ao, asid in a_steps[ia0 + 1 : ia1])
                if ref_mid == alt_mid:
                    continue

                raw_ref = anchor_base + ref_mid
                raw_alt = anchor_base + alt_mid
                t_pos0, t_ref, t_alt = trim_common_flanks(anchor_pos0, raw_ref, raw_alt)

                # Detect pure single-segment inversion bubble
                is_inv_bubble = (
                    ia1 - ia0 == 2
                    and ir1 - ir0 == 2
                    and a_steps[ia0 + 1][1] == ref_steps[ir0 + 1][1]
                    and a_steps[ia0 + 1][0] != ref_steps[ir0 + 1][0]
                )
                vid = f"GFA_{left_sid}_{ref_steps[ir1][1]}"
                if is_inv_bubble:
                    inv_pos0 = ref_node_pos0[ref_steps[ir0 + 1][1]][1]
                    sv_cls, sv_len = "INV_COMPLEX", len(ref_mid)
                    t_pos0, t_ref, t_alt = inv_pos0, ref_mid, alt_mid
                else:
                    sv_cls, sv_len = classify_sv_allele(vid, t_ref, t_alt)

                if min_sv_len <= sv_len <= max_sv_len:
                    per_hap_alleles.setdefault(sample_hap, []).append(
                        VariantAllele(
                            pos0=t_pos0,
                            ref=t_ref,
                            alt=t_alt,
                            var_id=vid,
                            is_sv=True,
                            sv_class=sv_cls,
                            sv_len=sv_len,
                        )
                    )

    return per_hap_alleles
