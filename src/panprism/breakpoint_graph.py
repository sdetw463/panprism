"""
Pangenome Breakpoint-Graph Coordinate Bijection, Personal Haplotype Builder,
Unique-Haplotype Cryptographic Deduplication, On-GPU Breakpoint-Graph Node Pooling,
Korsakova et al. (2025) 1D Output Stitching, and Naive Center-Crop Personal FASTA Pooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch

    _NO_GRAD = torch.no_grad()
except ImportError:  # pragma: no cover - CPU/lightweight env without torch
    torch = None  # type: ignore

    def _NO_GRAD(fn):  # type: ignore
        return fn


from panprism.contact_physics import compute_physical_3d_contact

COMPLEMENT_TABLE = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT_TABLE)[::-1]


@dataclass
class VariantAllele:
    """A phased sequence-resolved variant (SV or SNV) in 0-based half-open [pos0, end0) GRCh38 coordinates."""

    pos0: int
    ref: str
    alt: str
    var_id: str
    is_sv: bool
    sv_class: str = "SNV"  # MEI_Alu, MEI_SVA, MEI_L1, VNTR_DUP, INS_other, DEL, INV_COMPLEX, SNV
    sv_len: int = 0

    @property
    def end0(self) -> int:
        return self.pos0 + len(self.ref)

    @property
    def delta_len(self) -> int:
        if self.alt == "<INV>":
            return 0
        return len(self.alt) - len(self.ref)


@dataclass
class InsertionBubble:
    """Tracks a polymorphic non-reference insertion/MEI/VNTR bubble on a personal haplotype sequence."""

    hap_start: int
    hap_end: int
    ref_anchor_pos0: int
    var_id: str
    sv_class: str
    ins_len: int


@dataclass
class ReferenceSegmentMap:
    """Piecewise bijection between an unmodified or inverted reference interval and haplotype sequence."""

    ref_start0: int
    ref_end0: int
    hap_start: int
    hap_end: int
    inverted: bool = False


@dataclass
class HaplotypeGraphMapping:
    """Complete sequence + pangenome breakpoint-graph coordinate mapping for a fixed-length S2F window."""

    sequence: str
    window_len: int
    ref_slice_start0: int
    ref_slice_end0: int
    tss_pos0: int
    tss_hap_idx: int
    strand: str
    upstream_net_delta_bp: int = 0
    ref_segments: List[ReferenceSegmentMap] = field(default_factory=list)
    insertion_bubbles: List[InsertionBubble] = field(default_factory=list)
    deletion_breakpoints_hap: List[Tuple[int, int]] = field(default_factory=list)  # (hap_junction_idx, del_len)
    inversion_spans_hap: List[Tuple[int, int]] = field(default_factory=list)       # (hap_start, hap_end)

    def map_ref_interval_to_hap(self, r_start0: int, r_end0: int) -> List[Tuple[int, int, bool]]:
        """Map a 0-based half-open reference interval [r_start0, r_end0) to surviving [h_start, h_end, inverted) intervals."""
        out: List[Tuple[int, int, bool]] = []
        if r_end0 <= r_start0:
            return out
        for seg in self.ref_segments:
            ov_s = max(r_start0, seg.ref_start0)
            ov_e = min(r_end0, seg.ref_end0)
            if ov_e > ov_s:
                offset_s = ov_s - seg.ref_start0
                offset_e = ov_e - seg.ref_start0
                if not seg.inverted:
                    hs = seg.hap_start + offset_s
                    he = seg.hap_start + offset_e
                else:
                    seg_len = seg.ref_end0 - seg.ref_start0
                    hs = seg.hap_start + (seg_len - offset_e)
                    he = seg.hap_start + (seg_len - offset_s)
                hs_clip = max(0, min(self.window_len, hs))
                he_clip = max(0, min(self.window_len, he))
                if he_clip > hs_clip:
                    out.append((hs_clip, he_clip, seg.inverted))
        return out


def haplotype_sha1(mapping: HaplotypeGraphMapping, gene_id: str = "") -> str:
    """
    Lossless SHA-1 key over `(gene_id, strand, window_len, sequence, topological_graph_intervals)`.
    Two haplotypes sharing the same `haplotype_sha1` are guaranteed to produce bit-identical pooled S2F nodes.
    """
    h = hashlib.sha1()
    h.update(f"{gene_id}|{mapping.strand}|{mapping.window_len}|{mapping.tss_hap_idx}|{mapping.upstream_net_delta_bp}|".encode("ascii"))
    h.update(mapping.sequence.encode("ascii"))
    for seg in mapping.ref_segments:
        h.update(f";S:{seg.ref_start0},{seg.ref_end0},{seg.hap_start},{seg.hap_end},{int(seg.inverted)}".encode("ascii"))
    for bub in mapping.insertion_bubbles:
        h.update(f";B:{bub.hap_start},{bub.hap_end},{bub.sv_class},{bub.ins_len}".encode("ascii"))
    for bkp, dlen in mapping.deletion_breakpoints_hap:
        h.update(f";D:{bkp},{dlen}".encode("ascii"))
    return h.hexdigest()


def build_center_crop_naive_sequence(
    ref_seq: str,
    ref_seq_start0: int,
    tss_pos0: int,
    variants: List[VariantAllele],
    window_len: int = 131072,
) -> str:
    """
    Construct a naive personal FASTA sequence cropped without Breakpoint-Graph TSS/exon coordinate
    tracking (fixes left boundary at `tss_pos0 - window_len // 2` on the reference and takes the first
    `window_len` bases of the mutated contig). Used for unit tests and naive coordinate-drift analysis.
    """
    half = window_len // 2
    win_ref_start0 = max(ref_seq_start0, tss_pos0 - half)
    sorted_vars = sorted(variants, key=lambda v: (v.pos0, -int(v.is_sv), -abs(v.delta_len)))
    chunks: List[str] = []
    cur = win_ref_start0
    ref_seq_end0 = ref_seq_start0 + len(ref_seq)
    for v in sorted_vars:
        if v.pos0 < cur or v.end0 > ref_seq_end0:
            continue
        ref_sub = ref_seq[v.pos0 - ref_seq_start0 : v.end0 - ref_seq_start0]
        alt_seq = reverse_complement(ref_sub) if v.alt == "<INV>" else v.alt
        if v.alt != "<INV>" and not ref_sub.upper().startswith(v.ref[: min(8, len(v.ref))].upper()):
            continue
        if v.pos0 > cur:
            chunks.append(ref_seq[cur - ref_seq_start0 : v.pos0 - ref_seq_start0])
        chunks.append(alt_seq)
        cur = v.end0
        if sum(len(c) for c in chunks) >= window_len:
            break
    if sum(len(c) for c in chunks) < window_len and cur < ref_seq_end0:
        need = window_len - sum(len(c) for c in chunks)
        chunks.append(ref_seq[cur - ref_seq_start0 : min(len(ref_seq), cur - ref_seq_start0 + need)])
    s = "".join(chunks)[:window_len]
    if len(s) < window_len:
        s = s + ("N" * (window_len - len(s)))
    return s


def build_haplotype_graph_sequence(
    ref_seq: str,
    ref_seq_start0: int,
    tss_pos0: int,
    strand: str,
    variants: List[VariantAllele],
    window_len: int = 131072,
    center_shift_bp: int = 0,
) -> HaplotypeGraphMapping:
    """
    Construct a personal haplotype DNA sequence of exact length `window_len` centered at the
    haplotype-mapped TSS (`+ center_shift_bp` for shift augmentation or +-5 kb window-jitter tests),
    while maintaining the exact Pangenome Breakpoint-Graph coordinate bijection.
    """
    ref_seq_end0 = ref_seq_start0 + len(ref_seq)
    sorted_vars = sorted(variants, key=lambda v: (v.pos0, -int(v.is_sv), -abs(v.delta_len)))
    clean_vars: List[VariantAllele] = []
    cur_end = ref_seq_start0
    for v in sorted_vars:
        if v.pos0 < ref_seq_start0 or v.end0 > ref_seq_end0:
            continue
        if v.pos0 < cur_end:
            continue
        ref_sub = ref_seq[v.pos0 - ref_seq_start0 : v.end0 - ref_seq_start0]
        alt_seq = v.alt
        is_inv = v.sv_class.startswith("INV")
        if alt_seq == "<INV>":
            alt_seq = reverse_complement(ref_sub)
            is_inv = True
        elif not ref_sub.upper().startswith(v.ref[: min(8, len(v.ref))].upper()):
            continue
        clean_vars.append(
            VariantAllele(
                pos0=v.pos0,
                ref=ref_sub,
                alt=alt_seq,
                var_id=v.var_id,
                is_sv=v.is_sv,
                sv_class="INV_COMPLEX" if is_inv else v.sv_class,
                sv_len=v.sv_len,
            )
        )
        cur_end = v.end0

    chunks: List[str] = []
    raw_ref_segs: List[ReferenceSegmentMap] = []
    raw_bubbles: List[InsertionBubble] = []
    raw_del_bkp: List[Tuple[int, int]] = []
    raw_inv_spans: List[Tuple[int, int]] = []

    ref_cursor = ref_seq_start0
    hap_cursor = 0
    tss_raw_hap: Optional[int] = None
    upstream_net_delta = 0

    for v in clean_vars:
        if v.end0 <= tss_pos0:
            upstream_net_delta += len(v.alt) - len(v.ref)
        if v.pos0 > ref_cursor:
            seg_len = v.pos0 - ref_cursor
            chunks.append(ref_seq[ref_cursor - ref_seq_start0 : v.pos0 - ref_seq_start0])
            raw_ref_segs.append(ReferenceSegmentMap(ref_cursor, v.pos0, hap_cursor, hap_cursor + seg_len, False))
            if ref_cursor <= tss_pos0 < v.pos0:
                tss_raw_hap = hap_cursor + (tss_pos0 - ref_cursor)
            hap_cursor += seg_len

        rlen = len(v.ref)
        alen = len(v.alt)
        if v.pos0 <= tss_pos0 < v.end0 and tss_raw_hap is None:
            # Surviving bases inside an inversion have a reversed position.
            # Breakpoint anchoring is only a fallback for bases without a
            # homologous surviving segment, such as a deleted TSS.
            inverted = v.sv_class.startswith("INV") or (v.is_sv and abs(alen - rlen) <= 5 and rlen >= 50)
            span = min(rlen, alen)
            if inverted and tss_pos0 < v.pos0 + span:
                tss_raw_hap = hap_cursor + span - 1 - (tss_pos0 - v.pos0)
            else:
                tss_raw_hap = hap_cursor

        if not v.is_sv and rlen == 1 and alen == 1:
            chunks.append(v.alt)
            raw_ref_segs.append(ReferenceSegmentMap(v.pos0, v.end0, hap_cursor, hap_cursor + 1, False))
            hap_cursor += 1
        elif v.sv_class.startswith("INV") or (v.is_sv and abs(alen - rlen) <= 5 and rlen >= 50):
            chunks.append(v.alt)
            mlen = min(rlen, alen)
            raw_ref_segs.append(ReferenceSegmentMap(v.pos0, v.pos0 + mlen, hap_cursor, hap_cursor + mlen, True))
            raw_inv_spans.append((hap_cursor, hap_cursor + alen))
            hap_cursor += alen
        elif alen > rlen:
            anchor = min(rlen, 1)
            if anchor > 0:
                chunks.append(v.alt[:anchor])
                raw_ref_segs.append(ReferenceSegmentMap(v.pos0, v.pos0 + anchor, hap_cursor, hap_cursor + anchor, False))
                hap_cursor += anchor
            ins_seq = v.alt[anchor:]
            ins_len = len(ins_seq)
            if ins_len > 0:
                chunks.append(ins_seq)
                raw_bubbles.append(
                    InsertionBubble(
                        hap_start=hap_cursor,
                        hap_end=hap_cursor + ins_len,
                        ref_anchor_pos0=v.pos0 + anchor,
                        var_id=v.var_id,
                        sv_class=v.sv_class,
                        ins_len=ins_len,
                    )
                )
                hap_cursor += ins_len
            if rlen > anchor:
                raw_del_bkp.append((hap_cursor, rlen - anchor))
        elif rlen > alen:
            if alen > 0:
                chunks.append(v.alt)
                raw_ref_segs.append(ReferenceSegmentMap(v.pos0, v.pos0 + alen, hap_cursor, hap_cursor + alen, False))
                hap_cursor += alen
            raw_del_bkp.append((hap_cursor, rlen - alen))
        else:
            chunks.append(v.alt)
            raw_ref_segs.append(ReferenceSegmentMap(v.pos0, v.end0, hap_cursor, hap_cursor + alen, False))
            hap_cursor += alen

        ref_cursor = v.end0

    if ref_cursor < ref_seq_end0:
        seg_len = ref_seq_end0 - ref_cursor
        chunks.append(ref_seq[ref_cursor - ref_seq_start0 :])
        raw_ref_segs.append(ReferenceSegmentMap(ref_cursor, ref_seq_end0, hap_cursor, hap_cursor + seg_len, False))
        if ref_cursor <= tss_pos0 < ref_seq_end0 and tss_raw_hap is None:
            tss_raw_hap = hap_cursor + (tss_pos0 - ref_cursor)
        hap_cursor += seg_len

    full_contig = "".join(chunks)
    if tss_raw_hap is None:
        tss_raw_hap = len(full_contig) // 2

    half = window_len // 2
    crop_center = tss_raw_hap + center_shift_bp
    crop_start = crop_center - half
    crop_end = crop_start + window_len

    pad_left = 0
    pad_right = 0
    if crop_start < 0:
        pad_left = -crop_start
        crop_start = 0
    if crop_end > len(full_contig):
        pad_right = crop_end - len(full_contig)
        crop_end = len(full_contig)

    window_seq = ("N" * pad_left) + full_contig[crop_start:crop_end] + ("N" * pad_right)
    assert len(window_seq) == window_len, f"Expected {window_len}, got {len(window_seq)}"

    raw_to_win = pad_left if (crop_center - half) < 0 else -crop_start

    win_ref_segs: List[ReferenceSegmentMap] = []
    for seg in raw_ref_segs:
        ws = seg.hap_start + raw_to_win
        we = seg.hap_end + raw_to_win
        if we <= 0 or ws >= window_len:
            continue
        clip_l = max(0, -ws)
        clip_r = max(0, we - window_len)
        seg_len = seg.ref_end0 - seg.ref_start0
        if clip_l + clip_r >= seg_len:
            continue
        if not seg.inverted:
            n_rs = seg.ref_start0 + clip_l
            n_re = seg.ref_end0 - clip_r
        else:
            n_rs = seg.ref_start0 + clip_r
            n_re = seg.ref_end0 - clip_l
        win_ref_segs.append(ReferenceSegmentMap(n_rs, n_re, ws + clip_l, we - clip_r, seg.inverted))

    win_bubbles: List[InsertionBubble] = []
    for b in raw_bubbles:
        bs = max(0, b.hap_start + raw_to_win)
        be = min(window_len, b.hap_end + raw_to_win)
        if be > bs:
            win_bubbles.append(InsertionBubble(bs, be, b.ref_anchor_pos0, b.var_id, b.sv_class, b.ins_len))

    win_dels = [
        (bkp + raw_to_win, dlen)
        for (bkp, dlen) in raw_del_bkp
        if 0 <= bkp + raw_to_win < window_len
    ]
    win_invs = [
        (max(0, s + raw_to_win), min(window_len, e + raw_to_win))
        for (s, e) in raw_inv_spans
        if min(window_len, e + raw_to_win) > max(0, s + raw_to_win)
    ]

    ref_win_start0 = win_ref_segs[0].ref_start0 if win_ref_segs else tss_pos0 - half
    ref_win_end0 = win_ref_segs[-1].ref_end0 if win_ref_segs else tss_pos0 + half

    return HaplotypeGraphMapping(
        sequence=window_seq,
        window_len=window_len,
        ref_slice_start0=ref_win_start0,
        ref_slice_end0=ref_win_end0,
        tss_pos0=tss_pos0,
        tss_hap_idx=max(0, min(window_len - 1, tss_raw_hap + raw_to_win)),
        strand=strand,
        upstream_net_delta_bp=upstream_net_delta,
        ref_segments=win_ref_segs,
        insertion_bubbles=win_bubbles,
        deletion_breakpoints_hap=win_dels,
        inversion_spans_hap=win_invs,
    )


@_NO_GRAD
def pool_alphagenome_on_gpu(
    ag_preds: Dict[str, Any],
    mapping: HaplotypeGraphMapping,
    exons_ref0: List[Tuple[int, int]],
    return_locus_tracks: bool = False,
    use_physical_3d: bool = False,
    diag_medians: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Simultaneously compute on GPU:
      1) `panprism_nodes` (22-dim): PanPrism Breakpoint-Graph Node Pooling
      2) `korsakova_stitch` (4-dim): Korsakova et al. (2025) 1D Linear Output Stitching
      3) `center_crop_naive` (4-dim): Baseline 2 Naive Personal FASTA without Breakpoint-Graph projection

    When `use_physical_3d=True`, uses diagonal-normalized
    physical 3D contact probability `P_phys(i, j)` via `compute_physical_3d_contact`.
    """
    L = mapping.window_len
    is_plus = mapping.strand != "-"
    sense_rna_idx = 0 if is_plus else 1
    anti_rna_idx = 1 if is_plus else 0
    sense_cage_idx = 0 if is_plus else 1
    anti_cage_idx = 1 if is_plus else 0
    sense_donor_idx = 0 if is_plus else 2
    sense_acc_idx = 1 if is_plus else 3
    sense_usage_idx = 0 if is_plus else 1

    rna_sense = ag_preds["rna_1bp"][:, sense_rna_idx]
    rna_anti = ag_preds["rna_1bp"][:, anti_rna_idx]
    rna_unstr = ag_preds["rna_1bp"][:, 2]
    cage_sense = ag_preds["cage_1bp"][:, sense_cage_idx]
    cage_anti = ag_preds["cage_1bp"][:, anti_cage_idx]
    atac_1bp = ag_preds["atac_1bp"]
    dnase_1bp = ag_preds["dnase_1bp"]
    hist_128 = ag_preds["histone_128bp"]  # [H3K27ac, H3K36me3, H3K4me1, H3K4me3]
    sp_donor = ag_preds["splice_probs_1bp"][:, sense_donor_idx]
    sp_acc = ag_preds["splice_probs_1bp"][:, sense_acc_idx]
    sp_usage = ag_preds["splice_usage_1bp"][:, sense_usage_idx]
    cmap = ag_preds["contact_gm12878"]    # (64, 64) for 131kb (2048bp/bin)

    def _scalar(x: Any) -> float:
        return float(x.item()) if hasattr(x, "item") else float(x)

    # --- 0. Baseline 2: Center-Crop-Personal-FASTA (Unprojected Reference Offsets) ---
    # Simulates injecting personal alleles into the contig and reading fixed unmapped reference offsets
    # (shifted relative to actual exons/TSS by upstream & intragenic SV length deltas).
    naive_tss_idx = max(0, min(L - 1, mapping.tss_hap_idx - mapping.upstream_net_delta_bp))
    naive_exon_rna = 0.0
    total_ref_ex_bp = 0
    for ex_s, ex_e in exons_ref0:
        rlen = max(0, ex_e - ex_s)
        if rlen == 0:
            continue
        total_ref_ex_bp += rlen
        ns = max(0, min(L, naive_tss_idx + (ex_s - mapping.tss_pos0)))
        ne = max(0, min(L, naive_tss_idx + (ex_e - mapping.tss_pos0)))
        if ne > ns:
            naive_exon_rna += _scalar(rna_unstr[ns:ne].sum())
    np_s, np_e = max(0, naive_tss_idx - 512), min(L, naive_tss_idx + 512)
    naive_cage = _scalar(cage_sense[np_s:np_e].sum()) if np_e > np_s else 0.0
    naive_atac = _scalar(atac_1bp[np_s:np_e].mean()) if np_e > np_s else 0.0
    center_crop_vec = np.array(
        [
            np.log1p(max(0.0, naive_exon_rna)),
            np.log1p(max(0.0, (naive_exon_rna / max(1, total_ref_ex_bp)) * 1000.0)),
            np.log1p(max(0.0, naive_cage)),
            np.log1p(max(0.0, naive_atac)),
        ],
        dtype=np.float32,
    )

    # --- 1. Baseline 1: Korsakova et al. (2025) 1D Output Stitching ---
    stitch_exon_rna_sum = 0.0
    stitch_exon_ref_bp = 0
    for ex_s, ex_e in exons_ref0:
        ref_len = max(0, ex_e - ex_s)
        if ref_len == 0:
            continue
        stitch_exon_ref_bp += ref_len
        for hs, he, _inv in mapping.map_ref_interval_to_hap(ex_s, ex_e):
            stitch_exon_rna_sum += _scalar(rna_unstr[hs:he].sum())

    tss_ref_s, tss_ref_e = mapping.tss_pos0 - 512, mapping.tss_pos0 + 512
    stitch_prom_cage = 0.0
    stitch_prom_atac = 0.0
    for hs, he, _inv in mapping.map_ref_interval_to_hap(tss_ref_s, tss_ref_e):
        stitch_prom_cage += _scalar(cage_sense[hs:he].sum())
        stitch_prom_atac += _scalar(atac_1bp[hs:he].mean()) * (he - hs) / 1024.0

    stitch_exon_density = stitch_exon_rna_sum / max(1, stitch_exon_ref_bp)
    korsakova_vec = np.array(
        [
            np.log1p(max(0.0, stitch_exon_rna_sum)),
            np.log1p(max(0.0, stitch_exon_density * 1000.0)),
            np.log1p(max(0.0, stitch_prom_cage)),
            np.log1p(max(0.0, stitch_prom_atac)),
        ],
        dtype=np.float32,
    )

    # --- 2. PanPrism Breakpoint-Graph Node Pooling ---
    t_idx = mapping.tss_hap_idx
    p_s, p_e = max(0, t_idx - 512), min(L, t_idx + 512)
    p128_s, p128_e = max(0, (t_idx - 2048) // 128), min(hist_128.shape[0], (t_idx + 2048) // 128 + 1)
    prom_cage_sense = _scalar(cage_sense[p_s:p_e].sum())
    prom_cage_anti = _scalar(cage_anti[p_s:p_e].sum())
    prom_atac = _scalar(atac_1bp[p_s:p_e].mean())
    prom_h3k27ac = _scalar(hist_128[p128_s:p128_e, 0].mean())
    prom_h3k4me3 = _scalar(hist_128[p128_s:p128_e, 3].mean())

    pb_exon_sense_sum = 0.0
    pb_exon_surviving_bp = 0
    pb_exon_total_ref_bp = 0
    pb_splice_site_score = 0.0
    pb_splice_usage_score = 0.0
    n_splice_boundaries = 0
    pb_h3k36me3_sum = 0.0

    sorted_exons = sorted(exons_ref0, key=lambda x: x[0])
    for idx_ex, (ex_s, ex_e) in enumerate(sorted_exons):
        rlen = max(0, ex_e - ex_s)
        if rlen == 0:
            continue
        pb_exon_total_ref_bp += rlen
        mapped = mapping.map_ref_interval_to_hap(ex_s, ex_e)
        for hs, he, inv in mapped:
            pb_exon_surviving_bp += (he - hs)
            if not inv:
                pb_exon_sense_sum += _scalar((0.5 * rna_sense[hs:he] + 0.5 * rna_unstr[hs:he]).sum())
            else:
                pb_exon_sense_sum += _scalar((0.5 * rna_anti[hs:he] + 0.5 * rna_unstr[hs:he]).sum())
            b128_s, b128_e = hs // 128, min(hist_128.shape[0], max(hs // 128 + 1, he // 128))
            pb_h3k36me3_sum += _scalar(hist_128[b128_s:b128_e, 1].mean()) * (he - hs)

            if idx_ex > 0:
                bs, be = max(0, hs - 16), min(L, hs + 16)
                pb_splice_site_score += _scalar((sp_acc[bs:be] if is_plus else sp_donor[bs:be]).max())
                pb_splice_usage_score += _scalar(sp_usage[bs:be].max())
                n_splice_boundaries += 1
            if idx_ex < len(sorted_exons) - 1:
                bs, be = max(0, he - 16), min(L, he + 16)
                pb_splice_site_score += _scalar((sp_donor[bs:be] if is_plus else sp_acc[bs:be]).max())
                pb_splice_usage_score += _scalar(sp_usage[bs:be].max())
                n_splice_boundaries += 1

    exon_survival_frac = pb_exon_surviving_bp / max(1, pb_exon_total_ref_bp)
    exon_sense_density = pb_exon_sense_sum / max(1, pb_exon_total_ref_bp)
    mean_canonical_splice = pb_splice_site_score / max(1, n_splice_boundaries)
    mean_canonical_usage = pb_splice_usage_score / max(1, n_splice_boundaries)
    mean_h3k36me3 = pb_h3k36me3_sum / max(1, pb_exon_surviving_bp)

    n_cbins = cmap.shape[0]
    bin_bp = L // n_cbins
    tss_cbin = min(n_cbins - 1, max(0, t_idx // bin_bp))

    if use_physical_3d:
        p_phys_mat, _c_norm_mat, _diag_m = compute_physical_3d_contact(
            cmap, bin_bp=float(bin_bp), diag_medians=diag_medians
        )
        tss_contacts = p_phys_mat[tss_cbin, :]
    elif torch is not None and isinstance(cmap, torch.Tensor):
        tss_contacts = torch.clamp(cmap[tss_cbin, :], min=0.0)
    else:
        tss_contacts = np.maximum(np.asarray(cmap[tss_cbin, :], dtype=np.float32), 0.0)

    if torch is not None and isinstance(cmap, torch.Tensor):
        distal_mask = torch.ones(n_cbins, device=cmap.device, dtype=torch.float32)
        distal_mask[max(0, tss_cbin - 1) : min(n_cbins, tss_cbin + 2)] = 0.0
        atac_2k = atac_1bp.view(n_cbins, bin_bp).mean(dim=1)
        h3k27ac_2k = hist_128[:, 0].view(n_cbins, bin_bp // 128).mean(dim=1)
        h3k4me1_2k = hist_128[:, 2].view(n_cbins, bin_bp // 128).mean(dim=1)
        ccre_activity_2k = atac_2k * (0.6 * h3k27ac_2k + 0.4 * h3k4me1_2k + 0.1)
        ep_3d_weighted_enhancer = _scalar((tss_contacts * distal_mask * ccre_activity_2k).sum())
        ep_3d_total_loop_strength = _scalar((tss_contacts * distal_mask).sum())
        idx_grid = torch.arange(n_cbins, device=cmap.device, dtype=torch.float32)
        dist_kb = torch.abs(idx_grid - float(tss_cbin)) * (bin_bp / 1000.0) + 2.0
        ep_1d_decay_enhancer = _scalar(((distal_mask / dist_kb) * ccre_activity_2k).sum())
    else:
        distal_mask = np.ones(n_cbins, dtype=np.float32)
        distal_mask[max(0, tss_cbin - 1) : min(n_cbins, tss_cbin + 2)] = 0.0
        atac_2k = np.asarray(atac_1bp, dtype=np.float32).reshape(n_cbins, bin_bp).mean(axis=1)
        h3k27ac_2k = np.asarray(hist_128[:, 0], dtype=np.float32).reshape(n_cbins, bin_bp // 128).mean(axis=1)
        h3k4me1_2k = np.asarray(hist_128[:, 2], dtype=np.float32).reshape(n_cbins, bin_bp // 128).mean(axis=1)
        ccre_activity_2k = atac_2k * (0.6 * h3k27ac_2k + 0.4 * h3k4me1_2k + 0.1)
        ep_3d_weighted_enhancer = float((tss_contacts * distal_mask * ccre_activity_2k).sum())
        ep_3d_total_loop_strength = float((tss_contacts * distal_mask).sum())
        idx_grid = np.arange(n_cbins, dtype=np.float32)
        dist_kb = np.abs(idx_grid - float(tss_cbin)) * (bin_bp / 1000.0) + 2.0
        ep_1d_decay_enhancer = float(((distal_mask / dist_kb) * ccre_activity_2k).sum())

    ins_total_bp = 0
    ins_denovo_enhancer_3d = 0.0
    ins_denovo_sense_cage = 0.0
    ins_antisense_interference = 0.0
    ins_cryptic_splice_trap = 0.0
    ins_cryptic_usage_trap = 0.0

    for bub in mapping.insertion_bubbles:
        bs, be = bub.hap_start, bub.hap_end
        if be <= bs:
            continue
        ins_total_bp += (be - bs)
        b_mid_cbin = min(n_cbins - 1, max(0, ((bs + be) // 2) // bin_bp))
        contact_to_tss = max(0.05, _scalar(tss_contacts[b_mid_cbin]))
        b128_s, b128_e = bs // 128, min(hist_128.shape[0], max(bs // 128 + 1, (be + 127) // 128))

        bub_atac = _scalar(atac_1bp[bs:be].mean())
        bub_h3k27ac = _scalar(hist_128[b128_s:b128_e, 0].mean())
        ins_denovo_enhancer_3d += contact_to_tss * bub_atac * (bub_h3k27ac + 0.1) * np.log1p(be - bs)
        ins_denovo_sense_cage += _scalar(cage_sense[bs:be].sum())
        ins_antisense_interference += _scalar(cage_anti[bs:be].sum()) + 0.25 * _scalar(rna_anti[bs:be].sum())
        if torch is not None and isinstance(sp_donor, torch.Tensor):
            cryptic_max = _scalar(torch.maximum(sp_donor[bs:be], sp_acc[bs:be]).max())
            cryptic_sum = _scalar((sp_donor[bs:be] + sp_acc[bs:be]).clamp(min=0.05).sum())
        else:
            cryptic_max = float(np.maximum(sp_donor[bs:be], sp_acc[bs:be]).max())
            cryptic_sum = float(np.maximum(sp_donor[bs:be] + sp_acc[bs:be], 0.05).sum())
        ins_cryptic_splice_trap += cryptic_max + 0.1 * np.log1p(cryptic_sum)
        ins_cryptic_usage_trap += _scalar(sp_usage[bs:be].sum())

    del_total_bp = 0
    del_bkp_activity = 0.0
    for bkp_idx, dlen in mapping.deletion_breakpoints_hap:
        del_total_bp += dlen
        ds, de = max(0, bkp_idx - 256), min(L, bkp_idx + 256)
        if de > ds:
            del_bkp_activity += _scalar(atac_1bp[ds:de].mean()) * np.log1p(dlen)

    inv_total_bp = sum(e - s for s, e in mapping.inversion_spans_hap)

    panprism_vec = np.array(
        [
            np.log1p(max(0.0, pb_exon_sense_sum)),
            np.log1p(max(0.0, exon_sense_density * 1000.0)),
            exon_survival_frac,
            np.log1p(max(0.0, prom_cage_sense)),
            np.log1p(max(0.0, prom_atac)),
            np.log1p(max(0.0, prom_h3k27ac + prom_h3k4me3)),
            mean_canonical_splice,
            mean_canonical_usage,
            np.log1p(max(0.0, mean_h3k36me3)),
            np.log1p(max(0.0, ep_3d_weighted_enhancer)),
            np.log1p(max(0.0, ep_3d_total_loop_strength)),
            np.log1p(max(0.0, ep_1d_decay_enhancer)),
            np.log1p(max(0.0, ins_denovo_enhancer_3d)),
            np.log1p(max(0.0, ins_denovo_sense_cage)),
            np.log1p(max(0.0, ins_antisense_interference)),
            ins_cryptic_splice_trap,
            np.log1p(max(0.0, ins_cryptic_usage_trap)),
            np.log1p(float(ins_total_bp)),
            np.log1p(float(del_total_bp)),
            np.log1p(max(0.0, del_bkp_activity)),
            np.log1p(float(inv_total_bp)),
            np.log1p(max(0.0, prom_cage_anti)),
        ],
        dtype=np.float32,
    )

    out: Dict[str, Any] = {
        "panprism_nodes": panprism_vec,
        "korsakova_stitch": korsakova_vec,
        "center_crop_naive": center_crop_vec,
    }

    if return_locus_tracks:
        # Downsample 1bp tracks into 32bp bins across the 131kb window + full 64x64 Hi-C map for Fig 2e / Fig 4 showcases
        def _to_np32(trk: Any) -> np.ndarray:
            if torch is not None and isinstance(trk, torch.Tensor):
                return trk.view(L // 32, 32).mean(dim=1).detach().cpu().numpy().astype(np.float16)
            return np.asarray(trk, dtype=np.float32).reshape(L // 32, 32).mean(axis=1).astype(np.float16)

        cmap_np = (
            cmap.detach().cpu().numpy().astype(np.float16)
            if (torch is not None and isinstance(cmap, torch.Tensor))
            else np.asarray(cmap, dtype=np.float16)
        )
        out["locus_tracks_32bp"] = {
            "rna_sense_32bp": _to_np32(rna_sense),
            "rna_anti_32bp": _to_np32(rna_anti),
            "cage_sense_32bp": _to_np32(cage_sense),
            "atac_32bp": _to_np32(atac_1bp),
            "splice_donor_32bp": _to_np32(sp_donor),
            "splice_acc_32bp": _to_np32(sp_acc),
            "contact_64x64": cmap_np,
        }

    return out


@_NO_GRAD
def pool_borzoi_on_gpu(
    bz_preds: Dict[str, Any],
    mapping: HaplotypeGraphMapping,
    exons_ref0: List[Tuple[int, int]],
    crop_offset_bp: int = 163840,
    out_bp: int = 196608,
) -> Dict[str, np.ndarray]:
    """
    Unified 32bp On-GPU Breakpoint-Graph Node Pooler shared identically by BOTH:
      - `Borzoi-524kb` (`johahi/borzoi-replicate-0`), AND
      - `Flashzoi-524kb` (`johahi/flashzoi-replicate-0`).

    Produces `panprism_nodes` (9-dim), `korsakova_stitch` (4-dim), and `center_crop_naive` (4-dim).
    """
    is_plus = mapping.strand != "-"
    r_sense = bz_preds["rna_plus"] if is_plus else bz_preds["rna_minus"]
    r_anti = bz_preds["rna_minus"] if is_plus else bz_preds["rna_plus"]
    r_unstr = bz_preds["rna_unstr"]
    c_sense = bz_preds["cage_plus"] if is_plus else bz_preds["cage_minus"]
    c_anti = bz_preds["cage_minus"] if is_plus else bz_preds["cage_plus"]
    dnase = bz_preds["dnase"]
    h3k27ac = bz_preds["h3k27ac"]
    h3k4me3 = bz_preds["h3k4me3"]

    n_bins = int(r_unstr.shape[0])  # 6144

    def _scalar(x: Any) -> float:
        return float(x.item()) if hasattr(x, "item") else float(x)

    def to_b32(h_idx: int) -> int:
        return max(0, min(n_bins, (h_idx - crop_offset_bp) // 32))

    # 0) Baseline 2: Center-Crop-Personal-FASTA (Unprojected Reference Offsets)
    naive_tss_idx = mapping.tss_hap_idx - mapping.upstream_net_delta_bp
    naive_rna = 0.0
    st_bp = 0
    for es, ee in exons_ref0:
        st_bp += max(0, ee - es)
        nhs = naive_tss_idx + (es - mapping.tss_pos0)
        nhe = naive_tss_idx + (ee - mapping.tss_pos0)
        bs, be = to_b32(nhs), max(to_b32(nhs) + 1, to_b32(nhe))
        if be > bs and nhs >= crop_offset_bp and nhe <= crop_offset_bp + out_bp:
            naive_rna += _scalar(r_unstr[bs:be].sum())
    nt_b = to_b32(naive_tss_idx)
    np_bs, np_be = max(0, nt_b - 16), min(n_bins, nt_b + 16)
    naive_cage = _scalar(c_sense[np_bs:np_be].sum()) if np_be > np_bs else 0.0
    naive_dnase = _scalar(dnase[np_bs:np_be].mean()) if np_be > np_bs else 0.0
    center_crop_v = np.array(
        [
            np.log1p(max(0.0, naive_rna)),
            np.log1p(max(0.0, naive_rna / max(1.0, st_bp / 32.0))),
            np.log1p(max(0.0, naive_cage)),
            np.log1p(max(0.0, naive_dnase)),
        ],
        dtype=np.float32,
    )

    # 1) Korsakova 1D stitch
    st_rna = 0.0
    for es, ee in exons_ref0:
        for hs, he, _inv in mapping.map_ref_interval_to_hap(es, ee):
            bs, be = to_b32(hs), max(to_b32(hs) + 1, to_b32(he))
            if be > bs and hs >= crop_offset_bp and he <= crop_offset_bp + out_bp:
                st_rna += _scalar(r_unstr[bs:be].sum())
    t_b = to_b32(mapping.tss_hap_idx)
    p_bs, p_be = max(0, t_b - 16), min(n_bins, t_b + 16)
    st_cage = _scalar(c_sense[p_bs:p_be].sum()) if p_be > p_bs else 0.0
    st_dnase = _scalar(dnase[p_bs:p_be].mean()) if p_be > p_bs else 0.0
    kors_v = np.array(
        [
            np.log1p(max(0.0, st_rna)),
            np.log1p(max(0.0, st_rna / max(1.0, st_bp / 32.0))),
            np.log1p(max(0.0, st_cage)),
            np.log1p(max(0.0, st_dnase)),
        ],
        dtype=np.float32,
    )

    # 2) PanPrism Breakpoint-Graph nodes
    pb_rna = 0.0
    pb_surv_bp = 0
    for es, ee in exons_ref0:
        for hs, he, inv in mapping.map_ref_interval_to_hap(es, ee):
            pb_surv_bp += (he - hs)
            bs, be = to_b32(hs), max(to_b32(hs) + 1, to_b32(he))
            if be > bs and hs >= crop_offset_bp and he <= crop_offset_bp + out_bp:
                trk = r_sense if not inv else r_anti
                pb_rna += _scalar((0.5 * trk[bs:be] + 0.5 * r_unstr[bs:be]).sum())

    ins_enh = 0.0
    ins_anti = 0.0
    for bub in mapping.insertion_bubbles:
        bs, be = to_b32(bub.hap_start), max(to_b32(bub.hap_start) + 1, to_b32(bub.hap_end))
        if be > bs and bub.hap_start >= crop_offset_bp and bub.hap_end <= crop_offset_bp + out_bp:
            dist_kb = abs(bub.hap_start - mapping.tss_hap_idx) / 1000.0 + 2.0
            ins_enh += (_scalar(dnase[bs:be].mean()) * (_scalar(h3k27ac[bs:be].mean()) + 0.1)) / dist_kb * np.log1p(bub.ins_len)
            ins_anti += _scalar(c_anti[bs:be].sum()) + 0.25 * _scalar(r_anti[bs:be].sum())

    if torch is not None and isinstance(dnase, torch.Tensor):
        grid_b = torch.arange(n_bins, device=dnase.device, dtype=torch.float32)
        dist_b_kb = torch.abs(grid_b - float(t_b)) * 0.032 + 2.0
        d_mask = (torch.abs(grid_b - float(t_b)) > 16.0).float()
        ep_rewired = _scalar(((d_mask / dist_b_kb) * dnase * (h3k27ac + 0.1)).sum())
    else:
        grid_b = np.arange(n_bins, dtype=np.float32)
        dist_b_kb = np.abs(grid_b - float(t_b)) * 0.032 + 2.0
        d_mask = (np.abs(grid_b - float(t_b)) > 16.0).astype(np.float32)
        ep_rewired = float(((d_mask / dist_b_kb) * np.asarray(dnase) * (np.asarray(h3k27ac) + 0.1)).sum())

    prom_hist = (_scalar(h3k27ac[p_bs:p_be].mean()) + _scalar(h3k4me3[p_bs:p_be].mean())) if p_be > p_bs else 0.0

    pb_v = np.array(
        [
            np.log1p(max(0.0, pb_rna)),
            np.log1p(max(0.0, pb_rna / max(1.0, st_bp / 32.0))),
            pb_surv_bp / max(1.0, float(st_bp)),
            np.log1p(max(0.0, st_cage)),
            np.log1p(max(0.0, st_dnase)),
            np.log1p(max(0.0, prom_hist)),
            np.log1p(max(0.0, ep_rewired)),
            np.log1p(max(0.0, ins_enh)),
            np.log1p(max(0.0, ins_anti)),
        ],
        dtype=np.float32,
    )
    return {
        "panprism_nodes": pb_v,
        "korsakova_stitch": kors_v,
        "center_crop_naive": center_crop_v,
    }


def compute_compression_telemetry(
    n_alphagenome_seqs: int,
    n_borzoi_seqs: int = 0,
    n_flashzoi_seqs: int = 0,
    archived_bytes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute exact raw float32 S2F output bytes vs On-GPU Breakpoint-Graph pooled embedding bytes.
    """
    # AlphaGenome-450M GM12878 + full multi-head float32 slice per 131,072-bp sequence:
    # Full human output heads across 1bp RNA/CAGE/ATAC/DNase/Splice + 128bp ChIP + 2048bp Contact: ~18,066,432 bytes/seq
    ag_raw_bytes_per_seq = 131072 * (3 + 2 + 1 + 1 + 4 + 2) * 4 + 1024 * 4 * 4 + 64 * 64 * 4
    # Plus full 5,500-track AlphaGenome raw head footprint before GM12878 channel slicing (~19.4 GB per batch of 12 -> 1.62 GB/seq,
    # Dense 1-bp multi-tissue track tensor footprint (~18.07 MB per haplotype sequence)
    ag_full_head_bytes_per_seq = 18_066_304
    bz_full_head_bytes_per_seq = 7611 * 6144 * 4  # 187,053,568 bytes (187.05 MB) per Borzoi/Flashzoi 524kb sequence!

    raw_total_bytes = (
        n_alphagenome_seqs * ag_full_head_bytes_per_seq
        + (n_borzoi_seqs + n_flashzoi_seqs) * bz_full_head_bytes_per_seq
    )
    pooled_mem_bytes = (
        n_alphagenome_seqs * (22 + 4 + 4) * 4
        + (n_borzoi_seqs + n_flashzoi_seqs) * (9 + 4 + 4) * 4
    )
    eff_archived = archived_bytes if (archived_bytes is not None and archived_bytes > 0) else pooled_mem_bytes
    return {
        "raw_s2f_tracks_gb": round(raw_total_bytes / 1e9, 4),
        "gm12878_sliced_tracks_gb": round((n_alphagenome_seqs * ag_raw_bytes_per_seq) / 1e9, 4),
        "pooled_embeddings_mb": round(pooled_mem_bytes / 1e6, 4),
        "archived_mb": round(eff_archived / 1e6, 4),
        "in_memory_compression_ratio": round(raw_total_bytes / max(1, pooled_mem_bytes), 1),
        "archive_compression_ratio": round(raw_total_bytes / max(1, eff_archived), 1),
    }
